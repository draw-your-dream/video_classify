#!/usr/bin/env python
"""线A v0:合成缺陷监督轻头(2026-08-01 预注册)。

数据:bank_good 视频的抠像帧(评估侧 eval_good/bad 绝不参与训练)。
在线合成:每帧 p=0.5 注入随机缺陷(6族),否则 50% 良性贴图 / 50% 原图(标签=好)。
模型:DINOv2-base(本地缓存)冻结 -> 每 patch 2层MLP 缺陷 logit。
损失:patch 级 BCE(合成 mask 下采样 37x37)+ 图级 BCE(top-16 patch 均值)。
评估:帧分 = top-16 patch 概率均值;视频分 = 帧分 p75;AUC = 464 bad vs 156 eval_good。
晋级关 AUC>=0.70(真实 bad 上);合成数据只在训练侧。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axis_rself_local import split_goods, stem_seed, auc  # noqa: E402
from synth_defects import DEFECTS, benign_paste, char_mask, synthesize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID = 37
TOPK = 16
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(im_bgr: np.ndarray) -> torch.Tensor:
    im = cv2.resize(im_bgr, (518, 518), interpolation=cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im = (im - MEAN) / STD
    return torch.from_numpy(im.transpose(2, 0, 1))


class Head(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_backbone(device):
    from transformers import AutoModel
    local = ROOT / ".hf_cache/dinov2-base-ms"
    src = str(local) if local.exists() else "facebook/dinov2-base"
    m = AutoModel.from_pretrained(src, torch_dtype=torch.float16).to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def patch_feats(backbone, xs: torch.Tensor) -> torch.Tensor:
    out = backbone(pixel_values=xs.half()).last_hidden_state[:, 1:, :]
    return out.float()


def frame_score(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    return p.topk(TOPK, dim=-1).values.mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut-dir", default=str(ROOT / "data/sam3_cutouts"))
    ap.add_argument("--manifest", default=str(ROOT / "data/prod500/mech_subset.tsv"))
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "data/prod500/synth_head_v0.pt"))
    args = ap.parse_args()
    device = "cuda"

    rows = [l.split("\t") for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    goods_by_style = defaultdict(list)
    for rel, label in rows:
        if label == "good":
            goods_by_style[rel.split("/")[0]].append(rel)
    bank_rels, eval_rels = split_goods(goods_by_style)
    bads = [rel for rel, label in rows if label != "good"]

    train_frames = []
    for rel in bank_rels:
        train_frames += sorted((Path(args.cut_dir) / rel.replace(".mp4", "")).glob("f*.jpg"))
    print(f"train frames: {len(train_frames)} (bank {len(bank_rels)} videos)", flush=True)

    backbone = load_backbone(device)
    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    rng = np.random.default_rng(stem_seed("synth-train"))
    kinds = list(DEFECTS)

    def make_batch():
        xs, ms, ys = [], [], []
        while len(xs) < args.batch:
            p = train_frames[rng.integers(len(train_frames))]
            im = cv2.imread(str(p))
            if im is None:
                continue
            r = rng.random()
            if r < 0.5:
                got = synthesize(im, kinds[rng.integers(len(kinds))], rng)
                if got is None:
                    continue
                im2, dm = got
                y = 1.0
            elif r < 0.75:
                cm = char_mask(im)
                got = benign_paste(im, cm, rng)
                if got is None:
                    continue
                im2, dm = got
                y = 0.0
            else:
                im2, dm = im, np.zeros(im.shape[:2], np.float32)
                y = 0.0
            xs.append(to_tensor(im2))
            mm = cv2.resize(dm, (GRID, GRID), interpolation=cv2.INTER_AREA)
            ms.append(torch.from_numpy((mm > 0.15).astype(np.float32).reshape(-1)))
            ys.append(y)
        return (torch.stack(xs).to(device), torch.stack(ms).to(device),
                torch.tensor(ys, device=device))

    bce = nn.BCEWithLogitsLoss()
    for step in range(1, args.steps + 1):
        xs, ms, ys = make_batch()
        with torch.no_grad():
            feats = patch_feats(backbone, xs)
        logits = head(feats)
        # patch 级:正样本帧内 正patch权重升,负patch照学
        w = torch.ones_like(ms)
        w[ms > 0] = 4.0
        l_patch = nn.functional.binary_cross_entropy_with_logits(logits, ms, weight=w)
        l_img = bce(logits.topk(TOPK, dim=-1).values.mean(-1), ys)
        loss = l_patch + l_img
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} "
                  f"(patch {l_patch.item():.4f} img {l_img.item():.4f})", flush=True)

    torch.save(head.state_dict(), args.out)
    print("head saved ->", args.out, flush=True)

    # ---- 评估:真实 bad vs eval_good ----
    @torch.inference_mode()
    def video_score(rel):
        d = Path(args.cut_dir) / rel.replace(".mp4", "")
        jpgs = sorted(d.glob("f*.jpg"))
        if len(jpgs) < 4:
            return None
        scores = []
        for i in range(0, len(jpgs), 8):
            ims = [cv2.imread(str(j)) for j in jpgs[i:i + 8]]
            xs = torch.stack([to_tensor(im) for im in ims if im is not None]).to(device)
            fs = frame_score(head(patch_feats(backbone, xs)))
            scores += fs.cpu().tolist()
        return float(np.percentile(scores, 75)), float(np.max(scores))

    res = {}
    for grp, rels in (("bad", bads), ("eval_good", eval_rels)):
        vals = []
        for k, rel in enumerate(rels):
            s = video_score(rel)
            if s is not None:
                vals.append(s)
            if (k + 1) % 100 == 0:
                print(f"  eval {grp} {k+1}/{len(rels)}", flush=True)
        res[grp] = np.array(vals)
    for i, nme in ((0, "p75"), (1, "max")):
        a = auc(res["bad"][:, i], res["eval_good"][:, i])
        print(f"synth_head {nme} AUC = {a:.3f} "
              f"(bad {len(res['bad'])} / good {len(res['eval_good'])})")
    print("SYNTH_TRAIN_DONE")


if __name__ == "__main__":
    main()
