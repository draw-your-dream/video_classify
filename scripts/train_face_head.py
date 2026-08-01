#!/usr/bin/env python
"""线F v1:脸部合成缺陷监督头(2026-08-01 预注册)。

数据:bank_good 视频的脸裁剪(meta score>=0.5 的帧);评估侧不参与训练。
在线合成:p=0.55 脸缺陷族(spots/eyebrows/eye_extra/mouth_warp/smudge),
否则 benign(光度扰动+良性贴图,标签=好)。
模型:DINOv2-base 冻结 -> patch MLP;损失同线A(patch BCE + 图级 BCE)。
评估:帧分 top-16 patch 均值;视频分 = 可见脸帧的 p75 与 max;
AUC = bad vs eval_good(脸帧 >=3 的视频);报告覆盖率。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axis_rself_local import split_goods, stem_seed, auc  # noqa: E402
from synth_face_defects import FACE_DEFECTS, benign, synthesize_face  # noqa: E402
from train_synth_head import Head, load_backbone, patch_feats, to_tensor, frame_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-dir", default=str(ROOT / "data/face_crops"))
    ap.add_argument("--manifest", default=str(ROOT / "data/prod500/mech_subset.tsv"))
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "data/prod500/face_head_v1.pt"))
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
        d = Path(args.face_dir) / rel.replace(".mp4", "")
        train_frames += sorted(d.glob("f*.jpg"))
    print(f"train face frames: {len(train_frames)}", flush=True)

    backbone = load_backbone(device)
    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    rng = np.random.default_rng(stem_seed("face-train"))
    kinds = list(FACE_DEFECTS)
    GRID = 37

    def make_batch():
        xs, ms, ys = [], [], []
        while len(xs) < args.batch:
            p = train_frames[rng.integers(len(train_frames))]
            im = cv2.imread(str(p))
            if im is None:
                continue
            if rng.random() < 0.55:
                got = synthesize_face(im, kinds[rng.integers(len(kinds))], rng)
                if got is None:
                    continue
                im2, dm = got
                y = 1.0
            else:
                donor = cv2.imread(str(train_frames[rng.integers(len(train_frames))]))
                im2, dm = benign(im, rng, donor)
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
        feats = patch_feats(backbone, xs)
        logits = head(feats)
        w = torch.ones_like(ms)
        w[ms > 0] = 4.0
        l_patch = nn.functional.binary_cross_entropy_with_logits(logits, ms, weight=w)
        l_img = bce(logits.topk(16, dim=-1).values.mean(-1), ys)
        loss = l_patch + l_img
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)

    torch.save(head.state_dict(), args.out)
    print("head saved ->", args.out, flush=True)

    @torch.inference_mode()
    def video_score(rel):
        d = Path(args.face_dir) / rel.replace(".mp4", "")
        jpgs = sorted(d.glob("f*.jpg"))
        if len(jpgs) < 3:
            return None
        scores = []
        for i in range(0, len(jpgs), 8):
            ims = [cv2.imread(str(j)) for j in jpgs[i:i + 8]]
            xs = torch.stack([to_tensor(im) for im in ims if im is not None]).to(device)
            fs = frame_score(head(patch_feats(backbone, xs)))
            scores += fs.cpu().tolist()
        return float(np.percentile(scores, 75)), float(np.max(scores))

    res, cover = {}, {}
    for grp, rels in (("bad", bads), ("eval_good", eval_rels)):
        vals, n_tot = [], 0
        for k, rel in enumerate(rels):
            n_tot += 1
            s = video_score(rel)
            if s is not None:
                vals.append(s)
            if (k + 1) % 100 == 0:
                print(f"  eval {grp} {k+1}/{len(rels)}", flush=True)
        res[grp] = np.array(vals)
        cover[grp] = f"{len(vals)}/{n_tot}"
    for i, nme in ((0, "p75"), (1, "max")):
        a = auc(res["bad"][:, i], res["eval_good"][:, i])
        print(f"face_head {nme} AUC = {a:.3f} (覆盖 bad {cover['bad']} good {cover['eval_good']})")
    print("FACE_TRAIN_DONE")


if __name__ == "__main__":
    main()
