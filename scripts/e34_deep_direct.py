#!/usr/bin/env python
"""E34:端到端深监督直判(2026-08-07 预注册)。绕开320特征瓶颈,从原始帧直接学。
输入:crops_v3 的 8 帧(角色为中心,原画面裁剪)→ timm 主干逐帧编码 → 时序池化 → 多任务头。
多任务:主头 bad/good + 6 个缺陷类型辅助头(僵硬/卡顿/少动/还原/物理/画面,只在 train 有 reason 时监督)。
协议:train_v3 内 5 折 OOF(无泄漏),报 gn@95 与并栈增量;过门 0.3218 才谈 eval。
变体:--arch {effb0, convnext_tiny} --frames {8,12} --aux {0,1}"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/root/mech")
GROUPS = {
    "僵硬": ["僵硬"], "卡顿": ["卡顿/少活人感", "动作位移不连贯"],
    "少动": ["四肢不动", "静止不动", "运动主体", "慢动作"],
    "还原": ["还原度", "衣服/身体的时间一致性", "大小变化"],
    "物理": ["物理规律", "不合理的物体"], "画面": ["帧跳变", "首帧一致", "背景运动混乱"],
}
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


class Net(nn.Module):
    def __init__(self, arch, n_aux):
        super().__init__()
        import timm
        self.bb = timm.create_model(arch, pretrained=True, num_classes=0)
        d = self.bb.num_features
        self.head = nn.Linear(d * 2, 1)
        self.aux = nn.Linear(d * 2, n_aux) if n_aux else None

    def forward(self, x):  # x: (B,T,3,H,W)
        B, T = x.shape[:2]
        f = self.bb(x.flatten(0, 1)).view(B, T, -1)
        z = torch.cat([f.mean(1), f.max(1).values], 1)
        return self.head(z).squeeze(1), (self.aux(z) if self.aux is not None else None)


def load_clip(rel, n_frames, size=224, train=False):
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    jp = sorted(d.glob("f*.jpg"))
    if len(jp) < 4:
        return None
    idx = np.linspace(0, len(jp) - 1, n_frames).round().astype(int)
    xs = []
    flip = train and random.random() < 0.5
    for i in idx:
        im = cv2.imread(str(jp[i]))
        if im is None:
            return None
        im = cv2.resize(im, (size, size))
        if flip:
            im = im[:, ::-1]
        x = im[..., ::-1].astype(np.float32) / 255.0
        xs.append(((x - MEAN) / STD).transpose(2, 0, 1))
    return np.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--aux", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="e34")
    args = ap.parse_args()

    tr = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    reasons = {r["path"]: r["reasons"] for r in csv.DictReader(
        open(ROOT / "data/merged_labels.csv", encoding="utf-8-sig"))} if (ROOT / "data/merged_labels.csv").exists() else {}
    if not reasons:
        reasons = {r["path"]: r["reasons"] for r in csv.DictReader(
            open(ROOT / "data/s3/merged_labels.csv", encoding="utf-8-sig"))}
    rel_of = {}
    for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
        if l.strip():
            rel = l.split("\t")[0]
            rel_of[os.path.basename(rel)] = rel
    items = []
    for r in tr:
        rel = rel_of.get(r["video"])
        if not rel:
            continue
        y = 1 if r["label"] == "bad" else 0
        rs = reasons.get(r["video"], "")
        aux = [1.0 if (y == 1 and any(t in rs for t in tags)) else 0.0 for tags in GROUPS.values()]
        items.append((rel, y, aux))
    print(f"样本 {len(items)},bad {sum(1 for _, y, _ in items if y)}", flush=True)

    y_all = np.array([y for _, y, _ in items])
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=42).split(np.zeros(len(items)), y_all))
    oof = np.full(len(items), np.nan)

    def make_batch(idxs, train):
        xs, ys, aux = [], [], []
        for i in idxs:
            rel, y, a = items[i]
            c = load_clip(rel, args.frames, train=train)
            if c is None:
                continue
            xs.append(c); ys.append(y); aux.append(a)
        if not xs:
            return None
        return (torch.tensor(np.stack(xs)).float().cuda(),
                torch.tensor(ys).float().cuda(),
                torch.tensor(np.array(aux)).float().cuda())

    for fi, (a_idx, b_idx) in enumerate(folds):
        net = Net(args.arch, len(GROUPS) if args.aux else 0).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
        pw = torch.tensor([(len(a_idx) - y_all[a_idx].sum()) / max(1, y_all[a_idx].sum())]).cuda()
        lf = nn.BCEWithLogitsLoss(pos_weight=pw)
        lf_aux = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda")
        for ep in range(args.epochs):
            net.train()
            order = list(a_idx); random.shuffle(order)
            tot, nb = 0.0, 0
            for s in range(0, len(order), args.bs):
                bt = make_batch(order[s:s + args.bs], True)
                if bt is None:
                    continue
                x, y, a = bt
                opt.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out, aout = net(x)
                    loss = lf(out.float(), y)
                    if aout is not None:
                        loss = loss + 0.3 * lf_aux(aout.float(), a)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += float(loss); nb += 1
            print(f"  fold{fi} ep{ep}: loss {tot/max(1,nb):.4f}", flush=True)
        net.eval()
        with torch.inference_mode():
            for s in range(0, len(b_idx), args.bs):
                sel = list(b_idx[s:s + args.bs])
                bt = make_batch(sel, False)
                if bt is None:
                    continue
                x, _, _ = bt
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out, _ = net(x)
                pr = torch.sigmoid(out.float()).cpu().numpy()
                for j, i in enumerate(sel[:len(pr)]):
                    oof[i] = pr[j]
        ok = np.isfinite(oof)
        print(f"fold{fi} 累计OOF gn@95 = {gn(oof[ok], y_all[ok]):.4f} (覆盖{ok.sum()})", flush=True)

    ok = np.isfinite(oof)
    from scipy.stats import rankdata
    r = rankdata(oof[ok]); n1 = y_all[ok].sum()
    auc = (r[y_all[ok] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(r) - n1))
    print(f"[{args.tag}] 单独 OOF gn@95 = {gn(oof[ok], y_all[ok]):.4f}  AUC = {auc:.4f}", flush=True)
    np.save(ROOT / f"data/{args.tag}_oof.npy", oof)
    json.dump([it[0] for it in items], open(ROOT / f"data/{args.tag}_rels.json", "w"))
    print("E34_DONE", flush=True)


if __name__ == "__main__":
    main()
