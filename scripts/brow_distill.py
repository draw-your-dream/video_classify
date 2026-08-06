#!/usr/bin/env python
"""眉毛蒸馏小模型(2026-08-06 预注册):把 VLM 的眉毛旗标蒸馏成帧级小 CNN。
训练集:正=e12_v4∪v5 眉毛旗标帧(上部62%放大裁剪);负=v5 覆盖且零旗标的 good 视频抽帧。
锚(H086/H089/H101/7060/7241/6472)一律排除出训练,只做检验。
模型:timm efficientnet_b0 预训练,3 epoch。评估:留出帧AUC + 锚视频级 max 概率。"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/root/mech")
ANCHORS = {"基础款/5719.mp4", "基础款/6565.mp4", "基础款/6228.mp4", "基础款/7060.mp4",
           "基础款/7241.mp4", "花花款/6472.mp4", "基础款/5822.mp4"}


def upper(im448):
    H, W = im448.shape[:2]
    u = im448[0:int(H * 0.62), :]
    return cv2.resize(u, (224, 224))


def load_flags():
    pos, clean = {}, set()
    for fp in ("data/e12_v4.jsonl", "data/e12_v5.jsonl"):
        for l in open(ROOT / fp):
            j = json.loads(l)
            if j.get("error") or j.get("parse_error"):
                continue
            rel = j["rel"]
            if j.get("eyebrows"):
                pos.setdefault(rel, set()).update(int(x) for x in j["eyebrows"] if isinstance(x, int) and 0 <= x < 16)
            else:
                clean.add(rel)
    return pos, clean


def main():
    labels = {}
    for l in (ROOT / "corpus_full.tsv").read_text().splitlines():
        if l.strip():
            r, lab = l.split("\t"); labels[r] = lab
    for l in (ROOT / "manifest_rlhf.tsv").read_text().splitlines():
        if l.strip():
            parts = l.split("\t"); labels.setdefault(parts[0], parts[1] if len(parts) > 1 else "?")
    pos, clean = load_flags()
    pos = {r: f for r, f in pos.items() if r not in ANCHORS}
    clean_good = [r for r in clean if r not in ANCHORS and labels.get(r) != "bad"]
    rng = random.Random(7)
    rng.shuffle(clean_good)

    samples = []  # (rel, frame_idx, y)
    for r, fs in pos.items():
        d = ROOT / "data/crops_v3" / r.replace(".mp4", "")
        jp = sorted(d.glob("f*.jpg"))
        for fi in fs:
            if fi < len(jp):
                samples.append((str(jp[fi]), 1))
    n_pos = len(samples)
    neg_needed = min(n_pos * 4, 60000)
    for r in clean_good:
        d = ROOT / "data/crops_v3" / r.replace(".mp4", "")
        jp = sorted(d.glob("f*.jpg"))
        for p in jp[::3]:
            samples.append((str(p), 0))
        if len(samples) - n_pos >= neg_needed:
            break
    print(f"正 {n_pos} 帧 / 负 {len(samples)-n_pos} 帧(来自 {len(pos)} 旗标视频)", flush=True)
    rng.shuffle(samples)
    n_val = max(200, len(samples) // 10)
    val, tr = samples[:n_val], samples[n_val:]

    import timm
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    pw = torch.tensor([(len(tr) - sum(y for _, y in tr)) / max(1, sum(y for _, y in tr))]).to("cuda")
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

    def batch(items):
        xs, ys = [], []
        for p, y in items:
            im = cv2.imread(p)
            if im is None:
                continue
            x = upper(im)[..., ::-1].astype(np.float32) / 255.0
            xs.append(((x - MEAN) / STD).transpose(2, 0, 1))
            ys.append(y)
        return (torch.tensor(np.stack(xs)).float().to("cuda"),
                torch.tensor(ys).float().to("cuda"))

    BS = 64
    for ep in range(3):
        model.train()
        rng.shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr), BS):
            x, y = batch(tr[i:i + BS])
            opt.zero_grad()
            out = model(x).squeeze(1)
            loss = lossf(out, y)
            loss.backward(); opt.step()
            tot += float(loss) * len(y)
        model.eval()
        ps, ys = [], []
        with torch.inference_mode():
            for i in range(0, len(val), BS):
                x, y = batch(val[i:i + BS])
                ps += torch.sigmoid(model(x).squeeze(1)).cpu().tolist()
                ys += y.cpu().tolist()
        ps, ys = np.array(ps), np.array(ys)
        from scipy.stats import rankdata
        r = rankdata(ps); n1 = ys.sum()
        auc = (r[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(ys) - n1))
        print(f"ep{ep}: loss {tot/len(tr):.4f}  留出帧AUC {auc:.4f}", flush=True)

    torch.save(model.state_dict(), ROOT / "data/brow_distill_b0.pt")
    # 锚检验:视频级 = 各帧概率 max / top3均值
    print("== 锚检验(未参与训练) ==", flush=True)
    with torch.inference_mode():
        for rel in sorted(ANCHORS):
            d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
            jp = sorted(d.glob("f*.jpg"))[:16]
            if not jp:
                print(f"  {rel}: 无裁剪"); continue
            x, _ = batch([(str(p), 0) for p in jp])
            pr = torch.sigmoid(model(x).squeeze(1)).cpu().numpy()
            top3 = float(np.sort(pr)[-3:].mean())
            print(f"  {rel}: max {pr.max():.3f} top3 {top3:.3f} 帧>0.5: {list(np.where(pr>0.5)[0])}", flush=True)
    print("BROW_DISTILL_DONE", flush=True)


if __name__ == "__main__":
    main()
