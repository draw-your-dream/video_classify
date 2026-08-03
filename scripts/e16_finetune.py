#!/usr/bin/env python
"""E16:视频分类器端到端微调(2026-08-03 预注册,VMBench-CAS 配方)。

底座:transformers 原生 VideoMAE(MCG-NJU/videomae-base,16帧224)。
--fold 0/1:按 train 分层二折,训一半、给另一半打分(拼成无泄漏 train 分数);
--fold full:全 train 训练,给 eval_v3 打分(仅门检通过后使用)。
输出 csv: video,score 追加写。"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/root/mech")
N_FRAMES = 16
SIDE = 224


def read_clip(vp):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        cap.release(); return None
    idxs = [int(round(i * (n - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)]
    want = set(idxs); frames = {}; k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            fr = cv2.resize(fr, (SIDE, SIDE))
            frames[k] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        k += 1
    cap.release()
    if not frames:
        return None
    last = None
    out = []
    for i in idxs:
        if i in frames:
            last = frames[i]
        out.append(last if last is not None else next(iter(frames.values())))
    return np.stack(out).astype(np.float32) / 255.0


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class VidDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        clip = read_clip(ROOT / "data/corpus_videos" / rel)
        if clip is None:
            clip = np.zeros((N_FRAMES, SIDE, SIDE, 3), np.float32)
        clip = (clip - MEAN) / STD
        x = torch.from_numpy(clip.transpose(0, 3, 1, 2))  # (T,C,H,W)
        return x, torch.tensor(float(y)), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=["0", "1", "full", "smoke"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data/e16_scores.csv"))
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    rel_by_base = {}
    for f in ("corpus_full.tsv", "manifest_rlhf.tsv"):
        for l in (ROOT / f).read_text().splitlines():
            if l.strip():
                r = l.split("\t")[0]
                rel_by_base[r.rsplit("/", 1)[-1]] = r
    tr = [(json.loads(l)["video"], 1 if json.loads(l)["label"] == "bad" else 0)
          for l in open(ROOT / "train_v3.jsonl")]
    ev = [(json.loads(l)["video"], 1 if json.loads(l)["label"] == "bad" else 0)
          for l in open(ROOT / "eval_v3.jsonl")]
    tr = [(rel_by_base[v], y) for v, y in tr if v in rel_by_base]
    ev = [(rel_by_base[v], y) for v, y in ev if v in rel_by_base]
    rng = random.Random(42)
    bads = [t for t in tr if t[1] == 1]; goods = [t for t in tr if t[1] == 0]
    rng.shuffle(bads); rng.shuffle(goods)
    half_a = bads[::2] + goods[::2]
    half_b = bads[1::2] + goods[1::2]
    if args.fold == "0":
        train_set, score_set = half_a, half_b
    elif args.fold == "1":
        train_set, score_set = half_b, half_a
    elif args.fold == "full":
        train_set, score_set = tr, ev
    else:
        train_set, score_set = tr[:24], tr[24:32]
        args.epochs = 1
    rng.shuffle(train_set)
    print(f"fold={args.fold} train {len(train_set)} score {len(score_set)}", flush=True)

    from transformers import VideoMAEForVideoClassification
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base", num_labels=1, dtype=torch.bfloat16,
        problem_type="regression").to(device)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    steps = args.epochs * (len(train_set) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps))
    bce = torch.nn.BCEWithLogitsLoss()

    dl = DataLoader(VidDS(train_set), batch_size=args.batch, shuffle=True,
                    num_workers=6, pin_memory=True, drop_last=True)
    model.train()
    import time
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        for x, y, _ in dl:
            x = x.to(device, dtype=torch.bfloat16)
            y = y.to(device)
            logits = model(pixel_values=x).logits.squeeze(-1).float()
            loss = bce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                print(f"ep{ep} step{step}/{steps} loss {loss.item():.4f} "
                      f"{(time.time()-t0)/60:.1f}m", flush=True)
    print("train done", flush=True)

    model.eval()
    out_f = open(args.out, "a", newline="")
    w = csv.writer(out_f)
    dl2 = DataLoader(VidDS(score_set), batch_size=16, shuffle=False, num_workers=6)
    with torch.inference_mode():
        for x, y, idx in dl2:
            x = x.to(device, dtype=torch.bfloat16)
            p = torch.sigmoid(model(pixel_values=x).logits.squeeze(-1).float()).cpu().numpy()
            for i, pi in zip(idx.numpy(), p):
                w.writerow([score_set[i][0], f"{args.fold}", f"{pi:.6f}"])
    out_f.close()
    print(f"E16_FOLD_{args.fold}_DONE", flush=True)


if __name__ == "__main__":
    main()
