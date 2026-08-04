#!/usr/bin/env python
"""W4:通用+细节美学/质量分补充(2026-08-04,规则线美学辅助 + E13 家族)。

pyiqa 四个无参考指标逐帧打分:musiq(细节质量) clipiqa(语义质量)
brisque(传统统计,低=好) niqe(自然度,低=好)。每指标取 mean/min(或max) → 8 列。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

COLS = "musiq_mean musiq_min clipiqa_mean clipiqa_min brisque_mean brisque_max niqe_mean niqe_max".split()


def read_frames(vp, n_frames=8, side=448):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        cap.release(); return []
    idxs = {int(round(i * (n - 1) / (n_frames - 1))) for i in range(n_frames)}
    out, k = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in idxs:
            h, w = fr.shape[:2]
            s = side / max(h, w)
            fr = cv2.resize(fr, (int(w * s), int(h * s)))
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        k += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/root/mech/data/corpus_videos")
    ap.add_argument("--manifest", default="/root/mech/manifest_all.tsv")
    ap.add_argument("--out", default="/root/mech/data/w4_pyiqa.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rels = [l.split("\t")[0] for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    done = set()
    out_p = Path(args.out)
    if out_p.exists():
        done = {r[0] for r in csv.reader(open(out_p))}
    else:
        csv.writer(open(out_p, "w", newline="")).writerow(["rel"] + COLS)
    todo = [r for r in rels if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"todo {len(todo)}", flush=True)

    import pyiqa
    dev = "cuda"
    M = {name: pyiqa.create_metric(name, device=dev) for name in ("musiq", "clipiqa", "brisque", "niqe")}
    print("metrics loaded", flush=True)

    import time
    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    for k, rel in enumerate(todo):
        row = None
        try:
            ims = read_frames(Path(args.videos_dir) / rel)
            if len(ims) >= 4:
                x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(dev)
                vals = {}
                with torch.inference_mode():
                    for name, m in M.items():
                        s = m(x)
                        vals[name] = s.detach().float().cpu().numpy().reshape(-1)
                ft = dict(musiq_mean=vals["musiq"].mean(), musiq_min=vals["musiq"].min(),
                          clipiqa_mean=vals["clipiqa"].mean(), clipiqa_min=vals["clipiqa"].min(),
                          brisque_mean=vals["brisque"].mean(), brisque_max=vals["brisque"].max(),
                          niqe_mean=vals["niqe"].mean(), niqe_max=vals["niqe"].max())
                row = [f"{ft[c]:.5g}" for c in COLS]
        except Exception as e:
            if k < 3:
                print("ERR", rel, repr(e)[:120], flush=True)
        w.writerow([rel] + (row if row else ["nan"] * len(COLS)))
        if (k + 1) % 100 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    f.close()
    print("W4_DONE", flush=True)


if __name__ == "__main__":
    main()
