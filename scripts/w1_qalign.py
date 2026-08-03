#!/usr/bin/env python
"""W1:Q-Align 通用质量打分(2026-08-03 预注册,E13 现货打分器族)。

q-future/one-align(mPLUG-Owl2 底座,人类MOS训练)。视频口径:抽 8 帧,
用官方 score 接口按 image 批打分(quality 任务),取 mean/min/p25 三列;
另按 aesthetics 任务同样三列。csv 断点续跑。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

COLS = "q_mean q_min q_p25 a_mean a_min a_p25".split()


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
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        k += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/root/mech/data/corpus_videos")
    ap.add_argument("--manifest", default="/root/mech/manifest_all.tsv")
    ap.add_argument("--out", default="/root/mech/data/w1_qalign.csv")
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
    print(f"total {len(rels)} done {len(done)} todo {len(todo)}", flush=True)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "q-future/one-align", trust_remote_code=True,
        torch_dtype=torch.float16, device_map="cuda").eval()
    print("qalign loaded", flush=True)

    import time
    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    for k, rel in enumerate(todo):
        row = None
        try:
            ims = read_frames(Path(args.videos_dir) / rel)
            if len(ims) >= 4:
                with torch.inference_mode():
                    qs = model.score(ims, task_="quality", input_="image").cpu().float().numpy()
                    asc = model.score(ims, task_="aesthetics", input_="image").cpu().float().numpy()
                ft = dict(q_mean=qs.mean(), q_min=qs.min(), q_p25=np.percentile(qs, 25),
                          a_mean=asc.mean(), a_min=asc.min(), a_p25=np.percentile(asc, 25))
                row = [f"{ft[c]:.5g}" for c in COLS]
        except Exception as e:
            if k < 3:
                print("ERR", rel, repr(e)[:150], flush=True)
        w.writerow([rel] + (row if row else ["nan"] * len(COLS)))
        if (k + 1) % 100 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    f.close()
    print("W1_DONE", flush=True)


if __name__ == "__main__":
    main()
