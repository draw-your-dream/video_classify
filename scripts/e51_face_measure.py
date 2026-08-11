#!/usr/bin/env python
"""E51 续:真 face_crops(SAM3 原生分辨率 384px)上的面部活动测量,全语料。
与试点完全同口径(phaseCorrelate 对齐 → 中心区 absdiff;暗像素占比波动=眨眼)。
输出 data/face_act_full.csv。纯 CPU 并行。
"""
from __future__ import annotations

import csv
import glob
import os

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
import cv2
import numpy as np

cv2.setNumThreads(1)
ROOT = "/root/mech"


def face_feats(d):
    fs = sorted(glob.glob(os.path.join(d, "f*.jpg")))
    if len(fs) < 8:
        return None
    gs = []
    for f in fs:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        gs.append(im.astype(np.float32))
    if len(gs) < 8:
        return None
    act, dark = [], []
    for i in range(len(gs) - 1):
        (dx, dy), _ = cv2.phaseCorrelate(gs[i], gs[i + 1])
        if abs(dx) > 60 or abs(dy) > 60:
            dx = dy = 0.0
        Mx = np.float32([[1, 0, dx], [0, 1, dy]])
        w = cv2.warpAffine(gs[i], Mx, (gs[i].shape[1], gs[i].shape[0]))
        c = slice(64, 320)
        act.append(float(np.abs(w[c, c] - gs[i + 1][c, c]).mean()))
    for g in gs:
        dark.append(float((g[64:320, 64:320] < 60).mean()))
    dark = np.array(dark)
    return dict(face_act=float(np.median(act)), face_act_p90=float(np.percentile(act, 90)),
                blink=float(dark.std() / max(1e-4, dark.mean())),
                dark_med=float(np.median(dark)), n_face=len(act))


def one(args):
    rel, d = args
    row = {"rel": rel}
    try:
        ff = face_feats(d)
        if ff:
            row.update(ff)
    except Exception:
        pass
    return row


def main():
    dirs = {os.path.relpath(p, f"{ROOT}/data/face_crops") + ".mp4": p
            for p in glob.glob(f"{ROOT}/data/face_crops/*/*") if os.path.isdir(p)}
    jobs = sorted(dirs.items())
    print(f"face_crops 目录 {len(jobs)}", flush=True)
    import multiprocessing as mp
    cols = ["face_act", "face_act_p90", "blink", "dark_med", "n_face"]
    with open(f"{ROOT}/data/face_act_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel"] + cols)
        with mp.Pool(48) as pool:
            for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=16)):
                w.writerow([row["rel"]] + [row.get(c, "") for c in cols])
                if (i + 1) % 500 == 0:
                    print(f"[{i+1}/{len(jobs)}]", flush=True)
    print("E51_FACE_MEASURE_DONE", flush=True)


if __name__ == "__main__":
    main()
