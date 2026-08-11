#!/usr/bin/env python
"""E57:死寂窗口——静止的质量(2026-08-11 深夜,自主批三)。

机理:still_frac 失败因为它数"静止多少";但 good 的静止是活的(呼吸/微晃,微运动非零),
卡顿 bad 的静止是像素级冻结(帧复制,只剩编码噪声)。
测量:全帧率角色区帧差能量序列上,长度 W 的滑动窗口取均值的最小值 = 最死窗口能量;
再除以视频自身运动中位数得相对死寂度。W∈{8,15,30}(约0.25/0.5/1秒)。
另测:死窗内的帧间差是否达到编码噪声地板(与该视频静态背景区的帧差同量级=真冻结)。
输出 data/deadwin_full.csv。纯 CPU 48 进程。
"""
from __future__ import annotations

import csv
import glob
import json
import os

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
import cv2
import numpy as np

cv2.setNumThreads(1)
ROOT = "/root/mech"
COLS = ("dead8 dead15 dead30 rel_dead15 noise_floor dead_vs_floor n_fr").split()


def one(args):
    rel, geo = args
    row = {"rel": rel}
    vp = os.path.join(ROOT, "data/corpus_videos", rel)
    if not os.path.exists(vp):
        return row
    try:
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 40:
            cap.release()
            return row
        interp = None
        if geo and len(geo) >= 8:
            fr = np.array([x["frame"] for x in geo], float)
            fr = fr / max(1.0, fr.max()) * max(1, total - 1)
            B = np.array([x["bbox"] for x in geo], float)
            interp = (fr, B)
        prev = None
        dchar, dbg = [], []
        k = 0
        H0 = W0 = None
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if W0 is None:
                H0, W0 = im.shape[:2]
            g = cv2.cvtColor(cv2.resize(im, (320, 192)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                if interp is not None:
                    b = [float(np.interp(k, interp[0], interp[1][:, j])) for j in range(4)]
                    x0 = max(0, min(315, int(b[0] / W0 * 320)))
                    y0 = max(0, min(187, int(b[1] / H0 * 192)))
                    x1 = min(320, max(x0 + 4, int(b[2] / W0 * 320)))
                    y1 = min(192, max(y0 + 4, int(b[3] / H0 * 192)))
                    dchar.append(float(np.abs(g[y0:y1, x0:x1] - prev[y0:y1, x0:x1]).mean()))
                    # 背景 = 去掉 bbox 的边框区(取四条边带)
                    m = np.ones_like(g, bool)
                    m[y0:y1, x0:x1] = False
                    dbg.append(float(np.abs(g[m] - prev[m]).mean()))
                else:
                    dchar.append(float(np.abs(g - prev).mean()))
                    dbg.append(0.0)
            prev = g
            k += 1
        cap.release()
        d = np.asarray(dchar, float)
        if len(d) < 35:
            return row
        med = float(np.median(d))
        for W in (8, 15, 30):
            if len(d) >= W:
                sw = np.convolve(d, np.ones(W) / W, mode="valid")
                row[f"dead{W}"] = float(sw.min())
        row["rel_dead15"] = float(row.get("dead15", np.nan) / max(1e-6, med))
        bg = np.asarray(dbg, float)
        # 噪声地板 = 背景帧差的 p10(该视频编码噪声量级)
        nf = float(np.percentile(bg[bg > 0], 10)) if (bg > 0).any() else 0.0
        row["noise_floor"] = nf
        row["dead_vs_floor"] = float(row.get("dead15", np.nan) / max(1e-6, nf))
        row["n_fr"] = len(d)
    except Exception:
        pass
    return row


def main():
    geos = {}
    for p in glob.glob(f"{ROOT}/data/sam3_feat/*/*.npz"):
        rel = "/".join(p.split(os.sep)[-2:]).replace(".npz", ".mp4")
        try:
            z = np.load(p, allow_pickle=True)
            g = json.loads(str(z["geo"]))
            if isinstance(g, list):
                geos[rel] = g
        except Exception:
            pass
    rels = [l.split("\t")[0] for l in open(f"{ROOT}/manifest_all.tsv") if l.strip()]
    jobs = [(r, geos.get(r)) for r in rels]
    print(f"任务 {len(jobs)}", flush=True)
    import multiprocessing as mp
    with open(f"{ROOT}/data/deadwin_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel"] + COLS)
        with mp.Pool(48) as pool:
            for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
                w.writerow([row["rel"]] + [row.get(c, "") for c in COLS])
                if (i + 1) % 300 == 0:
                    print(f"[{i+1}/{len(jobs)}]", flush=True)
    print("E57_DEADWIN_DONE", flush=True)


if __name__ == "__main__":
    main()
