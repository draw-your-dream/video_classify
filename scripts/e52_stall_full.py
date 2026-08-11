#!/usr/bin/env python
"""E52:卡顿/时间箭头 全帧率帧差电池(2026-08-11,自主新想法批)。

机理:卡顿=帧差序列双峰(大量近零帧+突跳)——16 帧采样看不见(弹跳谱已证混叠),
须在原始视频全帧率(约150帧)上测;时间箭头=活物动作"预备-爆发-缓冲"不对称,
刚体漂移时间对称。128px 灰度,全帧 + 角色区(geo bbox 插值)双通道。
纯 CPU 48 进程。输出 data/stall_full.csv。
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
BASE = "still_frac spike_ratio jump_cnt bimod asym acf1 hf_ratio".split()
COLS = BASE + ["c_" + k for k in BASE] + ["n_fr"]


def series_feats(d):
    d = np.asarray(d, float)
    if len(d) < 30:
        return {}
    med = float(np.median(d[d > 0])) if (d > 0).any() else 1e-6
    out = {}
    out["still_frac"] = float((d < 0.15 * med).mean())
    out["spike_ratio"] = float(np.percentile(d, 99) / max(1e-9, np.percentile(d, 50)))
    out["jump_cnt"] = float((d > 4 * med).sum())
    h, _ = np.histogram(np.log(d + 1e-6), bins=24)
    h = h / max(1, h.sum())
    pk = [i for i in range(1, 23) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.03]
    bim = 0.0
    if len(pk) >= 2:
        bim = float(min(h[pk[0]], h[pk[-1]]) - h[pk[0]:pk[-1] + 1].min())
    out["bimod"] = bim
    s = np.convolve(d, np.ones(5) / 5, mode="same")
    pk2 = [i for i in range(3, len(s) - 3)
           if s[i] == s[max(0, i - 3):i + 4].max() and s[i] > 2 * med]
    asyms = []
    for i in pk2[:8]:
        l = r = 0
        while i - l - 1 >= 0 and s[i - l - 1] < s[i - l]:
            l += 1
        while i + r + 1 < len(s) and s[i + r + 1] < s[i + r]:
            r += 1
        if l + r > 2:
            asyms.append((r - l) / (l + r))
    out["asym"] = float(np.mean(np.abs(asyms))) if asyms else 0.0
    z = d - d.mean()
    ac = np.correlate(z, z, "full")[len(d) - 1:]
    out["acf1"] = float(ac[1] / max(1e-9, ac[0]))
    f = np.abs(np.fft.rfft(z)) ** 2
    out["hf_ratio"] = float(f[len(f) // 2:].sum() / max(1e-9, f[1:].sum()))
    return out


def one(args):
    rel, geo = args
    row = {"rel": rel}
    vp = os.path.join(ROOT, "data/corpus_videos", rel)
    if not os.path.exists(vp):
        return row
    try:
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 30:
            cap.release()
            return row
        interp = None
        if geo and len(geo) >= 8:
            fr = np.array([x["frame"] for x in geo], float)
            fr = fr / max(1.0, fr.max()) * max(1, total - 1)
            B = np.array([x["bbox"] for x in geo], float)
            interp = (fr, B)
        prev = None
        dfull, dchar = [], []
        k = 0
        W0 = H0 = None
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if W0 is None:
                H0, W0 = im.shape[:2]
            g = cv2.cvtColor(cv2.resize(im, (160, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                dfull.append(float(np.abs(g - prev).mean()))
                if interp is not None:
                    b = [float(np.interp(k, interp[0], interp[1][:, j])) for j in range(4)]
                    x0 = max(0, min(155, int(b[0] / W0 * 160)))
                    y0 = max(0, min(91, int(b[1] / H0 * 96)))
                    x1 = min(160, max(x0 + 4, int(b[2] / W0 * 160)))
                    y1 = min(96, max(y0 + 4, int(b[3] / H0 * 96)))
                    dchar.append(float(np.abs(g[y0:y1, x0:x1] - prev[y0:y1, x0:x1]).mean()))
            prev = g
            k += 1
        cap.release()
        f1 = series_feats(dfull)
        row.update(f1)
        f2 = series_feats(dchar)
        row.update({"c_" + k2: v for k2, v in f2.items()})
        row["n_fr"] = len(dfull)
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
    print(f"任务 {len(jobs)} (含geo {sum(1 for _, g in jobs if g)})", flush=True)
    import multiprocessing as mp
    with open(f"{ROOT}/data/stall_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel"] + COLS)
        with mp.Pool(48) as pool:
            for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
                w.writerow([row["rel"]] + [row.get(c, "") for c in COLS])
                if (i + 1) % 300 == 0:
                    print(f"[{i+1}/{len(jobs)}]", flush=True)
    print("E52_STALL_DONE", flush=True)


if __name__ == "__main__":
    main()
