#!/usr/bin/env python
"""E51:软硬度+面部活动 全语料测量(2026-08-11 预注册,纯 CPU 并行)。

轮廓软度:sam3_cutouts 白底掩码 → 逐对相似变换对齐(矩) → 1-IoU 残差(非刚性形变)。
面部活动:crops_v3 上半区 → phaseCorrelate 对齐 → 中心区 absdiff + 暗像素占比波动。
输出 data/softness_full.csv(每视频 11 列)。
"""
from __future__ import annotations

import csv
import glob
import os
import sys

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
import cv2
import numpy as np

cv2.setNumThreads(1)

ROOT = "/root/mech"
COLS = ("soft_med soft_p90 soft_when_move sticker mot_med "
        "face_act face_act_p90 blink dark_med n_mask n_face").split()


def mask_of(img):
    m = (~((img > 240).all(axis=2))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab2, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = 1 + np.argmax(stats[1:, 4])
    return (lab2 == k).astype(np.uint8)


def sim_params(m):
    M = cv2.moments(m)
    if M["m00"] < 50:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    a = M["mu20"] / M["m00"]; b2 = M["mu11"] / M["m00"]; c = M["mu02"] / M["m00"]
    theta = 0.5 * np.arctan2(2 * b2, (a - c))
    return cx, cy, np.sqrt(M["m00"]), theta


def soft_feats(d):
    fs = sorted(glob.glob(os.path.join(d, "f*.jpg")))
    if len(fs) < 8:
        return None
    ms = []
    for f in fs:
        im = cv2.imread(f)
        if im is None:
            return None
        m = mask_of(im)
        if m is None:
            return None
        ms.append(m)
    res, mot = [], []
    H, W = ms[0].shape
    for i in range(len(ms) - 1):
        p0, p1 = sim_params(ms[i]), sim_params(ms[i + 1])
        if p0 is None or p1 is None:
            continue
        s = p1[2] / max(1e-6, p0[2])
        dth = np.degrees(p1[3] - p0[3])
        if abs(dth) > 45:
            dth = 0.0
        Mx = cv2.getRotationMatrix2D((p0[0], p0[1]), -dth, s)
        Mx[0, 2] += p1[0] - p0[0]
        Mx[1, 2] += p1[1] - p0[1]
        w = cv2.warpAffine(ms[i], Mx, (W, H), flags=cv2.INTER_NEAREST)
        inter = np.logical_and(w, ms[i + 1]).sum()
        union = np.logical_or(w, ms[i + 1]).sum()
        res.append(1 - inter / max(1, union))
        mot.append(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / np.sqrt(ms[i].sum())
                   + abs(s - 1) + abs(np.radians(dth)))
    if len(res) < 6:
        return None
    res = np.array(res); mot = np.array(mot)
    hi = mot > np.median(mot)
    return dict(soft_med=float(np.median(res)), soft_p90=float(np.percentile(res, 90)),
                soft_when_move=float(np.median(res[hi])) if hi.sum() >= 3 else float(np.median(res)),
                sticker=float(np.median(mot) / max(1e-4, np.median(res))),
                mot_med=float(np.median(mot)), n_mask=len(res))


def face_feats(d):
    """crops_v3 上半 45% 作为脸区代理(角色紧裁剪,头在上部)。"""
    fs = sorted(glob.glob(os.path.join(d, "f*.jpg")))
    if len(fs) < 8:
        return None
    gs = []
    for f in fs:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        H, W = im.shape
        u = im[:int(H * 0.45), :]
        u = cv2.resize(u, (256, 128)).astype(np.float32)
        gs.append(u)
    act, dark = [], []
    for i in range(len(gs) - 1):
        (dx, dy), _ = cv2.phaseCorrelate(gs[i], gs[i + 1])
        if abs(dx) > 40 or abs(dy) > 40:
            dx = dy = 0.0
        Mx = np.float32([[1, 0, dx], [0, 1, dy]])
        w = cv2.warpAffine(gs[i], Mx, (256, 128))
        c = (slice(16, 112), slice(32, 224))
        act.append(float(np.abs(w[c] - gs[i + 1][c]).mean()))
    for g in gs:
        dark.append(float((g[16:112, 32:224] < 60).mean()))
    dark = np.array(dark)
    return dict(face_act=float(np.median(act)), face_act_p90=float(np.percentile(act, 90)),
                blink=float(dark.std() / max(1e-4, dark.mean())), dark_med=float(np.median(dark)),
                n_face=len(act))


def one(args):
    rel, cdir, fdir = args
    row = {"rel": rel}
    try:
        sf = soft_feats(cdir) if cdir else None
        if sf:
            row.update(sf)
    except Exception:
        pass
    try:
        ff = face_feats(fdir) if fdir else None
        if ff:
            row.update(ff)
    except Exception:
        pass
    return row


def main():
    cuts = {os.path.relpath(p, f"{ROOT}/data/sam3_cutouts"): p
            for p in glob.glob(f"{ROOT}/data/sam3_cutouts/*/*") if os.path.isdir(p)}
    crops = {os.path.relpath(p, f"{ROOT}/data/crops_v3"): p
             for p in glob.glob(f"{ROOT}/data/crops_v3/*/*") if os.path.isdir(p)}
    keys = sorted(set(cuts) | set(crops))
    jobs = [(k + ".mp4", cuts.get(k), crops.get(k)) for k in keys]
    print(f"cutouts {len(cuts)} | crops {len(crops)} | 任务 {len(jobs)}", flush=True)
    import multiprocessing as mp
    out = f"{ROOT}/data/softness_full.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel"] + COLS)
        with mp.Pool(48) as pool:
            for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=16)):
                w.writerow([row["rel"]] + [row.get(c, "") for c in COLS])
                if (i + 1) % 500 == 0:
                    print(f"[{i+1}/{len(jobs)}]", flush=True)
    print("E51_SOFTNESS_DONE", flush=True)


if __name__ == "__main__":
    main()
