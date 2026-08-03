#!/usr/bin/env python
"""E10 运动分解特征 v0(无掩码,2026-08-03 预注册)。

每相邻帧对:ORB+RANSAC 估全局相似变换(镜头运动;缩放|s-1|=拉近强度),
Farneback 稠密光流减去全局模型 = 残差(角色/主体)运动。
逐对:zoom、残差能量 res_mean、运动区占比 mov_frac、
     运动区方向离散度 coh(0=刚体式一致,1=关节式杂乱)、幅度变异 mag_cv。
视频级:各 mean/max + 卡顿度 jerk(残差能量帧间跳变)+ acf1 + 交互 zoom×刚体度。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

N_FRAMES = 16
MAX_SIDE = 448
MOV_THR = 0.4

COLS = ("zoom_mean zoom_max res_mean res_max mov_frac_mean mov_frac_max "
        "coh_mean coh_max magcv_mean jerk acf1 zoom_x_rigid n_pairs").split()


def read_frames(vp):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        cap.release(); return []
    idxs = sorted({int(round(i * (n - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)})
    out, k = [], 0
    want = set(idxs)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            h, w = fr.shape[:2]
            s = MAX_SIDE / max(h, w)
            g = cv2.cvtColor(cv2.resize(fr, (int(w * s), int(h * s))), cv2.COLOR_BGR2GRAY)
            out.append(g)
        k += 1
    cap.release()
    return out


def pair_feats(g0, g1, orb, bf):
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 12 or len(k1) < 12:
        return None
    ms = bf.match(d0, d1)
    if len(ms) < 12:
        return None
    src = np.float32([k0[m.queryIdx].pt for m in ms])
    dst = np.float32([k1[m.trainIdx].pt for m in ms])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=2.0)
    if M is None:
        return None
    scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    H, W = g0.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    gx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs
    gy = M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys
    rx, ry = flow[..., 0] - gx, flow[..., 1] - gy
    R = np.sqrt(rx ** 2 + ry ** 2)
    mov = R > MOV_THR
    res_mean = float(R.mean())
    mov_frac = float(mov.mean())
    if mov.sum() > 50:
        ux, uy = rx[mov] / (R[mov] + 1e-6), ry[mov] / (R[mov] + 1e-6)
        coh = float(1.0 - np.sqrt(ux.mean() ** 2 + uy.mean() ** 2))
        magcv = float(R[mov].std() / (R[mov].mean() + 1e-6))
    else:
        coh, magcv = np.nan, np.nan
    return dict(zoom=abs(scale - 1.0), res_mean=res_mean, mov_frac=mov_frac,
                coh=coh, magcv=magcv)


def video_feats(vp):
    gs = read_frames(vp)
    if len(gs) < 4:
        return None
    orb = cv2.ORB_create(nfeatures=1200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    rows = []
    for a, b in zip(gs[:-1], gs[1:]):
        f = pair_feats(a, b, orb, bf)
        if f:
            rows.append(f)
    if len(rows) < 3:
        return None
    g = lambda k: np.array([r[k] for r in rows], float)
    zoom, rm, mf = g("zoom"), g("res_mean"), g("mov_frac")
    coh = g("coh"); coh_v = coh[~np.isnan(coh)]
    magcv = g("magcv"); magcv_v = magcv[~np.isnan(magcv)]
    jerk = float(np.std(np.diff(rm)) / (rm.mean() + 1e-6))
    acf1 = float(np.corrcoef(rm[:-1], rm[1:])[0, 1]) if len(rm) > 3 and rm.std() > 1e-8 else 0.0
    rigid = 1.0 - (coh_v.mean() if len(coh_v) else np.nan)
    out = dict(zoom_mean=zoom.mean(), zoom_max=zoom.max(), res_mean=rm.mean(),
               res_max=rm.max(), mov_frac_mean=mf.mean(), mov_frac_max=mf.max(),
               coh_mean=coh_v.mean() if len(coh_v) else np.nan,
               coh_max=coh_v.max() if len(coh_v) else np.nan,
               magcv_mean=magcv_v.mean() if len(magcv_v) else np.nan,
               jerk=jerk, acf1=acf1,
               zoom_x_rigid=zoom.mean() * rigid if np.isfinite(rigid) else np.nan,
               n_pairs=float(len(rows)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, help="视频目录(可多个)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    vids = []
    for d in args.videos:
        vids += sorted(Path(d).glob("*.mp4"))
    done = set()
    out_p = Path(args.out)
    if out_p.exists():
        done = {r[0] for r in csv.reader(open(out_p))}
    else:
        csv.writer(open(out_p, "w", newline="")).writerow(["video"] + COLS)
    f = open(out_p, "a", newline="")
    w = csv.writer(f)
    import time
    t0 = time.time()
    for i, vp in enumerate(vids):
        if vp.name in done:
            continue
        ft = video_feats(vp)
        if ft is None:
            w.writerow([vp.name] + ["nan"] * len(COLS))
        else:
            w.writerow([vp.name] + [f"{ft[c]:.5g}" for c in COLS])
        if (i + 1) % 10 == 0:
            f.flush()
            print(f"[{i+1}/{len(vids)}] {(time.time()-t0)/(i+1):.2f}s/vid", flush=True)
    f.close()
    print("DECOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
