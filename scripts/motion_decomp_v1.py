#!/usr/bin/env python
"""E10-v1 掩码限定运动分解(2026-08-03 预注册)。

与 v0 的差别:利用 sam3_feat 的逐帧 bbox——
  背景配准:ORB 关键点仅取 bbox(外扩15%)之外;
  残差统计:仅统计 bbox 内像素。
特征名同 v0 的 12 列。无 bbox 帧对 → 跳过;有效帧对 <3 → 整条 nan。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

N_FRAMES = 16
MAX_SIDE = 448
MOV_THR = 0.4

COLS = ("zoom_mean zoom_max res_mean res_max mov_frac_mean mov_frac_max "
        "coh_mean coh_max magcv_mean jerk acf1 zoom_x_rigid n_pairs").split()

FEAT_DIR = None  # set in main


def read_frames_bgr(vp):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        cap.release(); return [], 1.0
    idxs = sorted({int(round(i * (n - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)})
    out, k, s = {}, 0, 1.0
    want = set(idxs)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            h, w = fr.shape[:2]
            s = MAX_SIDE / max(h, w)
            g = cv2.cvtColor(cv2.resize(fr, (int(w * s), int(h * s))), cv2.COLOR_BGR2GRAY)
            out[k] = g
        k += 1
    cap.release()
    return [out[i] for i in sorted(out)], s


def load_bboxes(rel, scale):
    p = FEAT_DIR / rel.replace(".mp4", ".npz")
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=True)
    geo = json.loads(str(z["geo"]))
    out = {}
    for g in geo:
        x0, y0, x1, y1 = g["bbox"]
        out[int(g["frame"])] = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
    return out


def nearest_bbox(bbs, i):
    if not bbs:
        return None
    j = min(bbs, key=lambda k: abs(k - i))
    if abs(j - i) > 4:
        return None
    return bbs[j]


def pair_feats(g0, g1, bb, orb, bf):
    H, W = g0.shape
    x0, y0, x1, y1 = bb
    mx = 0.15 * (x1 - x0); my = 0.15 * (y1 - y0)
    ex0, ey0, ex1, ey1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
    bgmask = np.full((H, W), 255, np.uint8)
    bgmask[int(ey0):int(ey1), int(ex0):int(ex1)] = 0
    k0, d0 = orb.detectAndCompute(g0, bgmask)
    k1, d1 = orb.detectAndCompute(g1, bgmask)
    if d0 is None or d1 is None or len(k0) < 10 or len(k1) < 10:
        return None
    ms = bf.match(d0, d1)
    if len(ms) < 10:
        return None
    src = np.float32([k0[m.queryIdx].pt for m in ms])
    dst = np.float32([k1[m.trainIdx].pt for m in ms])
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        return None
    scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    gx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs
    gy = M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys
    rx, ry = flow[..., 0] - gx, flow[..., 1] - gy
    # 只看角色框内
    sl = np.s_[int(max(0, y0)):int(min(H, y1)), int(max(0, x0)):int(min(W, x1))]
    rxc, ryc = rx[sl], ry[sl]
    if rxc.size < 400:
        return None
    R = np.sqrt(rxc ** 2 + ryc ** 2)
    mov = R > MOV_THR
    res_mean = float(R.mean())
    mov_frac = float(mov.mean())
    if mov.sum() > 50:
        ux, uy = rxc[mov] / (R[mov] + 1e-6), ryc[mov] / (R[mov] + 1e-6)
        coh = float(1.0 - np.sqrt(ux.mean() ** 2 + uy.mean() ** 2))
        magcv = float(R[mov].std() / (R[mov].mean() + 1e-6))
    else:
        coh, magcv = np.nan, np.nan
    return dict(zoom=abs(scale - 1.0), res_mean=res_mean, mov_frac=mov_frac,
                coh=coh, magcv=magcv)


def video_feats(args_tuple):
    vp, rel = args_tuple
    try:
        gs, scale = read_frames_bgr(vp)
        if len(gs) < 4:
            return rel, None
        bbs = load_bboxes(rel, scale)
        if not bbs:
            return rel, None
        orb = cv2.ORB_create(nfeatures=1200)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        rows = []
        for i, (a, b) in enumerate(zip(gs[:-1], gs[1:])):
            bb = nearest_bbox(bbs, i)
            if bb is None:
                continue
            f = pair_feats(a, b, bb, orb, bf)
            if f:
                rows.append(f)
        if len(rows) < 3:
            return rel, None
        g = lambda k: np.array([r[k] for r in rows], float)
        zoom, rm, mf = g("zoom"), g("res_mean"), g("mov_frac")
        coh = g("coh"); coh_v = coh[~np.isnan(coh)]
        magcv = g("magcv"); magcv_v = magcv[~np.isnan(magcv)]
        jerk = float(np.std(np.diff(rm)) / (rm.mean() + 1e-6))
        acf1 = float(np.corrcoef(rm[:-1], rm[1:])[0, 1]) if len(rm) > 3 and rm.std() > 1e-8 else 0.0
        rigid = 1.0 - (coh_v.mean() if len(coh_v) else np.nan)
        ft = dict(zoom_mean=zoom.mean(), zoom_max=zoom.max(), res_mean=rm.mean(),
                  res_max=rm.max(), mov_frac_mean=mf.mean(), mov_frac_max=mf.max(),
                  coh_mean=coh_v.mean() if len(coh_v) else np.nan,
                  coh_max=coh_v.max() if len(coh_v) else np.nan,
                  magcv_mean=magcv_v.mean() if len(magcv_v) else np.nan,
                  jerk=jerk, acf1=acf1,
                  zoom_x_rigid=zoom.mean() * rigid if np.isfinite(rigid) else np.nan,
                  n_pairs=float(len(rows)))
        return rel, [f"{ft[c]:.5g}" for c in COLS]
    except Exception:
        return rel, None


def main():
    global FEAT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/root/mech/data/corpus_videos")
    ap.add_argument("--feat-dir", default="/root/mech/data/sam3_feat")
    ap.add_argument("--manifest", default="/root/mech/manifest_all.tsv")
    ap.add_argument("--out", default="/root/mech/data/decomp_v1.csv")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    FEAT_DIR = Path(args.feat_dir)
    rels = [l.split("\t")[0] for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    done = set()
    out_p = Path(args.out)
    if out_p.exists():
        done = {r[0] for r in csv.reader(open(out_p))}
    else:
        csv.writer(open(out_p, "w", newline="")).writerow(["rel"] + COLS)
    todo = [(Path(args.videos_dir) / r, r) for r in rels if r not in done]
    print(f"total {len(rels)} done {len(done)} todo {len(todo)}", flush=True)
    cv2.setNumThreads(1)
    import time
    from multiprocessing import Pool
    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (rel, row) in enumerate(pool.imap_unordered(video_feats, todo, chunksize=8)):
            w.writerow([rel] + (row if row else ["nan"] * len(COLS)))
            if (i + 1) % 200 == 0:
                f.flush()
                print(f"[{i+1}/{len(todo)}] {(time.time()-t0)/(i+1):.2f}s/vid", flush=True)
    f.close()
    print("DECOMP_V1_DONE", flush=True)


if __name__ == "__main__":
    main()
