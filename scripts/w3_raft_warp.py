#!/usr/bin/env python
"""W3:RAFT warping error + 单应补偿残差(2026-08-03 预注册,E10 先验系+经典系GPU版)。

torchvision RAFT-large。每对相邻帧:
  warp_err:用流把 t 搬到 t+1 的像素 MAE(EvalCrafter 同款,全画面);
  背景单应(流场在 bbox 外像素上拟合 homography)→ 残差流;
  bbox 内残差统计:res_mean/coh/magcv(同 v1 定义但流质量高一档)。
特征 12 列,csv 断点续跑。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

N_FRAMES = 16
SIDE = 384

COLS = ("warp_err_mean warp_err_max r_zoom_mean r_zoom_max r_res_mean r_res_max "
        "r_movfrac r_coh r_magcv r_jerk r_acf1 r_zoom_x_rigid").split()


def read_frames(vp):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        cap.release(); return None, 1.0
    idxs = [int(round(i * (n - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)]
    want = set(idxs); frames = {}; k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            frames[k] = fr
        k += 1
    cap.release()
    if not frames:
        return None, 1.0
    first = next(iter(frames.values()))
    h, w = first.shape[:2]
    s = SIDE / max(h, w)
    W8, H8 = (int(w * s) // 8) * 8, (int(h * s) // 8) * 8
    out, last = [], None
    for i in idxs:
        if i in frames:
            last = frames[i]
        fr = last if last is not None else first
        out.append(cv2.resize(fr, (W8, H8)))
    return np.stack(out), s * W8 / (w * s)


def bbox_of(feat_dir, rel, scale):
    p = feat_dir / rel.replace(".mp4", ".npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    geo = json.loads(str(z["geo"]))
    out = {}
    for g in geo:
        out[int(g["frame"])] = tuple(v * scale for v in g["bbox"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/root/mech/data/corpus_videos")
    ap.add_argument("--feat-dir", default="/root/mech/data/sam3_feat")
    ap.add_argument("--manifest", default="/root/mech/manifest_all.tsv")
    ap.add_argument("--out", default="/root/mech/data/w3_raft.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    feat_dir = Path(args.feat_dir)
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

    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to("cuda").eval()
    tfm = weights.transforms()
    print("raft loaded", flush=True)

    import time
    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    for k, rel in enumerate(todo):
        row = None
        try:
            frames, scale = read_frames(Path(args.videos_dir) / rel)
            bbs = bbox_of(feat_dir, rel, scale) if frames is not None else None
            if frames is not None and bbs:
                T, H, W = frames.shape[:3]
                rgb = torch.from_numpy(frames[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0
                a, b = tfm(rgb[:-1], rgb[1:])
                with torch.inference_mode():
                    flows = model(a.to("cuda"), b.to("cuda"))[-1].cpu().numpy()
                xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
                warp_errs, zooms, res_means, res_maxs, movs, cohs, magcvs = [], [], [], [], [], [], []
                g0s = []
                for t in range(T - 1):
                    fl = flows[t]
                    mapx, mapy = (xs + fl[0]).astype(np.float32), (ys + fl[1]).astype(np.float32)
                    warped = cv2.remap(frames[t], mapx, mapy, cv2.INTER_LINEAR)
                    warp_errs.append(float(np.abs(warped.astype(np.float32) - frames[t + 1].astype(np.float32)).mean()))
                    bb = bbs.get(t) or bbs.get(min(bbs, key=lambda kk: abs(kk - t)))
                    x0, y0, x1, y1 = bb
                    bgmask = np.ones((H, W), bool)
                    mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
                    bgmask[int(max(0, y0 - my)):int(min(H, y1 + my)), int(max(0, x0 - mx)):int(min(W, x1 + mx))] = False
                    if bgmask.sum() < 500:
                        continue
                    src = np.stack([xs[bgmask], ys[bgmask]], 1)
                    dst = src + np.stack([fl[0][bgmask], fl[1][bgmask]], 1)
                    sel = np.random.default_rng(0).choice(len(src), min(1500, len(src)), replace=False)
                    Hm, _ = cv2.findHomography(src[sel], dst[sel], cv2.RANSAC, 3.0)
                    if Hm is None:
                        continue
                    s_est = float(np.sqrt(abs(np.linalg.det(Hm[:2, :2]))))
                    zooms.append(abs(s_est - 1))
                    pts = np.stack([xs, ys, np.ones_like(xs)], -1).reshape(-1, 3)
                    proj = pts @ Hm.T
                    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)
                    gx = proj[:, 0].reshape(H, W) - xs
                    gy = proj[:, 1].reshape(H, W) - ys
                    rx, ry = fl[0] - gx, fl[1] - gy
                    sl = np.s_[int(max(0, y0)):int(min(H, y1)), int(max(0, x0)):int(min(W, x1))]
                    R = np.sqrt(rx[sl] ** 2 + ry[sl] ** 2)
                    if R.size < 400:
                        continue
                    res_means.append(float(R.mean())); res_maxs.append(float(R.max()))
                    mov = R > 0.4
                    movs.append(float(mov.mean()))
                    if mov.sum() > 50:
                        ux, uy = rx[sl][mov] / (R[mov] + 1e-6), ry[sl][mov] / (R[mov] + 1e-6)
                        cohs.append(float(1 - np.sqrt(ux.mean() ** 2 + uy.mean() ** 2)))
                        magcvs.append(float(R[mov].std() / (R[mov].mean() + 1e-6)))
                if len(res_means) >= 3:
                    rm = np.array(res_means)
                    jerk = float(np.std(np.diff(rm)) / (rm.mean() + 1e-6))
                    acf1 = float(np.corrcoef(rm[:-1], rm[1:])[0, 1]) if rm.std() > 1e-8 else 0.0
                    rigid = 1 - (np.mean(cohs) if cohs else np.nan)
                    ft = dict(warp_err_mean=np.mean(warp_errs), warp_err_max=np.max(warp_errs),
                              r_zoom_mean=np.mean(zooms) if zooms else np.nan,
                              r_zoom_max=np.max(zooms) if zooms else np.nan,
                              r_res_mean=rm.mean(), r_res_max=np.max(res_maxs),
                              r_movfrac=np.mean(movs) if movs else np.nan,
                              r_coh=np.mean(cohs) if cohs else np.nan,
                              r_magcv=np.mean(magcvs) if magcvs else np.nan,
                              r_jerk=jerk, r_acf1=acf1,
                              r_zoom_x_rigid=(np.mean(zooms) * rigid) if zooms and np.isfinite(rigid) else np.nan)
                    row = [f"{ft[c]:.5g}" for c in COLS]
        except Exception as e:
            if k < 3:
                print("ERR", rel, repr(e)[:120], flush=True)
        w.writerow([rel] + (row if row else ["nan"] * len(COLS)))
        if (k + 1) % 100 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    f.close()
    print("W3_DONE", flush=True)


if __name__ == "__main__":
    main()
