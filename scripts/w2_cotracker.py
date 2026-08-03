#!/usr/bin/env python
"""W2:CoTracker3 轨迹特征(2026-08-03 预注册,E10 轨迹系)。

主体网格:SAM3 bbox 内铺 8x8 查询点(首帧);背景网格:bbox 外四周 12 点。
16 帧口径。特征:
  主体轨迹:成对距离变化率 pd_cv(刚体度反指标:低=刚体)、
            速度模均值 v_mean、二阶差分均值 jerk2、速度自相关 vacf、
            可见性丢失率 vis_loss;
  镜头(背景轨迹单应拟合):zoom_traj(缩放序列 |s-1| 均值/最大)、
            背景拟合残差 bg_res(背景本身乱动=场景崩);
  交互:zoom_traj_mean × (1 - pd_cv_norm)。
输出 csv 断点续跑。"""
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

COLS = ("pd_cv v_mean v_max jerk2 vacf vis_loss zoom_tr_mean zoom_tr_max "
        "bg_res sub_cam_ratio zoom_x_rigid_tr n_valid").split()


def read_frames(vp):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        cap.release(); return None, 1.0
    idxs = [int(round(i * (n - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)]
    want = set(idxs)
    frames, k = {}, 0
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
    out = []
    last = None
    for i in idxs:
        if i in frames:
            last = frames[i]
        fr = last if last is not None else first
        out.append(cv2.resize(fr, (int(w * s), int(h * s))))
    return np.stack(out), s


def bbox_of(feat_dir, rel, scale):
    p = feat_dir / rel.replace(".mp4", ".npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    geo = json.loads(str(z["geo"]))
    g0 = min(geo, key=lambda g: g["frame"]) if geo else None
    if g0 is None:
        return None
    x0, y0, x1, y1 = [v * scale for v in g0["bbox"]]
    return x0, y0, x1, y1


def make_queries(bb, H, W):
    x0, y0, x1, y1 = bb
    xs = np.linspace(x0 + 0.1 * (x1 - x0), x1 - 0.1 * (x1 - x0), 8)
    ys = np.linspace(y0 + 0.1 * (y1 - y0), y1 - 0.1 * (y1 - y0), 8)
    sub = [(x, y) for y in ys for x in xs]
    m = 0.06 * min(H, W)
    bxs = np.linspace(m, W - m, 6)
    bg = [(x, m) for x in bxs] + [(x, H - m) for x in bxs]
    bg = [(x, y) for x, y in bg if not (x0 - 20 < x < x1 + 20 and y0 - 20 < y < y1 + 20)]
    return np.array(sub, np.float32), np.array(bg, np.float32)


def traj_feats(tr_sub, vis_sub, tr_bg, vis_bg):
    T = tr_sub.shape[0]
    out = {c: np.nan for c in COLS}
    ok = vis_sub.mean(0) > 0.6
    out["vis_loss"] = float(1 - vis_sub.mean())
    P = tr_sub[:, ok]
    out["n_valid"] = float(P.shape[1])
    if P.shape[1] >= 8:
        d = P[:, :, None, :] - P[:, None, :, :]
        D = np.sqrt((d ** 2).sum(-1))
        iu = np.triu_indices(P.shape[1], 1)
        Dp = D[:, iu[0], iu[1]]
        mean_d = Dp.mean(0)
        keep = mean_d > 4.0
        if keep.sum() >= 10:
            cv = Dp[:, keep].std(0) / (mean_d[keep] + 1e-6)
            out["pd_cv"] = float(cv.mean())
        V = np.diff(P, axis=0)
        vmag = np.sqrt((V ** 2).sum(-1))
        out["v_mean"] = float(vmag.mean())
        out["v_max"] = float(vmag.mean(1).max())
        A = np.diff(V, axis=0)
        out["jerk2"] = float(np.sqrt((A ** 2).sum(-1)).mean())
        vm = vmag.mean(1)
        if len(vm) > 3 and vm.std() > 1e-8:
            out["vacf"] = float(np.corrcoef(vm[:-1], vm[1:])[0, 1])
    okb = vis_bg.mean(0) > 0.6
    B = tr_bg[:, okb]
    if B.shape[1] >= 6:
        zooms, resids = [], []
        for t in range(T - 1):
            src, dst = B[t].astype(np.float32), B[t + 1].astype(np.float32)
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=3.0)
            if M is None:
                continue
            s = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
            zooms.append(abs(s - 1))
            pred = src @ M[:, :2].T + M[:, 2]
            resids.append(float(np.sqrt(((pred - dst) ** 2).sum(-1)).mean()))
        if zooms:
            out["zoom_tr_mean"] = float(np.mean(zooms))
            out["zoom_tr_max"] = float(np.max(zooms))
            out["bg_res"] = float(np.mean(resids))
    if np.isfinite(out.get("v_mean", np.nan)) and np.isfinite(out.get("zoom_tr_mean", np.nan)):
        out["sub_cam_ratio"] = float(out["v_mean"] / (out["zoom_tr_mean"] * 100 + 1.0))
        if np.isfinite(out.get("pd_cv", np.nan)):
            out["zoom_x_rigid_tr"] = float(out["zoom_tr_mean"] * (1.0 / (out["pd_cv"] + 0.02)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/root/mech/data/corpus_videos")
    ap.add_argument("--feat-dir", default="/root/mech/data/sam3_feat")
    ap.add_argument("--manifest", default="/root/mech/manifest_all.tsv")
    ap.add_argument("--out", default="/root/mech/data/w2_cotracker.csv")
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

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda").eval()
    print("cotracker loaded", flush=True)

    import time
    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    for k, rel in enumerate(todo):
        row = None
        try:
            frames, scale = read_frames(Path(args.videos_dir) / rel)
            bb = bbox_of(feat_dir, rel, scale) if frames is not None else None
            if frames is not None and bb is not None:
                H, W = frames.shape[1:3]
                sub, bg = make_queries(bb, H, W)
                if len(sub) >= 8 and len(bg) >= 6:
                    video = torch.from_numpy(frames[..., ::-1].copy()).permute(0, 3, 1, 2)[None].float().to("cuda")
                    q = np.concatenate([sub, bg])
                    queries = torch.cat([torch.zeros(len(q), 1), torch.from_numpy(q)], 1)[None].float().to("cuda")
                    with torch.inference_mode():
                        tracks, vis = model(video, queries=queries)
                    tr = tracks[0].cpu().numpy()
                    vi = vis[0].cpu().numpy().astype(float)
                    ns = len(sub)
                    ft = traj_feats(tr[:, :ns], vi[:, :ns], tr[:, ns:], vi[:, ns:])
                    row = [f"{ft[c]:.5g}" for c in COLS]
        except Exception as e:
            if k < 3:
                print("ERR", rel, repr(e)[:120], flush=True)
        w.writerow([rel] + (row if row else ["nan"] * len(COLS)))
        if (k + 1) % 100 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    f.close()
    print("W2_DONE", flush=True)


if __name__ == "__main__":
    main()
