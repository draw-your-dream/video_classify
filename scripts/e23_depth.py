#!/usr/bin/env python
"""E23 深度物理专家(2026-08-04 预注册):DepthAnything-V2-Small 逐帧深度一致性。

12 帧。特征 10 列:
  cd_med_cv 角色深度中位轨迹变异 | cd_maxjump 角色深度最大突变
  gap_mean/gap_cv 角色底部 vs 支撑带深度差及稳定性(悬空=差大)
  bg_jit 背景深度时间抖动 | d_range 场景深度动态范围
  cd_grad 角色内部深度梯度(应为立体渐变,贴片=平) | cd_grad_cv 其时间变异
  n_ok 有效帧 | char_cov 角色框覆盖率"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")
N_FRAMES = 12
COLS = "cd_med_cv cd_maxjump gap_mean gap_cv bg_jit d_range cd_grad cd_grad_cv n_ok char_cov".split()


def read_frames(vp):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        cap.release(); return []
    idxs = [int(round(i * (n - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)]
    want = set(idxs); out = {}; k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            out[k] = fr
        k += 1
    cap.release()
    return [(i, out[i]) for i in sorted(out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--feat-dir", default=str(ROOT / "data/sam3_feat"))
    ap.add_argument("--manifest", default=str(ROOT / "manifest_all.tsv"))
    ap.add_argument("--out", default=str(ROOT / "data/e23_depth.csv"))
    ap.add_argument("--model", default="/root/mech/models/dav2-small")
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

    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    sp = AutoImageProcessor.from_pretrained(args.model)
    sm = AutoModelForDepthEstimation.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()

    @torch.inference_mode()
    def depth(ims):
        inp = sp(images=ims, return_tensors="pt").to("cuda")
        out = sm(pixel_values=inp["pixel_values"].half()).predicted_depth
        return out.float().cpu().numpy()  # (B,h,w) 相对深度,大=近

    import time
    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    for k, rel in enumerate(todo):
        row = None
        try:
            frames = read_frames(Path(args.videos_dir) / rel)
            bbs = {}
            p = Path(args.feat_dir) / rel.replace(".mp4", ".npz")
            if p.exists():
                z = np.load(p, allow_pickle=True)
                for g in json.loads(str(z["geo"])):
                    bbs[int(g["frame"])] = g["bbox"]
            if len(frames) >= 6 and bbs:
                ims, metas = [], []
                for j, (fi, fr) in enumerate(frames):
                    H, W = fr.shape[:2]
                    s = 336 / max(H, W)
                    ims.append(Image.fromarray(cv2.cvtColor(cv2.resize(fr, (int(W*s), int(H*s))), cv2.COLOR_BGR2RGB)))
                    bb = bbs.get(j) or (bbs.get(min(bbs, key=lambda kk: abs(kk-j))) if bbs else None)
                    metas.append((bb, s, H, W))
                D = depth(ims)
                cds, gaps, bgs, grads = [], [], [], []
                n_ok = 0
                for d, (bb, s, H, W) in zip(D, metas):
                    dh, dw = d.shape
                    if bb is None:
                        continue
                    sx, sy = dw / (W * s) * s, dh / (H * s) * s
                    x0, y0, x1, y1 = [int(v) for v in (bb[0]*sx, bb[1]*sy, bb[2]*sx, bb[3]*sy)]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(dw, x1), min(dh, y1)
                    if x1 - x0 < 4 or y1 - y0 < 4:
                        continue
                    n_ok += 1
                    char = d[y0:y1, x0:x1]
                    cds.append(float(np.median(char)))
                    gy = np.abs(np.diff(char, axis=0)).mean() + np.abs(np.diff(char, axis=1)).mean()
                    grads.append(float(gy / (np.abs(d).mean() + 1e-6)))
                    # 支撑带:角色框正下方一条
                    yb0, yb1 = min(dh - 1, y1), min(dh, y1 + max(4, (y1 - y0) // 4))
                    if yb1 > yb0 + 1:
                        below = d[yb0:yb1, x0:x1]
                        char_bottom = d[max(y0, y1 - max(4, (y1 - y0) // 6)):y1, x0:x1]
                        gaps.append(float(np.median(char_bottom) - np.median(below)))
                    m = np.ones_like(d, bool); m[y0:y1, x0:x1] = False
                    bgs.append(float(np.median(d[m])))
                if n_ok >= 5:
                    cds = np.array(cds); rng = np.abs(D).mean() + 1e-6
                    ft = dict(
                        cd_med_cv=float(cds.std() / (np.abs(cds.mean()) + 1e-6)),
                        cd_maxjump=float(np.abs(np.diff(cds)).max() / rng),
                        gap_mean=float(np.mean(gaps) / rng) if gaps else np.nan,
                        gap_cv=float(np.std(gaps) / (np.abs(np.mean(gaps)) + 1e-6)) if len(gaps) > 2 else np.nan,
                        bg_jit=float(np.abs(np.diff(bgs)).mean() / rng) if len(bgs) > 2 else np.nan,
                        d_range=float((D.max() - D.min()) / rng),
                        cd_grad=float(np.mean(grads)),
                        cd_grad_cv=float(np.std(grads) / (np.mean(grads) + 1e-6)),
                        n_ok=float(n_ok), char_cov=float(n_ok / len(frames)))
                    row = [f"{ft[c]:.5g}" for c in COLS]
        except Exception as e:
            if k < 3:
                print("ERR", rel, repr(e)[:100], flush=True)
        w.writerow([rel] + (row if row else ["nan"] * len(COLS)))
        if (k + 1) % 200 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    f.close()
    print("E23_DONE", flush=True)


if __name__ == "__main__":
    main()
