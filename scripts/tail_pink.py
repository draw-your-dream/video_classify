#!/usr/bin/env python
"""尾巴粉色特征(2026-08-06 预注册,非VLM):基于 sam3_cutouts 白底抠像。
逐帧:角色掩码下2/3内的粉色连通块(排除脸颊区),取贴轮廓的小块。
视频级 7 列:pk_persist(出现帧占比) pk_ratio_med(面积比中位) pk_edge(贴边率中位)
pk_pos_std(位置稳定性) pk_hue_med pk_frames pk_maxblob。校准锚:基础款/5822 必须高分。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/mech")
COLS = "pk_persist pk_ratio_med pk_edge pk_pos_std pk_hue_med pk_frames pk_maxblob".split()


def frame_pink(im):
    """返回 (blob面积比, 贴边率, 质心, hue) 或 None。im=白底抠像 BGR。"""
    mask = (im.max(2) < 246) | (im.std(2) > 6)  # 非白底=角色
    mask = mask.astype(np.uint8)
    if mask.sum() < 400:
        return None
    H, W = mask.shape
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    body_lo = y0 + int((y1 - y0) * 0.38)  # 排除头/脸颊区,只看下 62%
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pink = (((Hh >= 140) & (Hh <= 179)) | (Hh <= 8)) & (S >= 25) & (S <= 160) & (V >= 130)
    pink = pink & (mask > 0)
    pink[:body_lo] = False
    pink = pink.astype(np.uint8)
    if pink.sum() < 12:
        return None
    n, lab, stats, cents = cv2.connectedComponentsWithStats(pink, 8)
    if n < 2:
        return None
    bi = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[bi, cv2.CC_STAT_AREA]
    ratio = float(area / mask.sum())
    if ratio < 0.0008 or ratio > 0.08:  # 太小=噪声,太大=衣物/大面积
        return None
    blob = (lab == bi).astype(np.uint8)
    er = cv2.erode(mask, np.ones((7, 7), np.uint8))
    edge_band = (mask > 0) & (er == 0)
    edge_frac = float((blob & edge_band).sum() / (area + 1e-6))
    cy, cx = cents[bi][1], cents[bi][0]
    hue = float(np.median(Hh[(lab == bi)]))
    return ratio, edge_frac, (cx / W, cy / H), hue


def video_feats(d):
    jpgs = sorted(Path(d).glob("f*.jpg"))[:16]
    if len(jpgs) < 6:
        return None
    hits = []
    for p in jpgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        r = frame_pink(im)
        if r:
            hits.append(r)
    n = len(jpgs)
    if not hits:
        return dict(pk_persist=0.0, pk_ratio_med=0.0, pk_edge=0.0, pk_pos_std=np.nan,
                    pk_hue_med=np.nan, pk_frames=0.0, pk_maxblob=0.0)
    ratios = [h[0] for h in hits]; edges = [h[1] for h in hits]
    poss = np.array([h[2] for h in hits]); hues = [h[3] for h in hits]
    return dict(pk_persist=len(hits) / n, pk_ratio_med=float(np.median(ratios)),
                pk_edge=float(np.median(edges)),
                pk_pos_std=float(poss.std(0).mean()) if len(hits) > 2 else np.nan,
                pk_hue_med=float(np.median(hues)), pk_frames=float(len(hits)),
                pk_maxblob=float(np.max(ratios)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut-dir", default=str(ROOT / "data/sam3_cutouts"))
    ap.add_argument("--manifest", default=str(ROOT / "manifest_all.tsv"))
    ap.add_argument("--out", default=str(ROOT / "data/tail_pink.csv"))
    ap.add_argument("--workers", type=int, default=8)
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

    import time
    from concurrent.futures import ThreadPoolExecutor
    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()

    def one(rel):
        try:
            ft = video_feats(Path(args.cut_dir) / rel.replace(".mp4", ""))
        except Exception:
            ft = None
        return rel, ft

    with ThreadPoolExecutor(args.workers) as ex:
        for i, (rel, ft) in enumerate(ex.map(one, todo)):
            if ft:
                w.writerow([rel] + [f"{ft[c]:.5g}" for c in COLS])
            else:
                w.writerow([rel] + ["nan"] * len(COLS))
            if (i + 1) % 500 == 0:
                f.flush()
                print(f"[{i+1}/{len(todo)}] {(time.time()-t0)/(i+1):.3f}s/vid", flush=True)
    f.close()
    print("TAILPINK_DONE", flush=True)


if __name__ == "__main__":
    main()
