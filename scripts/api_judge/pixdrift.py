#!/usr/bin/env python3
"""像素级开局漂移:视频 frame0 vs 源图(有映射的 1126 条,其余置 0+has_src=0)。
特征:灰度 SSIM、RGB 直方图相关、HSV 色相直方图相关、均值色差。CPU 多进程。
输出 /workspace/r2/pixdrift_1233.csv,marker=PIXDRIFT_DONE。"""
import csv
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

R2 = Path("/workspace/r2")


def one(r):
    fn = r["filename"]
    vp = R2 / "videos" / fn
    if not vp.exists():
        return None
    cap = cv2.VideoCapture(str(vp))
    ok, f0 = cap.read()
    cap.release()
    if not ok:
        return None
    rec = {"filename": fn, "px_has_src": 0, "px_ssim": 0.0, "px_rgbcorr": 0.0,
           "px_huecorr": 0.0, "px_dcolor": 0.0}
    sfp = R2 / "qcimgs" / f"{r['image_dataset']}__SLASH__{r['image_sample_id']}.png"
    if not (r["image_sample_id"] and sfp.exists() and sfp.stat().st_size > 0):
        return rec
    src = cv2.imread(str(sfp))
    if src is None:
        return rec
    h, w = f0.shape[:2]
    src = cv2.resize(src, (w, h))
    rec["px_has_src"] = 1
    g1 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu1, mu2 = g1.mean(), g2.mean()
    v1, v2 = g1.var(), g2.var()
    cov = ((g1 - mu1) * (g2 - mu2)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    rec["px_ssim"] = float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) /
                           ((mu1 ** 2 + mu2 ** 2 + c1) * (v1 + v2 + c2)))
    h1 = cv2.calcHist([f0], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    h2 = cv2.calcHist([src], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    rec["px_rgbcorr"] = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    hs1 = cv2.calcHist([cv2.cvtColor(f0, cv2.COLOR_BGR2HSV)], [0], None, [36], [0, 180])
    hs2 = cv2.calcHist([cv2.cvtColor(src, cv2.COLOR_BGR2HSV)], [0], None, [36], [0, 180])
    rec["px_huecorr"] = float(cv2.compareHist(hs1, hs2, cv2.HISTCMP_CORREL))
    rec["px_dcolor"] = float(np.abs(f0.mean((0, 1)) - src.mean((0, 1))).mean())
    return rec


rows = list(csv.DictReader(open(R2 / "data/api_judge_video_image_map.csv", encoding="utf-8-sig")))
with Pool(24) as p:
    recs = [r for r in p.map(one, rows) if r]
with open(R2 / "pixdrift_1233.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
    w.writeheader()
    w.writerows(recs)
print(f"PIXDRIFT_DONE {len(recs)}", flush=True)
