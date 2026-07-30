#!/usr/bin/env python
"""裁剪对象核验 v2:图像侧离群检测(不依赖文本塔)。

v1(so400m 零样本)在 transformers 5.x 下文本塔 config 越界、编码损坏,
913/919 假 flag,经蒙太奇人工复核推翻。v2 只用已提取的 DINOv2 前景特征:
  视频嵌入 = 检出帧前景 patch 特征均值再归一 -> 全局中心 = 全体视频嵌入均值方向
  -> cos(视频, 中心) 低尾 = 疑似非蘑菇裁剪(假牙/齿轮/道具特写等)
输出排名 CSV + 最低 K 条的蒙太奇,交人工终审;不做自动剔除。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BOTTOM_K = 40


def read_frame0(path: Path, want_idx: int = 0) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def sq_crop_arr(fr: np.ndarray, b) -> np.ndarray:
    H, W = fr.shape[:2]
    if b is None or np.isnan(b).any():
        s = min(W, H)
        x0, y0 = (W - s) // 2, (H - s) // 2
        return fr[y0:y0 + s, x0:x0 + s]
    x0, y0, x1, y1 = b
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(max(x1 - x0, y1 - y0) * 1.25, 32)
    xa, ya = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    xb, yb = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    return fr[ya:yb, xa:xb]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/prod500/mech_subset.tsv"))
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--feat-dir", default=str(ROOT / "data/corpus_patch_feat"))
    ap.add_argument("--out", default=str(ROOT / "data/prod500/verify_crops2.csv"))
    ap.add_argument("--montage", default=str(ROOT / "data/prod500/flag_montage.jpg"))
    args = ap.parse_args()

    rows = [l.split("\t") for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    embs, recs = [], []
    for rel, label in rows:
        p = Path(args.feat_dir) / rel.replace(".mp4", ".npz")
        if not p.exists():
            continue
        z = np.load(p)
        det, fg, feat = z["det"], z["fg"], z["feat"]
        if not det.any():
            recs.append({"rel": rel, "label": label, "det_rate": 0.0}); embs.append(None)
            continue
        vs = []
        for i in np.where(det)[0]:
            m = fg[i]
            if m.any():
                v = feat[i][m].astype(np.float32).mean(0)
                vs.append(v / (np.linalg.norm(v) + 1e-9))
        if not vs:
            recs.append({"rel": rel, "label": label, "det_rate": float(det.mean())})
            embs.append(None)
            continue
        v = np.mean(vs, 0)
        v /= (np.linalg.norm(v) + 1e-9)
        embs.append(v)
        recs.append({"rel": rel, "label": label, "det_rate": float(det.mean())})

    ok_idx = [i for i, e in enumerate(embs) if e is not None]
    E = np.stack([embs[i] for i in ok_idx])
    c = E.mean(0); c /= np.linalg.norm(c) + 1e-9
    cos = E @ c
    for j, i in enumerate(ok_idx):
        recs[i]["cos_centroid"] = float(cos[j])

    import pandas as pd
    df = pd.DataFrame(recs)
    df = df.sort_values("cos_centroid", na_position="first")
    df.to_csv(args.out, index=False)
    lo = df.head(BOTTOM_K)
    print(f"verify2: {len(df)} 条;cos 中位 {df.cos_centroid.median():.3f} "
          f"P5 {df.cos_centroid.quantile(0.05):.3f} 最低 {df.cos_centroid.min():.3f}")
    print(lo[["rel", "label", "cos_centroid", "det_rate"]].head(15).to_string(index=False))

    thumbs = []
    for _, r in lo.iterrows():
        vp = Path(args.videos_dir) / r.rel
        fp = Path(args.feat_dir) / r.rel.replace(".mp4", ".npz")
        fr = read_frame0(vp)
        if fr is None:
            continue
        z = np.load(fp)
        b = z["boxes"][0] if z["det"][0] else None
        t = cv2.resize(sq_crop_arr(fr, b), (160, 160))
        cv2.putText(t, f"{r.cos_centroid:.2f}", (4, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        thumbs.append(t)
    if thumbs:
        while len(thumbs) % 8:
            thumbs.append(np.zeros((160, 160, 3), np.uint8))
        rows_img = [np.concatenate(thumbs[i:i + 8], 1) for i in range(0, len(thumbs), 8)]
        cv2.imwrite(args.montage, np.concatenate(rows_img, 0),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"flag montage -> {args.montage}")


if __name__ == "__main__":
    main()
