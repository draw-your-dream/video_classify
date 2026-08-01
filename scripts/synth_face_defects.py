#!/usr/bin/env python
"""线F 脸部合成缺陷族(2026-08-01)。输入 384px 原生脸裁剪(带背景)。

族:spots 黑点 / eyebrows 画眉(TUTU 标准无眉毛,出现眉=缺陷,盲审用户点名特征)
    eye_extra 眼部复制移位(第三只眼/错位) / mouth_warp 嘴部强扭曲
    smudge 五官融化 / benign 光度扰动+良性贴图(标签=好)
脸裁剪为 bbox×1.4 居中,眼线约在 0.30-0.40H、嘴约 0.55-0.70H(带抖动)。
"""
from __future__ import annotations

import numpy as np
import cv2


def center_mask(shape) -> np.ndarray:
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (W // 2, H // 2), (int(W * 0.40), int(H * 0.42)), 0, 0, 360, 1, -1)
    return m.astype(bool)


def _feather(mask_u8, r):
    return cv2.GaussianBlur(mask_u8.astype(np.float32), (0, 0), max(1.0, r / 2.5))


def spots(im, rng):
    H, W = im.shape[:2]
    out = im.astype(np.float32).copy()
    dm = np.zeros((H, W), np.float32)
    fm = center_mask(im.shape)
    ys, xs = np.where(fm)
    for _ in range(int(rng.integers(1, 4))):
        i = rng.integers(len(ys))
        y, x = int(ys[i]), int(xs[i])
        a, b = int(rng.integers(4, 16)), int(rng.integers(4, 16))
        blob = np.zeros((H, W), np.uint8)
        cv2.ellipse(blob, (x, y), (a, b), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        f = _feather(blob, int(rng.integers(2, 5)))
        col = rng.uniform(5, 70)
        out = out * (1 - f[..., None] * 0.92) + col * (f[..., None] * 0.92)
        dm = np.maximum(dm, f)
    return out.clip(0, 255).astype(np.uint8), dm


def eyebrows(im, rng):
    H, W = im.shape[:2]
    out = im.astype(np.float32).copy()
    dm = np.zeros((H, W), np.float32)
    ey = H * rng.uniform(0.26, 0.36)
    for ex_rel in (rng.uniform(0.28, 0.38), rng.uniform(0.60, 0.72)):
        ex = W * ex_rel
        w, h = int(W * rng.uniform(0.07, 0.12)), int(H * rng.uniform(0.015, 0.035))
        blob = np.zeros((H, W), np.uint8)
        cv2.ellipse(blob, (int(ex), int(ey)), (w, h),
                    float(rng.uniform(-18, 18)), 190, 350, 1,
                    thickness=int(max(2, H * rng.uniform(0.010, 0.022))))
        f = _feather(blob, 3)
        col = rng.uniform(15, 70)
        out = out * (1 - f[..., None] * 0.9) + col * (f[..., None] * 0.9)
        dm = np.maximum(dm, f)
    return out.clip(0, 255).astype(np.uint8), dm


def eye_extra(im, rng):
    H, W = im.shape[:2]
    # 眼源:上半张最暗小块附近取样
    top = cv2.cvtColor(im[:int(H * 0.55)], cv2.COLOR_BGR2GRAY)
    top = cv2.GaussianBlur(top, (0, 0), 3)
    _, _, minloc, _ = cv2.minMaxLoc(top)
    x0, y0 = minloc
    r = int(rng.integers(10, 20))
    ya, xa = max(0, y0 - r), max(0, x0 - r)
    patch = im[ya:ya + 2 * r, xa:xa + 2 * r]
    if patch.shape[0] != 2 * r or patch.shape[1] != 2 * r:
        return None
    fm = center_mask(im.shape)
    ys, xs = np.where(fm)
    i = rng.integers(len(ys))
    yd, xd = int(ys[i]), int(xs[i])
    if abs(yd - y0) < 2 * r and abs(xd - x0) < 2 * r:
        yd = min(H - r - 1, yd + 3 * r)
    blob = np.zeros((2 * r, 2 * r), np.uint8)
    cv2.circle(blob, (r, r), r - 2, 1, -1)
    f = _feather(blob, 4)
    out = im.astype(np.float32).copy()
    sub = out[yd - r:yd + r, xd - r:xd + r]
    if sub.shape[:2] != (2 * r, 2 * r):
        return None
    out[yd - r:yd + r, xd - r:xd + r] = sub * (1 - f[..., None]) + patch * f[..., None]
    dm = np.zeros((H, W), np.float32)
    dm[yd - r:yd + r, xd - r:xd + r] = f
    return out.clip(0, 255).astype(np.uint8), dm


def mouth_warp(im, rng):
    H, W = im.shape[:2]
    y = H * rng.uniform(0.52, 0.70)
    x = W * rng.uniform(0.38, 0.62)
    sigma = rng.uniform(0.06, 0.13) * W
    amp = rng.uniform(0.05, 0.11) * W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2)))
    th = rng.uniform(0, 2 * np.pi)
    mapx = (xx + amp * np.cos(th) * g).astype(np.float32)
    mapy = (yy + amp * np.sin(th) * g).astype(np.float32)
    out = cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, (g > 0.3).astype(np.float32)


def smudge(im, rng):
    H, W = im.shape[:2]
    fm = center_mask(im.shape)
    ys, xs = np.where(fm)
    i = rng.integers(len(ys))
    y, x = int(ys[i]), int(xs[i])
    r = int(rng.integers(18, 45))
    blob = np.zeros((H, W), np.uint8)
    cv2.circle(blob, (x, y), r, 1, -1)
    f = _feather(blob, 9)
    blur = cv2.GaussianBlur(im, (0, 0), rng.uniform(7, 16))
    out = im.astype(np.float32) * (1 - f[..., None]) + blur.astype(np.float32) * f[..., None]
    return out.clip(0, 255).astype(np.uint8), f


def benign(im, rng, donor=None):
    out = im.astype(np.float32)
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    if donor is not None and rng.random() < 0.5:
        H, W = im.shape[:2]
        r = int(rng.integers(10, 22))
        ys_, xs_ = int(rng.integers(r, H - r)), int(rng.integers(r, W - r))
        yd, xd = int(rng.integers(r, H - r)), int(rng.integers(r, W - r))
        patch = donor[ys_ - r:ys_ + r, xs_ - r:xs_ + r].astype(np.float32)
        sub = out[yd - r:yd + r, xd - r:xd + r]
        if patch.shape[:2] == (2 * r, 2 * r) and abs(patch.mean() - sub.mean()) < 25:
            blob = np.zeros((2 * r, 2 * r), np.uint8)
            cv2.circle(blob, (r, r), r - 2, 1, -1)
            f = _feather(blob, 5)
            out[yd - r:yd + r, xd - r:xd + r] = sub * (1 - f[..., None]) + patch * f[..., None]
    return out.clip(0, 255).astype(np.uint8), np.zeros(im.shape[:2], np.float32)


FACE_DEFECTS = {"spots": spots, "eyebrows": eyebrows, "eye_extra": eye_extra,
                "mouth_warp": mouth_warp, "smudge": smudge}


def synthesize_face(im, kind, rng):
    fn = FACE_DEFECTS.get(kind)
    if fn is None:
        return benign(im, rng)
    return fn(im, rng)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    jpgs = sorted(Path(args.face_dir).glob("*/*/f*.jpg"))
    picks = [jpgs[i] for i in rng.choice(len(jpgs), 8, replace=False)]
    rows = []
    for kind in list(FACE_DEFECTS) + ["benign"]:
        cells = []
        for p in picks:
            im = cv2.imread(str(p))
            got = synthesize_face(im, kind, rng)
            cell = got[0] if got else im
            cells.append(cv2.resize(cell, (180, 180)))
        strip = np.concatenate(cells, 1)
        strip = cv2.copyMakeBorder(strip, 22, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cv2.putText(strip, kind, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(strip)
    cv2.imwrite(args.out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("montage ->", args.out)
