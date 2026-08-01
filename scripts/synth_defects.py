#!/usr/bin/env python
"""线A 合成缺陷管线(2026-08-01,预注册于 P3 战役总结)。

输入:白底 448 抠像(好视频帧)。输出:缺陷变体 + 像素级缺陷 mask(免费标签)。
缺陷族(对齐用户清单):
  black_spots  脸/身黑点污斑(椭圆暗斑,羽化混合)
  smudge       五官融化/涂抹(局部高斯涂抹)
  warp         局部扭曲(嘴部异常代理:高斯位移场 remap)
  extra_limb   多余肢体(自体底部突起复制,贴到轮廓另一处)
  tail         尾巴/多余附属(体色长条附着轮廓)
  elongate     身体拉长(角色区纵向拉伸;mask=全角色,属全局缺陷)
  benign_paste 良性贴图(正常毛毡块贴入,标签=好;防"贴图痕迹"捷径)

设计要点:成对训练(同帧 ± 缺陷);benign_paste 让贴图伪影在两类同分布;
mask 供 patch 级密集监督。所有随机性走显式 rng(可复现)。
"""
from __future__ import annotations

import numpy as np
import cv2

WHITE_THR = 245


def char_mask(im: np.ndarray) -> np.ndarray:
    """白底抠像 -> 角色布尔掩码(非白像素,闭运算去噪)。"""
    m = (im < WHITE_THR).any(axis=2).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)


def _feather(mask_u8: np.ndarray, r: int) -> np.ndarray:
    return cv2.GaussianBlur(mask_u8.astype(np.float32), (0, 0), max(1, r / 2.5))


def _rand_pt_in(m: np.ndarray, rng, margin: int = 10):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    for _ in range(50):
        i = rng.integers(len(ys))
        y, x = int(ys[i]), int(xs[i])
        if margin <= y < m.shape[0] - margin and margin <= x < m.shape[1] - margin:
            return y, x
    return int(ys[0]), int(xs[0])


def black_spots(im, m, rng):
    out = im.astype(np.float32).copy()
    dmask = np.zeros(m.shape, np.float32)
    for _ in range(int(rng.integers(1, 5))):
        pt = _rand_pt_in(m, rng)
        if pt is None:
            return None
        y, x = pt
        a, b = int(rng.integers(3, 14)), int(rng.integers(3, 14))
        ang = float(rng.uniform(0, 180))
        blob = np.zeros(m.shape, np.uint8)
        cv2.ellipse(blob, (x, y), (a, b), ang, 0, 360, 1, -1)
        blob = blob & m.astype(np.uint8)
        f = _feather(blob, int(rng.integers(2, 6)))
        col = rng.uniform(5, 60)
        shade = np.array([col * rng.uniform(0.8, 1.2) for _ in range(3)])
        out = out * (1 - f[..., None] * rng.uniform(0.75, 0.98)) \
            + shade * (f[..., None] * rng.uniform(0.75, 0.98))
        dmask = np.maximum(dmask, f)
    return out.clip(0, 255).astype(np.uint8), dmask


def smudge(im, m, rng):
    pt = _rand_pt_in(m, rng)
    if pt is None:
        return None
    y, x = pt
    r = int(rng.integers(14, 40))
    blob = np.zeros(m.shape, np.uint8)
    cv2.circle(blob, (x, y), r, 1, -1)
    blob = blob & m.astype(np.uint8)
    f = _feather(blob, 8)
    blur = cv2.GaussianBlur(im, (0, 0), rng.uniform(6, 14))
    out = im.astype(np.float32) * (1 - f[..., None]) + blur.astype(np.float32) * f[..., None]
    return out.clip(0, 255).astype(np.uint8), f


def warp(im, m, rng):
    pt = _rand_pt_in(m, rng)
    if pt is None:
        return None
    y, x = pt
    H, W = m.shape
    sigma = rng.uniform(12, 30)
    amp = rng.uniform(8, 22)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2)))
    th = rng.uniform(0, 2 * np.pi)
    mapx = (xx + amp * np.cos(th) * g).astype(np.float32)
    mapy = (yy + amp * np.sin(th) * g).astype(np.float32)
    out = cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return out, (g > 0.3).astype(np.float32) * m


def _bottom_protrusion(im, m, rng):
    """取角色下 1/3 的一个突起块(近似手/脚)。"""
    ys, xs = np.where(m)
    y0 = int(np.percentile(ys, 66))
    band = np.zeros_like(m)
    band[y0:] = m[y0:]
    pt = _rand_pt_in(band, rng)
    if pt is None:
        return None
    y, x = pt
    r = int(rng.integers(18, 34))
    ya, yb = max(0, y - r), min(m.shape[0], y + r)
    xa, xb = max(0, x - r), min(m.shape[1], x + r)
    patch = im[ya:yb, xa:xb].copy()
    pm = m[ya:yb, xa:xb].astype(np.uint8)
    return patch, pm


def extra_limb(im, m, rng):
    """自体下部突起(含轮廓)整块复制,沿法向外移贴到身体下半轮廓——凸出如多肢。"""
    got = _bottom_protrusion(im, m, rng)
    if got is None:
        return None
    patch, pm = got
    if pm.mean() > 0.85 or pm.mean() < 0.15:   # 须同时含角色与留白,形状才像肢体
        return None
    ys, xs = np.where(m)
    y_lo = int(np.percentile(ys, 55))          # 附着点限定身体下半
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((7, 7), np.uint8))
    edge[:y_lo] = 0
    ph, pw = patch.shape[:2]
    pt = _rand_pt_in(edge.astype(bool), rng, margin=max(ph, pw) // 2 + 2)
    if pt is None:
        return None
    y, x = pt
    cx = float(xs.mean())
    dx = 1.0 if x >= cx else -1.0              # 沿水平法向外推,让块的一半悬在轮廓外
    y, x = int(y + rng.integers(-4, 5)), int(x + dx * pw * 0.35)
    ya, xa = y - ph // 2, x - pw // 2
    out = im.copy().astype(np.float32)
    dmask = np.zeros(m.shape, np.float32)
    sub = out[ya:ya + ph, xa:xa + pw]
    if sub.shape[:2] != (ph, pw):
        return None
    ang = float(rng.uniform(-40, 40))
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), ang, rng.uniform(0.9, 1.15))
    patch = cv2.warpAffine(patch, M, (pw, ph), borderValue=(255, 255, 255))
    pm = cv2.warpAffine(pm, M, (pw, ph)).astype(np.uint8)
    f = _feather(pm, 4)
    out[ya:ya + ph, xa:xa + pw] = sub * (1 - f[..., None]) + patch.astype(np.float32) * f[..., None]
    dmask[ya:ya + ph, xa:xa + pw] = f
    return out.clip(0, 255).astype(np.uint8), dmask


def tail(im, m, rng):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    # 体色采样
    i = rng.integers(len(ys), size=200)
    col = im[ys[i], xs[i]].mean(0)
    # 附着点:下半身中最宽的行(躯干),避免落在伞沿
    y_lo = int(np.percentile(ys, 60))
    widths = m[y_lo:].sum(1)
    if widths.max() == 0:
        return None
    y0 = y_lo + int(np.argmax(widths)) + int(rng.integers(-6, 7))
    y0 = min(max(y0, 0), m.shape[0] - 1)
    band = np.where(m[y0])[0]
    if len(band) == 0:
        return None
    side = rng.choice([0, 1])
    x0 = int(band[0]) if side == 0 else int(band[-1])
    L, w0 = int(rng.integers(45, 90)), int(rng.integers(10, 18))
    pts = []
    th = np.pi if side == 0 else 0.0
    th += rng.uniform(-0.5, 0.5)
    cx, cy = float(x0), float(y0)
    for t in range(L):
        cx += np.cos(th)
        cy += np.sin(th) * 0.3
        th += rng.uniform(-0.08, 0.08)
        pts.append((int(cx), int(cy), max(2, int(w0 * (1 - t / L)))))
    blob = np.zeros(m.shape, np.uint8)
    for x, y, w in pts:
        cv2.circle(blob, (x, y), w, 1, -1)
    f = _feather(blob, 4)
    noise = rng.normal(0, 6, im.shape).astype(np.float32)
    out = im.astype(np.float32) * (1 - f[..., None]) + (col + noise) * f[..., None]
    return out.clip(0, 255).astype(np.uint8), f


def elongate(im, m, rng):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    ya, yb, xa, xb = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = im[ya:yb, xa:xb]
    sy = rng.uniform(1.18, 1.5)
    if rng.random() < 0.3:
        sy = 1 / sy   # 矮胖方向
    nh = int((yb - ya) * sy)
    stretched = cv2.resize(crop, (xb - xa, nh))
    out = np.full_like(im, 255)
    y_new = min(nh, im.shape[0] - 4)
    yo = im.shape[0] - 4 - y_new  # 底部对齐(脚不动,往上长)
    out[yo:yo + y_new, xa:xb] = stretched[nh - y_new:]
    return out, char_mask(out).astype(np.float32)


def benign_paste(im, m, rng, donor=None):
    """良性贴图:同/异帧正常块贴入角色内部,标签=好。"""
    src = donor if donor is not None else im
    sm = char_mask(src)
    pt_s = _rand_pt_in(sm, rng, margin=20)
    pt_d = _rand_pt_in(m, rng, margin=20)
    if pt_s is None or pt_d is None:
        return None
    ys_, xs_ = pt_s
    yd, xd = pt_d
    r = int(rng.integers(10, 26))
    patch = src[ys_ - r:ys_ + r, xs_ - r:xs_ + r]
    if patch.shape[:2] != (2 * r, 2 * r):
        return None
    dst = im[yd - r:yd + r, xd - r:xd + r]
    if dst.shape[:2] != (2 * r, 2 * r):
        return None
    if abs(float(patch.mean()) - float(dst.mean())) > 28:   # 亮度失配(如采到眼睛)-> 弃,防假缺陷
        return None
    blob = np.zeros((2 * r, 2 * r), np.uint8)
    cv2.circle(blob, (r, r), r - 2, 1, -1)
    f = _feather(blob, 5)
    out = im.astype(np.float32).copy()
    sub = out[yd - r:yd + r, xd - r:xd + r]
    if sub.shape[:2] != (2 * r, 2 * r):
        return None
    out[yd - r:yd + r, xd - r:xd + r] = sub * (1 - f[..., None]) + patch * f[..., None]
    return out.clip(0, 255).astype(np.uint8), np.zeros(m.shape, np.float32)


DEFECTS = {"black_spots": black_spots, "smudge": smudge, "warp": warp,
           "extra_limb": extra_limb, "tail": tail, "elongate": elongate}


def synthesize(im: np.ndarray, kind: str, rng) -> tuple[np.ndarray, np.ndarray] | None:
    m = char_mask(im)
    if m.sum() < 2000:
        return None
    fn = DEFECTS.get(kind, benign_paste)
    return fn(im, m, rng)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="样例蒙太奇:每缺陷族一行")
    ap.add_argument("--cut-dir", default=str(Path(__file__).resolve().parents[1]
                                             / "data/sam3_cutouts"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data/prod500/synth_montage.jpg"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    jpgs = sorted(Path(args.cut_dir).glob("*/*/f00.jpg"))
    picks = [jpgs[i] for i in rng.choice(len(jpgs), 8, replace=False)]
    rows = []
    for kind in list(DEFECTS) + ["benign_paste"]:
        cells = []
        for p in picks:
            im = cv2.imread(str(p))
            got = synthesize(im, kind, rng)
            cell = got[0] if got else im
            if got is not None and kind != "benign_paste":
                mm = (got[1] > 0.3).astype(np.uint8)
                cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(cell, cnts, -1, (0, 0, 255), 1)
            cells.append(cv2.resize(cell, (180, 180)))
        strip = np.concatenate(cells, 1)
        strip = cv2.copyMakeBorder(strip, 22, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cv2.putText(strip, kind, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(strip)
    cv2.imwrite(args.out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("montage ->", args.out)
