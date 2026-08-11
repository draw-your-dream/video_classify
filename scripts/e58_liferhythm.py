#!/usr/bin/env python
"""E58:生命节律电池(2026-08-11 深夜,用户方向:没有生命力,非卡顿)。

生命力的物理签名,全部需要全帧率(弹跳谱曾死于 16 帧混叠):
①节律:呼吸/弹跳是 0.5-4Hz 的准周期振荡(vy 频谱带功率 + 自相关副峰);
②重力:活物跳跃/落下是抛物线弹道(cy 滑窗二次拟合 R²),没生命力的漂移是匀速;
③自发微动:静止段的微节律(低运动段的振荡功率)= 活的静止 vs 死的静止;
④升降不对称:重力下落更快(vy 上升/下降速度比)。
实现:角色区(geo bbox 插值)96×96 Farneback 光流,逐帧 vy/vx。
输出 data/liferhythm_full.csv。纯 CPU 48 进程,约 10 分钟。
"""
from __future__ import annotations

import csv
import glob
import json
import os

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
import cv2
import numpy as np

cv2.setNumThreads(1)
ROOT = "/root/mech"
COLS = ("rhy_pow rhy_peak micro_rhy grav_r2 fallrise vzc vmag fps n_fr").split()


def one(args):
    rel, geo = args
    row = {"rel": rel}
    vp = os.path.join(ROOT, "data/corpus_videos", rel)
    if not os.path.exists(vp) or not geo or len(geo) < 8:
        return row
    try:
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
        if total < 40:
            cap.release()
            return row
        fr = np.array([x["frame"] for x in geo], float)
        fr = fr / max(1.0, fr.max()) * max(1, total - 1)
        B = np.array([x["bbox"] for x in geo], float)
        prev = None
        vy, vx, vm = [], [], []
        k = 0
        H0 = W0 = None
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if W0 is None:
                H0, W0 = im.shape[:2]
            b = [float(np.interp(k, fr, B[:, j])) for j in range(4)]
            x0, y0, x1, y1 = (max(0, int(b[0])), max(0, int(b[1])),
                              min(W0, int(b[2])), min(H0, int(b[3])))
            if x1 - x0 < 16 or y1 - y0 < 16:
                prev = None
                k += 1
                continue
            g = cv2.cvtColor(cv2.resize(im[y0:y1, x0:x1], (96, 96)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 11, 2, 5, 1.1, 0)
                vy.append(float(fl[..., 1].mean()))
                vx.append(float(fl[..., 0].mean()))
                vm.append(float(np.hypot(fl[..., 0], fl[..., 1]).mean()))
            prev = g
            k += 1
        cap.release()
        vy = np.asarray(vy, float)
        vm = np.asarray(vm, float)
        if len(vy) < 40:
            return row
        n = len(vy)
        z = vy - vy.mean()
        # ①节律:0.5-4Hz 带功率占比
        f = np.abs(np.fft.rfft(z)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        band = (freqs >= 0.5) & (freqs <= 4.0)
        tot = f[1:].sum()
        row["rhy_pow"] = float(f[band].sum() / max(1e-9, tot))
        # 自相关副峰(0.25-2 秒滞后)
        ac = np.correlate(z, z, "full")[n - 1:]
        ac /= max(1e-9, ac[0])
        l0, l1 = int(0.25 * fps), min(n - 1, int(2.0 * fps))
        row["rhy_peak"] = float(ac[l0:l1].max()) if l1 > l0 else 0.0
        # ③静止段微节律:低 |v| 半段上的带功率
        lo = vm < np.median(vm)
        if lo.sum() >= 32:
            z2 = vy[lo] - vy[lo].mean()
            f2 = np.abs(np.fft.rfft(z2)) ** 2
            fq2 = np.fft.rfftfreq(len(z2), d=1.0 / fps)
            b2 = (fq2 >= 0.5) & (fq2 <= 4.0)
            row["micro_rhy"] = float(f2[b2].sum() / max(1e-9, f2[1:].sum()))
        # ②重力弹道:cy=累计 vy,滑窗 0.5s 二次拟合 R²(只在有运动的窗)
        cy = np.cumsum(vy)
        Wn = max(8, int(0.5 * fps))
        r2s = []
        for s in range(0, n - Wn, Wn // 2):
            seg = cy[s:s + Wn]
            if vm[s:s + Wn].mean() < np.percentile(vm, 40):
                continue
            t = np.arange(len(seg))
            co = np.polyfit(t, seg, 2)
            pred = np.polyval(co, t)
            ss = ((seg - seg.mean()) ** 2).sum()
            r2s.append(1 - ((seg - pred) ** 2).sum() / max(1e-9, ss))
        row["grav_r2"] = float(np.median(r2s)) if r2s else np.nan
        # ④升降不对称:下落(vy>0,图像坐标向下)速度 vs 上升速度
        up = vy[vy < -1e-4]
        dn = vy[vy > 1e-4]
        if len(up) >= 5 and len(dn) >= 5:
            row["fallrise"] = float(np.abs(dn).mean() / max(1e-6, np.abs(up).mean()))
        row["vzc"] = float((np.diff(np.sign(z)) != 0).mean())
        row["vmag"] = float(np.median(vm))
        row["fps"] = fps
        row["n_fr"] = n
    except Exception:
        pass
    return row


def main():
    geos = {}
    for p in glob.glob(f"{ROOT}/data/sam3_feat/*/*.npz"):
        rel = "/".join(p.split(os.sep)[-2:]).replace(".npz", ".mp4")
        try:
            z = np.load(p, allow_pickle=True)
            g = json.loads(str(z["geo"]))
            if isinstance(g, list):
                geos[rel] = g
        except Exception:
            pass
    rels = [l.split("\t")[0] for l in open(f"{ROOT}/manifest_all.tsv") if l.strip()]
    jobs = [(r, geos.get(r)) for r in rels]
    print(f"任务 {len(jobs)}", flush=True)
    import multiprocessing as mp
    with open(f"{ROOT}/data/liferhythm_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel"] + COLS)
        with mp.Pool(48) as pool:
            for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
                w.writerow([row["rel"]] + [row.get(c, "") for c in COLS])
                if (i + 1) % 300 == 0:
                    print(f"[{i+1}/{len(jobs)}]", flush=True)
    print("E58_LIFERHYTHM_DONE", flush=True)


if __name__ == "__main__":
    main()
