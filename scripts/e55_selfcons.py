#!/usr/bin/env python
"""E55:逐帧自一致性(v2:多进程预处理修复 GPU 空转,同 DataLoader 教训)。
worker 做 cv2 解码+dinov2 预处理(256 短边→224 中心裁→imagenet 归一),主进程只喂 GPU。
输出 data/selfcons_full.csv(断点续跑)。"""
from __future__ import annotations

import csv
import glob
import os

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
ROOT = "/root/mech"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep_video(d):
    rel = "/".join(d.split(os.sep)[-2:]) + ".mp4"
    fs = sorted(glob.glob(os.path.join(d, "f*.jpg")))[:16]
    if len(fs) < 8:
        return rel, None
    out = []
    for f in fs:
        im = cv2.imread(f)
        if im is None:
            continue
        H, W = im.shape[:2]
        s = 256 / min(H, W)
        im = cv2.resize(im, (int(round(W * s)), int(round(H * s))))
        H, W = im.shape[:2]
        y0 = (H - 224) // 2
        x0 = (W - 224) // 2
        im = im[y0:y0 + 224, x0:x0 + 224]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(((im - MEAN) / STD).transpose(2, 0, 1))
    if len(out) < 8:
        return rel, None
    return rel, np.stack(out)


def main():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(f"{ROOT}/models/dinov2-base").to("cuda").eval().half()
    dirs = sorted(p for p in glob.glob(f"{ROOT}/data/crops_v3/*/*") if os.path.isdir(p))
    outp = f"{ROOT}/data/selfcons_full.csv"
    done = set()
    if os.path.exists(outp):
        for r in csv.reader(open(outp)):
            done.add(r[0])
    mode = "a" if done else "w"
    dirs = [d for d in dirs if "/".join(d.split(os.sep)[-2:]) + ".mp4" not in done]
    print(f"视频 {len(dirs)}(已完成 {len(done)})", flush=True)
    out = open(outp, mode, newline="")
    w = csv.writer(out)
    if mode == "w":
        w.writerow(["rel", "max_selfdist", "med_selfdist", "fl_dist", "drift_mono", "jump_max", "n"])
    import multiprocessing as mp
    import time
    t0 = time.time()
    with mp.Pool(24) as pool:
        for i, (rel, arr) in enumerate(pool.imap(prep_video, dirs, chunksize=4)):
            if arr is None:
                w.writerow([rel] + [""] * 6)
                continue
            with torch.inference_mode():
                x = torch.from_numpy(arr).half().to("cuda", non_blocking=True)
                E = model(pixel_values=x).last_hidden_state[:, 0]
                E = torch.nn.functional.normalize(E, dim=1).float().cpu().numpy()
            med = np.median(E, axis=0)
            med /= max(1e-9, np.linalg.norm(med))
            sd = 1 - E @ med
            fl = float(1 - E[0] @ E[-1])
            step = 1 - (E[:-1] * E[1:]).sum(1)
            cum = 1 - E @ E[0]
            mono = float(abs(np.diff(cum).sum()) / (np.abs(np.diff(cum)).sum() + 1e-9))
            w.writerow([rel, f"{sd.max():.5f}", f"{np.median(sd):.5f}", f"{fl:.5f}",
                        f"{mono:.4f}", f"{step.max():.5f}", len(E)])
            if (i + 1) % 300 == 0:
                out.flush()
                print(f"[{i+1}/{len(dirs)}] {(time.time()-t0)/(i+1):.2f}s/vid", flush=True)
    out.close()
    print("E55_SELFCONS_DONE", flush=True)


if __name__ == "__main__":
    main()
