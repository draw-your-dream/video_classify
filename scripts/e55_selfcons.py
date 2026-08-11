#!/usr/bin/env python
"""E55:逐帧自一致性(2026-08-11 预注册,自主新想法批二)。

机理:衣服/配饰改变、穿模、突增物体、帧跳变——本质都是"某些帧与本视频其它帧不一致"。
视频级特征把单帧异常平均掉了;此处按帧算 dinov2 嵌入对本视频中位嵌入的距离:
max_selfdist(单帧走形)/ fl_dist(首末漂移=衣服类)/ drift_mono(单调漂移 vs 突变)。
用角色裁剪(crops_v3)排除背景干扰。GPU 批量,全语料约 10 分钟。
输出 data/selfcons_full.csv。
"""
from __future__ import annotations

import csv
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = "/root/mech"


def main():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(f"{ROOT}/models/dinov2-base")
    model = AutoModel.from_pretrained(f"{ROOT}/models/dinov2-base").to("cuda").eval().half()
    dirs = sorted(p for p in glob.glob(f"{ROOT}/data/crops_v3/*/*") if os.path.isdir(p))
    print(f"视频 {len(dirs)}", flush=True)
    out = open(f"{ROOT}/data/selfcons_full.csv", "w", newline="")
    w = csv.writer(out)
    w.writerow(["rel", "max_selfdist", "med_selfdist", "fl_dist", "drift_mono", "jump_max", "n"])
    import time
    t0 = time.time()
    for di, d in enumerate(dirs):
        rel = "/".join(d.split(os.sep)[-2:]) + ".mp4"
        fs = sorted(glob.glob(os.path.join(d, "f*.jpg")))[:16]
        if len(fs) < 8:
            w.writerow([rel] + [""] * 6)
            continue
        ims = []
        for f in fs:
            im = cv2.imread(f)
            if im is None:
                continue
            ims.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        if len(ims) < 8:
            w.writerow([rel] + [""] * 6)
            continue
        with torch.inference_mode():
            inp = proc(images=ims, return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].half()
            E = model(**inp).last_hidden_state[:, 0]          # CLS
            E = torch.nn.functional.normalize(E, dim=1).float().cpu().numpy()
        med = np.median(E, axis=0)
        med /= max(1e-9, np.linalg.norm(med))
        sd = 1 - E @ med
        fl = float(1 - E[0] @ E[-1])
        step = 1 - (E[:-1] * E[1:]).sum(1)
        cum = 1 - E @ E[0]
        mono = float(abs(np.diff(cum).sum()) / (np.abs(np.diff(cum)).sum() + 1e-9))
        w.writerow([rel, f"{sd.max():.5f}", f"{np.median(sd):.5f}", f"{fl:.5f}",
                    f"{mono:.4f}", f"{step.max():.5f}", len(ims)])
        if (di + 1) % 300 == 0:
            out.flush()
            print(f"[{di+1}/{len(dirs)}] {(time.time()-t0)/(di+1):.2f}s/vid", flush=True)
    out.close()
    print("E55_SELFCONS_DONE", flush=True)


if __name__ == "__main__":
    main()
