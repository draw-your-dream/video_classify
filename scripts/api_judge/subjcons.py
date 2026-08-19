#!/usr/bin/env python3
"""VBench-I2V 式 Subject Consistency:源图 vs 视频逐帧相似度曲线(SigLIP2 + DINOv2 双编码器)。
源图来自 video↔image 映射(1126/1233 有),缺失回退首帧并打 has_src=0。
顺序解码取 10 帧;输出 /workspace/r2/subjcons_1233.csv,marker=SUBJCONS_DONE。"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

R2 = Path("/workspace/r2")
N_FRAMES = 10

from transformers import AutoImageProcessor, AutoModel

encs = {}
for tag, mn in [("sig", "google/siglip2-so400m-patch14-384"),
                ("dino", "facebook/dinov2-base")]:
    proc = AutoImageProcessor.from_pretrained(mn)
    mod = AutoModel.from_pretrained(mn, torch_dtype=torch.float16).to("cuda").eval()
    encs[tag] = (proc, mod)


def embed(tag, pils):
    proc, mod = encs[tag]
    out = []
    for i in range(0, len(pils), 32):
        inputs = proc(images=pils[i:i + 32], return_tensors="pt").to("cuda")
        with torch.no_grad():
            if tag == "sig":
                o = mod.vision_model(pixel_values=inputs["pixel_values"].half())
                f = o.pooler_output
            else:
                o = mod(pixel_values=inputs["pixel_values"].half())
                f = o.last_hidden_state[:, 0]  # CLS
        out.append(f.float().cpu().numpy())
    E = np.concatenate(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def curve_feats(prefix, src, F):
    c = F @ src  # (T,)
    adj = (F[1:] * F[:-1]).sum(1)  # 相邻帧自相似
    T = len(c)
    slope = np.polyfit(np.arange(T), c, 1)[0] if T > 1 else 0.0
    return {
        f"{prefix}_mean": c.mean(), f"{prefix}_min": c.min(), f"{prefix}_last": c[-1],
        f"{prefix}_first": c[0], f"{prefix}_std": c.std(), f"{prefix}_slope": slope,
        f"{prefix}_drop": c.max() - c.min(), f"{prefix}_argmin": float(np.argmin(c)) / max(T - 1, 1),
        f"{prefix}_adj_mean": adj.mean(), f"{prefix}_adj_min": adj.min(),
    }


rows = list(csv.DictReader(open(R2 / "data/api_judge_video_image_map.csv", encoding="utf-8-sig")))
out_rows = []
done_fns = set()
outp = R2 / "subjcons_1233.csv"
if outp.exists():
    for r in csv.DictReader(open(outp)):
        done_fns.add(r["filename"])
        out_rows.append(r)

fieldnames = None
import time
t0 = time.time()
for i, r in enumerate(rows):
    fn = r["filename"]
    if fn in done_fns:
        continue
    vp = R2 / "videos" / fn
    if not vp.exists():
        continue
    cap = cv2.VideoCapture(str(vp))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    if len(frames) < 2:
        continue
    idx = np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)
    pils = []
    for j in idx:
        im = Image.fromarray(cv2.cvtColor(frames[j], cv2.COLOR_BGR2RGB))
        im.thumbnail((512, 512))
        pils.append(im)
    src_fp = R2 / "qcimgs" / f"{r['image_dataset']}__SLASH__{r['image_sample_id']}.png"
    has_src = 1 if (r["image_sample_id"] and src_fp.exists() and src_fp.stat().st_size > 0) else 0
    if has_src:
        try:
            sim = Image.open(src_fp).convert("RGB")
            sim.thumbnail((512, 512))
        except Exception:
            has_src = 0
    if not has_src:
        sim = pils[0]
    rec = {"filename": fn, "has_src": has_src}
    for tag in ("sig", "dino"):
        E = embed(tag, [sim] + pils)
        rec.update(curve_feats(tag, E[0], E[1:]))
    out_rows.append(rec)
    if fieldnames is None:
        fieldnames = list(rec.keys())
    if (len(out_rows) % 50 == 0) or (i == len(rows) - 1):
        fieldnames = fieldnames or list(out_rows[0].keys())
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)
        print(f"[{len(out_rows)}/{len(rows)}] {(time.time()-t0)/max(1,len(out_rows)-len(done_fns)):.2f}s/条", flush=True)

with open(outp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames or list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print("SUBJCONS_DONE", flush=True)
