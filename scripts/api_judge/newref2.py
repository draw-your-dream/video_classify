#!/usr/bin/env python3
"""newref v2:视频 16 帧 vs 官方视图,保真曲线加强版。
编码器:SigLIP2-so400m + SigLIP2-giant。特征:逐帧对本 SKU 视图取 max 的曲线统计
(mean/min/first/last/drop/std/slope)+ 最优视角切换次数 + 对全部 43 视图的 max。
输出 /workspace/r2/newref2_1233.csv,marker=NEWREF2_DONE。"""
import csv
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

R2 = Path("/workspace/r2")
NF = 16

from transformers import AutoImageProcessor, AutoModel

encs = {}
for tag, mn in [("s4", "google/siglip2-so400m-patch14-384"),
                ("sg", "google/siglip2-giant-opt-patch16-384")]:
    proc = AutoImageProcessor.from_pretrained(mn)
    mod = AutoModel.from_pretrained(mn, torch_dtype=torch.float16).to("cuda").eval()
    encs[tag] = (proc, mod)


def embed(tag, pils):
    proc, mod = encs[tag]
    out = []
    for i in range(0, len(pils), 32):
        inputs = proc(images=pils[i:i + 32], return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = mod.vision_model(pixel_values=inputs["pixel_values"].half())
        out.append(o.pooler_output.float().cpu().numpy())
    E = np.concatenate(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def load_pil(fp):
    im = Image.open(fp).convert("RGB")
    im.thumbnail((512, 512))
    return im


view_paths = sorted(glob.glob(str(R2 / "data/sku_ref_v2/views/*.jpg")))
view_sku = [Path(p).name.rsplit("_v", 1)[0] for p in view_paths]
V = {tag: embed(tag, [load_pil(p) for p in view_paths]) for tag in encs}
print(f"views {len(view_paths)}", flush=True)


def stats(prefix, c):
    T = len(c)
    slope = np.polyfit(np.arange(T), c, 1)[0] if T > 1 else 0.0
    return {f"{prefix}_mean": c.mean(), f"{prefix}_min": c.min(), f"{prefix}_first": c[0],
            f"{prefix}_last": c[-1], f"{prefix}_drop": c.max() - c.min(),
            f"{prefix}_std": c.std(), f"{prefix}_slope": slope}


rows = list(csv.DictReader(open(R2 / "data/api_judge_video_image_map.csv", encoding="utf-8-sig")))
outp = R2 / "newref2_1233.csv"
done, out_rows = set(), []
if outp.exists():
    for r in csv.DictReader(open(outp)):
        done.add(r["filename"])
        out_rows.append(r)
fieldnames = list(out_rows[0].keys()) if out_rows else None
t0 = time.time()
n_new = 0
for r in rows:
    fn = r["filename"]
    if fn in done:
        continue
    vp = R2 / "videos" / fn
    if not vp.exists():
        continue
    sku = fn.split("__")[2]
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
    idx = np.linspace(0, len(frames) - 1, NF).astype(int)
    pils = []
    for j in idx:
        im = Image.fromarray(cv2.cvtColor(frames[j], cv2.COLOR_BGR2RGB))
        im.thumbnail((512, 512))
        pils.append(im)
    rec = {"filename": fn}
    for tag in encs:
        F = embed(tag, pils)  # (NF, D)
        S = F @ V[tag].T  # (NF, 43)
        own = [i for i, s in enumerate(view_sku) if s == sku]
        c_own = S[:, own].max(1) if own else S.max(1)
        c_all = S.max(1)
        rec.update(stats(f"{tag}_own", c_own))
        rec[f"{tag}_all_min"] = c_all.min()
        rec[f"{tag}_all_mean"] = c_all.mean()
        if own:
            best = S[:, own].argmax(1)
            rec[f"{tag}_viewswitch"] = float((best[1:] != best[:-1]).sum()) / (NF - 1)
        else:
            rec[f"{tag}_viewswitch"] = 0.0
    out_rows.append(rec)
    n_new += 1
    if fieldnames is None:
        fieldnames = list(rec.keys())
    if n_new % 50 == 0:
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(out_rows)
        print(f"[{len(out_rows)}/{len(rows)}] {(time.time()-t0)/max(1,n_new):.2f}s/条", flush=True)

with open(outp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames or list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)
print("NEWREF2_DONE", flush=True)
