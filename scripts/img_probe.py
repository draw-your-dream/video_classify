#!/usr/bin/env python3
"""图片缺陷探针:4553 张带标图片(2962+0813)→ SigLIP2-so400m 嵌入 → LGBM/LR 探针;
再给 1233 条视频首帧打分 → /workspace/r2/imgprobe_1233.csv。
图片侧自评:按源 sha 分组 5 折 CV 的 AUC(bad vs 其余)。"""
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

R2 = Path("/workspace/r2")

from transformers import AutoImageProcessor, AutoModel
mn = "google/siglip2-so400m-patch14-384"
proc = AutoImageProcessor.from_pretrained(mn)
model = AutoModel.from_pretrained(mn, torch_dtype=torch.float16).to("cuda").eval()

def embed(pils):
    out = []
    B = 32
    for i in range(0, len(pils), B):
        inputs = proc(images=pils[i:i+B], return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.vision_model(pixel_values=inputs["pixel_values"].half())
            f = o.pooler_output.float().cpu().numpy()
        out.append(f)
    E = np.concatenate(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

# ---- 图片池 ----
rows = []
for f in ["data/tutu_image_annotations_2962.csv", "data/tutu_image_annotations_0813.csv"]:
    p = R2 / f
    if not p.exists():
        p = R2 / "data" / Path(f).name
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        rows.append((r["dataset"], r["sample_id"], r["label"]))
imgs, labs, shas = [], [], []
miss = 0
t0 = time.time()
E_list, meta = [], []
batch_pils = []
for d, s, l in rows:
    fp = R2 / "qcimgs" / f"{d}__SLASH__{s}.png"
    if not fp.exists() or fp.stat().st_size == 0:
        miss += 1
        continue
    try:
        im = Image.open(fp).convert("RGB")
        im.thumbnail((512, 512))
        batch_pils.append(im)
        meta.append((s.split("__")[0], l))
    except Exception:
        miss += 1
    if len(batch_pils) >= 256:
        E_list.append(embed(batch_pils))
        batch_pils = []
        print(f"embedded {sum(e.shape[0] for e in E_list)} ({time.time()-t0:.0f}s)", flush=True)
if batch_pils:
    E_list.append(embed(batch_pils))
E = np.concatenate(E_list)
print(f"images embedded: {E.shape}, missing {miss}", flush=True)
y = np.array([1 if l == "bad" else 0 for _, l in meta])
groups = np.array([s for s, _ in meta])

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import rankdata

oof = np.full(len(y), np.nan)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(E, y, groups):
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(E[tr], y[tr])
    gb = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                            min_child_samples=20, random_state=42, verbose=-1).fit(E[tr], y[tr])
    oof[te] = (rankdata(lr.predict_proba(E[te])[:, 1]) +
               rankdata(gb.predict_proba(E[te])[:, 1])) / 2 / len(te)
r = rankdata(oof); pos = r[y == 1]
auc = (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (y == 0).sum())
print(f"图片侧探针 CV AUC = {auc:.4f} (n={len(y)}, bad率={y.mean():.2f})", flush=True)

# 全量拟合 → 视频首帧打分
lr = LogisticRegression(C=1.0, max_iter=3000).fit(E, y)
gb = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                        min_child_samples=20, random_state=42, verbose=-1).fit(E, y)

targets = [json.loads(l) for l in open(R2 / "pbase/upstream/splits/train_v2.jsonl")]
out = []
pils, fns = [], []
for e in targets:
    fn = Path(e["video"]).name
    cap = cv2.VideoCapture(str(R2 / "videos" / fn))
    ok, im = cap.read()
    cap.release()
    if not ok:
        continue
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    pil.thumbnail((512, 512))
    pils.append(pil)
    fns.append(fn)
    if len(pils) >= 256:
        Ev = embed(pils)
        for f2, plr, pgb in zip(fns, lr.predict_proba(Ev)[:, 1], gb.predict_proba(Ev)[:, 1]):
            out.append((f2, float(plr), float(pgb)))
        pils, fns = [], []
        print(f"scored {len(out)}", flush=True)
if pils:
    Ev = embed(pils)
    for f2, plr, pgb in zip(fns, lr.predict_proba(Ev)[:, 1], gb.predict_proba(Ev)[:, 1]):
        out.append((f2, float(plr), float(pgb)))
with open(R2 / "imgprobe_1233.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "p_lr", "p_gbm"])
    w.writerows(out)
print(f"IMGPROBE_DONE {len(out)}", flush=True)
