#!/usr/bin/env python3
"""参考条件化图片探针:SigLIP2 嵌入 ⊕ 对 43 张官方视图的余弦特征。
对比三种特征集的分组 5 折 CV AUC:emb-only(基线 0.610)/ cos-only / emb⊕cos。
全量拟合后给 1233 视频首帧打分 → refprobe_1233.csv;嵌入缓存到 npz 以便复用。"""
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
CACHE = R2 / "qcimg_emb.npz"
FRAME_CACHE = R2 / "frame_emb.npz"

from transformers import AutoImageProcessor, AutoModel
mn = "google/siglip2-so400m-patch14-384"
proc = AutoImageProcessor.from_pretrained(mn)
model = AutoModel.from_pretrained(mn, torch_dtype=torch.float16).to("cuda").eval()


def embed(pils):
    out = []
    for i in range(0, len(pils), 32):
        inputs = proc(images=pils[i:i + 32], return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.vision_model(pixel_values=inputs["pixel_values"].half())
        out.append(o.pooler_output.float().cpu().numpy())
    E = np.concatenate(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def load_pil(fp):
    im = Image.open(fp).convert("RGB")
    im.thumbnail((512, 512))
    return im


# ---- 官方视图 ----
view_paths = sorted(glob.glob(str(R2 / "data/sku_ref_v2/views/*.jpg")))
V = embed([load_pil(p) for p in view_paths])
view_sku = [Path(p).name.rsplit("_v", 1)[0] for p in view_paths]
skus = sorted(set(view_sku))
print(f"views: {V.shape}, skus: {len(skus)}", flush=True)

# ---- 质检图嵌入(带缓存) ----
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    E, y, groups = z["E"], z["y"], z["groups"]
else:
    rows = []
    for f in ["tutu_image_annotations_2962.csv", "tutu_image_annotations_0813.csv"]:
        for r in csv.DictReader(open(R2 / "data" / f, encoding="utf-8-sig")):
            rows.append((r["dataset"], r["sample_id"], r["label"]))
    E_list, meta, pils = [], [], []
    t0 = time.time()
    for d, s, l in rows:
        fp = R2 / "qcimgs" / f"{d}__SLASH__{s}.png"
        if not fp.exists() or fp.stat().st_size == 0:
            continue
        try:
            pils.append(load_pil(fp))
            meta.append((s.split("__")[0], l))
        except Exception:
            continue
        if len(pils) >= 256:
            E_list.append(embed(pils))
            pils = []
            print(f"embedded {sum(e.shape[0] for e in E_list)} ({time.time()-t0:.0f}s)", flush=True)
    if pils:
        E_list.append(embed(pils))
    E = np.concatenate(E_list)
    y = np.array([1 if l == "bad" else 0 for _, l in meta])
    groups = np.array([s for s, _ in meta])
    np.savez_compressed(CACHE, E=E, y=y, groups=groups)
print(f"qcimgs: {E.shape}, bad率={y.mean():.2f}", flush=True)

# ---- 特征集 ----
C = E @ V.T  # 对 43 视图的余弦
per_sku_max = np.stack([C[:, [i for i, s in enumerate(view_sku) if s == sk]].max(1)
                        for sk in skus], axis=1)
cos_feats = np.hstack([C, per_sku_max,
                       C.max(1, keepdims=True), C.mean(1, keepdims=True),
                       per_sku_max.max(1, keepdims=True) - np.sort(per_sku_max, 1)[:, -2:-1]])
FEATSETS = {"emb": E, "cos": cos_feats, "emb+cos": np.hstack([E, cos_feats])}

import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


def cv_auc(X):
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups):
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr])
        gb = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                min_child_samples=20, random_state=42, verbose=-1).fit(X[tr], y[tr])
        oof[te] = (rankdata(lr.predict_proba(X[te])[:, 1]) +
                   rankdata(gb.predict_proba(X[te])[:, 1])) / 2 / len(te)
    r = rankdata(oof)
    pos = r[y == 1]
    return (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (y == 0).sum()), oof


best_name, best_auc, oofs = None, -1, {}
for name, X in FEATSETS.items():
    auc, oof = cv_auc(X)
    oofs[name] = oof
    print(f"refprobe CV AUC [{name}] = {auc:.4f}", flush=True)
    if auc > best_auc:
        best_name, best_auc = name, auc
np.savez_compressed(R2 / "refprobe_oof.npz", **oofs, y=y, groups=groups)
print(f"BEST: {best_name} {best_auc:.4f}", flush=True)

# ---- 全量拟合最优特征集 → 1233 首帧 ----
Xb = FEATSETS[best_name]
lr = LogisticRegression(C=1.0, max_iter=3000).fit(Xb, y)
gb = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                        min_child_samples=20, random_state=42, verbose=-1).fit(Xb, y)

if FRAME_CACHE.exists():
    z = np.load(FRAME_CACHE, allow_pickle=True)
    Ev_all, fns = z["E"], list(z["fns"])
else:
    targets = [json.loads(l) for l in open(R2 / "pbase/upstream/splits/train_v2.jsonl")]
    pils, fns, E_list = [], [], []
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
            E_list.append(embed(pils))
            pils = []
            print(f"frames embedded {sum(e.shape[0] for e in E_list)}", flush=True)
    if pils:
        E_list.append(embed(pils))
    Ev_all = np.concatenate(E_list)
    np.savez_compressed(FRAME_CACHE, E=Ev_all, fns=np.array(fns))

Cv = Ev_all @ V.T
psm = np.stack([Cv[:, [i for i, s in enumerate(view_sku) if s == sk]].max(1)
                for sk in skus], axis=1)
cosv = np.hstack([Cv, psm, Cv.max(1, keepdims=True), Cv.mean(1, keepdims=True),
                  psm.max(1, keepdims=True) - np.sort(psm, 1)[:, -2:-1]])
Xv = {"emb": Ev_all, "cos": cosv, "emb+cos": np.hstack([Ev_all, cosv])}[best_name]
with open(R2 / "refprobe_1233.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "p_lr", "p_gbm", "max_cos"])
    for fn, a, b, mc in zip(fns, lr.predict_proba(Xv)[:, 1], gb.predict_proba(Xv)[:, 1],
                            Cv.max(1)):
        w.writerow([fn, float(a), float(b), float(mc)])
print("REFPROBE_DONE", flush=True)
