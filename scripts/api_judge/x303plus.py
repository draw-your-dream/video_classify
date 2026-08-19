#!/usr/bin/env python3
"""X303+ 基座:X303 原始特征 ⊕ newref ⊕ subjcons ⊕ prop,冠军 LGBM 配置,
分组 10 折 OOF 标量 → data/pbase/out/x303plus_oof.npy(与 r1a 同协议、可直接替换)。"""
import csv
import json
from pathlib import Path

import numpy as np

D = Path.home() / "tutu-video-eval/data"
OUT = D / "pbase/out"

vids = json.load(open(OUT / "X303_vids.json"))
vids = [v if v.endswith(".mp4") else v + ".mp4" for v in vids]
idx = {v: i for i, v in enumerate(vids)}
N = len(vids)
mapr = {r["filename"]: r for r in csv.DictReader(open(D / "api_judge_video_image_map.csv", encoding="utf-8-sig"))}
y = np.array([1 if mapr[v]["grade"] == "bad" else 0 for v in vids])
groups = np.array([mapr[v]["source_sha"] for v in vids])

X = np.load(OUT / "X303_new.npy")
ext = []
for fn in ["newref_feats.csv", "subjcons_1233.csv", "prop_timeline.csv"]:
    rows = list(csv.DictReader(open(OUT / fn)))
    def is_num(x):
        try:
            float(x or 0); return True
        except ValueError:
            return False
    keys = [k for k in rows[0] if k not in ("filename", "label") and is_num(rows[0][k])]
    M = np.zeros((N, len(keys)))
    for r in rows:
        if r["filename"] in idx:
            M[idx[r["filename"]]] = [float(r[k] or 0) for k in keys]
    ext.append(M)
    print(f"{fn}: +{len(keys)}")
Xp = np.hstack([X] + ext)
print("X303+ shape:", Xp.shape)

import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

CFG = dict(num_leaves=3, n_estimators=181, learning_rate=0.0117, min_child_samples=108,
           scale_pos_weight=0.8539, feature_fraction=0.6769, bagging_fraction=0.769,
           bagging_freq=1, random_state=42, verbose=-1)

oof = np.full(N, np.nan)
for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(Xp, y, groups):
    m = lgb.LGBMClassifier(**CFG).fit(Xp[tr], y[tr])
    oof[te] = m.predict_proba(Xp[te])[:, 1]
np.save(OUT / "x303plus_oof.npy", oof)

from scipy.stats import rankdata
r = rankdata(oof); pos = r[y == 1]
a = (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (y == 0).sum())
r0 = np.load(OUT / "r1a_tuned_oof.npy")
rr = rankdata(r0); pos0 = rr[y == 1]
a0 = (pos0.sum() - len(pos0) * (len(pos0) + 1) / 2) / (len(pos0) * (y == 0).sum())
print(f"X303+ OOF AUC={a:.4f} (旧 r1a={a0:.4f})")
