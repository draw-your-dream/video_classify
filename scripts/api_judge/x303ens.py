#!/usr/bin/env python3
"""基座集成:sweep1/sweep2 最优配置 × 多随机种,OOF 秩平均 → x303ens_oof.npy。"""
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
for fn in ["newref_feats.csv", "subjcons_1233.csv", "prop_timeline.csv", "newref2_1233.csv"]:
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
Xp = np.hstack([X] + ext)

import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

cfg1 = json.load(open(OUT / "x303plus_cfg.json"))
cfg2 = json.load(open(OUT / "x303plus2_cfg.json"))
oofs = []
for cfg, seeds in [(cfg2, (42, 7, 13)), (cfg1, (42, 7))]:
    for rs in seeds:
        c = dict(cfg); c["random_state"] = rs
        oof = np.full(N, np.nan)
        for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(Xp, y, groups):
            m = lgb.LGBMClassifier(**c).fit(Xp[tr], y[tr])
            oof[te] = m.predict_proba(Xp[te])[:, 1]
        oofs.append(rankdata(oof) / N)
        print(f"member cfg={cfg is cfg2 and 2 or 1} rs={rs} done", flush=True)
ens = np.mean(oofs, 0)
np.save(OUT / "x303ens_oof.npy", ens)
r = rankdata(ens); pos = r[y == 1]
a = (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (y == 0).sum())
print(f"ensemble base AUC={a:.4f}", flush=True)
print("ENS_DONE", flush=True)
