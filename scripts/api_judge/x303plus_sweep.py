#!/usr/bin/env python3
"""X303+ 346 维基座的 LGBM 超参随机搜索(60 trial,以 dev 内嵌套 br@80 为准)。
输出最优配置 + 用它重写 x303plus_oof.npy。"""
import csv
import json
import random
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
lb = json.load(open(D / "lockbox_split.json"))
dev_set = set(lb["dev"])
dev_mask = np.array([(g in dev_set) or (v in dev_set) for g, v in zip(groups, vids)])

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
Xp = np.hstack([X] + ext)

import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

Xd, yd, gd = Xp[dev_mask], y[dev_mask], groups[dev_mask]


def br80(scores, yy):
    gn = np.sort(scores[yy == 0]); b = scores[yy == 1]
    k = int(np.floor(0.8 * len(gn)))
    t = gn[k - 1]
    n_below = (gn < t).sum(); n_eq = (gn == t).sum()
    fr = (k - n_below) / n_eq
    return ((b > t).sum() + (b == t).sum() * (1 - fr)) / len(b)


rnd = random.Random(0)
best = None
for trial in range(60):
    cfg = dict(num_leaves=rnd.choice([3, 5, 7, 15]),
               n_estimators=rnd.randint(100, 500),
               learning_rate=10 ** rnd.uniform(-2.2, -1.2),
               min_child_samples=rnd.randint(30, 150),
               scale_pos_weight=rnd.uniform(0.6, 1.2),
               feature_fraction=rnd.uniform(0.4, 0.9),
               bagging_fraction=rnd.uniform(0.6, 0.95),
               bagging_freq=1, random_state=42, verbose=-1)
    oof = np.full(len(yd), np.nan)
    for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(Xd, yd, gd):
        m = lgb.LGBMClassifier(**cfg).fit(Xd[tr], yd[tr])
        oof[te] = m.predict_proba(Xd[te])[:, 1]
    b = br80(oof, yd)
    r = rankdata(oof); pos = r[yd == 1]
    a = (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (yd == 0).sum())
    if best is None or b > best[0]:
        best = (b, a, cfg)
        print(f"trial {trial}: br80={b:.4f} auc={a:.4f} {cfg}", flush=True)
print("BEST", best, flush=True)

# 用最优配置重写全量 OOF
cfg = best[2]
oof = np.full(N, np.nan)
for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(Xp, y, groups):
    m = lgb.LGBMClassifier(**cfg).fit(Xp[tr], y[tr])
    oof[te] = m.predict_proba(Xp[te])[:, 1]
np.save(OUT / "x303plus_oof.npy", oof)
json.dump(cfg, open(OUT / "x303plus_cfg.json", "w"))
print("SWEEP_DONE", flush=True)
