#!/usr/bin/env python
"""E62:组合器超参重扫(2026-08-12,用户指令:仔细看调参能否提升)。

与 E18 原搜索的差异:①选择判据 = 3 折种子 OOF gn@95 均值(E18 是单种子 trial,
E42 已证单种子 +0.8pt 可为噪声);②空间放宽(leaves/est/lr/mcs/spw/ff/bf 随机 90 配置)。
晋级判准(冻结):3 种子均值 > 冠军同判据均值 + 0.010(可验证分辨率),才做 eval 单发。
纯 CPU。输出 data/hparam_sweep.csv。
"""
from __future__ import annotations

import csv
import json
import pickle

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

ROOT = "/root/mech"


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


oof15, _e, y_tr, *_ = pickle.load(open(f"{ROOT}/upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15 = np.asarray(oof15, float)
y = np.asarray(y_tr, int)
z = np.load(f"{ROOT}/upstream/cache_v3/_full_raw_v2.npz")
X = z["X_tr"].astype(float)
md = np.nanmedian(X, axis=0)
ii = np.where(~np.isfinite(X))
X[ii] = np.take(md, ii[1])
B = np.hstack([oof15, X])
SEEDS = (42, 101, 202)


def score(cfg):
    out = []
    for sd in SEEDS:
        o = np.zeros(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=sd).split(B, y):
            m = lgb.LGBMClassifier(verbose=-1, random_state=42, n_jobs=40, bagging_freq=1, **cfg)
            m.fit(B[a], y[a])
            o[b] = m.predict_proba(B[b])[:, 1]
        out.append(gn(o, y))
    return out


champ = dict(num_leaves=7, n_estimators=122, learning_rate=0.0317, min_child_samples=60,
             scale_pos_weight=1.0689, feature_fraction=0.5030, bagging_fraction=0.9871)
cs = score(champ)
print(f"冠军 3种子: {[round(x,4) for x in cs]} 均值 {np.mean(cs):.4f}", flush=True)
BAR = np.mean(cs) + 0.010
print(f"晋级线 = {BAR:.4f}", flush=True)

rng = np.random.RandomState(0)
rows = []
best = (np.mean(cs), "champ", champ)
with open(f"{ROOT}/data/hparam_sweep.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["i", "mean3", "s42", "s101", "s202", "cfg"])
    w.writerow([-1, round(np.mean(cs), 4)] + [round(x, 4) for x in cs] + [json.dumps(champ)])
    for i in range(90):
        cfg = dict(
            num_leaves=int(rng.choice([5, 7, 10, 15, 23])),
            n_estimators=int(rng.choice([80, 122, 180, 260, 350])),
            learning_rate=float(rng.choice([0.02, 0.032, 0.05, 0.08])),
            min_child_samples=int(rng.choice([20, 40, 60, 90, 130])),
            scale_pos_weight=float(rng.choice([1.0, 1.07, 1.3, 1.6])),
            feature_fraction=float(rng.choice([0.35, 0.5, 0.65, 0.8])),
            bagging_fraction=float(rng.choice([0.7, 0.85, 0.99])),
        )
        ss = score(cfg)
        m3 = float(np.mean(ss))
        w.writerow([i, round(m3, 4)] + [round(x, 4) for x in ss] + [json.dumps(cfg)])
        f.flush()
        if m3 > best[0]:
            best = (m3, f"cfg{i}", cfg)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/90] 当前最优 {best[0]:.4f} ({best[1]})", flush=True)
print(f"最优 {best[0]:.4f} {best[1]} {best[2]}", flush=True)
print(f"判决: {'✔ 过晋级线' if best[0] > BAR else '✘ 未过晋级线'} (线 {BAR:.4f})", flush=True)
print("E62_DONE", flush=True)
