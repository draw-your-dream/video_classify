#!/usr/bin/env python3
"""P2 特征贪心反向剪枝:逐列试删(3-seed dev br@80 为准),删到无提升;
最终配置用另一组 seed(47,48,49)复核,防选择噪声自欺。"""
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
tracks = np.array([mapr[v]["track"] for v in vids])
skus = np.array([v.split("__")[2] for v in vids])
lb = json.load(open(D / "lockbox_split.json"))
dev_set = set(lb["dev"])
dev_mask = np.array([(g in dev_set) or (v in dev_set) for g, v in zip(groups, vids)])

from scipy.stats import rankdata


def rankpct(x):
    return rankdata(x) / len(x)


cols, names = [], []


def add(name, arr):
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    for j in range(arr.shape[1]):
        cols.append(arr[:, j])
        names.append(f"{name}{'' if arr.shape[1]==1 else '_'+str(j)}")


add("x303plus", np.load(OUT / "x303plus_oof.npy"))
z = np.load(OUT / "r1_oof.npz", allow_pickle=True)
add("r1b", z["r1b"]); add("r1c", z["r1c"])
s1 = json.load(open(OUT / "flash_full_1233.json"))
s2 = json.load(open(OUT / "flash_run2_1233.json"))
f1 = np.array([s1.get(v, 50) for v in vids], float)
f2 = np.array([s2.get(v, 50) for v in vids], float)
flash_avg = (rankpct(f1) + rankpct(f2)) / 2
fl_pct = np.zeros(N)
for t in set(tracks):
    m = tracks == t
    fl_pct[m] = rankpct(flash_avg[m])
add("flash_avg", flash_avg); add("flash_trkpct", fl_pct)
ip = np.zeros((N, 2))
for r in csv.DictReader(open(OUT / "imgprobe_1233.csv")):
    if r["filename"] in idx:
        ip[idx[r["filename"]]] = [float(r["p_lr"]), float(r["p_gbm"])]
add("imgprobe_lr", ip[:, 0]); add("imgprobe_gbm", ip[:, 1])
rows = list(csv.DictReader(open(OUT / "newref_feats.csv")))
keys = [k for k in rows[0] if k not in ("filename", "label")]
M = np.zeros((N, len(keys)))
for r in rows:
    if r["filename"] in idx:
        M[idx[r["filename"]]] = [float(r[k] or 0) for k in keys]
for j, k in enumerate(keys):
    add(f"nr_{k}", M[:, j])
for t in sorted(set(tracks)):
    add(f"TR_{t}", (tracks == t).astype(float))
for s in sorted(set(skus)):
    add(f"SKU_{s}", (skus == s).astype(float))

X = np.stack(cols, 1)
print(f"start cols: {X.shape[1]}")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

Xd_all, yd, gd = X[dev_mask], y[dev_mask], groups[dev_mask]


def br80(scores, yy, rel=0.8):
    gn = np.sort(scores[yy == 0]); b = scores[yy == 1]
    k = int(np.floor(rel * len(gn)))
    t = gn[k - 1]
    nb = (gn < t).sum(); ne = (gn == t).sum()
    fr = (k - nb) / ne
    return ((b > t).sum() + (b == t).sum() * (1 - fr)) / len(b)


def evaluate(active, seeds):
    Xd = Xd_all[:, active]
    vals = []
    for seed in seeds:
        oof = np.full(len(yd), np.nan)
        for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(Xd, yd, gd):
            sc = StandardScaler().fit(Xd[tr])
            lr = LogisticRegression(C=100, max_iter=5000, class_weight={0: 1, 1: 2})
            lr.fit(sc.transform(Xd[tr]), yd[tr])
            oof[te] = lr.predict_proba(sc.transform(Xd[te]))[:, 1]
        vals.append(br80(oof, yd))
    return np.mean(vals)


SEL = (42, 43, 44)
CONF = (47, 48, 49)
active = list(range(X.shape[1]))
base = evaluate(active, SEL)
print(f"P2 baseline (3-seed): {base:.4f}", flush=True)
improved = True
while improved and len(active) > 5:
    improved = False
    best_gain, best_drop = 0.0, None
    for c in list(active):
        cand = [a for a in active if a != c]
        v = evaluate(cand, SEL)
        if v - base > best_gain + 1e-6:
            best_gain, best_drop, best_val = v - base, c, v
    if best_drop is not None:
        active = [a for a in active if a != best_drop]
        base = base + best_gain
        print(f"drop {names[best_drop]:<22} -> {base:.4f} ({len(active)} cols)", flush=True)
        improved = True

print("FINAL cols:", [names[a] for a in active], flush=True)
print(f"selection-seed br@80: {base:.4f}", flush=True)
conf = evaluate(active, CONF)
full_conf = evaluate(list(range(X.shape[1])), CONF)
print(f"confirm-seed (47,48,49): pruned={conf:.4f} vs full={full_conf:.4f}", flush=True)
json.dump([names[a] for a in active], open(OUT / "prune_p2_cols.json", "w"), ensure_ascii=False)
print("PRUNE_DONE", flush=True)
