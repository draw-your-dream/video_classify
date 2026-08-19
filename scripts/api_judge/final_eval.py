#!/usr/bin/env python3
"""终评:P2-pruned(30列)× {x303plus, x303plus2} × 8 seeds(42-49)。"""
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


def build(base_file):
    cols, names = [], []

    def add(name, arr):
        arr = np.asarray(arr, float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        for j in range(arr.shape[1]):
            cols.append(arr[:, j])
            names.append(f"{name}{'' if arr.shape[1]==1 else '_'+str(j)}")

    add("base", np.load(OUT / base_file))
    z = np.load(OUT / "r1_oof.npz", allow_pickle=True)
    add("r1b", z["r1b"])  # r1c 已剪
    s1 = json.load(open(OUT / "flash_full_1233.json"))
    s2 = json.load(open(OUT / "flash_run2_1233.json"))
    f1 = np.array([s1.get(v, 50) for v in vids], float)
    f2 = np.array([s2.get(v, 50) for v in vids], float)
    flash_avg = (rankpct(f1) + rankpct(f2)) / 2
    fl_pct = np.zeros(N)
    for t in set(tracks):
        m = tracks == t
        fl_pct[m] = rankpct(flash_avg[m])
    add("flash_avg", flash_avg)
    add("flash_trkpct", fl_pct)
    ip = np.zeros((N, 2))
    for r in csv.DictReader(open(OUT / "imgprobe_1233.csv")):
        if r["filename"] in idx:
            ip[idx[r["filename"]]] = [float(r["p_lr"]), float(r["p_gbm"])]
    add("imgprobe", ip)
    rows = list(csv.DictReader(open(OUT / "newref_feats.csv")))
    keys = [k for k in rows[0] if k not in ("filename", "label")]
    M = np.zeros((N, len(keys)))
    for r in rows:
        if r["filename"] in idx:
            M[idx[r["filename"]]] = [float(r[k] or 0) for k in keys]
    add("newref", M)
    for t in sorted(set(tracks)):
        if t != "base-props":  # 已剪
            add(f"TR_{t}", (tracks == t).astype(float))
    for s in sorted(set(skus)):
        add(f"SKU_{s}", (skus == s).astype(float))
    return np.stack(cols, 1)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def br_at(scores, yy, rel):
    gn = np.sort(scores[yy == 0]); b = scores[yy == 1]
    k = int(np.floor(rel * len(gn)))
    t = gn[k - 1]
    nb = (gn < t).sum(); ne = (gn == t).sum()
    fr = (k - nb) / ne
    return ((b > t).sum() + (b == t).sum() * (1 - fr)) / len(b)


def auc_of(scores, yy):
    r = rankdata(scores); pos = r[yy == 1]
    return (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (yy == 0).sum())


for base_file in ["x303plus_oof.npy", "x303ens_oof.npy"]:
    X = build(base_file)
    Xd, yd, gd = X[dev_mask], y[dev_mask], groups[dev_mask]
    b80s, b70s, aucs = [], [], []
    for seed in range(42, 50):
        oof = np.full(len(yd), np.nan)
        for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(Xd, yd, gd):
            sc = StandardScaler().fit(Xd[tr])
            lr = LogisticRegression(C=100, max_iter=5000, class_weight={0: 1, 1: 2})
            lr.fit(sc.transform(Xd[tr]), yd[tr])
            oof[te] = lr.predict_proba(sc.transform(Xd[te]))[:, 1]
        b80s.append(br_at(oof, yd, 0.8))
        b70s.append(br_at(oof, yd, 0.7))
        aucs.append(auc_of(oof, yd))
    print(f"{base_file}: br@80={np.mean(b80s):.4f}±{np.std(b80s):.4f} "
          f"br@70={np.mean(b70s):.4f} AUC={np.mean(aucs):.4f} "
          f"seeds={[round(x,3) for x in b80s]}", flush=True)
print("FINAL_EVAL_DONE", flush=True)
