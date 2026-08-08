"""E43:交互约束的 eval 单发(2026-08-08,E42 双判准达成后执行)。
仅此一发,不调参、不挑种子;冠军配方为唯一对照。"""
import csv
import json
import pickle

import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


oof15, ev15, y_tr, y_ev, *_ = pickle.load(open("upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15, ev15 = np.asarray(oof15, float), np.asarray(ev15, float)
y_tr, y_ev = np.asarray(y_tr, int), np.asarray(y_ev, int)
z = np.load("upstream/cache_v3/_full_raw_v2.npz")
X_tr, X_ev = z["X_tr"].astype(float), z["X_ev"].astype(float)
md = np.nanmedian(X_tr, axis=0)
for A in (X_tr, X_ev):
    ii = np.where(~np.isfinite(A))
    A[ii] = np.take(md, ii[1])
B_tr, B_ev = np.hstack([oof15, X_tr]), np.hstack([ev15, X_ev])
n15 = oof15.shape[1]
D = B_tr.shape[1]
champ = json.load(open("data/s3/e18_champion.json"))["params"]
kw = dict(num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
          min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"],
          feature_fraction=champ["ff"], bagging_fraction=champ["bf"], bagging_freq=1,
          random_state=42, verbose=-1)
m = lgb.LGBMClassifier(**kw, interaction_constraints=[list(range(n15)), list(range(n15, D))])
m.fit(B_tr, y_tr)
p = m.predict_proba(B_ev)[:, 1]
print(f"E43 交互约束 eval单发: ev@95={gn(p, y_ev):.4f} ev@100={gn(p, y_ev, 1.0):.4f} "
      f"ev@90={gn(p, y_ev, 0.90):.4f} AUC={auc(p[y_ev == 1], p[y_ev == 0]):.4f}")
print("对照 E18 冠军:     ev@95=0.2913 ev@100=0.0409 ev@90=0.4334 AUC=0.7593")
ev_meta = [json.loads(l) for l in open("splits/eval_v3.jsonl")]
with open("data/s3/predictions_e43_eval.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["video", "label", "p_e43"])
    for r, pi in zip(ev_meta, p):
        w.writerow([r["video"], r["label"], f"{float(pi):.6f}"])
print("E43_DONE")
