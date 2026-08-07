"""E37:E36b(VLM微调)与 E34a(端到端CNN)并入冠军的增量测试。基准 0.3218。"""
import json
import os
import pickle

import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


oof15, ev15, y_tr, y_ev, *_ = pickle.load(open("upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15 = np.asarray(oof15, float)
y_tr = np.asarray(y_tr, int)
z = np.load("upstream/cache_v3/_full_raw_v2.npz")
X_tr = z["X_tr"].astype(float)
md = np.nanmedian(X_tr, axis=0)
ii = np.where(~np.isfinite(X_tr))
X_tr[ii] = np.take(md, ii[1])
tr = [json.loads(l) for l in open("splits/train_v3.jsonl")]
B = np.hstack([oof15, X_tr])
champ = json.load(open("data/s3/e18_champion.json"))["params"]


def mk():
    return lgb.LGBMClassifier(
        num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
        min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"],
        feature_fraction=champ["ff"], bagging_fraction=champ["bf"], bagging_freq=1,
        random_state=42, verbose=-1)


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))
cols = {}
for t in ("e36b", "e34a"):
    o = np.load(f"data/s3/{t}_oof.npy")
    rels = json.load(open(f"data/s3/{t}_rels.json"))
    m = {os.path.basename(r): v for r, v in zip(rels, o)}
    c = np.array([m.get(r["video"], np.nan) for r in tr], float)
    fin = np.isfinite(c)
    c[~fin] = np.nanmedian(c[fin])
    cols[t] = c
    rk = rankdata(c)
    n1 = y_tr.sum()
    print(f"{t}: 对齐AUC {(rk[y_tr==1].sum()-n1*(n1+1)/2)/(n1*(len(y_tr)-n1)):.4f} 覆盖{fin.mean():.0%}", flush=True)
print(f"e36b vs e34a 秩相关 {np.corrcoef(rankdata(cols['e36b']), rankdata(cols['e34a']))[0,1]:.3f}", flush=True)

for name, TR in (("冠军基准", B),
                 ("冠军+e36b(VLM微调)", np.hstack([B, cols["e36b"][:, None]])),
                 ("冠军+e36b+e34a(双深度信号)", np.hstack([B, cols["e36b"][:, None], cols["e34a"][:, None]]))):
    o = np.zeros(len(y_tr))
    for a, b in folds:
        m2 = mk()
        m2.fit(TR[a], y_tr[a])
        o[b] = m2.predict_proba(TR[b])[:, 1]
    print(f"{name}: train-OOF gn@95 = {gn(o, y_tr):.4f}", flush=True)

base = np.load("data/s3/e18_champion_train_oof.npy")
for lam in (0.05, 0.1, 0.2):
    print(f"秩融合 冠军+{lam}×e36b: {gn(rankdata(base) + lam*rankdata(cols['e36b']), y_tr):.4f}", flush=True)
print("E37_DONE", flush=True)
