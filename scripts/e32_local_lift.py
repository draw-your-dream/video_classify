#!/usr/bin/env python
"""E32:零新特征的放行率提升五路(2026-08-07 预注册)。全部 train-OOF 过门制,基准 0.3218。
A 种子集成x10 | B 异构集成(冠军+LR原始+中容量LGBM) | C 分源分位归一 | D 冠军⊕C5秩融合 | E 剪枝重训。"""
import json, pickle
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())

oof15, ev15, y_tr, y_ev, *_ = pickle.load(open("upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15 = np.asarray(oof15, float); y_tr = np.asarray(y_tr, int)
z = np.load("upstream/cache_v3/_full_raw_v2.npz")
X_tr = z["X_tr"].astype(float)
md = np.nanmedian(X_tr, axis=0)
ii = np.where(~np.isfinite(X_tr)); X_tr[ii] = np.take(md, ii[1])
tr = [json.loads(l) for l in open("splits/train_v3.jsonl")]
src = np.array([0 if "ti2i2v" in r["abs_path"] else (1 if "rlhf" in r["abs_path"] else 2) for r in tr])
B = np.hstack([oof15, X_tr])
champ = json.load(open("data/s3/e18_champion.json"))["params"]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))

def champ_model(seed=42):
    return lgb.LGBMClassifier(num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
                              min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"],
                              feature_fraction=champ["ff"], bagging_fraction=champ["bf"], bagging_freq=1,
                              random_state=seed, verbose=-1)

def oof_with(mk):
    o = np.zeros(len(y_tr))
    for a, b in folds:
        m = mk(); m.fit(B[a], y_tr[a]); o[b] = m.predict_proba(B[b])[:, 1]
    return o

base = np.load("data/s3/e18_champion_train_oof.npy")
print(f"基准 {gn(base, y_tr):.4f}", flush=True)

# A 种子集成
os_ = []
for sd in range(10):
    os_.append(rankdata(oof_with(lambda: champ_model(sd))))
    if sd in (2, 4, 9):
        gA = gn(np.mean(os_, 0), y_tr)
        print(f"[A 种子集成x{sd+1}] {gA:.4f}", flush=True)
ensA = np.mean(os_, 0)

# B 异构集成
o_lr = np.zeros(len(y_tr))
sc = StandardScaler().fit(B)
Bs = sc.transform(B)
for a, b in folds:
    m = LogisticRegression(C=0.05, max_iter=4000, class_weight={0: 1, 1: 2})
    m.fit(Bs[a], y_tr[a]); o_lr[b] = m.predict_proba(Bs[b])[:, 1]
print(f"  (LR原始单独 {gn(o_lr, y_tr):.4f})", flush=True)
o_mid = oof_with(lambda: lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                            min_child_samples=20, scale_pos_weight=2.0,
                                            random_state=42, verbose=-1))
print(f"  (中容量LGBM单独 {gn(o_mid, y_tr):.4f})", flush=True)
for w in ((3, 1, 1), (2, 1, 1), (1, 1, 1), (4, 1, 0), (3, 0, 1)):
    e = w[0]*ensA/10 + w[1]*rankdata(o_lr) + w[2]*rankdata(o_mid)
    print(f"[B 异构 w={w}] {gn(e, y_tr):.4f}", flush=True)

# C 分源分位归一(基准分数 + 种子集成分数两版)
for name, s in (("基准", base), ("种子集成", ensA)):
    ns = np.zeros(len(s))
    for g in (0, 1, 2):
        m2 = src == g
        ns[m2] = rankdata(s[m2]) / m2.sum()
    print(f"[C 分源归一({name})] {gn(ns, y_tr):.4f}", flush=True)

# D 冠军⊕C5
c5 = pickle.load(open("data/s3/c5_lgbm_final.pkl", "rb")) if False else None
o_c5 = oof_with(lambda: lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                           min_child_samples=20, scale_pos_weight=2.0,
                                           random_state=7, verbose=-1))
for lam in (0.2, 0.35, 0.5):
    print(f"[D 冠军⊕C5 λ={lam}] {gn(rankdata(base) + lam*rankdata(o_c5), y_tr):.4f}", flush=True)

# E 剪枝重训
mimp = champ_model(); mimp.fit(B, y_tr)
imp = mimp.feature_importances_
for K in (200, 150, 100):
    keep = np.argsort(-imp)[:K]
    o = np.zeros(len(y_tr))
    for a, b in folds:
        m = champ_model(); m.fit(B[a][:, keep], y_tr[a]); o[b] = m.predict_proba(B[b][:, keep])[:, 1]
    print(f"[E 剪枝top{K}] {gn(o, y_tr):.4f}", flush=True)
np.save("data/s3/e32_seed_ens_oof.npy", ensA)
print("E32_DONE", flush=True)
