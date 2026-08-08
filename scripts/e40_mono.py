"""E40:单调约束与交互约束(2026-08-08 预注册)。基准 0.3218。
先用冠军的特征重要度+方向性(该特征高时 bad 率是升是降)自动推方向,再加 monotone_constraints。
变体:仅对 top-K 高重要度且方向稳定(5折内符号一致)的特征加约束;K ∈ {20, 50, 100}。
附:interaction_constraints 变体——限制 15 专家分与 320 原始特征各自成组交互。"""
import json
import pickle

import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata, spearmanr
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
B = np.hstack([oof15, X_tr])
n15 = oof15.shape[1]
champ = json.load(open("data/s3/e18_champion.json"))["params"]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))


def run(mono=None, inter=None, tag=""):
    o = np.zeros(len(y_tr))
    for a, b in folds:
        kw = dict(num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
                  min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"],
                  feature_fraction=champ["ff"], bagging_fraction=champ["bf"], bagging_freq=1,
                  random_state=42, verbose=-1)
        if mono is not None:
            kw["monotone_constraints"] = mono
        if inter is not None:
            kw["interaction_constraints"] = inter
        m = lgb.LGBMClassifier(**kw)
        m.fit(B[a], y_tr[a])
        o[b] = m.predict_proba(B[b])[:, 1]
    print(f"[E40 {tag}] train-OOF gn@95 = {gn(o, y_tr):.4f}", flush=True)
    return o


run(tag="基准复现")

# 方向性:5 折内 spearman(feature, y) 的符号一致性
signs = np.zeros((len(folds), B.shape[1]))
for fi, (a, _) in enumerate(folds):
    for j in range(B.shape[1]):
        r = spearmanr(B[a][:, j], y_tr[a]).statistic
        signs[fi, j] = 0 if not np.isfinite(r) else np.sign(r) * (abs(r) > 0.03)
stable = (np.abs(signs.sum(0)) == len(folds))
direction = np.sign(signs.sum(0)).astype(int)
m0 = lgb.LGBMClassifier(num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
                        min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"],
                        feature_fraction=champ["ff"], bagging_fraction=champ["bf"], bagging_freq=1,
                        random_state=42, verbose=-1)
m0.fit(B, y_tr)
imp = m0.feature_importances_
print(f"方向稳定特征 {int(stable.sum())}/{B.shape[1]}", flush=True)

for K in (20, 50, 100):
    cand = [j for j in np.argsort(-imp) if stable[j]][:K]
    mono = [0] * B.shape[1]
    for j in cand:
        mono[j] = int(direction[j])
    run(mono=mono, tag=f"单调top{K}")

inter = [list(range(n15)), list(range(n15, B.shape[1]))]
run(inter=inter, tag="交互分组(专家分|原始特征)")
print("E40_DONE", flush=True)
