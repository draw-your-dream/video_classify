"""E41:交互约束(E40 唯一过基准者 0.3271)的稳健性验证与深挖(2026-08-08 预注册)。
关卡①多种子:5 个不同折种子,基准 vs 交互约束成对比较,须多数胜且均值占优;
关卡②分组变体:换几种分组方式,看增益是否依赖具体分组(若只有一种分组灵→可疑);
关卡③双新切分保险(E20 式):两个全新切分上整套流程重演。
三关全过才申请 eval 单发。"""
import json
import pickle

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


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


def mk(inter=None):
    kw = dict(num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
              min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"],
              feature_fraction=champ["ff"], bagging_fraction=champ["bf"], bagging_freq=1,
              random_state=42, verbose=-1)
    if inter is not None:
        kw["interaction_constraints"] = inter
    return lgb.LGBMClassifier(**kw)


def oof_score(X, y, inter, seed):
    o = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = mk(inter)
        m.fit(X[a], y[a])
        o[b] = m.predict_proba(X[b])[:, 1]
    return gn(o, y)


GROUP2 = [list(range(n15)), list(range(n15, D))]
print("== 关卡①多种子(基准 vs 分两组) ==", flush=True)
wins, diffs = 0, []
for seed in (42, 101, 202, 303, 404):
    g0 = oof_score(B_tr, y_tr, None, seed)
    g1 = oof_score(B_tr, y_tr, GROUP2, seed)
    diffs.append(g1 - g0)
    wins += g1 > g0
    print(f"  seed{seed}: 基准 {g0:.4f} | 交互约束 {g1:.4f} | Δ {g1-g0:+.4f}", flush=True)
print(f"  胜 {wins}/5,平均 Δ {np.mean(diffs):+.4f}", flush=True)

print("== 关卡②分组变体(seed42) ==", flush=True)
rng = np.random.RandomState(0)
perm = rng.permutation(D)
VARIANTS = {
    "随机等分两组(对照)": [sorted(perm[:D // 2].tolist()), sorted(perm[D // 2:].tolist())],
    "三组(专家|原始前半|原始后半)": [list(range(n15)), list(range(n15, n15 + (D - n15) // 2)),
                                list(range(n15 + (D - n15) // 2, D))],
    "专家分独立(其余全交互)": [list(range(n15)), list(range(n15, D)), list(range(D))],
}
for name, iv in VARIANTS.items():
    print(f"  {name}: {oof_score(B_tr, y_tr, iv, 42):.4f}", flush=True)

print("== 关卡③双新切分保险 ==", flush=True)
X_all = np.vstack([X_tr, X_ev])
O_all = np.vstack([oof15, ev15])
y_all = np.concatenate([y_tr, y_ev])
tr_meta = [json.loads(l) for l in open("splits/train_v3.jsonl")]
ev_meta = [json.loads(l) for l in open("splits/eval_v3.jsonl")]
srcs = ["t" if "ti2i2v" in r["abs_path"] else ("r" if "rlhf" in r["abs_path"] else "s")
        for r in tr_meta + ev_meta]
strat = np.array([f"{y}{s}" for y, s in zip(y_all, srcs)])
for seed in (101, 202):
    ia, ib = train_test_split(np.arange(len(y_all)), test_size=0.2, random_state=seed, stratify=strat)
    Ba = np.hstack([O_all[ia], X_all[ia]])
    Bb = np.hstack([O_all[ib], X_all[ib]])
    ya, yb = y_all[ia], y_all[ib]
    m0 = mk(None); m0.fit(Ba, ya); p0 = m0.predict_proba(Bb)[:, 1]
    m1 = mk(GROUP2); m1.fit(Ba, ya); p1 = m1.predict_proba(Bb)[:, 1]
    print(f"  切分{seed}: 基准 {gn(p0, yb):.4f} | 交互约束 {gn(p1, yb):.4f} | Δ {gn(p1,yb)-gn(p0,yb):+.4f}", flush=True)
print("E41_DONE", flush=True)
