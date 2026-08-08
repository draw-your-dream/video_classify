"""E42:交互约束效应量的高统计力检验(2026-08-08 预注册)。
E41 证据弱正(多种子4/5、均值+0.83pt;但双切分1/2)。本实验提高统计力后终判。
①20 个折种子的配对比较 → 均值、标准误、95%CI、Wilcoxon;
②6 个全新切分保险 → 胜负与均值;
③随机分组对照同样跑 20 种子(证伪"任意约束皆有效")。
判准(发车前冻结):20 种子配对 95%CI 下界 > 0 且 6 切分中 ≥4 胜 → 申请 eval 单发;
否则记为噪声级,不改汇报口径。"""
import json
import pickle

import numpy as np
import lightgbm as lgb
from scipy.stats import wilcoxon
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
B_tr = np.hstack([oof15, X_tr])
n15 = oof15.shape[1]
D = B_tr.shape[1]
champ = json.load(open("data/s3/e18_champion.json"))["params"]
GROUP2 = [list(range(n15)), list(range(n15, D))]
rng = np.random.RandomState(0)


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


print("== ①20 折种子配对(基准 vs 语义分组 vs 随机分组) ==", flush=True)
d_sem, d_rnd = [], []
for k, seed in enumerate(range(1000, 1020)):
    g0 = oof_score(B_tr, y_tr, None, seed)
    g1 = oof_score(B_tr, y_tr, GROUP2, seed)
    perm = rng.permutation(D)
    grnd = [sorted(perm[:D // 2].tolist()), sorted(perm[D // 2:].tolist())]
    g2 = oof_score(B_tr, y_tr, grnd, seed)
    d_sem.append(g1 - g0)
    d_rnd.append(g2 - g0)
    print(f"  seed{seed}: 基准{g0:.4f} 语义{g1:.4f}({g1-g0:+.4f}) 随机{g2:.4f}({g2-g0:+.4f})", flush=True)

for name, d in (("语义分组", d_sem), ("随机分组(对照)", d_rnd)):
    d = np.array(d)
    se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
    try:
        p = wilcoxon(d).pvalue
    except Exception:
        p = np.nan
    print(f"  [{name}] 均值 {d.mean():+.4f} ± {se:.4f}(SE) 95%CI [{lo:+.4f}, {hi:+.4f}] "
          f"胜 {(d>0).sum()}/{len(d)} Wilcoxon p={p:.4f}", flush=True)

print("== ②6 个全新切分保险 ==", flush=True)
X_all = np.vstack([X_tr, X_ev])
O_all = np.vstack([oof15, ev15])
y_all = np.concatenate([y_tr, y_ev])
tr_meta = [json.loads(l) for l in open("splits/train_v3.jsonl")]
ev_meta = [json.loads(l) for l in open("splits/eval_v3.jsonl")]
srcs = ["t" if "ti2i2v" in r["abs_path"] else ("r" if "rlhf" in r["abs_path"] else "s")
        for r in tr_meta + ev_meta]
strat = np.array([f"{y}{s}" for y, s in zip(y_all, srcs)])
ds = []
for seed in (101, 202, 303, 404, 505, 606):
    ia, ib = train_test_split(np.arange(len(y_all)), test_size=0.2, random_state=seed, stratify=strat)
    Ba = np.hstack([O_all[ia], X_all[ia]])
    Bb = np.hstack([O_all[ib], X_all[ib]])
    ya, yb = y_all[ia], y_all[ib]
    m0 = mk(None); m0.fit(Ba, ya); g0 = gn(m0.predict_proba(Bb)[:, 1], yb)
    m1 = mk(GROUP2); m1.fit(Ba, ya); g1 = gn(m1.predict_proba(Bb)[:, 1], yb)
    ds.append(g1 - g0)
    print(f"  切分{seed}: 基准{g0:.4f} 交互{g1:.4f} Δ{g1-g0:+.4f}", flush=True)
ds = np.array(ds)
print(f"  [保险] 胜 {(ds>0).sum()}/6 均值 {ds.mean():+.4f}", flush=True)
print("E42_DONE", flush=True)
