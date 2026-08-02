#!/usr/bin/env python
"""E3:新特征并栈重训(2026-08-02 预注册)。

输入:upstream/cache_v3/_stack_15expert.pkl + data/s3/corpus_new_feats.tsv(盒上回传)。
两个预注册变体(配方冻结,不调参):
  E3-a 直接并入 meta:LR 输入 = [15专家分, 21新特征(train z-score, NaN=train中位)],
        boost 输入同;配方与 train.py 完全一致(C=100 pw=2, LGBM 4/40/0.04/30, 5x5 OOF)。
  E3-b 第16专家:21特征上 LGBM 分类器(同 boost 超参族:leaves=8 est=200 lr=0.05 mcs=30,
        固定不调),5 折 OOF 出 train 分,全量训出 eval 分;并入 meta 走原配方。
按族消融(预注册):A几何计数 / B参照相似 / C+D双头。
判准:Δev@95 ≥ +0.03 且 ev@100 不降。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMS = {
    "A_geo": ["sc_mean", "sc_min", "multi_frac", "zero_frac", "err_frac", "hc_med",
              "hc_drift", "area_slope", "area_range", "cxy_drift", "n_geo"],
    "B_ref": ["rso_mean", "rso_p75", "rso_max", "rds_mean", "rds_p75", "rds_max"],
    "CD_heads": ["bh_p75", "bh_max", "fh_p75", "fh_max", "fh_cover"],
}
ALL = [c for f in FAMS.values() for c in f]


def gn_at_recall(p, y, rec):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def meta_boost(X_tr, X_ev, y_tr, y_ev, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    import lightgbm as lgb
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    oof = np.zeros(len(y_tr)); cnt = np.zeros(len(y_tr))
    for i_tr, i_va in rkf.split(X_tr, y_tr):
        m = LogisticRegression(C=100.0, class_weight={0: 1, 1: 2.0}, max_iter=10000)
        m.fit(X_tr[i_tr], y_tr[i_tr])
        oof[i_va] += m.predict_proba(X_tr[i_va])[:, 1]; cnt[i_va] += 1
    meta = LogisticRegression(C=100.0, class_weight={0: 1, 1: 2.0}, max_iter=10000)
    meta.fit(X_tr, y_tr)
    boost = lgb.LGBMRegressor(n_estimators=40, num_leaves=4, learning_rate=0.04,
                              min_child_samples=30, random_state=42, verbose=-1)
    boost.fit(X_tr, y_tr - oof / cnt)
    p = meta.predict_proba(X_ev)[:, 1] + boost.predict(X_ev)
    a = auc(p[y_ev == 1], p[y_ev == 0])
    e95, e100 = gn_at_recall(p, y_ev, 0.95), gn_at_recall(p, y_ev, 1.0)
    print(f"{tag:28s}: AUC={a:.4f} ev@95={e95:.4f} ev@100={e100:.4f}")
    return e95


def main():
    oof15, ev15, y_tr, y_ev, *_ = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15, ev15 = np.asarray(oof15, float), np.asarray(ev15, float)
    y_tr, y_ev = np.asarray(y_tr, int), np.asarray(y_ev, int)

    feats = {}
    for l in (ROOT / "data/s3/corpus_new_feats.tsv").read_text().splitlines()[1:]:
        p = l.split("\t")
        feats[p[0].rsplit("/", 1)[-1]] = np.array([float(x) for x in p[1:]])
    tr_v = [json.loads(l)["video"] for l in open(ROOT / "splits/train_v3.jsonl")]
    ev_v = [json.loads(l)["video"] for l in open(ROOT / "splits/eval_v3.jsonl")]
    nan = np.full(len(ALL), np.nan)
    F_tr = np.stack([feats.get(v, nan) for v in tr_v])
    F_ev = np.stack([feats.get(v, nan) for v in ev_v])
    cov_tr = 1 - np.isnan(F_tr).all(1).mean()
    cov_ev = 1 - np.isnan(F_ev).all(1).mean()
    print(f"feats coverage: train {cov_tr:.3f} eval {cov_ev:.3f}  cols={len(ALL)}")

    med = np.nanmedian(F_tr, axis=0)
    mu = np.nanmean(F_tr, axis=0)
    sd = np.nanstd(F_tr, axis=0); sd[sd < 1e-9] = 1.0

    def prep(F):
        G = F.copy()
        idx = np.where(np.isnan(G))
        G[idx] = np.take(med, idx[1])
        return (G - mu) / sd

    Z_tr, Z_ev = prep(F_tr), prep(F_ev)

    print("== 基线(E1 复现口径) ==")
    base95 = meta_boost(oof15, ev15, y_tr, y_ev, "base 15-expert")

    print("== E3-a 直接并入 ==")
    cols = {f: [ALL.index(c) for c in cs] for f, cs in FAMS.items()}
    for name, ci in [("+ALL", list(range(len(ALL))))] + [(f"+{f}", ci) for f, ci in cols.items()]:
        e95 = meta_boost(np.hstack([oof15, Z_tr[:, ci]]), np.hstack([ev15, Z_ev[:, ci]]),
                         y_tr, y_ev, f"E3-a {name}")

    print("== E3-b 第16专家 ==")
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    Xf_tr, Xf_ev = prep(F_tr), prep(F_ev)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof16 = np.zeros(len(y_tr))
    for i_tr, i_va in skf.split(Xf_tr, y_tr):
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=8, learning_rate=0.05,
                               min_child_samples=30, random_state=42, verbose=-1)
        m.fit(Xf_tr[i_tr], y_tr[i_tr])
        oof16[i_va] = m.predict_proba(Xf_tr[i_va])[:, 1]
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=8, learning_rate=0.05,
                           min_child_samples=30, random_state=42, verbose=-1)
    m.fit(Xf_tr, y_tr)
    ev16 = m.predict_proba(Xf_ev)[:, 1]
    a16 = auc(ev16[y_ev == 1], ev16[y_ev == 0])
    print(f"第16专家单独: AUC={a16:.4f} ev@95={gn_at_recall(ev16, y_ev, 0.95):.4f}")
    meta_boost(np.hstack([oof15, oof16[:, None]]), np.hstack([ev15, ev16[:, None]]),
               y_tr, y_ev, "E3-b 15+1expert")


if __name__ == "__main__":
    main()
