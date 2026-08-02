#!/usr/bin/env python
"""E7 训练侧模型选择(OOF gn@95 选优,胜者单次 eval)+ E8 跨切分机理复核。
2026-08-03 预注册后执行,菜单与超参冻结。"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def gn(p, y, rec):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def source_of(abs_path):
    if "ti2i2v" in abs_path:
        return "ti2i2v"
    if "rlhf" in abs_path:
        return "rlhf"
    return "skus"


def lr_boost_recipe(X_tr, y_tr, X_ev):
    """前人 meta 配方:LR(C=100,pw=2) 5x5 OOF + LGBM残差boost。返回 eval 分。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    import lightgbm as lgb
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    oof = np.zeros(len(y_tr)); cnt = np.zeros(len(y_tr))
    for a, b in rkf.split(X_tr, y_tr):
        m = LogisticRegression(C=100.0, class_weight={0: 1, 1: 2.0}, max_iter=10000)
        m.fit(X_tr[a], y_tr[a])
        oof[b] += m.predict_proba(X_tr[b])[:, 1]; cnt[b] += 1
    meta = LogisticRegression(C=100.0, class_weight={0: 1, 1: 2.0}, max_iter=10000).fit(X_tr, y_tr)
    boost = lgb.LGBMRegressor(n_estimators=40, num_leaves=4, learning_rate=0.04,
                              min_child_samples=30, random_state=42, verbose=-1)
    boost.fit(X_tr, y_tr - oof / cnt)
    return meta.predict_proba(X_ev)[:, 1] + boost.predict(X_ev)


def main():
    import lightgbm as lgb
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import rankdata

    oof15, ev15, y_tr, y_ev, *_ = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15, ev15 = np.asarray(oof15, float), np.asarray(ev15, float)
    y_tr, y_ev = np.asarray(y_tr, int), np.asarray(y_ev, int)
    z = np.load(ROOT / "upstream/cache_v3/_full_raw_v2.npz")
    X_tr, X_ev = z["X_tr"].astype(float), z["X_ev"].astype(float)
    med = np.nanmedian(X_tr, axis=0)
    for A in (X_tr, X_ev):
        idx = np.where(~np.isfinite(A)); A[idx] = np.take(med, idx[1])

    # F22 + 源
    tr_meta = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    ev_meta = [json.loads(l) for l in open(ROOT / "splits/eval_v3.jsonl")]
    feats = {}
    hdr = None
    for i, l in enumerate((ROOT / "data/s3/corpus_new_feats.tsv").read_text().splitlines()):
        p = l.split("\t")
        if i == 0:
            hdr = p[1:]; continue
        feats[p[0].rsplit("/", 1)[-1]] = np.array([float(x) for x in p[1:]])
    nan = np.full(len(hdr), np.nan)
    F_tr = np.stack([feats.get(r["video"], nan) for r in tr_meta])
    F_ev = np.stack([feats.get(r["video"], nan) for r in ev_meta])
    fmed = np.nanmedian(F_tr, axis=0)
    for A in (F_tr, F_ev):
        idx = np.where(~np.isfinite(A)); A[idx] = np.take(fmed, idx[1])
    SRC = ("skus", "ti2i2v", "rlhf")
    s_tr = np.stack([[1.0 * (source_of(r["abs_path"]) == s) for s in SRC] for r in tr_meta])
    s_ev = np.stack([[1.0 * (source_of(r["abs_path"]) == s) for s in SRC] for r in ev_meta])

    B_tr, B_ev = np.hstack([oof15, X_tr]), np.hstack([ev15, X_ev])

    def mk_lgbm(lv, est, lr, mcs, spw=None):
        kw = dict(num_leaves=lv, n_estimators=est, learning_rate=lr,
                  min_child_samples=mcs, random_state=42, verbose=-1)
        if spw:
            kw["scale_pos_weight"] = spw
        return lgb.LGBMClassifier(**kw)

    CANDS = {
        "C1_M1": (lambda: mk_lgbm(31, 300, 0.05, 20), B_tr, B_ev),
        "C2_+F22": (lambda: mk_lgbm(31, 300, 0.05, 20), np.hstack([B_tr, F_tr]), np.hstack([B_ev, F_ev])),
        "C3_63l600e": (lambda: mk_lgbm(63, 600, 0.03, 20), B_tr, B_ev),
        "C4_15l40m": (lambda: mk_lgbm(15, 300, 0.05, 40), B_tr, B_ev),
        "C5_spw2": (lambda: mk_lgbm(31, 300, 0.05, 20, 2.0), B_tr, B_ev),
        "C6_hgbt": (lambda: HistGradientBoostingClassifier(max_leaf_nodes=31, max_iter=300,
                                                           learning_rate=0.05, random_state=42), B_tr, B_ev),
        "C8_+F22+src": (lambda: mk_lgbm(31, 300, 0.05, 20),
                        np.hstack([B_tr, F_tr, s_tr]), np.hstack([B_ev, F_ev, s_ev])),
    }

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(skf.split(B_tr, y_tr))
    oof_scores = {}
    for name, (mkm, TR, EV) in CANDS.items():
        oof = np.zeros(len(y_tr))
        for a, b in folds:
            m = mkm(); m.fit(TR[a], y_tr[a])
            oof[b] = m.predict_proba(TR[b])[:, 1]
        oof_scores[name] = oof
        print(f"E7 {name:12s}: train-OOF gn@95={gn(oof, y_tr, 0.95):.4f} gn@100={gn(oof, y_tr, 1.0):.4f}", flush=True)
    # C7 秩平均集成:C1 OOF 与 M3配方 OOF(折内算 M3 太贵,用同折 LR+boost 简化=预注册配方)
    oofc7 = np.zeros(len(y_tr))
    for a, b in folds:
        p_l = mk_lgbm(31, 300, 0.05, 20).fit(B_tr[a], y_tr[a]).predict_proba(B_tr[b])[:, 1]
        mu, sd = X_tr[a].mean(0), X_tr[a].std(0); sd[sd < 1e-9] = 1
        Xa = np.hstack([oof15[a], (X_tr[a] - mu) / sd]); Xb = np.hstack([oof15[b], (X_tr[b] - mu) / sd])
        p_m = lr_boost_recipe(Xa, y_tr[a], Xb)
        oofc7[b] = (rankdata(p_l) + rankdata(p_m)) / (2 * len(p_l))
    oof_scores["C7_rankavg"] = oofc7
    print(f"E7 C7_rankavg  : train-OOF gn@95={gn(oofc7, y_tr, 0.95):.4f} gn@100={gn(oofc7, y_tr, 1.0):.4f}", flush=True)

    winner = max(oof_scores, key=lambda k: gn(oof_scores[k], y_tr, 0.95))
    print(f"\nE7 OOF 胜者: {winner} -> 唯一 eval 出手:")
    if winner == "C7_rankavg":
        p_l = mk_lgbm(31, 300, 0.05, 20).fit(B_tr, y_tr).predict_proba(B_ev)[:, 1]
        mu, sd = X_tr.mean(0), X_tr.std(0); sd[sd < 1e-9] = 1
        p_m = lr_boost_recipe(np.hstack([oof15, (X_tr - mu) / sd]), y_tr,
                              np.hstack([ev15, (X_ev - mu) / sd]))
        p = (rankdata(p_l) + rankdata(p_m)) / (2 * len(p_l))
    else:
        mkm, TR, EV = CANDS[winner]
        p = mkm().fit(TR, y_tr).predict_proba(EV)[:, 1]
    print(f"  eval: ev@95={gn(p, y_ev, 0.95):.4f} ev@100={gn(p, y_ev, 1.0):.4f} "
          f"ev@90={gn(p, y_ev, 0.90):.4f} ev@80={gn(p, y_ev, 0.80):.4f}")

    # ---------------- E8 跨切分机理复核 ----------------
    print("\nE8 跨切分(LGBM[X320] vs LR+boost[zX320]):")
    X_all = np.vstack([X_tr, X_ev]); y_all = np.concatenate([y_tr, y_ev])
    strat = np.array([f"{y}-{source_of(r['abs_path'])}" for y, r in
                      zip(y_all, tr_meta + ev_meta)])
    from sklearn.model_selection import train_test_split
    for seed in (0, 1, 2):
        a, b = train_test_split(np.arange(len(y_all)), test_size=0.2,
                                random_state=seed, stratify=strat)
        p_g = mk_lgbm(31, 300, 0.05, 20).fit(X_all[a], y_all[a]).predict_proba(X_all[b])[:, 1]
        mu, sd = X_all[a].mean(0), X_all[a].std(0); sd[sd < 1e-9] = 1
        p_m = lr_boost_recipe((X_all[a] - mu) / sd, y_all[a], (X_all[b] - mu) / sd)
        print(f"  seed{seed}: GBM ev@95={gn(p_g, y_all[b], 0.95):.4f} | "
              f"线性meta ev@95={gn(p_m, y_all[b], 0.95):.4f}")


if __name__ == "__main__":
    main()
