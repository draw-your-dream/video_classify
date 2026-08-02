#!/usr/bin/env python
"""E5 分源阈值 + E6 组合器菜单(2026-08-02 预注册后执行,超参冻结)。"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def auc_tie(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gn(p, y, rec):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def source_of(abs_path: str) -> str:
    if "ti2i2v" in abs_path:
        return "ti2i2v"
    if "rlhf" in abs_path:
        return "rlhf"
    return "skus"


def train_meta_boost(X_tr, y_tr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    import lightgbm as lgb
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    oof = np.zeros(len(y_tr)); cnt = np.zeros(len(y_tr))
    for i_tr, i_va in rkf.split(X_tr, y_tr):
        m = LogisticRegression(C=100.0, class_weight={0: 1, 1: 2.0}, max_iter=10000)
        m.fit(X_tr[i_tr], y_tr[i_tr])
        oof[i_va] += m.predict_proba(X_tr[i_va])[:, 1]; cnt[i_va] += 1
    oof_meta = oof / cnt
    meta = LogisticRegression(C=100.0, class_weight={0: 1, 1: 2.0}, max_iter=10000)
    meta.fit(X_tr, y_tr)
    boost = lgb.LGBMRegressor(n_estimators=40, num_leaves=4, learning_rate=0.04,
                              min_child_samples=30, random_state=42, verbose=-1)
    boost.fit(X_tr, y_tr - oof_meta)
    return meta, boost, oof_meta


def main():
    oof15, ev15, y_tr, y_ev, *_ = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15, ev15 = np.asarray(oof15, float), np.asarray(ev15, float)
    y_tr, y_ev = np.asarray(y_tr, int), np.asarray(y_ev, int)
    tr = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    ev = [json.loads(l) for l in open(ROOT / "splits/eval_v3.jsonl")]
    src_tr = np.array([source_of(r["abs_path"]) for r in tr])
    src_ev = np.array([source_of(r["abs_path"]) for r in ev])

    # ---------- E5:用 E1 配方的 train OOF 分 + eval 分 ----------
    meta, boost, oof_meta = train_meta_boost(oof15, y_tr)
    # train 侧部署分:OOF meta + boost 的 OOF 近似(boost在全train拟合,残差泄漏小;
    # 阈值只用 bad 分位,与 E1 同风险面)。冻结:train 分 = oof_meta + boost.predict(oof15)
    p_tr = oof_meta + boost.predict(oof15)
    p_ev = meta.predict_proba(ev15)[:, 1] + boost.predict(ev15)
    print(f"全局基线: ev@95={gn(p_ev, y_ev, 0.95):.4f} ev@100={gn(p_ev, y_ev, 1.0):.4f}")
    for s in ("skus", "ti2i2v", "rlhf"):
        m_ev = src_ev == s
        print(f"  eval {s}: n={m_ev.sum()} bad={y_ev[m_ev].sum()} "
              f"good均分={p_ev[m_ev][y_ev[m_ev]==0].mean() if (y_ev[m_ev]==0).any() else float('nan'):.3f}")

    # (a) 保守版:每源 train-bad 95% 分位阈值
    def eval_with_T(T_by_src):
        caught = rel = n_good = 0
        for s, T in T_by_src.items():
            m = src_ev == s
            caught += (p_ev[m][y_ev[m] == 1] >= T).sum()
            rel += (p_ev[m][y_ev[m] == 0] < T).sum()
            n_good += (y_ev[m] == 0).sum()
        # rlhf 无 train bad 的源:T=+inf 表示全放行(下面单独处理)
        return caught / y_ev.sum(), rel / (y_ev == 0).sum()

    Ta = {}
    for s in ("skus", "ti2i2v", "rlhf"):
        b = np.sort(p_tr[(src_tr == s) & (y_tr == 1)])
        if len(b) == 0:
            Ta[s] = np.inf  # 该源train无bad -> 全放行
            continue
        k = int(np.ceil(0.95 * len(b)))
        Ta[s] = b[len(b) - k]
    rec_a, rel_a = eval_with_T(Ta)
    print(f"E5a 每源95%: eval召回={rec_a:.4f} 放行={rel_a:.4f}   阈值={ {k: round(v,4) if np.isfinite(v) else 'inf' for k,v in Ta.items()} }")

    # (b) 预算版:train 全局 5% 漏检预算,贪心分配(每次放过"换取放行最多"的下一条最低分bad)
    bads = {s: np.sort(p_tr[(src_tr == s) & (y_tr == 1)]) for s in ("skus", "ti2i2v", "rlhf")}
    goods = {s: np.sort(p_tr[(src_tr == s) & (y_tr == 0)]) for s in ("skus", "ti2i2v", "rlhf")}
    n_bad_tr = y_tr.sum()
    budget = int(np.floor(0.05 * n_bad_tr))
    skip = {s: 0 for s in bads}  # 每源放过的最低分bad数
    def release_gain(s):
        b, k = bads[s], skip[s]
        if k + 1 > len(b) - 1:
            return -1, None
        T_now = b[k]; T_next = b[k + 1]
        g = goods[s]
        return np.searchsorted(g, T_next) - np.searchsorted(g, T_now), T_next
    for _ in range(budget):
        gains = {s: release_gain(s)[0] for s in bads if len(bads[s])}
        s_best = max(gains, key=gains.get)
        if gains[s_best] < 0:
            break
        skip[s_best] += 1
    Tb = {}
    for s in bads:
        b = bads[s]
        Tb[s] = b[skip[s]] if len(b) else np.inf
    rec_b, rel_b = eval_with_T(Tb)
    print(f"E5b 预算分配: train放过={skip} eval召回={rec_b:.4f} 放行={rel_b:.4f}")

    # ---------- E6:组合器菜单 ----------
    z = np.load(ROOT / "upstream/cache_v3/_full_raw_v2.npz")
    X_tr, X_ev = z["X_tr"].astype(float), z["X_ev"].astype(float)
    med = np.nanmedian(X_tr, axis=0)
    for A in (X_tr, X_ev):
        idx = np.where(~np.isfinite(A)); A[idx] = np.take(med, idx[1])
    import lightgbm as lgb
    for tag, tr_X, ev_X in (
        ("M1 LGBM[oof15+X320]", np.hstack([oof15, X_tr]), np.hstack([ev15, X_ev])),
        ("M2 LGBM[X320]", X_tr, X_ev),
    ):
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                               min_child_samples=20, random_state=42, verbose=-1)
        m.fit(tr_X, y_tr)
        p = m.predict_proba(ev_X)[:, 1]
        print(f"E6 {tag:20s}: AUC={auc_tie(p[y_ev==1], p[y_ev==0]):.4f} "
              f"ev@95={gn(p, y_ev, 0.95):.4f} ev@100={gn(p, y_ev, 1.0):.4f}")
    mu, sd = X_tr.mean(0), X_tr.std(0); sd[sd < 1e-9] = 1
    meta3, boost3, oof_m3 = train_meta_boost(np.hstack([oof15, (X_tr - mu) / sd]), y_tr)
    Xe3 = np.hstack([ev15, (X_ev - mu) / sd])
    p3 = meta3.predict_proba(Xe3)[:, 1] + boost3.predict(Xe3)
    print(f"E6 M3 LR+boost[oof15+zX320]: AUC={auc_tie(p3[y_ev==1], p3[y_ev==0]):.4f} "
          f"ev@95={gn(p3, y_ev, 0.95):.4f} ev@100={gn(p3, y_ev, 1.0):.4f}")


if __name__ == "__main__":
    main()
