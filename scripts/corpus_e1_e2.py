#!/usr/bin/env python
"""语料放行率战役 E1+E2(2026-08-02 预注册后执行)。

E1 复现:_stack_15expert.pkl + 前人 train.py 配方(LR meta C=100 pw=2 + LGBM残差 boost),
    验证 p_base ev@95≈0.188(±0.02)。
E2 分位OR门:ev15 专家 eval 内百分位秩,susp=max;QOR-15 与 QOR-8(非per-src);
    报 ev@95/ev@100(阈值口径同基线)+ LOO 版。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream"))


def gn_at_recall(p, y, rec):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean()), float(T)


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    oof15, ev15, y_tr, y_ev, *rest = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15, ev15 = np.asarray(oof15, float), np.asarray(ev15, float)
    y_tr, y_ev = np.asarray(y_tr, int), np.asarray(y_ev, int)
    print(f"oof15 {oof15.shape} ev15 {ev15.shape} bads tr={y_tr.sum()} ev={y_ev.sum()}", flush=True)

    # ---------------- E1 复现 ----------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    import lightgbm as lgb

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    oof = np.zeros(len(y_tr)); cnt = np.zeros(len(y_tr))
    for i_tr, i_va in rkf.split(oof15, y_tr):
        m = LogisticRegression(C=100.0, class_weight={0: 1, 1: 2.0}, max_iter=10000)
        m.fit(oof15[i_tr], y_tr[i_tr])
        oof[i_va] += m.predict_proba(oof15[i_va])[:, 1]; cnt[i_va] += 1
    meta = LogisticRegression(C=100.0, class_weight={0: 1, 1: 2.0}, max_iter=10000)
    meta.fit(oof15, y_tr)
    boost = lgb.LGBMRegressor(n_estimators=40, num_leaves=4, learning_rate=0.04,
                              min_child_samples=30, random_state=42, verbose=-1)
    boost.fit(oof15, y_tr - oof / cnt)

    p_meta = meta.predict_proba(ev15)[:, 1]
    p_mb = p_meta + boost.predict(ev15)
    for name, p in (("meta(LR)", p_meta), ("meta+boost", p_mb)):
        a = auc(p[y_ev == 1], p[y_ev == 0])
        e95, _ = gn_at_recall(p, y_ev, 0.95)
        e100, _ = gn_at_recall(p, y_ev, 1.0)
        print(f"E1 {name:11s}: AUC={a:.4f} ev@95={e95:.4f} ev@100={e100:.4f}")

    # ---------------- E2 分位OR ----------------
    # eval 内百分位秩(0-1),方向:专家分数越大越bad(oof AUC 校验方向)
    dirs = []
    for j in range(ev15.shape[1]):
        a = auc(oof15[y_tr == 1, j], oof15[y_tr == 0, j])
        dirs.append(1.0 if a >= 0.5 else -1.0)
    dirs = np.array(dirs)
    print("专家方向(按train OOF AUC):", dirs.astype(int).tolist())

    def pct_rank(col):
        order = np.argsort(col)
        r = np.empty(len(col)); r[order] = np.arange(len(col))
        return r / (len(col) - 1)

    R = np.stack([pct_rank(dirs[j] * ev15[:, j]) for j in range(ev15.shape[1])], 1)
    base_idx = list(range(8))          # 前8=非per-src基础专家(expert_definitions 顺序)
    for name, idx in (("QOR-15", list(range(15))), ("QOR-8", base_idx)):
        susp = R[:, idx].max(1)
        a = auc(susp[y_ev == 1], susp[y_ev == 0])
        e95, _ = gn_at_recall(susp, y_ev, 0.95)
        e100, _ = gn_at_recall(susp, y_ev, 1.0)
        # LOO 版:每个bad用其余bad定阈,统计其是否被抓;release用全bad阈
        b = np.sort(susp[y_ev == 1]); caught95 = caught100 = 0
        bs = susp[y_ev == 1]
        for i in range(len(bs)):
            rest_ = np.sort(np.delete(bs, i))
            k95 = int(np.ceil(0.95 * len(rest_)))
            caught95 += bs[i] >= rest_[len(rest_) - k95]
            caught100 += bs[i] >= rest_[0]
        print(f"E2 {name:7s}: AUC={a:.4f} ev@95={e95:.4f} ev@100={e100:.4f} "
              f"LOO召回@95阈={caught95/len(bs):.3f} @100阈={caught100/len(bs):.3f}")


if __name__ == "__main__":
    main()
