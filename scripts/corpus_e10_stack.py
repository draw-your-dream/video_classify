#!/usr/bin/env python
"""E10 并栈判决(2026-08-03 预注册):C5 输入追加 12 列运动分解特征。
协议同 E7:train 5折 OOF gn@95 与 C5 基准(0.3103)比较,不低于基准才单发 eval;
判准:Δev@95 ≥ +0.03 且 ev@100 不降(与 E3 同准)。"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DCOLS = ("zoom_mean zoom_max res_mean res_max mov_frac_mean mov_frac_max "
         "coh_mean coh_max magcv_mean jerk acf1 zoom_x_rigid").split()


def gn(p, y, rec):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def main():
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    oof15, ev15, y_tr, y_ev, *_ = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15, ev15 = np.asarray(oof15, float), np.asarray(ev15, float)
    y_tr, y_ev = np.asarray(y_tr, int), np.asarray(y_ev, int)
    z = np.load(ROOT / "upstream/cache_v3/_full_raw_v2.npz")
    X_tr, X_ev = z["X_tr"].astype(float), z["X_ev"].astype(float)
    med = np.nanmedian(X_tr, axis=0)
    for A in (X_tr, X_ev):
        idx = np.where(~np.isfinite(A)); A[idx] = np.take(med, idx[1])

    rows = list(csv.reader(open(ROOT / "data/s3/decomp_full.csv")))
    hdr = rows[0][1:]
    ci = [hdr.index(c) for c in DCOLS]
    dfeat = {}
    for r in rows[1:]:
        vals = np.array([float(x) if x != "nan" else np.nan for x in r[1:]])
        dfeat[r[0]] = vals[ci]
    tr_v = [json.loads(l)["video"] for l in open(ROOT / "splits/train_v3.jsonl")]
    ev_v = [json.loads(l)["video"] for l in open(ROOT / "splits/eval_v3.jsonl")]
    nan = np.full(len(DCOLS), np.nan)
    D_tr = np.stack([dfeat.get(v, nan) for v in tr_v])
    D_ev = np.stack([dfeat.get(v, nan) for v in ev_v])
    print(f"decomp 覆盖: train {1-np.isnan(D_tr).all(1).mean():.3f} eval {1-np.isnan(D_ev).all(1).mean():.3f}")
    dmed = np.nanmedian(D_tr, axis=0)
    for A in (D_tr, D_ev):
        idx = np.where(~np.isfinite(A)); A[idx] = np.take(dmed, idx[1])

    def mk():
        return lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                  min_child_samples=20, scale_pos_weight=2.0,
                                  random_state=42, verbose=-1)

    B_tr = np.hstack([oof15, X_tr, D_tr])
    B_ev = np.hstack([ev15, X_ev, D_ev])
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_tr))
    for a, b in skf.split(B_tr, y_tr):
        m = mk(); m.fit(B_tr[a], y_tr[a])
        oof[b] = m.predict_proba(B_tr[b])[:, 1]
    g = gn(oof, y_tr, 0.95)
    print(f"E10 C5+decomp train-OOF gn@95 = {g:.4f}  (C5 基准 0.3103)")
    if g < 0.3103 - 1e-9:
        print("OOF 低于基准 -> 不动 eval,判决:并栈无增量(记录收档)")
        return
    m = mk(); m.fit(B_tr, y_tr)
    p = m.predict_proba(B_ev)[:, 1]
    from scipy.stats import rankdata
    def auc(pos, neg):
        r = rankdata(np.concatenate([pos, neg]))
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    print(f"E10 eval: ev@95={gn(p, y_ev, 0.95):.4f} ev@100={gn(p, y_ev, 1.0):.4f} "
          f"ev@90={gn(p, y_ev, 0.90):.4f} ev@80={gn(p, y_ev, 0.80):.4f} "
          f"AUC={auc(p[y_ev==1], p[y_ev==0]):.4f}")
    print("对照 C5 eval: ev@95=0.2647 ev@100=0.1208 ev@90=0.3570 ev@80=0.5240")
    # 新特征贡献占比
    imp = m.feature_importances_
    n15, nX = 15, X_tr.shape[1]
    print(f"重要度: 专家{imp[:n15].sum()} X320:{imp[n15:n15+nX].sum()} decomp:{imp[n15+nX:].sum()}")


if __name__ == "__main__":
    main()
