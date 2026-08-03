#!/usr/bin/env python
"""E11:V-JEPA2-giant 嵌入 -> 第16专家 -> 并栈(2026-08-03 预注册)。
协议:嵌入上 LGBM 头(leaves8/est200/lr0.05/mcs30,冻结)5折 OOF 出 train 专家分,
全量训出 eval 分;并入 C5 输入。train-OOF gn@95 >= 0.3103 才单发 eval。
另报:该专家单独 AUC(midrank,对照前人 vjepa 0.63)。"""
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


def main():
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import rankdata

    z = np.load(ROOT / "data/s3/vjepa2g_f16.npz", allow_pickle=True)
    emb_by_base = {str(r).rsplit("/", 1)[-1]: e for r, e in zip(z["rels"], z["emb"].astype(np.float32))}
    print("embeddings:", len(emb_by_base), "dim:", z["emb"].shape[1])

    oof15, ev15, y_tr, y_ev, *_ = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15, ev15 = np.asarray(oof15, float), np.asarray(ev15, float)
    y_tr, y_ev = np.asarray(y_tr, int), np.asarray(y_ev, int)
    zz = np.load(ROOT / "upstream/cache_v3/_full_raw_v2.npz")
    X_tr, X_ev = zz["X_tr"].astype(float), zz["X_ev"].astype(float)
    med = np.nanmedian(X_tr, axis=0)
    for A in (X_tr, X_ev):
        idx = np.where(~np.isfinite(A)); A[idx] = np.take(med, idx[1])

    tr_v = [json.loads(l)["video"] for l in open(ROOT / "splits/train_v3.jsonl")]
    ev_v = [json.loads(l)["video"] for l in open(ROOT / "splits/eval_v3.jsonl")]
    dim = z["emb"].shape[1]
    nanv = np.full(dim, np.nan, np.float32)
    E_tr = np.stack([emb_by_base.get(v, nanv) for v in tr_v])
    E_ev = np.stack([emb_by_base.get(v, nanv) for v in ev_v])
    cov_tr = 1 - np.isnan(E_tr).all(1).mean(); cov_ev = 1 - np.isnan(E_ev).all(1).mean()
    print(f"覆盖: train {cov_tr:.3f} eval {cov_ev:.3f}")
    emed = np.nanmedian(E_tr, axis=0)
    for A in (E_tr, E_ev):
        idx = np.where(~np.isfinite(A)); A[idx] = np.take(emed, idx[1])

    def head():
        return lgb.LGBMClassifier(n_estimators=200, num_leaves=8, learning_rate=0.05,
                                  min_child_samples=30, random_state=42, verbose=-1)

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof16 = np.zeros(len(y_tr))
    for a, b in skf.split(E_tr, y_tr):
        m = head(); m.fit(E_tr[a], y_tr[a])
        oof16[b] = m.predict_proba(E_tr[b])[:, 1]
    m = head(); m.fit(E_tr, y_tr)
    ev16 = m.predict_proba(E_ev)[:, 1]

    def auc(pos, neg):
        r = rankdata(np.concatenate([pos, neg]))
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    print(f"E11 vjepa2-g 专家单独: train-OOF AUC={auc(oof16[y_tr==1], oof16[y_tr==0]):.4f} "
          f"eval AUC={auc(ev16[y_ev==1], ev16[y_ev==0]):.4f} (前人 vjepa 专家 ~0.63)")

    def mk():
        return lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                  min_child_samples=20, scale_pos_weight=2.0,
                                  random_state=42, verbose=-1)
    B_tr = np.hstack([oof15, X_tr, oof16[:, None]])
    B_ev = np.hstack([ev15, X_ev, ev16[:, None]])
    oof = np.zeros(len(y_tr))
    for a, b in skf.split(B_tr, y_tr):
        mm = mk(); mm.fit(B_tr[a], y_tr[a])
        oof[b] = mm.predict_proba(B_tr[b])[:, 1]
    g = gn(oof, y_tr, 0.95)
    print(f"E11 C5+vjepa2g专家 train-OOF gn@95 = {g:.4f}  (C5 基准 0.3103)")
    if g < 0.3103 - 1e-9:
        print("OOF 低于基准 -> 不动 eval。判决:E11 并栈无增量(记录收档)")
        return
    mm = mk(); mm.fit(B_tr, y_tr)
    p = mm.predict_proba(B_ev)[:, 1]
    print(f"E11 eval: ev@95={gn(p, y_ev, 0.95):.4f} ev@100={gn(p, y_ev, 1.0):.4f} "
          f"ev@90={gn(p, y_ev, 0.90):.4f} AUC={auc(p[y_ev==1], p[y_ev==0]):.4f}")
    print("对照 C5 eval: ev@95=0.2647 ev@100=0.1208 ev@90=0.3570")


if __name__ == "__main__":
    main()
