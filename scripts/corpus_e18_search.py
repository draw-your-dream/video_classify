#!/usr/bin/env python
"""E18:训练侧系统性搜索(2026-08-04 预注册)。Optuna 150 trial,目标=5折 OOF gn@95。
胜者(最优单配置 vs 前5秩平均)单发 eval。"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def main():
    import lightgbm as lgb
    import optuna
    from scipy.stats import rankdata
    from sklearn.model_selection import StratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    oof15, ev15, y_tr, y_ev, *_ = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15, ev15 = np.asarray(oof15, float), np.asarray(ev15, float)
    y_tr, y_ev = np.asarray(y_tr, int), np.asarray(y_ev, int)
    z = np.load(ROOT / "upstream/cache_v3/_full_raw_v2.npz")
    X_tr, X_ev = z["X_tr"].astype(float), z["X_ev"].astype(float)
    md = np.nanmedian(X_tr, axis=0)
    for A in (X_tr, X_ev):
        ii = np.where(~np.isfinite(A)); A[ii] = np.take(md, ii[1])
    tr_meta = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    ev_meta = [json.loads(l) for l in open(ROOT / "splits/eval_v3.jsonl")]
    def src_oh(rows):
        S = ("ti2i2v", "skus", "rlhf")
        def s_of(p):
            return "ti2i2v" if "ti2i2v" in p else ("rlhf" if "rlhf" in p else "skus")
        return np.array([[1.0 * (s_of(r["abs_path"]) == s) for s in S] for r in rows])
    S_tr, S_ev = src_oh(tr_meta), src_oh(ev_meta)

    INPUTS = {
        "full": (np.hstack([oof15, X_tr]), np.hstack([ev15, X_ev])),
        "raw": (X_tr, X_ev),
        "full_src": (np.hstack([oof15, X_tr, S_tr]), np.hstack([ev15, X_ev, S_ev])),
    }
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(skf.split(X_tr, y_tr))

    def make_model(t):
        return lgb.LGBMClassifier(
            num_leaves=t.suggest_int("leaves", 7, 63),
            n_estimators=t.suggest_int("est", 100, 800),
            learning_rate=t.suggest_float("lr", 0.02, 0.1, log=True),
            min_child_samples=t.suggest_int("mcs", 5, 60),
            scale_pos_weight=t.suggest_float("spw", 1.0, 3.0),
            feature_fraction=t.suggest_float("ff", 0.5, 1.0),
            bagging_fraction=t.suggest_float("bf", 0.5, 1.0),
            bagging_freq=1, random_state=42, verbose=-1)

    trial_oofs = {}

    def objective(t):
        inp = t.suggest_categorical("input", list(INPUTS))
        TR, _ = INPUTS[inp]
        o = np.zeros(len(y_tr))
        for a, b in folds:
            m = make_model(t)
            m.fit(TR[a], y_tr[a])
            o[b] = m.predict_proba(TR[b])[:, 1]
        g = gn(o, y_tr)
        trial_oofs[t.number] = (g, inp, o)
        return g

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(objective, n_trials=150, n_jobs=1, show_progress_bar=False)
    top = sorted(trial_oofs.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
    print("top5 OOF:", [(n, round(v[0], 4), v[1]) for n, v in top])
    best_single = top[0][1][0]
    ens = np.mean([rankdata(v[2]) for _, v in top], axis=0)
    g_ens = gn(ens, y_tr)
    print(f"最优单配置 OOF={best_single:.4f}  前5秩平均 OOF={g_ens:.4f}  (C5 基准 0.3103)")

    use_ens = g_ens > best_single
    print("胜者:", "前5集成" if use_ens else "最优单配置")
    from scipy.stats import rankdata as rd
    if use_ens:
        preds = []
        for n, (g, inp, _) in top:
            t = study.trials[n]
            TR, EV = INPUTS[inp]
            m = lgb.LGBMClassifier(num_leaves=t.params["leaves"], n_estimators=t.params["est"],
                                   learning_rate=t.params["lr"], min_child_samples=t.params["mcs"],
                                   scale_pos_weight=t.params["spw"], feature_fraction=t.params["ff"],
                                   bagging_fraction=t.params["bf"], bagging_freq=1,
                                   random_state=42, verbose=-1)
            m.fit(TR, y_tr)
            preds.append(rd(m.predict_proba(EV)[:, 1]))
        p = np.mean(preds, axis=0)
    else:
        n, (g, inp, _) = top[0]
        t = study.trials[n]
        TR, EV = INPUTS[inp]
        m = lgb.LGBMClassifier(num_leaves=t.params["leaves"], n_estimators=t.params["est"],
                               learning_rate=t.params["lr"], min_child_samples=t.params["mcs"],
                               scale_pos_weight=t.params["spw"], feature_fraction=t.params["ff"],
                               bagging_fraction=t.params["bf"], bagging_freq=1,
                               random_state=42, verbose=-1)
        m.fit(TR, y_tr)
        p = m.predict_proba(EV)[:, 1]
    from scipy.stats import rankdata as rk
    def auc(pos, neg):
        r = rk(np.concatenate([pos, neg]))
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    print(f"E18 eval 单发: ev@95={gn(p, y_ev, 0.95):.4f} ev@100={gn(p, y_ev, 1.0):.4f} "
          f"ev@90={gn(p, y_ev, 0.90):.4f} AUC={auc(p[y_ev==1], p[y_ev==0]):.4f}")
    print("对照 C5: ev@95=0.2647 ev@100=0.1208 ev@90=0.3570")
    n0, (g0, inp0, oof0) = top[0]
    t0p = study.trials[n0].params
    import csv as _csv
    np.save(ROOT / "data/s3/e18_champion_train_oof.npy", oof0)
    with open(ROOT / "data/s3/e18_champion.json", "w") as f:
        json.dump({"trial": n0, "oof_gn95": g0, "input": inp0, "params": t0p}, f, indent=1)
    with open(ROOT / "data/s3/predictions_e18_eval.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["video", "label", "p_e18"])
        for r, pi in zip(ev_meta, p):
            w.writerow([r["video"], r["label"], f"{float(pi):.6f}"])
    print("champion saved")


if __name__ == "__main__":
    main()
