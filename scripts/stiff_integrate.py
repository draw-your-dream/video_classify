#!/usr/bin/env python
"""E30:僵硬信号结合总分四路线(2026-08-06 预注册)。全部 train 侧,过门 0.3218 才谈 eval。
A 带新特征重搜索(60 trial) | B 交互特征(动∧刚) | C 僵硬头OOF当单列 | D 秩加权融合。
附:僵硬报告器top60 与 82尾部bad 的重叠。"""
from __future__ import annotations

import csv
import json
import os
import pickle

import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def main():
    oof15, ev15, y_tr, y_ev, *_ = pickle.load(open("upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15 = np.asarray(oof15, float); y_tr = np.asarray(y_tr, int)
    z = np.load("upstream/cache_v3/_full_raw_v2.npz")
    X_tr = z["X_tr"].astype(float)
    md = np.nanmedian(X_tr, axis=0)
    ii = np.where(~np.isfinite(X_tr)); X_tr[ii] = np.take(md, ii[1])
    tr = [json.loads(l) for l in open("splits/train_v3.jsonl")]
    reasons = {r["path"]: r["reasons"] for r in csv.DictReader(open("data/s3/merged_labels.csv", encoding="utf-8-sig"))}
    rd = list(csv.reader(open("data/s3/rigid_feats.csv")))
    rcols = rd[0][1:]
    rmap = {os.path.basename(r[0]): [float(v) if v not in ("", "nan") else np.nan for v in r[1:]] for r in rd[1:]}
    R = np.array([rmap.get(r["video"], [np.nan] * len(rcols)) for r in tr], float)
    rmd = np.nanmedian(R, axis=0)
    ri = np.where(~np.isfinite(R)); R[ri] = np.take(rmd, ri[1])
    B = np.hstack([oof15, X_tr])
    champ = json.load(open("data/s3/e18_champion.json"))["params"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))

    def champ_model(**kw):
        p = dict(num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
                 min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"],
                 feature_fraction=champ["ff"], bagging_fraction=champ["bf"], bagging_freq=1,
                 random_state=42, verbose=-1)
        p.update(kw)
        return lgb.LGBMClassifier(**p)

    def oof_of(TR, mk=champ_model):
        o = np.zeros(len(y_tr))
        for a, b in folds:
            m = mk(); m.fit(TR[a], y_tr[a]); o[b] = m.predict_proba(TR[b])[:, 1]
        return o

    # 僵硬头 full-train OOF(僵硬vs good 训练,预测全体)
    stiff = np.array([1 if (r["label"] == "bad" and "僵硬" in reasons.get(r["video"], "")) else 0 for r in tr])
    good = np.array([1 if r["label"] != "bad" else 0 for r in tr])
    sub = (stiff == 1) | (good == 1)
    BR = np.hstack([B, R])
    sh = np.zeros(len(y_tr))
    for a, b in folds:
        a2 = a[sub[a]]
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                               min_child_samples=30, random_state=42, verbose=-1)
        m.fit(BR[a2], stiff[a2])
        sh[b] = m.predict_proba(BR[b])[:, 1]

    # 报告器×尾部重叠
    ch_oof = np.load("data/s3/e18_champion_train_oof.npy")
    bad_i = np.where(y_tr == 1)[0]
    tail = set(bad_i[np.argsort(ch_oof[bad_i])[:82]])
    top60 = set(np.argsort(-sh)[:60])
    print(f"[重叠] 僵硬头top60 ∩ 82尾部bad = {len(top60 & tail)}", flush=True)

    print(f"[基准] 冠军 OOF = {gn(ch_oof, y_tr):.4f}", flush=True)

    # B 交互特征
    def col(name):
        return R[:, rcols.index(name)]
    mnorm = rankdata(col("m_mean")) / len(y_tr)
    rr_inv = 1 - rankdata(col("rr_mean")) / len(y_tr)
    qd_inv = 1 - rankdata(col("quad_dis")) / len(y_tr)
    hf_inv = 1 - rankdata(col("hf_static")) / len(y_tr)
    inter = np.stack([
        mnorm * rr_inv,                      # 在动∧残差低
        mnorm * qd_inv,                      # 在动∧象限同向
        mnorm * rr_inv * qd_inv,             # 三重
        col("rr_lowfrac") * mnorm,           # 刚性帧占比×动
        hf_inv * (1 - mnorm),                # 静∧无微颤(死寂静止)
    ], 1)
    gB = gn(oof_of(np.hstack([B, inter])), y_tr)
    print(f"[B 交互5列] {gB:.4f}", flush=True)

    # C 头分数单列
    gC = gn(oof_of(np.hstack([B, sh[:, None]])), y_tr)
    print(f"[C 僵硬头单列] {gC:.4f}", flush=True)

    # D 秩融合
    rc = rankdata(ch_oof); rs = rankdata(sh)
    best = (0, 0.0)
    for lam in (0.02, 0.05, 0.08, 0.12, 0.18, 0.25):
        g = gn(rc + lam * rs, y_tr)
        if g > best[0]:
            best = (g, lam)
        print(f"[D 融合 λ={lam}] {g:.4f}", flush=True)

    # A 带新特征重搜索
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    TRA = np.hstack([B, R, inter, sh[:, None]])
    def obj(t):
        m = lgb.LGBMClassifier(
            num_leaves=t.suggest_int("leaves", 7, 63),
            n_estimators=t.suggest_int("est", 100, 800),
            learning_rate=t.suggest_float("lr", 0.02, 0.1, log=True),
            min_child_samples=t.suggest_int("mcs", 5, 60),
            scale_pos_weight=t.suggest_float("spw", 1.0, 3.0),
            feature_fraction=t.suggest_float("ff", 0.3, 1.0),
            bagging_fraction=t.suggest_float("bf", 0.5, 1.0),
            bagging_freq=1, random_state=42, verbose=-1)
        o = np.zeros(len(y_tr))
        for a, b in folds:
            m.fit(TRA[a], y_tr[a]); o[b] = m.predict_proba(TRA[b])[:, 1]
        return gn(o, y_tr)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
    st.optimize(obj, n_trials=60, show_progress_bar=False)
    print(f"[A 重搜索60trial(含刚体+交互+头)] best={st.best_value:.4f} params={st.best_params}", flush=True)
    print("E30_DONE", flush=True)


if __name__ == "__main__":
    main()
