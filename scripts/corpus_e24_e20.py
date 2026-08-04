#!/usr/bin/env python
"""E24 分缺陷多头(结构性)+ E20 过拟合保险(2026-08-04 预注册)。"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


GROUPS = {
    "僵硬": ["僵硬"],
    "卡顿": ["卡顿/少活人感", "动作位移不连贯"],
    "少动": ["四肢不动", "静止不动", "运动主体", "慢动作"],
    "还原": ["还原度", "衣服/身体的时间一致性", "大小变化"],
    "物理": ["物理规律", "不合理的物体"],
    "画面": ["帧跳变", "首帧一致", "背景运动混乱"],
}


def main():
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    oof15, ev15, y_tr, y_ev, *_ = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15, ev15 = np.asarray(oof15, float), np.asarray(ev15, float)
    y_tr, y_ev = np.asarray(y_tr, int), np.asarray(y_ev, int)
    z = np.load(ROOT / "upstream/cache_v3/_full_raw_v2.npz")
    X_tr, X_ev = z["X_tr"].astype(float), z["X_ev"].astype(float)
    md = np.nanmedian(X_tr, axis=0)
    for A in (X_tr, X_ev):
        ii = np.where(~np.isfinite(A)); A[ii] = np.take(md, ii[1])
    tr_meta = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    reasons = {r["path"]: r["reasons"] for r in csv.DictReader(
        open(ROOT / "data/s3/merged_labels.csv", encoding="utf-8-sig"))}
    champ = json.load(open(ROOT / "data/s3/e18_champion.json"))["params"]

    def champ_model():
        return lgb.LGBMClassifier(num_leaves=champ["leaves"], n_estimators=champ["est"],
                                  learning_rate=champ["lr"], min_child_samples=champ["mcs"],
                                  scale_pos_weight=champ["spw"], feature_fraction=champ["ff"],
                                  bagging_fraction=champ["bf"], bagging_freq=1,
                                  random_state=42, verbose=-1)

    B_tr, B_ev = np.hstack([oof15, X_tr]), np.hstack([ev15, X_ev])
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(skf.split(B_tr, y_tr))

    # ---------- E24 多头 ----------
    print("== E24 分缺陷多头 ==", flush=True)
    heads_oof = []
    good_mask = y_tr == 0
    for gname, tags in GROUPS.items():
        pos = np.array([y == 1 and any(t in reasons.get(r["video"], "") for t in tags)
                        for y, r in zip(y_tr, tr_meta)])
        sub = pos | good_mask
        o = np.full(len(y_tr), np.nan)
        for a, b in folds:
            a2 = a[sub[a]]
            m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                   min_child_samples=30, random_state=42, verbose=-1)
            m.fit(B_tr[a2], pos[a2].astype(int))
            o[b] = m.predict_proba(B_tr[b])[:, 1]
        heads_oof.append(o)
        from scipy.stats import rankdata
        def auc(p, yy):
            r = rankdata(p)
            n1 = yy.sum()
            return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))
        print(f"  {gname}: 正类 {pos.sum()},头 OOF AUC(该类vs全体) = {auc(o, pos.astype(int)):.3f}", flush=True)
    H_tr = np.stack(heads_oof, 1)
    TR24 = np.hstack([B_tr, H_tr])
    o24 = np.zeros(len(y_tr))
    for a, b in folds:
        m = champ_model(); m.fit(TR24[a], y_tr[a])
        o24[b] = m.predict_proba(TR24[b])[:, 1]
    g24 = gn(o24, y_tr)
    print(f"E24 冠军+6头 train-OOF gn@95 = {g24:.4f} (冠军基准 0.3218)", flush=True)
    # 备选结构:6头 max-rank 直接融合
    from scipy.stats import rankdata
    fuse = np.max(np.stack([rankdata(h) for h in heads_oof], 1), 1)
    print(f"E24b 6头max秩融合 train-OOF gn@95 = {gn(fuse, y_tr):.4f}", flush=True)

    # ---------- E20 保险 ----------
    print("\n== E20 新切分重演(60-trial × 2 切分) ==", flush=True)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_all = np.vstack([X_tr, X_ev])
    O_all = np.vstack([oof15, ev15])
    y_all = np.concatenate([y_tr, y_ev])
    srcs = []
    for r in tr_meta + [json.loads(l) for l in open(ROOT / "splits/eval_v3.jsonl")]:
        p = r["abs_path"]
        srcs.append("t" if "ti2i2v" in p else ("r" if "rlhf" in p else "s"))
    strat = np.array([f"{y}{s}" for y, s in zip(y_all, srcs)])
    from sklearn.model_selection import train_test_split
    for seed in (101, 202):
        ia, ib = train_test_split(np.arange(len(y_all)), test_size=0.2,
                                  random_state=seed, stratify=strat)
        TRa = np.hstack([O_all[ia], X_all[ia]]); EVb = np.hstack([O_all[ib], X_all[ib]])
        ya, yb = y_all[ia], y_all[ib]
        f2 = list(StratifiedKFold(5, shuffle=True, random_state=42).split(TRa, ya))
        best = {"g": -1}
        def obj(t):
            m = lgb.LGBMClassifier(
                num_leaves=t.suggest_int("leaves", 7, 63),
                n_estimators=t.suggest_int("est", 100, 800),
                learning_rate=t.suggest_float("lr", 0.02, 0.1, log=True),
                min_child_samples=t.suggest_int("mcs", 5, 60),
                scale_pos_weight=t.suggest_float("spw", 1.0, 3.0),
                feature_fraction=t.suggest_float("ff", 0.5, 1.0),
                bagging_fraction=t.suggest_float("bf", 0.5, 1.0),
                bagging_freq=1, random_state=42, verbose=-1)
            o = np.zeros(len(ya))
            for a, b in f2:
                m.fit(TRa[a], ya[a]); o[b] = m.predict_proba(TRa[b])[:, 1]
            g = gn(o, ya)
            if g > best["g"]:
                best.update(g=g, params=t.params)
            return g
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
        st.optimize(obj, n_trials=60, show_progress_bar=False)
        p = best["params"]
        m = lgb.LGBMClassifier(num_leaves=p["leaves"], n_estimators=p["est"], learning_rate=p["lr"],
                               min_child_samples=p["mcs"], scale_pos_weight=p["spw"],
                               feature_fraction=p["ff"], bagging_fraction=p["bf"],
                               bagging_freq=1, random_state=42, verbose=-1)
        m.fit(TRa, ya)
        pe = m.predict_proba(EVb)[:, 1]
        # 基线配方(C5固定超参)同切分对照
        mb = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                min_child_samples=20, scale_pos_weight=2.0, random_state=42, verbose=-1)
        mb.fit(TRa, ya)
        pb = mb.predict_proba(EVb)[:, 1]
        # 前人配方(meta LR 仅专家分)同切分对照
        from sklearn.linear_model import LogisticRegression
        ml = LogisticRegression(C=100.0, class_weight={0: 1, 1: 2.0}, max_iter=10000)
        ml.fit(O_all[ia], ya)
        pl = ml.predict_proba(O_all[ib])[:, 1]
        print(f"切分{seed}: 搜索冠军 ev@95={gn(pe, yb):.4f} | C5配方={gn(pb, yb):.4f} | 前人式meta={gn(pl, yb):.4f}", flush=True)
    print("E20_E24_DONE", flush=True)


if __name__ == "__main__":
    main()
