#!/usr/bin/env python
"""第三波训练门判决(2026-08-04 预注册):E21/E22/E23 各自并入冠军输入,
5折 OOF(折固定 seed42)对比冠军基准 0.3218;过门者合并单发 eval。
附:每列 midrank AUC(可解释性)+ 用户35条盲区bad的新特征命中核验。"""
from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHAMP_OOF = 0.3218


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def midrank_auc(x, y):
    from scipy.stats import rankdata
    m = np.isfinite(x)
    if m.sum() < 50 or y[m].sum() < 10:
        return np.nan
    r = rankdata(x[m])
    yy = y[m]
    n1 = yy.sum()
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1)))


def load_family(name):
    p = ROOT / f"data/s3/{name}.csv"
    rd = list(csv.reader(open(p)))
    cols = rd[0][1:]
    d = {os.path.basename(r[0]): [float(v) if v not in ("", "nan") else np.nan for v in r[1:]]
         for r in rd[1:]}
    return cols, d


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
    ev_meta = [json.loads(l) for l in open(ROOT / "splits/eval_v3.jsonl")]
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

    def oof_gate(TR):
        o = np.zeros(len(y_tr))
        for a, b in folds:
            m = champ_model(); m.fit(TR[a], y_tr[a])
            o[b] = m.predict_proba(TR[b])[:, 1]
        return gn(o, y_tr), o

    fams = {}
    for name in ("e21_bitstream", "e22_drift", "e23_depth"):
        cols, d = load_family(name)
        F_tr = np.array([d.get(r["video"], [np.nan] * len(cols)) for r in tr_meta], float)
        F_ev = np.array([d.get(r["video"], [np.nan] * len(cols)) for r in ev_meta], float)
        keep = [j for j in range(len(cols)) if np.isfinite(F_tr[:, j]).mean() > 0.5]
        cols2 = [cols[j] for j in keep]
        F_tr, F_ev = F_tr[:, keep], F_ev[:, keep]
        fmd = np.nanmedian(F_tr, axis=0)
        for A in (F_tr, F_ev):
            ii = np.where(~np.isfinite(A)); A[ii] = np.take(fmd, ii[1])
        cov = np.isfinite(np.array([d.get(r["video"], [np.nan] * len(cols)) for r in tr_meta], float)).all(1).mean()
        aucs = sorted(((midrank_auc(F_tr[:, j], y_tr), cols2[j]) for j in range(len(cols2))),
                      key=lambda t: -abs((t[0] or 0.5) - 0.5))
        print(f"\n== {name}: {len(cols2)}列 覆盖{cov:.0%} ==", flush=True)
        print("  单列AUC top5:", [(c, round(a, 3)) for a, c in aucs[:5] if np.isfinite(a)], flush=True)
        g, o = oof_gate(np.hstack([B_tr, F_tr]))
        print(f"  冠军+{name} train-OOF gn@95 = {g:.4f} (基准 {CHAMP_OOF})  "
              f"{'过门' if g > CHAMP_OOF else '不过'}", flush=True)
        fams[name] = dict(g=g, F_tr=F_tr, F_ev=F_ev, cols=cols2)

    passers = [n for n, v in fams.items() if v["g"] > CHAMP_OOF]
    print(f"\n过门家族: {passers or '无'}", flush=True)
    # 合并(全部三族)也报一个 OOF,供诚实记录;eval 只发过门组合
    g_all, _ = oof_gate(np.hstack([B_tr] + [fams[n]["F_tr"] for n in fams]))
    print(f"冠军+三族全并 train-OOF gn@95 = {g_all:.4f}", flush=True)
    if passers:
        TR = np.hstack([B_tr] + [fams[n]["F_tr"] for n in passers])
        EV = np.hstack([B_ev] + [fams[n]["F_ev"] for n in passers])
        g_p, _ = oof_gate(TR)
        print(f"冠军+过门组合 train-OOF gn@95 = {g_p:.4f}", flush=True)
        m = champ_model(); m.fit(TR, y_tr)
        p = m.predict_proba(EV)[:, 1]
        from scipy.stats import rankdata as rk
        def auc(pos, neg):
            r = rk(np.concatenate([pos, neg]))
            return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
        print(f"EVAL单发: ev@95={gn(p, y_ev):.4f} ev@100={gn(p, y_ev, 1.0):.4f} "
              f"ev@90={gn(p, y_ev, 0.90):.4f} AUC={auc(p[y_ev == 1], p[y_ev == 0]):.4f}", flush=True)
        print("对照 E18: ev@95=0.2913 ev@100=0.0409 ev@90=0.4334 AUC=0.7593", flush=True)
        with open(ROOT / "data/s3/predictions_wave3_eval.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["video", "label", "p_wave3"])
            for r, pi in zip(ev_meta, p):
                w.writerow([r["video"], r["label"], f"{float(pi):.6f}"])

    # ---- 用户35条盲区bad核验:对应新特征在全train分布中的分位 ----
    print("\n== 用户35条判读 × 新特征命中核验(特征值在train bad+good全体中的分位) ==", flush=True)
    mani = json.load(open("/mnt/c/Users/Lenovo/Downloads/TUTU难bad增补gallery/manifest.json"))
    CHECK = {
        "21": [("e21_bitstream", "mv_zero_frac", "背景全静→零MV占比"),
               ("e21_bitstream", "mv_mag_mean", "背景全静→MV幅值(低)")],
        "20": [("e23_depth", "cd_med_cv", "大小/远近违透视→角色深度轨迹变异")],
        "22": [("e23_depth", "gap_mean", "悬空→角色-支撑深度差")],
        "31": [("e23_depth", "gap_mean", "无支撑→角色-支撑深度差")],
        "26": [("e21_bitstream", "i_frac", "镜头切换→I帧占比"), ("e22_drift", "g_maxjump", "场景突变→全局漂移突变")],
        "27": [("e21_bitstream", "i_frac", "镜头切换→I帧占比"), ("e22_drift", "g_maxjump", "场景突变→全局漂移突变")],
        "28": [("e21_bitstream", "i_frac", "镜头切换→I帧占比"), ("e22_drift", "g_maxjump", "场景突变→全局漂移突变")],
        "29": [("e21_bitstream", "i_frac", "镜头切换→I帧占比"), ("e22_drift", "g_maxjump", "场景突变→全局漂移突变")],
        "30": [("e21_bitstream", "i_frac", "镜头切换→I帧占比"), ("e22_drift", "g_maxjump", "场景突变→全局漂移突变")],
        "35": [("e22_drift", "c_end", "换装→角色漂移终值"), ("e22_drift", "c_maxjump", "换装→角色漂移突变")],
        "1": [("e21_bitstream", "mv_mag_cv", "卡顿→MV时间变异")],
        "3": [("e21_bitstream", "mv_mag_cv", "卡顿→MV时间变异")],
        "4": [("e21_bitstream", "mv_mag_cv", "卡顿→MV时间变异")],
        "5": [("e21_bitstream", "mv_mag_cv", "卡顿→MV时间变异")],
    }
    fam_data = {}
    for name in fams:
        cols, d = load_family(name)
        fam_data[name] = (cols, d)
    tr_bns = [r["video"] for r in tr_meta]
    for row in mani:
        if "_note" in row or not row.get("num"):
            continue
        n = str(int(row["num"]))
        if n not in CHECK:
            continue
        bn = row["video"]
        outs = []
        for fam, col, why in CHECK[n]:
            cols, d = fam_data[fam]
            j = cols.index(col)
            v = d.get(bn, [np.nan] * len(cols))[j]
            pop = np.array([d.get(b, [np.nan] * len(cols))[j] for b in tr_bns], float)
            pop = pop[np.isfinite(pop)]
            if np.isfinite(v) and len(pop):
                pct = float((pop < v).mean())
                outs.append(f"{col}={v:.3g}(P{pct*100:.0f}) [{why}]")
            else:
                outs.append(f"{col}=缺失 [{why}]")
        print(f"  #{row['num']} {bn} ({row.get('user_audit', row.get('grp',''))}): " + "; ".join(outs), flush=True)


if __name__ == "__main__":
    main()
