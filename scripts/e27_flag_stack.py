#!/usr/bin/env python
"""E27:最新一代检测器旗标(眉毛v4∪v5 / 悬空双钥匙确认 / 场景突变确认)作为3列特征
并入冠军输入,train-OOF 过门测试(基准 0.3218)。附尾部82条覆盖统计。盒上运行。"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/root/mech")


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def main():
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    oof15, ev15, y_tr, y_ev, *_ = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15 = np.asarray(oof15, float)
    y_tr = np.asarray(y_tr, int)
    z = np.load(ROOT / "upstream/cache_v3/_full_raw_v2.npz")
    X_tr = z["X_tr"].astype(float)
    md = np.nanmedian(X_tr, axis=0)
    ii = np.where(~np.isfinite(X_tr)); X_tr[ii] = np.take(md, ii[1])
    tr = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    bns = [r["video"] for r in tr]

    def load_flags(fp, cat):
        s = set()
        for l in open(fp):
            j = json.loads(l)
            if j.get(cat):
                s.add(os.path.basename(j["rel"]))
        return s

    eb = load_flags(ROOT / "data/e12_v4.jsonl", "eyebrows") | load_flags(ROOT / "data/e12_v5.jsonl", "eyebrows")
    fl = {os.path.basename(r["rel"]) for r in json.load(open(ROOT / "data/s3/twokey_v3_floating_hits.json"))}
    sj = {os.path.basename(r["rel"]) for r in json.load(open(ROOT / "data/s3/twokey_v2_scenejump_hits.json"))}
    F = np.array([[1.0 * (b in eb), 1.0 * (b in fl), 1.0 * (b in sj)] for b in bns])
    hit_b = [int(((F[:, j] == 1) & (y_tr == 1)).sum()) for j in range(3)]
    hit_g = [int(((F[:, j] == 1) & (y_tr == 0)).sum()) for j in range(3)]
    print(f"旗标覆盖(train) 眉毛 {int(F[:,0].sum())}(bad{hit_b[0]}/good{hit_g[0]}) "
          f"悬空 {int(F[:,1].sum())}(bad{hit_b[1]}/good{hit_g[1]}) "
          f"突变 {int(F[:,2].sum())}(bad{hit_b[2]}/good{hit_g[2]})", flush=True)
    oof_ch = np.load(ROOT / "data/s3/e18_champion_train_oof.npy")
    bad_i = np.where(y_tr == 1)[0]
    tail = bad_i[np.argsort(oof_ch[bad_i])[:82]]
    print(f"82条尾部难bad中旗标命中: {int(F[tail].any(1).sum())}", flush=True)

    champ = json.load(open(ROOT / "data/s3/e18_champion.json"))["params"]

    def champ_model():
        return lgb.LGBMClassifier(num_leaves=champ["leaves"], n_estimators=champ["est"],
                                  learning_rate=champ["lr"], min_child_samples=champ["mcs"],
                                  scale_pos_weight=champ["spw"], feature_fraction=champ["ff"],
                                  bagging_fraction=champ["bf"], bagging_freq=1,
                                  random_state=42, verbose=-1)

    B = np.hstack([oof15, X_tr])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))
    for name, TR in (("冠军基准", B), ("冠军+3列新旗标", np.hstack([B, F]))):
        o = np.zeros(len(y_tr))
        for a, b in folds:
            m = champ_model(); m.fit(TR[a], y_tr[a]); o[b] = m.predict_proba(TR[b])[:, 1]
        print(f"{name}: train-OOF gn@95 = {gn(o, y_tr):.4f}", flush=True)
    print("E27_DONE", flush=True)


if __name__ == "__main__":
    main()
