#!/usr/bin/env python3
"""R1b/R1c:30.89% 系统在新 1233 分布上全面重训(严格嵌套,source_sha 分组 10 外折)。

外折内完整复刻前人协议:
  每专家:5 内折(分组)三模型(LR+LGBM31+LGBM15)秩平均 → 内 OOF;全外训 3 模型 → 外测概率
  R1b = meta LR(C=100,pw=2)+ LGBM 残差 boost(前人 p_base 结构)
  R1c = 冠军 LGBM 吃 [oof15 ⊕ X303](E18 结构)
外折并行(fork 共享特征矩阵)。输出:data/pbase/out/r1_oof.npz + r1_stacks.pkl
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream"))
from training.expert_definitions import EXPERTS  # noqa: E402
from training import feature_loader as FL  # noqa: E402

FL.FEATURE_REGISTRY["asr"] = lambda c, l, s, a: np.zeros(14, np.float32)
CACHE = ROOT / "data/pbase/upstream/data/cache"
OUT = ROOT / "data/pbase/out"

G = {}  # fork 前填充,子进程共享


def train_three(X, y, seed=42):
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
    g31 = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                             min_child_samples=20, random_state=seed, verbose=-1).fit(X, y)
    g15 = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.05,
                             min_child_samples=30, random_state=seed, verbose=-1).fit(X, y)
    return [lr, g31, g15]


def run_fold(args_):
    k, tr, te = args_
    Xs, y, groups, X303, PC = G["Xs"], G["y"], G["groups"], G["X303"], G["PC"]
    tk = time.time()
    oof_tr = np.zeros((len(tr), 15))
    prob_te = np.zeros((len(te), 15))
    for j, spec in enumerate(EXPERTS):
        X = Xs[spec["name"]]
        # 内折 OOF(分组防泄漏;前人用普通分层,此处从严)
        inner = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        oof_j = np.zeros(len(tr))
        for i_tr, i_va in inner.split(X[tr], y[tr], groups[tr]):
            ms = train_three(X[tr][i_tr], y[tr][i_tr])
            ranks = np.zeros(len(i_va))
            for m in ms:
                ranks += rankdata(m.predict_proba(X[tr][i_va])[:, 1])
            oof_j[i_va] = ranks / 3.0 / len(i_va)
        oof_tr[:, j] = oof_j
        ms = train_three(X[tr], y[tr])
        prob_te[:, j] = np.mean([m.predict_proba(X[te])[:, 1] for m in ms], 0)

    # R1b: meta LR + 残差 boost(前人 train.py 配方)
    meta = LogisticRegression(C=100, class_weight={0: 1, 1: 2.0}, max_iter=10000)
    meta.fit(oof_tr, y[tr])
    oof_meta = meta.predict_proba(oof_tr)[:, 1]
    residual = y[tr].astype(float) - oof_meta
    boost = lgb.LGBMRegressor(num_leaves=4, n_estimators=40, learning_rate=0.04,
                              min_child_samples=30, random_state=42, verbose=-1)
    boost.fit(oof_tr, residual)
    pb = np.clip(meta.predict_proba(prob_te)[:, 1] + boost.predict(prob_te), 0, 1)

    # R1c: 冠军 LGBM [oof15 ⊕ X303]
    mc = lgb.LGBMClassifier(num_leaves=PC["leaves"], n_estimators=PC["est"],
                            learning_rate=PC["lr"], min_child_samples=PC["mcs"],
                            scale_pos_weight=PC["spw"], feature_fraction=PC["ff"],
                            bagging_fraction=PC["bf"], bagging_freq=1,
                            random_state=42, verbose=-1)
    mc.fit(np.hstack([oof_tr, X303[tr]]), y[tr])
    pc = mc.predict_proba(np.hstack([prob_te, X303[te]]))[:, 1]
    print(f"outer fold {k}: {time.time()-tk:.0f}s", flush=True)
    return k, te, pb, pc, oof_tr, prob_te, tr


def main():
    import csv
    targets = [json.loads(l) for l in open(ROOT / "data/pbase/upstream/splits/train_v2.jsonl")]
    lab = {r["filename"]: r["grade"] for r in csv.DictReader(
        open(ROOT / "data/tutu_task1_annotations_1233.csv", encoding="utf-8-sig"))}
    shas = {r["filename"]: r.get("source_sha", "") for r in csv.DictReader(
        open(ROOT / "data/api_judge_video_image_map.csv"))}

    vids = [e["video"] for e in targets]
    y = np.array([1 if lab[Path(v).name] == "bad" else 0 for v in vids])

    def sha_of(v):
        fn = Path(v).name
        s = shas.get(fn, "")
        if s:
            return s
        p = fn.split("__")
        return p[1] if len(p) > 1 else fn

    groups = np.array([sha_of(v) for v in vids])

    # 预计算 15 专家特征矩阵(新数据全 rlhf,per_source 塌缩为全局,忠实于回退逻辑)
    Xs = {}
    t0 = time.time()
    last_feats, last_name = None, None
    for spec in EXPERTS:
        nm = spec["name"]
        if spec["features"] == last_feats:
            Xs[nm] = Xs[last_name]
            continue
        rows = [FL.featurize_one(CACHE, e["label"], Path(e["video"]).stem,
                                 e["abs_path"], spec["features"]) for e in targets]
        Xs[nm] = np.stack(rows).astype(np.float64)
        last_feats, last_name = spec["features"], nm
        print(f"feats {nm}: {Xs[nm].shape} ({time.time()-t0:.0f}s)", flush=True)
    X303 = np.load(OUT / "X303_new.npy")
    PC = json.load(open(ROOT / "data/s3/e18_champion.json"))["params"]

    n = len(y)
    r1b = np.full(n, np.nan)
    r1c = np.full(n, np.nan)
    G.update(Xs=Xs, y=y, groups=groups, X303=X303, PC=PC)

    lb = json.load(open(ROOT / "data/lockbox_split.json"))
    name_idx = {Path(v).name: i for i, v in enumerate(vids)}
    dev_idx = np.array([name_idx[f] for f in lb["dev"]])
    test_idx = np.array([name_idx[f] for f in lb["test"]])
    outer = StratifiedGroupKFold(10, shuffle=True, random_state=42)
    fold_args = [(k, dev_idx[tr], dev_idx[te]) for k, (tr, te) in
                 enumerate(outer.split(X303[dev_idx], y[dev_idx], groups[dev_idx]))]
    fold_args.append((99, dev_idx, test_idx))  # 终枪折:dev全量→test
    import multiprocessing as mp
    stacks = {}
    with mp.get_context("fork").Pool(10) as pool:
        for k, te, pb, pc, oof_tr, prob_te, tr in pool.imap_unordered(run_fold, fold_args):
            r1b[te] = pb
            r1c[te] = pc
            stacks[k] = {"tr": tr, "te": te, "oof_tr": oof_tr, "prob_te": prob_te}
    pickle.dump(stacks, open(OUT / "r1_lockbox_stacks.pkl", "wb"))

    # 锁箱版:dev 行有 OOF,test 行有 dev→test 预测,全 1233 覆盖

    def br_at(y, s, rel=0.80):
        quota = rel * (y == 0).sum(); rg = rb_ = 0.0; nb = (y == 1).sum()
        for v in np.unique(np.sort(s)):
            g = ((y == 0) & (s == v)).sum(); bb = ((y == 1) & (s == v)).sum()
            if rg + g <= quota:
                rg += g
            else:
                f = (quota - rg) / max(1e-9, g) if g else 0.0
                rb_ += bb * (1 - f); rb_ += ((y == 1) & (s > v)).sum(); break
        else:
            return 0.0
        return float(rb_ / max(1, nb))

    def auc(y, s):
        r = rankdata(s); pos = r[y == 1]
        return float((pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (y == 0).sum()))

    for tag, s in [("R1b p_base结构重训", r1b), ("R1c E18结构重训 ", r1c)]:
        print(f"{tag}: AUC={auc(y,s):.4f} br@70={br_at(y,s,.70):.4f} "
              f"br@80={br_at(y,s,.80):.4f} br@90={br_at(y,s,.90):.4f}")
    np.savez(OUT / "r1_lockbox.npz", r1b=r1b, r1c=r1c, y=y,
             videos=np.array(vids), groups=groups)
    print("R1_DONE")


if __name__ == "__main__":
    main()
