#!/usr/bin/env python3
"""dev150 组合器装配:运动特征 ⊕ flash 三档 ⊕ pro 三档 → 交叉验证 OOF 分 → br@80%。

对照组:flash 单独 / pro 单独 / still_frac 单独 / 运动特征组 / 全组合。
n=150(bad 108 / gn 42),10 折分层 CV,逻辑回归 + 小 LGBM 双模型取稳者。
holdout 1083 不碰。
"""
import csv
import json
import collections
import numpy as np
from itertools import product

ANN = {r["filename"]: r for r in csv.DictReader(
    open("data/tutu_task1_annotations_1233.csv", encoding="utf-8-sig"))}
DEV = json.load(open("data/api_judge_split.json"))["dev"]
ORD = {"good": 0.0, "normal": 0.25, "abstain": 0.5, "bad": 1.0}


def api_col(path):
    d = {}
    for l in open(path):
        r = json.loads(l)
        if "result" in r:
            d[r["filename"]] = ORD.get(r["result"]["grade"], 0.5)
    return d


def br_at(y, s, rel=0.80):
    """并列组按比例放行的公平记账:放行恰好 rel 比例的 gn,报被拦 bad 比例。"""
    y = np.asarray(y)
    s = np.asarray(s, float)
    quota = rel * (y == 0).sum()
    released_gn = 0.0
    removed_bad = 0.0
    nb = (y == 1).sum()
    for v in np.unique(np.sort(s)):
        g = ((y == 0) & (s == v)).sum()
        b = ((y == 1) & (s == v)).sum()
        if released_gn + g <= quota:
            released_gn += g          # 整组放行,组内 bad 漏过(不计 removed)
        else:
            f = (quota - released_gn) / max(1e-9, g) if g else 0.0
            removed_bad += b * (1 - f)
            released_gn = quota
            # 之后的组全拦
            removed_bad += ((y == 1) & (s > v)).sum()
            break
    else:
        pass
    return float(removed_bad / max(1, nb))


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0
                          for a, b in product(pos, neg)]))


def main():
    flash = api_col("data/dev150_flash_v2.jsonl")
    pro = api_col("data/dev150_pro_v6.jsonl")
    mot = {r["filename"]: r for r in csv.DictReader(open("data/motion_feats_1233.csv"))}
    MCOLS = ["still_frac", "bimod", "spike_ratio", "jump_cnt", "acf1", "hf_ratio",
             "c_still_frac", "c_bimod", "c_spike_ratio", "c_jump_cnt", "c_acf1", "c_hf_ratio",
             "rhy_pow", "rhy_peak", "micro_rhy", "grav_r2", "fallrise", "vzc", "vmag"]

    fns = [f for f in DEV if f in flash and f in pro and f in mot]
    y = np.array([1 if ANN[f]["grade"] == "bad" else 0 for f in fns])
    print(f"样本 {len(fns)} | bad {y.sum()} | gn {(y==0).sum()}")

    def mfeat(f):
        out = []
        for c in MCOLS:
            try:
                out.append(float(mot[f][c]))
            except (ValueError, KeyError):
                out.append(np.nan)
        return out

    X_m = np.array([mfeat(f) for f in fns])
    med = np.nanmedian(X_m, axis=0)
    ii = np.where(~np.isfinite(X_m))
    X_m[ii] = np.take(med, ii[1])
    x_f = np.array([flash[f] for f in fns])
    x_p = np.array([pro[f] for f in fns])

    sets = {
        "flash 单独": x_f.reshape(-1, 1),
        "pro 单独": x_p.reshape(-1, 1),
        "flash+pro": np.stack([x_f, x_p], 1),
        "运动特征组": X_m,
        "运动+flash": np.hstack([X_m, x_f.reshape(-1, 1)]),
        "运动+pro": np.hstack([X_m, x_p.reshape(-1, 1)]),
        "全组合": np.hstack([X_m, x_f.reshape(-1, 1), x_p.reshape(-1, 1)]),
    }

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    print(f"\n{'方案':<12} {'br@70':>7} {'br@80':>7} {'br@90':>7} {'AUC':>7}  (10折OOF, LR/LGBM取优)")
    for name, X in sets.items():
        best = None
        for kind in ("lr", "gbm"):
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(10, shuffle=True, random_state=42).split(X, y):
                if kind == "lr":
                    sc = StandardScaler().fit(X[tr])
                    m = LogisticRegression(C=0.5, max_iter=1000).fit(sc.transform(X[tr]), y[tr])
                    oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
                else:
                    m = lgb.LGBMClassifier(num_leaves=5, n_estimators=60, learning_rate=0.05,
                                           min_child_samples=15, verbose=-1, random_state=42)
                    m.fit(X[tr], y[tr])
                    oof[te] = m.predict_proba(X[te])[:, 1]
            r = (br_at(y, oof, 0.70), br_at(y, oof, 0.80), br_at(y, oof, 0.90), auc(y, oof))
            if best is None or r[1] > best[1]:
                best = r
                bk = kind
        print(f"{name:<12} {best[0]:7.3f} {best[1]:7.3f} {best[2]:7.3f} {best[3]:7.3f}  [{bk}]")

    print("\n参照(不同分布不可直比):eval_v3 最好基线 br@80=0.556;"
          "本数据 bad 率 72%(dev150),难度高得多")


if __name__ == "__main__":
    main()
