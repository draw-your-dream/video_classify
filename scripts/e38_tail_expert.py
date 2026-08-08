"""E38:条件式尾部专家(2026-08-08 预注册)。基准 0.3218。
思路:放行率只由"冠军低分区"决定,该区分布与全局不同。用嵌套 OOF 在低分区训专家,
只对该区重排序,高分区保持冠军原序(拼接式两段排序,无泄漏)。
变体:低分区比例 q ∈ {0.3, 0.4, 0.5};专家 = 小LGBM;专家输入 = 冠军输入。
无泄漏协议:外层5折决定评分,内层在训练折上按同一分位切区并训练专家。"""
import json
import pickle

import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


oof15, ev15, y_tr, y_ev, *_ = pickle.load(open("upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15 = np.asarray(oof15, float)
y_tr = np.asarray(y_tr, int)
z = np.load("upstream/cache_v3/_full_raw_v2.npz")
X_tr = z["X_tr"].astype(float)
md = np.nanmedian(X_tr, axis=0)
ii = np.where(~np.isfinite(X_tr))
X_tr[ii] = np.take(md, ii[1])
B = np.hstack([oof15, X_tr])
champ = json.load(open("data/s3/e18_champion.json"))["params"]


def champ_model():
    return lgb.LGBMClassifier(
        num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
        min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"],
        feature_fraction=champ["ff"], bagging_fraction=champ["bf"], bagging_freq=1,
        random_state=42, verbose=-1)


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))
base = np.load("data/s3/e18_champion_train_oof.npy")
print(f"基准 {gn(base, y_tr):.4f}", flush=True)

for q in (0.3, 0.4, 0.5):
    for leaves, est in ((7, 150), (15, 300)):
        final = np.zeros(len(y_tr))
        for a, b in folds:
            # 折内冠军(给 a 折内部打分,决定低分区)
            in_oof = np.zeros(len(a))
            for a2, b2 in StratifiedKFold(4, shuffle=True, random_state=7).split(B[a], y_tr[a]):
                m = champ_model()
                m.fit(B[a][a2], y_tr[a][a2])
                in_oof[b2] = m.predict_proba(B[a][b2])[:, 1]
            thr = np.quantile(in_oof, q)
            low = in_oof <= thr
            if y_tr[a][low].sum() < 20:
                final[b] = base[b]
                continue
            exp = lgb.LGBMClassifier(num_leaves=leaves, n_estimators=est, learning_rate=0.05,
                                     min_child_samples=20, scale_pos_weight=3.0,
                                     random_state=42, verbose=-1)
            exp.fit(B[a][low], y_tr[a][low])
            # 测试折:冠军全量模型定分区,专家重排低分区
            mc = champ_model()
            mc.fit(B[a], y_tr[a])
            pb = mc.predict_proba(B[b])[:, 1]
            thr_b = np.quantile(pb, q)
            lo_b = pb <= thr_b
            pe = exp.predict_proba(B[b][lo_b])[:, 1]
            out = np.zeros(len(b))
            # 高分区保持冠军排序(整体在上),低分区用专家分排序(整体在下)
            out[~lo_b] = 1.0 + rankdata(pb[~lo_b]) / max(1, (~lo_b).sum())
            out[lo_b] = rankdata(pe) / max(1, lo_b.sum())
            final[b] = out
        print(f"[E38 q={q} leaves={leaves}] train-OOF gn@95 = {gn(final, y_tr):.4f}", flush=True)
print("E38_DONE", flush=True)
