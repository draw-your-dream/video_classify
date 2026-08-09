"""E47:把 32B VLM 的满折 OOF 作为第 16 个专家入栈(2026-08-09 预注册)。

前置:E45 中 32B+三档 是唯一过尾部门者(fold0 tail-AUC 0.5727 > 0.55 判准);
八个 8B 配置(目标格式/训练分布/文本信号/时序分辨率/空间分辨率)全部钉在 0.5155~0.5324。

**判准(发车前冻结)**:train-OOF gn@95 > 0.3218(E18 冠军基准)。
这一关才是历史上所有 VLM/深度专家倒下的地方——E34(AUC 0.665)、E36(0.649)整体 AUC
都不比 32B 的 0.6587 差,入栈后无一过线。故 AUC 不作数,只认 gn@95。

**方法学限制(须随结果一并报告)**:32B 用 2 折 OOF 而组合器用 5 折,折结构不一致会带来
轻微乐观偏差;若本实验过线,须在 eval 单发前用一致折结构复核。

变体:A 仅加分数列;B 分数列 + 缺失指示列(缺失非随机,源自 crops_v3 覆盖率);
      C 分数列的秩变换(消除尺度差异);D 仅在有覆盖子集上诊断(不作判准,仅看信号强弱)。
"""
import json
import pickle
import sys

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

ROOT = "/root/mech"


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


tag = sys.argv[1] if len(sys.argv) > 1 else "e45_32b_full"
score = np.load(f"{ROOT}/data/{tag}_score.npy")

# 重建 items 索引映射(e45 脚本未保存映射,此处按同一确定性逻辑复原)
tr = [json.loads(l) for l in open(f"{ROOT}/splits/train_v3.jsonl")]
rel_of = {}
for l in open(f"{ROOT}/manifest_all.tsv"):
    if l.strip():
        rel = l.split("\t")[0]
        rel_of[rel.split("/")[-1]] = rel
keep = [i for i, r in enumerate(tr) if rel_of.get(r["video"])]
assert len(keep) == len(score), f"映射长度不符 {len(keep)} vs {len(score)}"
full = np.full(len(tr), np.nan)
full[np.array(keep)] = score
miss = ~np.isfinite(full)
print(f"32B OOF 覆盖 {int((~miss).sum())}/{len(tr)}(缺失 {int(miss.sum())},源自 crops_v3 覆盖率)",
      flush=True)

oof15, _ev, y_tr, *_ = pickle.load(open(f"{ROOT}/upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15 = np.asarray(oof15, float)
y_tr = np.asarray(y_tr, int)
z = np.load(f"{ROOT}/upstream/cache_v3/_full_raw_v2.npz")
X = z["X_tr"].astype(float)
md = np.nanmedian(X, axis=0)
ii = np.where(~np.isfinite(X))
X[ii] = np.take(md, ii[1])
B0 = np.hstack([oof15, X])
c = json.load(open(f"{ROOT}/data/s3/e18_champion.json"))["params"]


def mk():
    return lgb.LGBMClassifier(
        num_leaves=c["leaves"], n_estimators=c["est"], learning_rate=c["lr"],
        min_child_samples=c["mcs"], scale_pos_weight=c["spw"], feature_fraction=c["ff"],
        bagging_fraction=c["bf"], bagging_freq=1, random_state=42, verbose=-1)


def run(Bm, tag_, seeds=(42,)):
    out = []
    for sd in seeds:
        o = np.zeros(len(y_tr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=sd).split(Bm, y_tr):
            m = mk()
            m.fit(Bm[a], y_tr[a])
            o[b] = m.predict_proba(Bm[b])[:, 1]
        out.append(gn(o, y_tr))
    s = float(np.mean(out))
    flag = "✔ 过基准" if s > 0.3218 else ""
    print(f"[E47 {tag_:34s}] gn@95 = {s:.4f}  {flag}"
          + (f"  (各种子 {[round(x,4) for x in out]})" if len(out) > 1 else ""), flush=True)
    return s


run(B0, "基准(15专家⊕320,复现)")

filled = np.where(miss, np.nanmedian(full), full)
run(np.hstack([oof15, filled[:, None], X]), "A 仅加 32B 分数列")
run(np.hstack([oof15, filled[:, None], miss[:, None].astype(float), X]), "B +缺失指示列")

from scipy.stats import rankdata
rk = np.full(len(tr), 0.5)
rk[~miss] = rankdata(full[~miss]) / (~miss).sum()
run(np.hstack([oof15, rk[:, None], X]), "C 秩变换")

sub = ~miss
o = np.zeros(int(sub.sum()))
Bs = np.hstack([oof15[sub], filled[sub, None], X[sub]])
ys = y_tr[sub]
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(Bs, ys):
    m = mk(); m.fit(Bs[a], ys[a]); o[b] = m.predict_proba(Bs[b])[:, 1]
Bs0 = np.hstack([oof15[sub], X[sub]])
o0 = np.zeros(int(sub.sum()))
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(Bs0, ys):
    m = mk(); m.fit(Bs0[a], ys[a]); o0[b] = m.predict_proba(Bs0[b])[:, 1]
print(f"[E47 D 仅覆盖子集(诊断,非判准)] 加32B {gn(o, ys):.4f} vs 同子集基准 {gn(o0, ys):.4f}",
      flush=True)
print("E47_DONE", flush=True)
