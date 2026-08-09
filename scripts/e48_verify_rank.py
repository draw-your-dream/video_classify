"""E48:E47-C(秩变换 +2.35pt)的证伪性验证(2026-08-09 预注册,发车前冻结判准)。

背景:E47 中"32B 分数秩变换"列使 train-OOF gn@95 从 0.3218 → 0.3453。
但存在一个足以完全解释该增益的替代假说:该列把缺失样本赋 0.5、覆盖样本赋 [0,1] 秩,
**同时编码了"该视频有没有 crops_v3"**;若覆盖与标签相关,增益来自这条捷径而非 32B 的判别力。

**四个对照(判准发车前冻结)**:
 ①置换检验:在**覆盖样本内部**随机打乱 32B 分数,缺失模式原封不动。
   → 若打乱后仍显著高于基准,增益即来自缺失模式,32B 信号被证伪。跑 10 次取分布。
 ②纯噪声列:用均匀随机数替代 32B 分数(同缺失模式),同样跑 10 次。
 ③多种子:真 32B 秩变换列在 5 个折种子下的稳定性(单种子 +2.35pt 可能是折划分运气,
   本项目已有前车之鉴:E42 训练侧 +0.78pt 统计显著仍在 eval 归零)。
 ④覆盖子集内对照:仅用有覆盖的 2849 条,真分数 vs 打乱分数(此时无缺失可利用,
   增益若存活即为 32B 的净贡献)。

**通过条件**:真分数的多种子均值须显著高于①②的分布上沿,且④中真分数须胜出。
任一不满足,判 E47-C 为缺失捷径伪影,不进 eval。
"""
import json
import pickle

import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

ROOT = "/root/mech"


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


score = np.load(f"{ROOT}/data/e45_32b_full_score.npy")
tr = [json.loads(l) for l in open(f"{ROOT}/splits/train_v3.jsonl")]
rel_of = {}
for l in open(f"{ROOT}/manifest_all.tsv"):
    if l.strip():
        rel = l.split("\t")[0]
        rel_of[rel.split("/")[-1]] = rel
keep = np.array([i for i, r in enumerate(tr) if rel_of.get(r["video"])])
full = np.full(len(tr), np.nan)
full[keep] = score
miss = ~np.isfinite(full)

oof15, _ev, y_tr, *_ = pickle.load(open(f"{ROOT}/upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15 = np.asarray(oof15, float)
y_tr = np.asarray(y_tr, int)
z = np.load(f"{ROOT}/upstream/cache_v3/_full_raw_v2.npz")
X = z["X_tr"].astype(float)
md = np.nanmedian(X, axis=0)
ii = np.where(~np.isfinite(X))
X[ii] = np.take(md, ii[1])
c = json.load(open(f"{ROOT}/data/s3/e18_champion.json"))["params"]

print(f"覆盖 {int((~miss).sum())}/{len(tr)};覆盖内 bad 率 {y_tr[~miss].mean():.4f} "
      f"vs 缺失内 {y_tr[miss].mean():.4f}  ← 若两者差异大,缺失本身就是强特征", flush=True)


def mk():
    return lgb.LGBMClassifier(
        num_leaves=c["leaves"], n_estimators=c["est"], learning_rate=c["lr"],
        min_child_samples=c["mcs"], scale_pos_weight=c["spw"], feature_fraction=c["ff"],
        bagging_fraction=c["bf"], bagging_freq=1, random_state=42, verbose=-1)


def score_of(col, seed=42, sub=None):
    Bm = np.hstack([oof15, col[:, None], X]) if col is not None else np.hstack([oof15, X])
    yy = y_tr
    if sub is not None:
        Bm, yy = Bm[sub], y_tr[sub]
    o = np.zeros(len(yy))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Bm, yy):
        m = mk(); m.fit(Bm[a], yy[a]); o[b] = m.predict_proba(Bm[b])[:, 1]
    return gn(o, yy)


def rankcol(vals, mask_miss):
    r = np.full(len(vals), 0.5)
    ok = ~mask_miss
    r[ok] = rankdata(vals[ok]) / ok.sum()
    return r


real = rankcol(full, miss)
print("\n=== ③ 多种子:真 32B 秩变换列 vs 基准 ===", flush=True)
rs, bs = [], []
for sd in (42, 101, 202, 303, 404):
    a, b = score_of(real, sd), score_of(None, sd)
    rs.append(a); bs.append(b)
    print(f"  seed{sd}: 真列 {a:.4f} | 基准 {b:.4f} | Δ {a-b:+.4f}", flush=True)
print(f"  均值 真列 {np.mean(rs):.4f} 基准 {np.mean(bs):.4f} Δ {np.mean(rs)-np.mean(bs):+.4f}",
      flush=True)

print("\n=== ① 置换检验(覆盖内打乱,缺失模式不变,10 次)===", flush=True)
perm = []
for t in range(10):
    v = full.copy()
    idx = np.where(~miss)[0]
    v[idx] = full[np.random.RandomState(t).permutation(idx)]
    perm.append(score_of(rankcol(v, miss), 42))
print(f"  分布 {np.min(perm):.4f}~{np.max(perm):.4f} 均值 {np.mean(perm):.4f}", flush=True)

print("\n=== ② 纯噪声列(同缺失模式,10 次)===", flush=True)
noise = []
for t in range(10):
    v = np.full(len(tr), np.nan)
    v[keep] = np.random.RandomState(100 + t).rand(len(keep))
    noise.append(score_of(rankcol(v, miss), 42))
print(f"  分布 {np.min(noise):.4f}~{np.max(noise):.4f} 均值 {np.mean(noise):.4f}", flush=True)

print("\n=== ④ 仅覆盖子集(无缺失可利用):真 vs 打乱 ===", flush=True)
sub = ~miss
sc_real = score_of(rankcol(full, miss), 42, sub)
sc_perm = []
for t in range(5):
    v = full.copy()
    idx = np.where(~miss)[0]
    v[idx] = full[np.random.RandomState(t).permutation(idx)]
    sc_perm.append(score_of(rankcol(v, miss), 42, sub))
sc_base = score_of(None, 42, sub)
print(f"  真 {sc_real:.4f} | 打乱 {np.mean(sc_perm):.4f}({np.min(sc_perm):.4f}~{np.max(sc_perm):.4f})"
      f" | 无该列基准 {sc_base:.4f}", flush=True)

ok13 = np.mean(rs) > max(np.max(perm), np.max(noise))
ok4 = sc_real > np.mean(sc_perm)
print(f"\n判决:①②上沿 {max(np.max(perm), np.max(noise)):.4f} vs 真列均值 {np.mean(rs):.4f} "
      f"→ {'通过' if ok13 else '未通过'};④ {'通过' if ok4 else '未通过'}", flush=True)
print(f"总判决:{'✔ 32B 有净贡献,可申请 eval' if (ok13 and ok4) else '✘ 判为缺失捷径伪影,不进 eval'}",
      flush=True)
print("E48_DONE", flush=True)
