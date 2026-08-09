"""E44:三档序数监督(2026-08-09 预注册,用户指令:利用 good/normal/bad 档位信息)。

动机:splits 里一直存着三档标注(train bad1621/good1302/normal954),历来被压成二分类
0/1 喂入,丢弃了 954 条 normal 的档位信息。冠军在未见 normal 标签的情况下自发学出
good(0.304) < normal(0.378) < bad(0.555) 的序,说明序真实存在于数据中。
序数监督与 E39(失败的尾部损失)的关键区别:E39 人为定义尾部,本实验用的是标注者
给出的真实边界;二分类中"bad判成good"与"bad判成normal"惩罚等价,序数框架下前者跨两档。

基准:冠军二分类 train-OOF gn@95 = 0.3218(同折 seed42,同参数)。
判准(发车前冻结):train-OOF gn@95 > 0.3218,且需在 E44b 多种子复核中稳定,才谈 eval。

变体:
 A 3类softmax:score = P(bad);A2: P(bad)+0.5*P(normal);A3: 期望档位 0*Pg+1*Pn+2*Pb
 B Frank-Hall 序数分解:M1=P(y>=normal), M2=P(y=bad),score = w*M1+(1-w)*M2,w∈{0.2,0.35,0.5}
 C 不对称样本权重(二分类框架):normal 权重 β∈{0.5,0.75,1.5,2.0},good 固定 1.0
 D 序数回归:LGBMRegressor 拟合 y∈{0,1,2}
 E 两阶段融合:S1=good vs (normal+bad),S2=(good+normal) vs bad,score=平均秩
 F normal 概率作为新特征:3类模型 OOF 的 P(normal) 拼进原特征再训二分类(嵌套无泄漏)
"""
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

# 三档标签:splits 顺序与 pkl 一致(e43 已依此写出 predictions),用二分类一致性断言校验
tr_meta = [json.loads(l) for l in open("splits/train_v3.jsonl")]
TIER = {"good": 0, "normal": 1, "bad": 2}
t3 = np.array([TIER[r["label"]] for r in tr_meta])
assert len(t3) == len(y_tr), f"长度不匹配 {len(t3)} vs {len(y_tr)}"
assert ((t3 == 2).astype(int) == y_tr).all(), "顺序错位:三档标签与 pkl 的 y_tr 不一致"
print(f"三档对齐校验通过 good={int((t3==0).sum())} normal={int((t3==1).sum())} bad={int((t3==2).sum())}",
      flush=True)

champ = json.load(open("data/s3/e18_champion.json"))["params"]
BASE = dict(num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
            min_child_samples=champ["mcs"], feature_fraction=champ["ff"],
            bagging_fraction=champ["bf"], bagging_freq=1, random_state=42, verbose=-1)
SPW = champ["spw"]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))
RES = {}


def report(tag, o):
    s = gn(o, y_tr)
    RES[tag] = s
    flag = "✔ 过基准" if s > 0.3218 else ""
    print(f"[E44 {tag:38s}] train-OOF gn@95 = {s:.4f}  {flag}", flush=True)
    return s


# ---- 基准复现 ----
o = np.zeros(len(y_tr))
for a, b in folds:
    m = lgb.LGBMClassifier(**BASE, scale_pos_weight=SPW)
    m.fit(B[a], y_tr[a])
    o[b] = m.predict_proba(B[b])[:, 1]
report("基准 二分类(复现)", o)

# ---- A 3类 softmax ----
P3 = np.zeros((len(y_tr), 3))
for a, b in folds:
    m = lgb.LGBMClassifier(**BASE, objective="multiclass", num_class=3)
    m.fit(B[a], t3[a])
    P3[b] = m.predict_proba(B[b])
report("A1 3类softmax P(bad)", P3[:, 2])
for lam in (0.25, 0.5, 0.75):
    report(f"A2 3类 P(bad)+{lam}*P(normal)", P3[:, 2] + lam * P3[:, 1])
report("A3 3类 期望档位", P3[:, 1] + 2 * P3[:, 2])

# ---- B Frank-Hall 序数分解 ----
m1o, m2o = np.zeros(len(y_tr)), np.zeros(len(y_tr))
for a, b in folds:
    ma = lgb.LGBMClassifier(**BASE)
    ma.fit(B[a], (t3[a] >= 1).astype(int))
    m1o[b] = ma.predict_proba(B[b])[:, 1]
    mb = lgb.LGBMClassifier(**BASE, scale_pos_weight=SPW)
    mb.fit(B[a], (t3[a] == 2).astype(int))
    m2o[b] = mb.predict_proba(B[b])[:, 1]
report("B0 M1单独 P(y>=normal)", m1o)
for w in (0.2, 0.35, 0.5):
    report(f"B  Frank-Hall w={w}", w * m1o + (1 - w) * m2o)
r1, r2 = rankdata(m1o) / len(m1o), rankdata(m2o) / len(m2o)
report("B4 Frank-Hall 秩平均", 0.5 * r1 + 0.5 * r2)

# ---- C 不对称样本权重 ----
for beta in (0.5, 0.75, 1.5, 2.0):
    o = np.zeros(len(y_tr))
    for a, b in folds:
        w = np.ones(len(a))
        w[t3[a] == 1] = beta
        m = lgb.LGBMClassifier(**BASE, scale_pos_weight=SPW)
        m.fit(B[a], y_tr[a], sample_weight=w)
        o[b] = m.predict_proba(B[b])[:, 1]
    report(f"C  normal权重 β={beta}", o)

# ---- D 序数回归 ----
o = np.zeros(len(y_tr))
for a, b in folds:
    m = lgb.LGBMRegressor(**BASE)
    m.fit(B[a], t3[a].astype(float))
    o[b] = m.predict(B[b])
report("D  序数回归 y∈{0,1,2}", o)

# ---- E 两阶段秩融合 ----
s1, s2 = np.zeros(len(y_tr)), np.zeros(len(y_tr))
for a, b in folds:
    ma = lgb.LGBMClassifier(**BASE)
    ma.fit(B[a], (t3[a] >= 1).astype(int))
    s1[b] = ma.predict_proba(B[b])[:, 1]
    mb = lgb.LGBMClassifier(**BASE, scale_pos_weight=SPW)
    mb.fit(B[a], (t3[a] == 2).astype(int))
    s2[b] = mb.predict_proba(B[b])[:, 1]
report("E  两阶段秩平均", 0.5 * rankdata(s1) / len(s1) + 0.5 * rankdata(s2) / len(s2))

# ---- F P(normal) 作为新特征(嵌套无泄漏)----
pn = np.zeros(len(y_tr))
for a, b in folds:
    inner = np.zeros(len(a))
    for ia, ib in StratifiedKFold(4, shuffle=True, random_state=7).split(B[a], t3[a]):
        mi = lgb.LGBMClassifier(**BASE, objective="multiclass", num_class=3)
        mi.fit(B[a][ia], t3[a][ia])
        inner[ib] = mi.predict_proba(B[a][ib])[:, 1]
    mo = lgb.LGBMClassifier(**BASE, objective="multiclass", num_class=3)
    mo.fit(B[a], t3[a])
    pn[b] = mo.predict_proba(B[b])[:, 1]
    # 折内:用 inner OOF 作为训练侧的该列
    Ba = np.hstack([B[a], inner[:, None]])
    Bb = np.hstack([B[b], pn[b][:, None]])
    mf = lgb.LGBMClassifier(**BASE, scale_pos_weight=SPW)
    mf.fit(Ba, y_tr[a])
    o[b] = mf.predict_proba(Bb)[:, 1]
report("F  +P(normal)列 二分类", o)

print("\n=== E44 汇总(降序)===", flush=True)
for k, v in sorted(RES.items(), key=lambda x: -x[1]):
    print(f"  {v:.4f}  {k}", flush=True)
best = max(RES.items(), key=lambda x: x[1])
print(f"\n最佳 {best[0]} = {best[1]:.4f}  基准 0.3218  Δ={best[1]-0.3218:+.4f}", flush=True)
json.dump(RES, open("data/s3/e44_ordinal_results.json", "w"), ensure_ascii=False, indent=1)
print("E44_DONE", flush=True)
