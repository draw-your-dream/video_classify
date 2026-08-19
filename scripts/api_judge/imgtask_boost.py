#!/usr/bin/env python3
"""图片质检任务本身能不能做得更好(用现有嵌入 + 现有元信息)。
口径:按源图 sha 分组 5 折 CV,报 AUC。所有条件化统计量只在训练折内计算。"""
import csv
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
def auc(s,y):
    r=rankdata(s); pos=r[y==1]; return (pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum())
z=np.load(OUT/"qcimg_emb.npz",allow_pickle=True); E=z["E"]; y=z["y"]; g=z["groups"].astype(str)
rows=[]
for f in ("tutu_image_annotations_2962.csv","tutu_image_annotations_0813.csv"):
    for r in csv.DictReader(open(D/f,encoding="utf-8-sig")): rows.append((r["dataset"],r["sample_id"],r["label"]))
# 对齐:npz 是按 CSV 顺序过滤缺失文件后的结果
tok=[]; ds=[]; p=0
for d,s,l in rows:
    if p>=len(g): break
    if s.split("__")[0]==g[p] and (1 if l=="bad" else 0)==y[p]:
        parts=s.split("__"); tok.append(parts[1] if len(parts)>1 else ""); ds.append(d); p+=1
tok=np.array(tok); ds=np.array(ds)
print(f"对齐 {p}/{len(g)};token 种类 {len(set(tok))};批次 {len(set(ds))}")
assert p==len(g)
def cv_auc(feat_fn, head="both", seeds=(42,)):
    outs=[]
    for sd in seeds:
        oof=np.zeros(len(y))
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=sd).split(E,y,g):
            A,B=feat_fn(tr,te)
            ps=[]
            if head in ("lr","both"):
                m=LogisticRegression(C=1.0,max_iter=3000).fit(A,y[tr]); ps.append(rankdata(m.predict_proba(B)[:,1]))
            if head in ("gb","both"):
                m=lgb.LGBMClassifier(n_estimators=300,num_leaves=31,learning_rate=0.05,min_child_samples=20,
                    random_state=42,verbose=-1,n_jobs=8).fit(A,y[tr]); ps.append(rankdata(m.predict_proba(B)[:,1]))
            if head=="mlp":
                m=MLPClassifier((256,),alpha=1e-3,max_iter=400,random_state=sd).fit(A,y[tr]); ps.append(rankdata(m.predict_proba(B)[:,1]))
            oof[te]=sum(ps)/len(ps)/len(te)
        outs.append(auc(oof,y))
    return float(np.mean(outs))
def f_raw(tr,te): return E[tr],E[te]
def centered(tr,te,keys):
    cen={}; gm=E[tr].mean(0)
    for k in set(keys[tr]):
        m=keys[tr]==k
        if m.sum()>=5: cen[k]=E[tr][m].mean(0)
    C=lambda I: np.stack([E[i]-cen.get(keys[i],gm) for i in I])
    return C(tr),C(te)
def f_cen_tok(tr,te): return centered(tr,te,tok)
def f_both_tok(tr,te):
    a,b=centered(tr,te,tok); return np.hstack([E[tr],a]),np.hstack([E[te],b])
def f_proto(tr,te):
    """每 token 的 good/bad 原型余弦 + 全局原型余弦(仅训练折)"""
    def build(mask):
        d={}
        for k in set(tok[tr]):
            m=(tok[tr]==k)&mask
            if m.sum()>=3:
                v=E[tr][m].mean(0); d[k]=v/np.linalg.norm(v)
        v=E[tr][mask].mean(0); d[None]=v/np.linalg.norm(v)
        return d
    G=build(y[tr]==0); Bd=build(y[tr]==1)
    def feat(I):
        out=[]
        for i in I:
            k=tok[i]; gg=G.get(k,G[None]); bb=Bd.get(k,Bd[None])
            out.append([E[i]@gg, E[i]@bb, E[i]@bb-E[i]@gg, E[i]@G[None], E[i]@Bd[None]])
        return np.array(out)
    return np.hstack([E[tr],feat(tr)]),np.hstack([E[te],feat(te)])
def f_knn(tr,te):
    """同 token 内对训练集 good/bad 的 k 近邻平均相似度(仅训练折;训练行排除自身)"""
    K=10
    def feat(I,excl):
        S=E[I]@E[tr].T
        if excl:
            for a,i in enumerate(I):
                w=np.where(np.array(tr)==i)[0]
                if len(w): S[a,w[0]]=-9
        gm=(y[tr]==0); bm=(y[tr]==1); out=[]
        for a in range(len(I)):
            sg=np.sort(S[a][gm])[-K:]; sb=np.sort(S[a][bm])[-K:]
            out.append([sg.mean(),sb.mean(),sb.mean()-sg.mean(),sg.max(),sb.max()])
        return np.array(out)
    return np.hstack([E[tr],feat(tr,True)]),np.hstack([E[te],feat(te,False)])
tests=[("基线 emb",f_raw,"both"),("emb(LR)",f_raw,"lr"),("emb(GBM)",f_raw,"gb"),("emb(MLP)",f_raw,"mlp"),
       ("按token去中心",f_cen_tok,"both"),("emb⊕去中心",f_both_tok,"both"),
       ("emb⊕good/bad原型",f_proto,"both"),("emb⊕kNN相似",f_knn,"both")]
for nm,fn,h in tests:
    print(f"  {nm:20s} AUC={cv_auc(fn,h):.4f}", flush=True)
