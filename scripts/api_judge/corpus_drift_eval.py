#!/usr/bin/env python3
"""旧语料 eval968:在 [p_base ⊕ 15专家OOF ⊕ cotracker ⊕ tail_pink] 之上加 DINOv2 漂移特征,
按 2.0 上验证过的两个子集(src9 / src9+patch4)以及全 57 列测增量。选型 42-49,复核 50-54。"""
import csv, json, pickle
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
R=Path.home()/"tutu-video-eval"; S3=R/"data/s3"; OUT=R/"data/pbase/out"
def br_at(p,y,rel=0.8):
    gn=np.sort(p[y==0]); b=p[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return float(((b>t).sum()+(b==t).sum()*(1-fr))/len(b))
def ev_at(p,y,rec=0.95):
    b=np.sort(p[y==1]); T=b[int(np.floor((1-rec)*len(b)))]
    return float((p[y==0]<T).mean())
def auc(p,y):
    r=rankdata(p); pos=r[y==1]; return float((pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum()))
def rp(x):
    x=np.asarray(x,float)
    if np.isfinite(x).any(): x=np.nan_to_num(x,nan=np.nanmedian(x[np.isfinite(x)]),posinf=0.0,neginf=0.0)
    else: x=np.zeros_like(x)
    return rankdata(x)/len(x)
ev=[json.loads(l) for l in open(R/"splits/eval_v3.jsonl")]
def relkey(a):
    p=a.split("/data/s3/")[-1]
    return p.replace("v-0430-skus/","").replace("v-0430-ti2i2v/","ti2i2v/").replace("rlhf-0430/","rlhf/")
rels=[relkey(e["abs_path"]) for e in ev]; vids=[e["video"] for e in ev]
y=np.array([1 if e["label"]=="bad" else 0 for e in ev]); N=len(y)
o_tr,o_ev,ytr,yev,_=pickle.load(open(R/"upstream/cache_v3/_stack_15expert.pkl","rb")); assert (yev==y).all()
pb=np.load(OUT/"corpus_base_eval_p.npy")
def load(fn,sep=","):
    rows=list(csv.DictReader(open(S3/fn),delimiter=sep)); key=list(rows[0])[0]
    cols=[k for k in rows[0] if k!=key and not any(x in (k or "") for x in ("frames","list","ids"))]
    idx={r[key]:r for r in rows}; M=np.zeros((N,len(cols)))
    for i,rl in enumerate(rels):
        r=idx.get(rl)
        if r:
            v=[]
            for c in cols:
                try: v.append(float(r[c]))
                except Exception: v.append(0.0)
            M[i]=v
    return M
ct=load("w2_cotracker.csv"); tp=load("tail_pink.csv")
rows=list(csv.DictReader(open(OUT/"corpus_drift_968.csv")))
K=[k for k in rows[0] if k not in ("video","label")]
idx={r["video"]:r for r in rows}; DR=np.zeros((N,len(K))); hit=0
for i,v in enumerate(vids):
    r=idx.get(v)
    if r:
        hit+=1
        DR[i]=[float(r[k]) if r[k] not in ("","nan") else 0.0 for k in K]
print(f"漂移特征 {len(K)} 列,覆盖 {hit}/{N}")
SRC9=['adj_min','src_drop','src_end_drop','src_first','src_lowfrac','src_max','src_mean','src_min','src_slope']
P4=['al_slope','al_p10','al_last','al10_min']
def sub(names): return [DR[:,K.index(n)] for n in names if n in K]
base=[rp(pb)]+[rp(o_ev[:,j]) for j in range(o_ev.shape[1])]+[rp(ct[:,j]) for j in range(ct.shape[1])]+[rp(tp[:,j]) for j in range(tp.shape[1])]
def run(extra,seeds):
    X=np.stack(base+[rp(c) for c in extra],1); oofs=[]
    for sd in seeds:
        oof=np.zeros(N)
        for tr,te in StratifiedKFold(10,shuffle=True,random_state=sd).split(X,y):
            sc=StandardScaler().fit(X[tr])
            m=LogisticRegression(C=100,max_iter=6000).fit(sc.transform(X[tr]),y[tr])
            oof[te]=m.predict_proba(sc.transform(X[te]))[:,1]
        oofs.append(oof)
    per=[br_at(o,y) for o in oofs]; bag=np.mean([rp(o) for o in oofs],0)
    return np.mean(per),np.std(per),br_at(bag,y),ev_at(bag,y),auc(bag,y)
S8=tuple(range(42,50)); CONF=(50,51,52,53,54)
for nm,ex in (("基线(含cotracker+tail_pink)",[]),("+src9(2.0验证过)",sub(SRC9)),
              ("+src9+patch4(2.0终配)",sub(SRC9)+sub(P4)),("+全部漂移57列",[DR[:,j] for j in range(len(K))])):
    m,s,b,e,a=run(ex,S8); m2,s2,b2,e2,a2=run(ex,CONF)
    print(f"  {nm:26s} 8种子 单{m:.4f}±{s:.4f} 装袋{b:.4f} ev@95={e:.4f} AUC={a:.4f} | 复核 单{m2:.4f} 装袋{b2:.4f}",flush=True)
