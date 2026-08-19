#!/usr/bin/env python3
"""旧 0430 语料基线复现:LGBM[15专家OOF ⊕ X320],train 3877 → eval 968 单发。
同时报历史口径 ev@95 与新口径 br@80,便于与 2.0 数据集对照。"""
import json, pickle
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
import lightgbm as lgb
R=Path.home()/"tutu-video-eval"
z=np.load(R/"upstream/cache_v3/_full_raw_v2.npz",allow_pickle=True)
Xtr,Xev,ytr,yev=z["X_tr"],z["X_ev"],z["y_tr"],z["y_ev"]
o_tr,o_ev,y2tr,y2ev,_=pickle.load(open(R/"upstream/cache_v3/_stack_15expert.pkl","rb"))
assert (ytr==y2tr).all() and (yev==y2ev).all()
PC=json.load(open(R/"data/s3/e18_champion.json"))["params"]
def ev_at(p,y,rec=0.95):
    """历史口径:95% bad 召回阈值下,good+normal 的放行率"""
    b=np.sort(p[y==1]); T=b[int(np.floor((1-rec)*len(b)))]
    gn=p[y==0]; return float((gn<T).mean())
def br_at(p,y,rel=0.8):
    gn=np.sort(p[y==0]); bb=p[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return float(((bb>t).sum()+(bb==t).sum()*(1-fr))/len(bb))
def auc(p,y):
    r=rankdata(p); pos=r[y==1]; return float((pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum()))
print(f"train {Xtr.shape} bad {ytr.mean():.3f} | eval {Xev.shape} bad {yev.mean():.3f}")
def fit_eval(A,B,tag,seeds=(42,43,44,45,46)):
    ps=[]
    for sd in seeds:
        p=dict(PC); p.update(random_state=sd,verbose=-1,n_jobs=8)
        m=lgb.LGBMClassifier(**p).fit(A,ytr)
        ps.append(rankdata(m.predict_proba(B)[:,1]))
    p=np.mean(ps,0)/len(yev)
    print(f"  {tag:22s} ev@95={ev_at(p,yev):.4f} ev@100={ev_at(p,yev,1.0):.4f} br@80={br_at(p,yev):.4f} AUC={auc(p,yev):.4f}",flush=True)
    return p
fit_eval(o_tr,o_ev,"仅15专家OOF")
fit_eval(Xtr,Xev,"仅X320")
p0=fit_eval(np.hstack([o_tr,Xtr]),np.hstack([o_ev,Xev]),"OOF15⊕X320(基线)")
np.save(R/"data/pbase/out/corpus_base_eval_p.npy",p0)
