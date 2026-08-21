#!/usr/bin/env python3
"""融合层实验室 2:装袋种子数 / 组内分位归一 / 放行线邻域加权。"""
import sys, numpy as np
sys.path.insert(0,str(__import__("pathlib").Path(__file__).parent))
import _base43 as B
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
rp=B.rp; br_at=B.br_at; auc=B.auc; y=B.y; groups=B.groups; N=B.N; NAMES=B.NAMES
def oof_for(X,mask,nf,sd,C=100,w=None):
    Xd=X[mask]; yd=y[mask]; gd=groups[mask]; oof=np.zeros(len(yd))
    for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=sd).split(Xd,yd,gd):
        sc=StandardScaler().fit(Xd[tr]); m=LogisticRegression(C=C,max_iter=8000)
        ww=None if w is None else w[mask][tr]
        m.fit(sc.transform(Xd[tr]),yd[tr],sample_weight=ww)
        oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
    return oof,yd
def report(nm,X,seeds=tuple(range(42,50)),**kw):
    for tag,mask,nf in (("dev867/10折",B.dm,10),("全1233/20折",B.ALL,20)):
        os_=[oof_for(X,mask,nf,sd,**kw) for sd in seeds]; yd=os_[0][1]
        per=[br_at(o,yd) for o,_ in os_]; bag=np.mean([rp(o) for o,_ in os_],0)
        s3=[br_at(o,yd) for o,_ in os_[:3]]; s2=[br_at(o,yd) for o,_ in os_[-3:]]
        b3=br_at(np.mean([rp(o) for o,_ in os_[:3]],0),yd); b2=br_at(np.mean([rp(o) for o,_ in os_[-3:]],0),yd)
        print(f"  {nm:26s} {tag:12s} {len(seeds)}seed 单{np.mean(per):.4f}±{np.std(per):.4f} 装袋{br_at(bag,yd):.4f} AUC{auc(bag,yd):.4f} | 前3 装袋{b3:.4f} | 后3 装袋{b2:.4f}",flush=True)
X=B.X43
print("=== 4 装袋种子数(基线配置)===",flush=True)
report("8 seed",X)
report("32 seed",X,seeds=tuple(range(42,74)))
print("=== 5 组内分位归一(把 flash trkpct 的做法推广)===",flush=True)
STRONG=["r1b","r1c","imgprobe_lr","max_cos","src_mean","src_drop","al_p10","clip_min"]
def grppct(v,key):
    o=np.zeros(N)
    for g in set(key):
        m=key==g; o[m]=rp(v[m])
    return o
for keyname,key in (("轨内",B.tracks),("款式内",B.skus)):
    extra=[grppct(B.COLS[NAMES.index(k)],key) for k in STRONG if k in NAMES]
    report(f"+{keyname}分位×{len(extra)}",B.stack(B.COLS+extra))
extra=[grppct(B.COLS[NAMES.index(k)],B.tracks) for k in STRONG if k in NAMES]+\
      [grppct(B.COLS[NAMES.index(k)],B.skus) for k in STRONG if k in NAMES]
report("+两种分位×16",B.stack(B.COLS+extra))
print("=== 6 放行线邻域加权(两阶段,权重来自折内 5 折 OOF)===",flush=True)
def weights(X,mask,nf,sd,h):
    Xd=X[mask]; yd=y[mask]; gd=groups[mask]; inner=np.zeros(len(yd))
    for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=sd).split(Xd,yd,gd):
        sc=StandardScaler().fit(Xd[a]); m=LogisticRegression(C=100,max_iter=8000).fit(sc.transform(Xd[a]),yd[a])
        inner[b]=m.predict_proba(sc.transform(Xd[b]))[:,1]
    r=rp(inner); gnr=np.sort(r[yd==0]); k=int(np.floor(0.8*len(gnr))); rt=gnr[k-1]
    w=np.exp(-((r-rt)/h)**2)+0.05
    out=np.zeros(N); out[mask]=w/w.mean(); return out
for h in (0.10,0.20,0.35):
    for tag,mask,nf in (("dev867/10折",B.dm,10),("全1233/20折",B.ALL,20)):
        per=[];bags=[]
        for sd in range(42,50):
            w=weights(X,mask,nf,sd,h)
            o,yd=oof_for(X,mask,nf,sd,w=w); per.append(br_at(o,yd)); bags.append(rp(o))
        bag=np.mean(bags,0)
        print(f"  {'邻域加权 h='+str(h):26s} {tag:12s} 8seed 单{np.mean(per):.4f}±{np.std(per):.4f} 装袋{br_at(bag,yd):.4f} AUC{auc(bag,yd):.4f}",flush=True)
