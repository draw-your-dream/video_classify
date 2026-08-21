#!/usr/bin/env python3
"""放行线邻域加权,权重严格在外层训练集内部生成(无泄漏)。
每个外层折:tr 内做 5 折 OOF → 由 tr 的合格样本定阈值 → 高斯核给 tr 加权 → 加权重训 → 预测 te。"""
import sys, numpy as np
sys.path.insert(0,str(__import__("pathlib").Path(__file__).parent))
import _base43 as B
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
rp=B.rp; br_at=B.br_at; auc=B.auc; y=B.y; groups=B.groups
def oof_weighted(X,mask,nf,sd,h,C=100,rel=0.8,floor=0.05):
    Xd=X[mask]; yd=y[mask]; gd=groups[mask]; oof=np.zeros(len(yd))
    for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=sd).split(Xd,yd,gd):
        Xt=Xd[tr]; yt=yd[tr]; gt=gd[tr]
        inner=np.zeros(len(tr))
        for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=sd).split(Xt,yt,gt):
            sc=StandardScaler().fit(Xt[a]); m=LogisticRegression(C=C,max_iter=8000).fit(sc.transform(Xt[a]),yt[a])
            inner[b]=m.predict_proba(sc.transform(Xt[b]))[:,1]
        r=rp(inner); gnr=np.sort(r[yt==0]); k=int(np.floor(rel*len(gnr))); rt=gnr[k-1]
        w=np.exp(-((r-rt)/h)**2)+floor; w=w/w.mean()
        sc=StandardScaler().fit(Xt); m=LogisticRegression(C=C,max_iter=8000).fit(sc.transform(Xt),yt,sample_weight=w)
        oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
    return oof,yd
def base(X,mask,nf,sd,C=100):
    Xd=X[mask]; yd=y[mask]; gd=groups[mask]; oof=np.zeros(len(yd))
    for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=sd).split(Xd,yd,gd):
        sc=StandardScaler().fit(Xd[tr]); m=LogisticRegression(C=C,max_iter=8000).fit(sc.transform(Xd[tr]),yd[tr])
        oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
    return oof,yd
X=B.X43
def show(nm,fn):
    for tag,mask,nf in (("dev867/10折",B.dm,10),("全1233/20折",B.ALL,20)):
        res=[fn(X,mask,nf,sd) for sd in range(42,50)]; yd=res[0][1]
        per=[br_at(o,yd) for o,_ in res]; bag=np.mean([rp(o) for o,_ in res],0)
        b3=br_at(np.mean([rp(o) for o,_ in res[:3]],0),yd); b2=br_at(np.mean([rp(o) for o,_ in res[-3:]],0),yd)
        print(f"  {nm:22s} {tag:12s} 8seed 单{np.mean(per):.4f}±{np.std(per):.4f} 装袋{br_at(bag,yd):.4f} AUC{auc(bag,yd):.4f} | SEL装袋{b3:.4f} CONF装袋{b2:.4f}",flush=True)
show("基线(无加权)",base)
for h in (0.10,0.15,0.20,0.30):
    show(f"邻域加权 h={h}",lambda X,m,nf,sd,h=h: oof_weighted(X,m,nf,sd,h))
