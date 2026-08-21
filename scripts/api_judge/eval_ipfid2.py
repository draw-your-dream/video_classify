#!/usr/bin/env python3
"""ipfid 正规评估:列子集按机制预先指定(不看成绩挑),并试"替换整帧 max_cos"而非追加。
机制来源(对方 repo):fid=掩膜裁剪对官方图余弦(还原度主分)、gap=定位健康度、
self=裁切后自参照漂移、capar/capcb=伞盖两条判据(变圆球 / 缩到与身体同宽)。"""
import sys, csv
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import _base43 as B
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
rp=B.rp; br_at=B.br_at; auc=B.auc; y=B.y; groups=B.groups; N=B.N; idx=B.idx; NAMES=B.NAMES
OUT=Path.home()/"tutu-video-eval/data/pbase/out"
rows=list(csv.DictReader(open(OUT/"ipfid_1233.csv",encoding="utf-8")))
K=[k for k in rows[0] if k!="filename"]
M=np.full((N,len(K)),np.nan)
for r in rows:
    if r["filename"] in idx:
        M[idx[r["filename"]]]=[float(r[k]) if r[k] not in ("","nan","None") else np.nan for k in K]
M=np.where(np.isnan(M),np.nanmedian(M,0)[None,:],M)
col=lambda k: M[:,K.index(k)]
def run(X,mask,nf,seeds):
    Xd=X[mask]; yd=y[mask]; gd=groups[mask]; oofs=[]
    for sd in seeds:
        oof=np.zeros(len(yd))
        for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=sd).split(Xd,yd,gd):
            sc=StandardScaler().fit(Xd[tr]); m=LogisticRegression(C=100,max_iter=8000).fit(sc.transform(Xd[tr]),yd[tr])
            oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
        oofs.append(oof)
    per=[br_at(o,yd) for o in oofs]; bag=np.mean([rp(o) for o in oofs],0)
    return np.mean(per),np.std(per),br_at(bag,yd),auc(bag,yd)
def report(nm,cols):
    X=B.stack(cols)
    for tag,mask,nf in (("dev867/10折",B.dm,10),("全1233/20折",B.ALL,20)):
        m,s,b,a=run(X,mask,nf,tuple(range(42,50)))
        m1,_,b1,_=run(X,mask,nf,(42,43,44)); m2,_,b2,_=run(X,mask,nf,(47,48,49))
        print(f"  {nm:24s} {tag:12s} 列{X.shape[1]:3d} 8seed 单{m:.4f}±{s:.4f} 装袋{b:.4f} AUC{a:.4f} | SEL 单{m1:.4f} | CONF 单{m2:.4f} 装袋{b2:.4f}",flush=True)
BASE=list(B.COLS)
NOMC=[c for i,c in enumerate(B.COLS) if NAMES[i]!="max_cos"]
S4=["fid_min","fid_drop","raw_first","self_min"]
S6=S4+["capar_min","capcb_min"]
S3=["fid_min","raw_first","self_min"]
print("=== 机制预设子集(未看成绩挑选)===",flush=True)
report("基线 43列",BASE)
report("+机制3列",BASE+[col(k) for k in S3])
report("+机制4列",BASE+[col(k) for k in S4])
report("+机制6列",BASE+[col(k) for k in S6])
print("=== 替换整帧 max_cos(列数不增)===",flush=True)
report("max_cos→fid_min",NOMC+[col("fid_min")])
report("max_cos→raw_first",NOMC+[col("raw_first")])
report("max_cos→机制3列(净+2)",NOMC+[col(k) for k in S3])
