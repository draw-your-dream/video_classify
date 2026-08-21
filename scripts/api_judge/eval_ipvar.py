#!/usr/bin/env python3
"""变体对照评估:哪一处补齐是关键(局部放大 / 白底方形 pad / 自动掩膜候选 / 完整 ip_score),
以及官方视角逐张的有用程度。统一按"替换整帧 max_cos"读,列数不增;16 种子。"""
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
rows=list(csv.DictReader(open(OUT/"ipvar_1233.csv",encoding="utf-8")))
K=[k for k in rows[0] if k!="filename"]
M=np.full((N,len(K)),np.nan)
for r in rows:
    if r["filename"] in idx:
        M[idx[r["filename"]]]=[float(r[k]) if r[k] not in ("","nan","None","NaN") else np.nan for k in K]
print(f"覆盖 {(~np.isnan(M[:,K.index('v_box')])).sum()}/{N}")
for j,k in enumerate(K):
    print(f"  {k:10s} 缺失 {int(np.isnan(M[:,j]).sum()):4d}")
M=np.where(np.isnan(M),np.nanmedian(M,0)[None,:],M)
col=lambda k: M[:,K.index(k)]
ipf=np.full(N,np.nan)
for r in csv.DictReader(open(OUT/"ipfid_1233.csv",encoding="utf-8")):
    if r["filename"] in idx: ipf[idx[r["filename"]]]=float(r["raw_first"])
ipf=np.where(np.isnan(ipf),np.nanmedian(ipf),ipf)
print("\n== 单列强度(全1233,余弦低=坏)==")
cands=[("旧 max_cos(整帧)",B.COLS[NAMES.index("max_cos")]),
       ("v_box 裸框裁剪",col("v_box")),
       ("v_pad 白底方形pad768",col("v_pad")),
       ("v_local 局部放大+SAM",col("v_local")),
       ("v_seg 自动掩膜候选",col("v_seg")),
       ("v_ipscore 完整两路并集",col("v_ipscore")),
       ("(6帧版)raw_first",ipf)]
for nm,v in cands:
    print(f"  {nm:24s} br@80={br_at(rp(-v),y):.4f} AUC={auc(-v,y):.4f}")
print("\n== 官方视角逐张(该框对第 j 张视角的余弦)==")
for j in range(5):
    k=f"view{j}"
    if k in K: print(f"  {k}  br@80={br_at(rp(-col(k)),y):.4f} AUC={auc(-col(k),y):.4f}")
vm=np.stack([col(f"view{j}") for j in range(5) if f"view{j}" in K],1)
print(f"  5视角取最大  br@80={br_at(rp(-vm.max(1)),y):.4f} AUC={auc(-vm.max(1),y):.4f}")
print(f"  5视角取均值  br@80={br_at(rp(-vm.mean(1)),y):.4f} AUC={auc(-vm.mean(1),y):.4f}")
def run(cols,mask,nf,seeds):
    X=B.stack(cols); Xd=X[mask]; yd=y[mask]; gd=groups[mask]; oofs=[]
    for sd in seeds:
        oof=np.zeros(len(yd))
        for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=sd).split(Xd,yd,gd):
            sc=StandardScaler().fit(Xd[tr]); m=LogisticRegression(C=100,max_iter=8000).fit(sc.transform(Xd[tr]),yd[tr])
            oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
        oofs.append(oof)
    per=[br_at(o,yd) for o in oofs]; bag=np.mean([rp(o) for o in oofs],0)
    return np.mean(per),np.std(per),br_at(bag,yd),auc(bag,yd)
NOMC=[c for i,c in enumerate(B.COLS) if NAMES[i]!="max_cos"]
print("\n== 融合层:替换整帧 max_cos(16 种子,列数不增)==",flush=True)
CFG=[("基线(留 max_cos)",B.COLS)]+[(f"→{nm}",NOMC+[v]) for nm,v in cands[1:]]
CFG.append(("→v_local+v_ipscore(净+1)",NOMC+[col("v_local"),col("v_ipscore")]))
CFG.append(("→raw_first+v_local(净+1)",NOMC+[ipf,col("v_local")]))
for nm,cols in CFG:
    for tag,mask,nf in (("dev867/10折",B.dm,10),("全1233/20折",B.ALL,20)):
        m,s,b,a=run(cols,mask,nf,tuple(range(42,58)))
        m2,_,b2,_=run(cols,mask,nf,(47,48,49))
        print(f"  {nm:26s} {tag:12s} 16seed 单{m:.4f}±{s:.4f} 装袋{b:.4f} AUC{a:.4f} | CONF 单{m2:.4f}",flush=True)
