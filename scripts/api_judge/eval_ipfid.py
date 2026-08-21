#!/usr/bin/env python3
"""评估 ipfid 块:单列强度 → 是否补上判官的盲区 → 加进 43 列终配。
盲区检验是重点:判官三遍全打 0 分的 208 条 bad,这个特征能不能把它们排上来。"""
import sys, csv, json, collections
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import _base43 as B
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
rp=B.rp; br_at=B.br_at; auc=B.auc; y=B.y; groups=B.groups; N=B.N; vids=B.vids; idx=B.idx
OUT=Path.home()/"tutu-video-eval/data/pbase/out"
TAG=sys.argv[1] if len(sys.argv)>1 else ""
rows=list(csv.DictReader(open(OUT/f"ipfid{TAG}_1233.csv",encoding="utf-8")))
K=[k for k in rows[0] if k!="filename"]
M=np.full((N,len(K)),np.nan)
for r in rows:
    if r["filename"] in idx:
        M[idx[r["filename"]]]=[float(r[k]) if r[k] not in ("","nan","None") else np.nan for k in K]
cov=(~np.isnan(M[:,0])).mean()
print(f"覆盖 {cov*100:.1f}%  列 {len(K)}")
med=np.nanmedian(M,0); M=np.where(np.isnan(M),med[None,:],M)
# 判官盲区
def raw(f):
    m={}
    for l in open(f,encoding="utf-8"):
        try: o=json.loads(l)
        except: continue
        r=o.get("result") or {}
        if "bad_score" in r: m[o["filename"]]=r["bad_score"]
    return np.array([m.get(v,np.nan) for v in vids],float)
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
S=[np.array([s1.get(v,np.nan) for v in vids],float),
   np.array([s2.get(v,np.nan) for v in vids],float), raw(OUT/"flash_v8_raw.jsonl")]
S=[np.where(np.isnan(x),np.nanmedian(x),x) for x in S]
blind=np.all(np.stack([x==0 for x in S],1),1)&(y==1)
print(f"判官三遍全漏的 bad:{blind.sum()} 条")
print(f"\n{'列':22s} {'br@80':>7s} {'AUC':>7s}  盲区召回(该列前20%里占比)")
best=[]
for j,k in enumerate(K):
    v=M[:,j]
    for sgn in (1,-1):
        s=rp(sgn*v); a=auc(sgn*v,y)
        if a>=0.5:
            top=s>=np.quantile(s,0.80)
            br=br_at(s,y)
            hit=(top&blind).sum()/max(1,blind.sum())
            best.append((br,a,k,sgn,hit))
            break
best.sort(reverse=True)
for br,a,k,sgn,hit in best[:14]:
    print(f"{k:22s} {br:7.4f} {a:7.4f}  {hit*100:5.1f}%   {'(取负)' if sgn<0 else ''}")
print(f"\n(基线参照:图片探针单列 br@80 0.2487 / 判官三票 0.3533 / 随机盲区召回 20%)")
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
def fold1(Bk,mask,nf,seed):
    out=np.zeros(N); Xd=Bk[mask]; yd=y[mask]; gd=groups[mask]; o=np.zeros(len(yd))
    for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=seed).split(Xd,yd,gd):
        inner=np.zeros(len(tr))
        for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(Xd[tr],yd[tr],gd[tr]):
            sc=StandardScaler().fit(Xd[tr][a]); m=LogisticRegression(C=1,max_iter=4000).fit(sc.transform(Xd[tr][a]),yd[tr][a])
            inner[b]=m.predict_proba(sc.transform(Xd[tr][b]))[:,1]
        sc=StandardScaler().fit(Xd[tr]); m=LogisticRegression(C=1,max_iter=4000).fit(sc.transform(Xd[tr]),yd[tr])
        o[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
    out[mask]=o; return out
TOP=[k for _,_,k,_,_ in best[:6]]
CFG=[("基线 43列",()),
     (f"+全部 ipfid({len(K)}列)",tuple(M[:,j] for j in range(len(K)))),
     (f"+ipfid 前6列",tuple(M[:,K.index(k)] for k in TOP))]
print("\n=== 融合层 ===",flush=True)
for nm,ex in CFG:
    X=B.stack(B.COLS+list(ex))
    for tag,mask,nf in (("dev867/10折",B.dm,10),("全1233/20折",B.ALL,20)):
        m,s,b,a=run(X,mask,nf,tuple(range(42,50)))
        m2,_,b2,_=run(X,mask,nf,(47,48,49))
        print(f"  {nm:20s} {tag:12s} 列{X.shape[1]:3d} 8seed 单{m:.4f}±{s:.4f} 装袋{b:.4f} AUC{a:.4f} | CONF 单{m2:.4f} 装袋{b2:.4f}",flush=True)
print("  —— 块折成单列(嵌套) ——",flush=True)
for tag,mask,nf in (("dev867/10折",B.dm,10),("全1233/20折",B.ALL,20)):
    res=[]
    for sd in range(42,50):
        col=fold1(M,mask,nf,sd); X=B.stack(B.COLS+[col])
        Xd=X[mask]; yd=y[mask]; gd=groups[mask]; oof=np.zeros(len(yd))
        for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=sd).split(Xd,yd,gd):
            sc=StandardScaler().fit(Xd[tr]); m=LogisticRegression(C=100,max_iter=8000).fit(sc.transform(Xd[tr]),yd[tr])
            oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
        res.append(oof)
    per=[br_at(o,yd) for o in res]; bag=np.mean([rp(o) for o in res],0)
    print(f"  {'ipfid块折1列':20s} {tag:12s} 8seed 单{np.mean(per):.4f}±{np.std(per):.4f} 装袋{br_at(bag,yd):.4f} AUC{auc(bag,yd):.4f}",flush=True)
