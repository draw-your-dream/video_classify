#!/usr/bin/env python3
"""正确的块折叠:训练行用内层 5 折的样本外预测,测试行用训练部分全量拟合的预测。
(此前用样本内预测,融合层过度信任该列,系统性低估所有块 —— 该批结论作废,以本脚本为准。)"""
import csv, json, sys
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
def br_at(s,y,rel=0.8):
    gn=np.sort(s[y==0]); b=s[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return ((b>t).sum()+(b==t).sum()*(1-fr))/len(b)
def auc(s,y):
    r=rankdata(s); pos=r[y==1]; return (pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum())
def rp(x): return rankdata(x)/len(x)
def isnum(v):
    try: float(v); return True
    except: return False
vids=[v if v.endswith(".mp4") else v+".mp4" for v in json.load(open(OUT/"X303_vids.json"))]
idx={v:i for i,v in enumerate(vids)}; N=len(vids)
mapr={r["filename"]:r for r in csv.DictReader(open(D/"api_judge_video_image_map.csv",encoding="utf-8-sig"))}
y=np.array([1 if mapr[v]["grade"]=="bad" else 0 for v in vids]); groups=np.array([mapr[v]["source_sha"] for v in vids])
tracks=np.array([mapr[v]["track"] for v in vids]); skus=np.array([v.split("__")[2] for v in vids])
lb=json.load(open(D/"lockbox_split.json")); dev=set(lb["dev"])
dm=np.array([(g in dev) or (v in dev) for g,v in zip(groups,vids)]); yd=y[dm]; gd=groups[dm]
def load_csv(name):
    rows=list(csv.DictReader(open(OUT/name)))
    keys=[k for k in rows[0] if k not in ("filename","label","video") and isnum(rows[0][k])]
    M=np.zeros((N,len(keys)))
    for r in rows:
        f=r.get("filename") or r.get("video")
        if f in idx: M[idx[f]]=[float(r[k] or 0) for k in keys]
    return M
z=np.load(OUT/"r1_oof.npz",allow_pickle=True)
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
f1=np.array([s1.get(v,50) for v in vids],float); f2=np.array([s2.get(v,50) for v in vids],float)
flash=(rp(f1)+rp(f2))/2; fl=np.zeros(N)
for t in set(tracks):
    m=tracks==t; fl[m]=rp(flash[m])
cl=np.load(OUT/"imgprobe_clean.npz",allow_pickle=True); cmap={f:i for i,f in enumerate(cl["fns"].astype(str))}
ipc=np.array([cl["p_gbm"][cmap[v]] for v in vids])
NRK=["clip_mean","clip_min","clip_last","clip_drop","clip_std","sig_mean","sig_min","sig_first","sig_last","sig_drop","sig_std"]
NR=np.zeros((N,len(NRK)))
for r in csv.DictReader(open(OUT/"newref_feats.csv")):
    if r["filename"] in idx: NR[idx[r["filename"]]]=[float(r[k] or 0) for k in NRK]
mc=np.zeros(N)
for r in csv.DictReader(open(OUT/"refprobe_1233.csv")):
    if r["filename"] in idx: mc[idx[r["filename"]]]=float(r["max_cos"])
fr=np.load(OUT/"frame_emb.npz",allow_pickle=True); fmap={f:i for i,f in enumerate(fr["fns"].astype(str))}
FE=np.stack([fr["E"][fmap[v]] for v in vids])
BLOCKS={"X303":(np.load(OUT/"X303_new.npy"),"lgb"), "frameemb":(FE,"pca"),
        "subjcons":(load_csv("subjcons_1233.csv"),"lr"), "prop":(load_csv("prop_timeline.csv"),"lr"),
        "newref2":(load_csv("newref2_1233.csv"),"lr"), "phys32":(load_csv("phys32_1233.csv"),"lr"),
        "newref_full":(load_csv("newref_feats.csv"),"lr")}
def core():
    cols=[z["r1b"].astype(float),z["r1c"].astype(float),flash,fl,ipc,mc]+[NR[:,j] for j in range(len(NRK))]
    for t in sorted(set(tracks)):
        if t!="base-props": cols.append((tracks==t).astype(float))
    for s in sorted(set(skus)): cols.append((skus==s).astype(float))
    return np.stack(cols,1)
def submodel(kind, Xa, ya, Xb, seed):
    if kind=="lr":
        sc=StandardScaler().fit(Xa); m=LogisticRegression(C=1.0,max_iter=3000).fit(sc.transform(Xa),ya)
        return m.predict_proba(sc.transform(Xb))[:,1]
    if kind=="pca":
        pc=PCA(64,random_state=0).fit(Xa); m=LogisticRegression(C=1.0,max_iter=3000).fit(pc.transform(Xa),ya)
        return m.predict_proba(pc.transform(Xb))[:,1]
    m=lgb.LGBMClassifier(n_estimators=250,num_leaves=15,learning_rate=0.05,min_child_samples=25,
        colsample_bytree=0.3,subsample=0.8,subsample_freq=1,random_state=seed,verbose=-1,n_jobs=4).fit(Xa,ya)
    return m.predict_proba(Xb)[:,1]
def run(X, blocks=(), seeds=(42,43,44)):
    Xd=X[dm]; oofs=[]
    for sd in seeds:
        oof=np.zeros(len(yd))
        for tr,te in StratifiedGroupKFold(10,shuffle=True,random_state=sd).split(Xd,yd,gd):
            Xtr,Xte=Xd[tr],Xd[te]
            for nm in blocks:
                M,kind=BLOCKS[nm]; Md=M[dm]
                inner=np.zeros(len(tr))
                for itr,ite in StratifiedGroupKFold(5,shuffle=True,random_state=sd).split(Md[tr],yd[tr],gd[tr]):
                    inner[ite]=submodel(kind,Md[tr][itr],yd[tr][itr],Md[tr][ite],sd)
                Xtr=np.hstack([Xtr,inner.reshape(-1,1)])
                Xte=np.hstack([Xte,submodel(kind,Md[tr],yd[tr],Md[te],sd).reshape(-1,1)])
            sc=StandardScaler().fit(Xtr)
            m=LogisticRegression(C=100,max_iter=4000,class_weight={0:1,1:2}).fit(sc.transform(Xtr),yd[tr])
            oof[te]=m.predict_proba(sc.transform(Xte))[:,1]
        oofs.append(oof)
    per=[br_at(o,yd) for o in oofs]; bag=np.mean([rp(o) for o in oofs],0)
    return np.mean(per),np.std(per),br_at(bag,yd),auc(bag,yd)
X=core()
m0,s0,b0,a0=run(X); print(f"基线(29列) SEL 单seed {m0:.4f}±{s0:.4f} 装袋 {b0:.4f} AUC {a0:.4f}",flush=True)
for nm in (sys.argv[1:] or list(BLOCKS)):
    m,s,b,a=run(X,(nm,)); print(f"  +{nm:12s} SEL 单seed {m:.4f}±{s:.4f} 装袋 {b:.4f} AUC {a:.4f}  Δ单seed {m-m0:+.4f}",flush=True)
