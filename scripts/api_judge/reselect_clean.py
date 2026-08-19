#!/usr/bin/env python3
"""无泄漏口径下重新选型:探针塌掉后地形变了,把当初被剪掉/判死的特征块逐个重试。
纪律:选型 seed 42-44,复核 seed 47-49,终读 8 seed;高维块一律折成嵌套 OOF 标量再进融合层。
"""
import csv, json, sys
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
SEL=(42,43,44); CONF=(47,48,49)
def br_at(s,y,rel=0.8):
    gn=np.sort(s[y==0]); b=s[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return ((b>t).sum()+(b==t).sum()*(1-fr))/len(b)
def auc(s,y):
    r=rankdata(s); pos=r[y==1]
    return (pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum())
def rankpct(x): return rankdata(x)/len(x)
def isnum(v):
    try: float(v); return True
    except: return False

vids=[v if v.endswith(".mp4") else v+".mp4" for v in json.load(open(OUT/"X303_vids.json"))]
idx={v:i for i,v in enumerate(vids)}; N=len(vids)
mapr={r["filename"]:r for r in csv.DictReader(open(D/"api_judge_video_image_map.csv",encoding="utf-8-sig"))}
y=np.array([1 if mapr[v]["grade"]=="bad" else 0 for v in vids])
groups=np.array([mapr[v]["source_sha"] for v in vids])
tracks=np.array([mapr[v]["track"] for v in vids]); skus=np.array([v.split("__")[2] for v in vids])
lb=json.load(open(D/"lockbox_split.json")); dev=set(lb["dev"])
dm=np.array([(g in dev) or (v in dev) for g,v in zip(groups,vids)])
yd=y[dm]; gd=groups[dm]

def load_csv(name):
    rows=list(csv.DictReader(open(OUT/name)))
    keys=[k for k in rows[0] if k not in ("filename","label","video") and isnum(rows[0][k])]
    M=np.zeros((N,len(keys)))
    for r in rows:
        f=r.get("filename") or r.get("video")
        if f in idx: M[idx[f]]=[float(r[k] or 0) for k in keys]
    return M

z=np.load(OUT/"r1_oof.npz",allow_pickle=True); r1b=z["r1b"].astype(float); r1c=z["r1c"].astype(float)
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
f1=np.array([s1.get(v,50) for v in vids],float); f2=np.array([s2.get(v,50) for v in vids],float)
flash=(rankpct(f1)+rankpct(f2))/2
fl=np.zeros(N)
for t in set(tracks):
    m=tracks==t; fl[m]=rankpct(flash[m])
cl=np.load(OUT/"imgprobe_clean.npz",allow_pickle=True)
cmap={f:i for i,f in enumerate(cl["fns"].astype(str))}
ipc=np.array([cl["p_gbm"][cmap[v]] for v in vids])
ipc_lr=np.array([cl["p_lr"][cmap[v]] for v in vids])
NRK=["clip_mean","clip_min","clip_last","clip_drop","clip_std","sig_mean","sig_min","sig_first","sig_last","sig_drop","sig_std"]
NRall=load_csv("newref_feats.csv")
nrmap={k:i for i,k in enumerate([k for k in csv.DictReader(open(OUT/"newref_feats.csv")).fieldnames if k not in ("filename","label")])}
NR=np.stack([NRall[:,nrmap[k]] for k in NRK],1)
BLOCKS={
 "subjcons": load_csv("subjcons_1233.csv"),
 "prop": load_csv("prop_timeline.csv"),
 "phys32": load_csv("phys32_1233.csv"),
 "refprobe": load_csv("refprobe_1233.csv"),
 "newref2": load_csv("newref2_1233.csv"),
 "X303": np.load(OUT/"X303_new.npy"),
}
BASE_SCALARS={"x303plus": np.load(OUT/"x303plus_oof.npy"), "x303ens": np.load(OUT/"x303ens_oof.npy")}

CORE=[("r1b",r1b),("r1c",r1c),("flash_avg",flash),("flash_trkpct",fl),("imgprobe",ipc)]+\
     [(f"nr_{k}",NR[:,j]) for j,k in enumerate(NRK)]
def meta_cols(extra_scalars=(), extra_raw=(), drop=()):
    cols=[(n,v) for n,v in CORE if n not in drop]+list(extra_scalars)
    for t in sorted(set(tracks)):
        if t!="base-props" and f"TR_{t}" not in drop: cols.append((f"TR_{t}",(tracks==t).astype(float)))
    for s in sorted(set(skus)): cols.append((f"SKU_{s}",(skus==s).astype(float)))
    names=[n for n,_ in cols]; X=np.stack([v for _,v in cols],1)
    for nm,M in extra_raw:
        X=np.hstack([X,M]); names+= [f"{nm}_{i}" for i in range(M.shape[1])]
    return X,names

def run(X, fold_blocks=(), seeds=SEL, ret_oof=False):
    """fold_blocks: [(name, M, kind)] 在每个训练折内拟合子模型 → 该折测试样本的一列。"""
    Xd=X[dm]; oofs=[]
    for sd in seeds:
        oof=np.zeros(len(yd))
        for tr,te in StratifiedGroupKFold(10,shuffle=True,random_state=sd).split(Xd,yd,gd):
            Xtr,Xte=Xd[tr],Xd[te]
            for nm,M,kind in fold_blocks:
                Md=M[dm]
                if kind=="lgb":
                    sub=lgb.LGBMClassifier(n_estimators=200,num_leaves=15,learning_rate=0.05,
                        min_child_samples=25,colsample_bytree=0.6,subsample=0.8,subsample_freq=1,
                        random_state=sd,verbose=-1,n_jobs=4).fit(Md[tr],yd[tr])
                else:
                    sc0=StandardScaler().fit(Md[tr])
                    sub=LogisticRegression(C=1.0,max_iter=3000).fit(sc0.transform(Md[tr]),yd[tr])
                    class W:
                        def __init__(s,m,sc): s.m,s.sc=m,sc
                        def predict_proba(s,A): return s.m.predict_proba(s.sc.transform(A))
                    sub=W(sub,sc0)
                Xtr=np.hstack([Xtr,sub.predict_proba(Md[tr])[:,1:2]])
                Xte=np.hstack([Xte,sub.predict_proba(Md[te])[:,1:2]])
            sc=StandardScaler().fit(Xtr)
            m=LogisticRegression(C=100,max_iter=4000,class_weight={0:1,1:2}).fit(sc.transform(Xtr),yd[tr])
            oof[te]=m.predict_proba(sc.transform(Xte))[:,1]
        oofs.append(oof)
    oofs=np.array(oofs)
    per=[br_at(o,yd) for o in oofs]
    bag=np.mean([rankpct(o) for o in oofs],0)
    return (np.mean(per), np.std(per), br_at(bag,yd), auc(bag,yd), oofs) if ret_oof else (np.mean(per),np.std(per),br_at(bag,yd),auc(bag,yd))

X0,n0=meta_cols()
print(f"[基线 clean-28] 列数={X0.shape[1]}")
m,s,b,a=run(X0); print(f"  SEL 单seed {m:.4f}±{s:.4f}  装袋 {b:.4f}  AUC {a:.4f}", flush=True)

print("\n[候选块:折内子模型折成一列]")
for nm,M in BLOCKS.items():
    for kind in (("lgb",) if nm=="X303" else ("lr","lgb")):
        m2,s2,b2,a2=run(X0, fold_blocks=[(nm,M,kind)])
        print(f"  +{nm}({kind},{M.shape[1]}维): 单seed {m2:.4f}  装袋 {b2:.4f}  AUC {a2:.4f}   Δ单seed {m2-m:+.4f}", flush=True)

print("\n[候选:现成基座标量 / 探针 LR 列]")
for nm,v in list(BASE_SCALARS.items())+[("imgprobe_lr",ipc_lr)]:
    X1,_=meta_cols(extra_scalars=[(nm,v)])
    m2,s2,b2,a2=run(X1)
    print(f"  +{nm}: 单seed {m2:.4f}  装袋 {b2:.4f}  AUC {a2:.4f}   Δ {m2-m:+.4f}", flush=True)

print("\n[候选:把 base-props 轨也加回来]")
cols=[(n,v) for n,v in CORE]
X1,_=meta_cols(); X1=np.hstack([X1,(tracks=="base-props").astype(float).reshape(-1,1)])
m2,s2,b2,a2=run(X1); print(f"  +TR_base-props: 单seed {m2:.4f}  装袋 {b2:.4f}  Δ {m2-m:+.4f}", flush=True)
