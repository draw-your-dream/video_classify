#!/usr/bin/env python3
"""newref3(32帧×官方视角曲线+跨款判别余量+首帧漂移)进融合层的评估。
口径:无泄漏探针(LR头)+ max_cos + 旧栈;选型 42-44 / 复核 47-49 / 终读 8seed。"""
import csv, json, sys
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
TAG=sys.argv[1] if len(sys.argv)>1 else "b"
def br_at(s,y,rel=0.8):
    gn=np.sort(s[y==0]); b=s[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return ((b>t).sum()+(b==t).sum()*(1-fr))/len(b)
def auc(s,y):
    r=rankdata(s); pos=r[y==1]; return (pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum())
def rp(x): return rankdata(x)/len(x)
vids=[v if v.endswith(".mp4") else v+".mp4" for v in json.load(open(OUT/"X303_vids.json"))]
idx={v:i for i,v in enumerate(vids)}; N=len(vids)
mapr={r["filename"]:r for r in csv.DictReader(open(D/"api_judge_video_image_map.csv",encoding="utf-8-sig"))}
y=np.array([1 if mapr[v]["grade"]=="bad" else 0 for v in vids]); groups=np.array([mapr[v]["source_sha"] for v in vids])
tracks=np.array([mapr[v]["track"] for v in vids]); skus=np.array([v.split("__")[2] for v in vids])
lb=json.load(open(D/"lockbox_split.json")); dev=set(lb["dev"])
dm=np.array([(g in dev) or (v in dev) for g,v in zip(groups,vids)]); yd=y[dm]; gd=groups[dm]
z=np.load(OUT/"r1_oof.npz",allow_pickle=True)
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
f1=np.array([s1.get(v,50) for v in vids],float); f2=np.array([s2.get(v,50) for v in vids],float)
flash=(rp(f1)+rp(f2))/2; fl=np.zeros(N)
for t in set(tracks):
    m=tracks==t; fl[m]=rp(flash[m])
cl=np.load(OUT/"imgprobe_clean.npz",allow_pickle=True); cmap={f:i for i,f in enumerate(cl["fns"].astype(str))}
ip=np.array([cl["p_lr"][cmap[v]] for v in vids])
NRK=["clip_mean","clip_min","clip_last","clip_drop","clip_std","sig_mean","sig_min","sig_first","sig_last","sig_drop","sig_std"]
NR=np.zeros((N,len(NRK)))
for r in csv.DictReader(open(OUT/"newref_feats.csv")):
    if r["filename"] in idx: NR[idx[r["filename"]]]=[float(r[k] or 0) for k in NRK]
mc=np.zeros(N)
for r in csv.DictReader(open(OUT/"refprobe_1233.csv")):
    if r["filename"] in idx: mc[idx[r["filename"]]]=float(r["max_cos"])
rows=list(csv.DictReader(open(OUT/f"newref3{TAG}_1233.csv")))
K3=[k for k in rows[0] if k!="filename"]
M3=np.zeros((N,len(K3)))
for r in rows:
    if r["filename"] in idx:
        M3[idx[r["filename"]]]=[float(r[k]) if r[k] not in ("","nan") else 0.0 for k in K3]
print(f"newref3 {len(K3)} 列: {', '.join(K3[:8])} ...")
GRP={"marg":[i for i,k in enumerate(K3) if k.startswith("marg")]+[K3.index("oth_mean")] if "oth_mean" in K3 else [],
     "own":[i for i,k in enumerate(K3) if k.startswith("own")],
     "src":[i for i,k in enumerate(K3) if k.startswith("src") or k.startswith("adj")]}
def build(extra=(), keep_old=True):
    cols=[z["r1b"].astype(float),z["r1c"].astype(float),flash,fl,ip,mc]
    if keep_old: cols+= [NR[:,j] for j in range(len(NRK))]
    cols+=list(extra)
    for t in sorted(set(tracks)):
        if t!="base-props": cols.append((tracks==t).astype(float))
    for s in sorted(set(skus)): cols.append((skus==s).astype(float))
    return np.stack(cols,1)
def run(X,seeds):
    Xd=X[dm]; oofs=[]
    for sd in seeds:
        oof=np.zeros(len(yd))
        for tr,te in StratifiedGroupKFold(10,shuffle=True,random_state=sd).split(Xd,yd,gd):
            sc=StandardScaler().fit(Xd[tr])
            m=LogisticRegression(C=100,max_iter=6000).fit(sc.transform(Xd[tr]),yd[tr])
            oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
        oofs.append(oof)
    per=[br_at(o,yd) for o in oofs]; bag=np.mean([rp(o) for o in oofs],0)
    return np.mean(per),br_at(bag,yd),auc(bag,yd),bag
def show(nm,X):
    o=[]
    for tag,sd in (("SEL",(42,43,44)),("CONF",(47,48,49)),("8seed",tuple(range(42,50)))):
        m,b,a,bag=run(X,sd); o.append(f"{tag} 单{m:.4f} 装袋{b:.4f}")
        if tag=="8seed": last=(m,b,a,bag)
    print(f"  {nm:24s} "+" | ".join(o), flush=True); return last
base=show("基线(29列,LR探针)",build())
for gname,ids in GRP.items():
    if ids: show(f"+{gname}({len(ids)}列)",build([M3[:,i] for i in ids]))
show(f"+全部 newref3({len(K3)}列)",build([M3[:,i] for i in range(len(K3))]))
show(f"newref3 取代旧 newref",build([M3[:,i] for i in range(len(K3))],keep_old=False))
