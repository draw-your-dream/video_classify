#!/usr/bin/env python3
"""flash v5 问法评估:总分单独 / 五维度分 / 风格字段 / 与现有两遍的组合。
终配基线 = 42 列(r1b,r1c,flash两遍,imgprobe_lr,max_cos,newref11,src9,patch4,TR4,SKU8)。"""
import csv, json, collections
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
def br_at(s,y,rel=0.8):
    gn=np.sort(s[y==0]); b=s[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return float(((b>t).sum()+(b==t).sum()*(1-fr))/len(b))
def auc(s,y):
    r=rankdata(s); pos=r[y==1]; return float((pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum()))
def rp(x): return rankdata(x)/len(x)
vids=[v if v.endswith(".mp4") else v+".mp4" for v in json.load(open(OUT/"X303_vids.json"))]
idx={v:i for i,v in enumerate(vids)}; N=len(vids)
mapr={r["filename"]:r for r in csv.DictReader(open(D/"api_judge_video_image_map.csv",encoding="utf-8-sig"))}
y=np.array([1 if mapr[v]["grade"]=="bad" else 0 for v in vids]); groups=np.array([mapr[v]["source_sha"] for v in vids])
tracks=np.array([mapr[v]["track"] for v in vids]); skus=np.array([v.split("__")[2] for v in vids])
lb=json.load(open(D/"lockbox_split.json")); dev=set(lb["dev"])
dm=np.array([(g in dev) or (v in dev) for g,v in zip(groups,vids)])
V5={}
for l in open(OUT/"flash_v6_raw.jsonl"):
    o=json.loads(l); r=o.get("result") or {}
    if "bad_score" in r: V5[o["filename"]]=r
print(f"v5 有效 {len(V5)}/{N}")
b5=np.array([V5.get(v,{}).get("bad_score",np.nan) for v in vids],float)
med=np.nanmedian(b5); b5=np.where(np.isnan(b5),med,b5)
grade5=np.array([V5.get(v,{}).get("grade","") for v in vids])
print("  判档分布",dict(collections.Counter(grade5)),"(真实 good164/normal234/bad835)")
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
f1=np.array([s1.get(v,50) for v in vids],float); f2=np.array([s2.get(v,50) for v in vids],float)
A2=(rp(f1)+rp(f2))/2
print(f"  秩相关 v5-第1遍 {spearmanr(b5,f1).statistic:.3f} v5-第2遍 {spearmanr(b5,f2).statistic:.3f} (第1-2遍 0.712)")
print(f"  v5 总分单独 dev br@80={br_at(rp(b5)[dm],y[dm]):.4f} AUC={auc(b5[dm],y[dm]):.4f}")
print(f"  两遍均值单独 dev br@80={br_at(A2[dm],y[dm]):.4f} AUC={auc(A2[dm],y[dm]):.4f}")

z=np.load(OUT/"r1_oof.npz",allow_pickle=True)
cl=np.load(OUT/"imgprobe_clean.npz",allow_pickle=True); cmap={f:i for i,f in enumerate(cl["fns"].astype(str))}
ip=np.array([cl["p_lr"][cmap[v]] for v in vids])
NRK=["clip_mean","clip_min","clip_last","clip_drop","clip_std","sig_mean","sig_min","sig_first","sig_last","sig_drop","sig_std"]
NR=np.zeros((N,len(NRK)))
for r in csv.DictReader(open(OUT/"newref_feats.csv")):
    if r["filename"] in idx: NR[idx[r["filename"]]]=[float(r[k] or 0) for k in NRK]
mc=np.zeros(N)
for r in csv.DictReader(open(OUT/"refprobe_1233.csv")):
    if r["filename"] in idx: mc[idx[r["filename"]]]=float(r["max_cos"])
def load3(tag):
    rows=list(csv.DictReader(open(OUT/f"newref{tag}_1233.csv"))); K=[k for k in rows[0] if k!="filename"]
    M=np.zeros((N,len(K)))
    for r in rows:
        if r["filename"] in idx: M[idx[r["filename"]]]=[float(r[k]) if r[k] not in("","nan") else 0.0 for k in K]
    return M,K
M3,K3=load3("3d"); M4,K4=load3("4")
SRC9=['adj_min','src_drop','src_end_drop','src_first','src_lowfrac','src_max','src_mean','src_min','src_slope']
P4=['al_slope','al_p10','al_last','al10_min']
drift=[M3[:,K3.index(k)] for k in SRC9]+[M4[:,K4.index(k)] for k in P4]
def build(flash_cols, extra=()):
    cols=[z["r1b"].astype(float),z["r1c"].astype(float)]+list(flash_cols)+[ip,mc]+[NR[:,j] for j in range(len(NRK))]+drift+list(extra)
    for t in sorted(set(tracks)):
        if t!="base-props": cols.append((tracks==t).astype(float))
    for s in sorted(set(skus)): cols.append((skus==s).astype(float))
    return np.stack(cols,1)
def trkpct(v):
    o=np.zeros(N)
    for t in set(tracks):
        m=tracks==t; o[m]=rp(v[m])
    return o
def run(X,mask,nf,seeds):
    Xd=X[mask]; yd=y[mask]; gd=groups[mask]; oofs=[]
    for sd in seeds:
        oof=np.zeros(len(yd))
        for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=sd).split(Xd,yd,gd):
            sc=StandardScaler().fit(Xd[tr])
            m=LogisticRegression(C=100,max_iter=6000).fit(sc.transform(Xd[tr]),yd[tr])
            oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
        oofs.append(oof)
    per=[br_at(o,yd) for o in oofs]; bag=np.mean([rp(o) for o in oofs],0)
    return np.mean(per),np.std(per),br_at(bag,yd),auc(bag,yd)
ALL=np.ones(N,bool)
R5=rp(b5)
A3=(rp(f1)+rp(f2)+R5)/3
CFG=[("现配:两遍",[A2,trkpct(A2)],()),
     ("v6 替换两遍",[R5,trkpct(R5)],()),
     ("两遍+v6列",[A2,trkpct(A2),R5],()),
     ("三遍均值(含v6)",[A3,trkpct(A3)],())]
print("\n=== 融合层 ===")
for nm,fc,ex in CFG:
    for tag,mask,nf in (("dev867/10折",dm,10),("全1233/20折",ALL,20)):
        m,s,b,a=run(build(fc,ex),mask,nf,tuple(range(42,50)))
        m2,_,b2,_=run(build(fc,ex),mask,nf,(47,48,49))
        print(f"  {nm:18s} {tag:12s} 8seed 单{m:.4f}±{s:.4f} 装袋{b:.4f} AUC{a:.4f} | CONF 单{m2:.4f} 装袋{b2:.4f}",flush=True)
