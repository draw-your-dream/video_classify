#!/usr/bin/env python3
"""判官第三遍进融合层:两遍秩均值 vs 三遍秩均值(以及三遍的自一致性)。
口径:隔离探针 + max_cos + src9 + patch4(2.0 终配),dev867/10折 与 全1233/20折,选型/复核/8种子。"""
import csv, json
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
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
r3={}
for l in open(OUT/"flash_run3_raw.jsonl"):
    o=json.loads(l); res=o.get("result") or {}
    if "bad_score" in res: r3[o["filename"]]=res["bad_score"]
json.dump(r3,open(OUT/"flash_run3_1233.json","w"))
print(f"第三遍有效 {len(r3)}/{N}")
f1=np.array([s1.get(v,50) for v in vids],float); f2=np.array([s2.get(v,50) for v in vids],float)
f3=np.array([r3.get(v,np.nan) for v in vids],float)
ok=~np.isnan(f3); f3f=np.where(ok,f3,np.nanmedian(f3))
print("三遍两两秩相关: 1-2 %.3f  1-3 %.3f  2-3 %.3f"%(
    spearmanr(f1,f2).statistic, spearmanr(f1[ok],f3[ok]).statistic, spearmanr(f2[ok],f3[ok]).statistic))
for nm,arr in (("第1遍",f1),("第2遍",f2),("第3遍",f3f)):
    print(f"  {nm} 单独 dev br@80={br_at(arr[dm],y[dm]):.4f} AUC={auc(arr[dm],y[dm]):.4f}")
A2=(rp(f1)+rp(f2))/2; A3=(rp(f1)+rp(f2)+rp(f3f))/3
for nm,arr in (("两遍均值",A2),("三遍均值",A3)):
    print(f"  {nm} dev br@80={br_at(arr[dm],y[dm]):.4f} AUC={auc(arr[dm],y[dm]):.4f}")
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
extra=[M3[:,K3.index(k)] for k in SRC9]+[M4[:,K4.index(k)] for k in P4]
def build(favg):
    fl=np.zeros(N)
    for t in set(tracks):
        m=tracks==t; fl[m]=rp(favg[m])
    cols=[z["r1b"].astype(float),z["r1c"].astype(float),favg,fl,ip,mc]+[NR[:,j] for j in range(len(NRK))]+extra
    for t in sorted(set(tracks)):
        if t!="base-props": cols.append((tracks==t).astype(float))
    for s in sorted(set(skus)): cols.append((skus==s).astype(float))
    return np.stack(cols,1)
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
print("\n=== 融合层 ===")
for nm,favg in (("两遍(现配)",A2),("三遍",A3)):
    for tag,mask,nf in (("dev867/10折",dm,10),("全1233/20折",ALL,20)):
        for sname,seeds in (("8seed",tuple(range(42,50))),("CONF",(47,48,49))):
            m,s,b,a=run(build(favg),mask,nf,seeds)
            print(f"  {nm:10s} {tag:12s} {sname:6s} 单{m:.4f}±{s:.4f} 装袋{b:.4f} AUC{a:.4f}",flush=True)
