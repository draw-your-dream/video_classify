#!/usr/bin/env python3
"""周报口径重算(无泄漏图片探针):给某条视频打分的探针不许见过该视频的源图。
产出周报所需全部数字,并保存无泄漏融合 OOF 供画图。"""
import csv, json
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
def br_at(s,y,rel=0.8):
    gn=np.sort(s[y==0]); b=s[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return ((b>t).sum()+(b==t).sum()*(1-fr))/len(b)
def auc(s,y):
    r=rankdata(s); pos=r[y==1]
    return (pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum())
def rankpct(x): return rankdata(x)/len(x)

vids=[v if v.endswith(".mp4") else v+".mp4" for v in json.load(open(OUT/"X303_vids.json"))]
idx={v:i for i,v in enumerate(vids)}; N=len(vids)
mapr={r["filename"]:r for r in csv.DictReader(open(D/"api_judge_video_image_map.csv",encoding="utf-8-sig"))}
y=np.array([1 if mapr[v]["grade"]=="bad" else 0 for v in vids])
grade=np.array([mapr[v]["grade"] for v in vids])
groups=np.array([mapr[v]["source_sha"] for v in vids])
tracks=np.array([mapr[v]["track"] for v in vids]); skus=np.array([v.split("__")[2] for v in vids])
lb=json.load(open(D/"lockbox_split.json")); dev=set(lb["dev"])
dm=np.array([(g in dev) or (v in dev) for g,v in zip(groups,vids)])
yd=y[dm]

z=np.load(OUT/"r1_oof.npz",allow_pickle=True); r1b=z["r1b"].astype(float); r1c=z["r1c"].astype(float)
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
f1=np.array([s1.get(v,50) for v in vids],float); f2=np.array([s2.get(v,50) for v in vids],float)
flash=(rankpct(f1)+rankpct(f2))/2
fl=np.zeros(N)
for t in set(tracks):
    m=tracks==t; fl[m]=rankpct(flash[m])
cl=np.load(OUT/"imgprobe_clean.npz",allow_pickle=True)
cfns=cl["fns"].astype(str); cmap={f:i for i,f in enumerate(cfns)}
ipc=np.array([cl["p_gbm"][cmap[v]] for v in vids])
NRK=["clip_mean","clip_min","clip_last","clip_drop","clip_std","sig_mean","sig_min","sig_first","sig_last","sig_drop","sig_std"]
NR=np.zeros((N,len(NRK)))
for r in csv.DictReader(open(OUT/"newref_feats.csv")):
    if r["filename"] in idx: NR[idx[r["filename"]]]=[float(r[k] or 0) for k in NRK]

def cv(X,seeds=range(42,50)):
    Xd=X[dm]; gd=groups[dm]; oofs=[]
    for sd in seeds:
        oof=np.zeros(len(yd))
        for tr,te in StratifiedGroupKFold(10,shuffle=True,random_state=sd).split(Xd,yd,gd):
            sc=StandardScaler().fit(Xd[tr])
            m=LogisticRegression(C=100,max_iter=4000,class_weight={0:1,1:2}).fit(sc.transform(Xd[tr]),yd[tr])
            oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
        oofs.append(oof)
    return np.array(oofs)

print("== 单路(dev 867) ==")
for nm,v in (("专家栈 R1c",r1c),("专家栈 R1b",r1b),("flash 双跑均值",flash),("图片探针(无泄漏)",ipc)):
    print(f"  {nm:20s} br@80={br_at(v[dm],yd):.4f} AUC={auc(v[dm],yd):.4f}")
o=cv(NR,(42,43,44)); print(f"  {'官方形象图对照11列':20s} br@80={np.mean([br_at(x,yd) for x in o]):.4f} AUC={np.mean([auc(x,yd) for x in o]):.4f}")

def build(with_probe=True):
    cols=[r1b,r1c,flash,fl]+([ipc] if with_probe else [])+[NR[:,j] for j in range(len(NRK))]
    for t in sorted(set(tracks)):
        if t!="base-props": cols.append((tracks==t).astype(float))
    for s in sorted(set(skus)): cols.append((skus==s).astype(float))
    return np.stack(cols,1)

for nm,wp in (("融合(含无泄漏探针,28列)",True),("融合(去掉探针,27列)",False)):
    o=cv(build(wp)); per=[br_at(x,yd) for x in o]; bag=np.mean([rankpct(x) for x in o],0)
    print(f"\n== {nm} ==")
    print("  单seed:", " ".join(f"{v:.4f}" for v in per))
    print(f"  单seed 均值 {np.mean(per):.4f} ± {np.std(per):.4f}")
    print(f"  装袋 br@70={br_at(bag,yd,0.7):.4f} br@80={br_at(bag,yd,0.8):.4f} br@90={br_at(bag,yd,0.9):.4f} AUC={auc(bag,yd):.4f}")
    if wp:
        np.save(OUT/"weekly_clean_bag.npy",bag)
        g=grade[dm]; gn=np.sort(bag[yd==0]); k=int(np.floor(0.8*len(gn))); t=gn[k-1]
        held=(bag>t)&(yd==0)
        print(f"  扣下的合格视频 {int(held.sum())} 条: good {int((held&(g=='good')).sum())} / normal {int((held&(g=='normal')).sum())}")
        for tr_ in sorted(set(tracks)):
            m=(tracks[dm]==tr_)
            if m.sum()>40 and yd[m].sum()>5:
                print(f"    轨 {tr_:12s} n={int(m.sum()):4d} bad={int(yd[m].sum()):3d} br@80={br_at(bag[m],yd[m]):.3f}")
