#!/usr/bin/env python3
"""零成本试验:把两遍冠军输出里没用过的结构化字段(reason_labels / evidence / normal_play_visible /
两遍分歧)做成特征块,加到 43 列终配上。不花任何 API 钱。
终配 = r1b,r1c,flash两遍(A2+trkpct),v8列,imgprobe_lr,max_cos,newref11,src9,patch4,TR4,SKU8 = 43 列。"""
import csv, json, collections
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
ALL=np.ones(N,bool)

# ---- 两份冠军配置的结构化原始输出 ----
LABELS=["还原度","衣服/身体的时间一致性","大小变化","僵硬","卡顿/少活人感","四肢不动",
        "动作位移不连贯","运动主体","静止不动","慢动作","物理规律","不合理的物体",
        "帧跳变","首帧一致","背景运动混乱"]
def load_struct(files):
    m={}
    for f in files:
        for l in open(f,encoding="utf-8"):
            try: o=json.loads(l)
            except: continue
            r=o.get("result") or {}
            if "bad_score" in r: m[o["filename"]]=r
    return m
RUNS=[load_struct([D/"out_holdout_full.jsonl", D/"out_d150_newrefs_ref.jsonl"]),
      load_struct([OUT/"flash_v8_raw.jsonl"])]
print("两份结构化输出覆盖:", [sum(1 for v in vids if v in R) for R in RUNS])

nL=len(LABELS)
LV=np.zeros((N,nL)); EV=np.zeros((N,4)); DIS=np.zeros((N,3))
for i,v in enumerate(vids):
    rs=[R[v] for R in RUNS if v in R]
    if not rs: continue
    k=len(rs)
    for r in rs:
        for L in r.get("reason_labels") or []:
            if L in LABELS: LV[i,LABELS.index(L)]+=1.0/k
    nev=[len(r.get("evidence") or []) for r in rs]
    dur=[]; first=[]
    for r in rs:
        ev=r.get("evidence") or []
        tot=0.0; fs=9.0
        for e in ev:
            try:
                a=float(e.get("start_sec",0)); b=float(e.get("end_sec",0))
                tot+=max(0.0,b-a); fs=min(fs,a)
            except: pass
        dur.append(tot); first.append(fs)
    EV[i]=[np.mean(nev), np.mean(dur), np.mean(first), np.mean([0.0 if r.get("normal_play_visible") else 1.0 for r in rs])]
    ss=[r["bad_score"] for r in rs]; gg=[r.get("grade") for r in rs]
    DIS[i]=[np.std(ss), 1.0 if len(set(gg))>1 else 0.0, np.mean([1.0 if g=="bad" else 0.0 for g in gg])]
keep=[j for j in range(nL) if (LV[:,j]>0).sum()>=20]
LV=LV[:,keep]; LNAMES=[LABELS[j] for j in keep]
print(f"标签列保留 {len(LNAMES)}: {LNAMES}")
print("evidence 列: n_ev, 总时长, 最早时刻, npv_false 比例 | 分歧列: score_std, grade_disagree, bad票率")
for nm,M in (("标签票",LV),("evidence",EV),("分歧",DIS)):
    print(f"  {nm}: 单列最好 br@80(dev) = "+", ".join(f"{br_at(rp(M[dm,j]),y[dm]):.3f}" for j in range(M.shape[1])))

# ---- 43 列终配 ----
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
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
f1=np.array([s1.get(v,50) for v in vids],float); f2=np.array([s2.get(v,50) for v in vids],float)
A2=(rp(f1)+rp(f2))/2
b8=np.array([RUNS[1].get(v,{}).get("bad_score",np.nan) for v in vids],float)
b8=np.where(np.isnan(b8),np.nanmedian(b8),b8); R8=rp(b8)
def trkpct(v):
    o=np.zeros(N)
    for t in set(tracks):
        m=tracks==t; o[m]=rp(v[m])
    return o
FLASH=[A2,trkpct(A2),R8]
def build(extra=()):
    cols=[z["r1b"].astype(float),z["r1c"].astype(float)]+FLASH+[ip,mc]+[NR[:,j] for j in range(len(NRK))]+drift+list(extra)
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
def fold_block(B,mask,nf,seed):
    """把一个块用折内 5 折 OOF 折成一列(训练行用样本外预测,遵守嵌套规则)"""
    out=np.zeros(N)
    Xd=B[mask]; yd=y[mask]; gd=groups[mask]
    o=np.zeros(len(yd))
    for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=seed).split(Xd,yd,gd):
        inner=np.zeros(len(tr))
        for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(Xd[tr],yd[tr],gd[tr]):
            sc=StandardScaler().fit(Xd[tr][a]); m=LogisticRegression(C=1,max_iter=4000).fit(sc.transform(Xd[tr][a]),yd[tr][a])
            inner[b]=m.predict_proba(sc.transform(Xd[tr][b]))[:,1]
        sc=StandardScaler().fit(Xd[tr]); m=LogisticRegression(C=1,max_iter=4000).fit(sc.transform(Xd[tr]),yd[tr])
        o[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
    out[mask]=o
    return out
CFG=[("基线 43列",()),
     (f"+标签票({len(LNAMES)}列)",tuple(LV[:,j] for j in range(LV.shape[1]))),
     ("+evidence(4列)",tuple(EV[:,j] for j in range(4))),
     ("+分歧(3列)",tuple(DIS[:,j] for j in range(3))),
     ("+全部结构化",tuple(LV[:,j] for j in range(LV.shape[1]))+tuple(EV[:,j] for j in range(4))+tuple(DIS[:,j] for j in range(3)))]
print("\n=== 融合层 ===",flush=True)
for nm,ex in CFG:
    X=build(ex)
    for tag,mask,nf in (("dev867/10折",dm,10),("全1233/20折",ALL,20)):
        m,s,b,a=run(X,mask,nf,tuple(range(42,50)))
        m2,_,b2,_=run(X,mask,nf,(47,48,49))
        print(f"  {nm:20s} {tag:12s} 列{X.shape[1]:3d} 8seed 单{m:.4f}±{s:.4f} 装袋{b:.4f} AUC{a:.4f} | CONF 单{m2:.4f} 装袋{b2:.4f}",flush=True)

print("\n=== 块折成单列(折内 5 折 OOF) ===",flush=True)
BLOCKS={"标签票块":LV,"evidence块":EV,"分歧块":DIS,"全结构化块":np.hstack([LV,EV,DIS])}
for nm,B in BLOCKS.items():
    for tag,mask,nf in (("dev867/10折",dm,10),("全1233/20折",ALL,20)):
        res=[]
        for sd in range(42,50):
            col=fold_block(B,mask,nf,sd)
            X=build((col,)); Xd=X[mask]; yd=y[mask]; gd=groups[mask]
            oof=np.zeros(len(yd))
            for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=sd).split(Xd,yd,gd):
                sc=StandardScaler().fit(Xd[tr]); m=LogisticRegression(C=100,max_iter=6000).fit(sc.transform(Xd[tr]),yd[tr])
                oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
            res.append(oof)
        per=[br_at(o,yd) for o in res]; bag=np.mean([rp(o) for o in res],0)
        print(f"  {nm:12s} {tag:12s} 8seed 单{np.mean(per):.4f}±{np.std(per):.4f} 装袋{br_at(bag,yd):.4f} AUC{auc(bag,yd):.4f}",flush=True)
