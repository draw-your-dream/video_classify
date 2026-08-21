"""43 列终配的共用加载器。import 后拿 vids/y/groups/tracks/skus/dm/COLS/NAMES。"""
import csv, json
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
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
y=np.array([1 if mapr[v]["grade"]=="bad" else 0 for v in vids])
groups=np.array([mapr[v]["source_sha"] for v in vids])
tracks=np.array([mapr[v]["track"] for v in vids]); skus=np.array([v.split("__")[2] for v in vids])
lb=json.load(open(D/"lockbox_split.json")); dev=set(lb["dev"])
dm=np.array([(g in dev) or (v in dev) for g,v in zip(groups,vids)]); ALL=np.ones(N,bool)
def trkpct(v):
    o=np.zeros(N)
    for t in set(tracks):
        m=tracks==t; o[m]=rp(v[m])
    return o
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
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
f1=np.array([s1.get(v,50) for v in vids],float); f2=np.array([s2.get(v,50) for v in vids],float)
A2=(rp(f1)+rp(f2))/2
v8={}
for l in open(OUT/"flash_v8_raw.jsonl"):
    o=json.loads(l); r=o.get("result") or {}
    if "bad_score" in r: v8[o["filename"]]=r["bad_score"]
b8=np.array([v8.get(v,np.nan) for v in vids],float); b8=np.where(np.isnan(b8),np.nanmedian(b8),b8); R8=rp(b8)
# 43 列(名字与顺序对应 final_cols_20260819.json)
NAMES=["r1b","r1c","flash_avg","flash_trkpct","flash_v8","imgprobe_lr","max_cos"]+NRK+SRC9+P4
COLS=[z["r1b"].astype(float),z["r1c"].astype(float),A2,trkpct(A2),R8,ip,mc]+[NR[:,j] for j in range(len(NRK))] \
     +[M3[:,K3.index(k)] for k in SRC9]+[M4[:,K4.index(k)] for k in P4]
ONEHOT=[]
for t in sorted(set(tracks)):
    if t!="base-props": ONEHOT.append(((tracks==t).astype(float),f"TR_{t}"))
for s in sorted(set(skus)): ONEHOT.append(((skus==s).astype(float),f"SKU_{s}"))
def stack(cols, onehot=True):
    c=list(cols)+([o for o,_ in ONEHOT] if onehot else [])
    return np.stack(c,1)
X43=stack(COLS)
