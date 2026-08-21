#!/usr/bin/env python3
"""在已完成的部分行上给早期读数:单列强度 + 判官盲区召回。只用覆盖到的行,不插补。"""
import sys, csv, json
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import _base43 as B
OUT=Path.home()/"tutu-video-eval/data/pbase/out"
TAG=sys.argv[1] if len(sys.argv)>1 else ""
rp=B.rp; br_at=B.br_at; auc=B.auc; vids=B.vids; idx=B.idx; y=B.y
rows=[r for r in csv.DictReader(open(OUT/f"ipfid{TAG}_1233.csv",encoding="utf-8")) if r["filename"] in idx]
K=[k for k in rows[0] if k!="filename"]
ii=np.array([idx[r["filename"]] for r in rows])
M=np.array([[float(r[k]) if r[k] not in ("","nan","None") else np.nan for k in K] for r in rows])
yy=y[ii]
print(f"已覆盖 {len(rows)} 条  bad {yy.sum()} ({yy.mean()*100:.0f}%)  列 {len(K)}")
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
blind=(np.all(np.stack([x==0 for x in S],1),1))[ii]&(yy==1)
flash=np.mean([rp(x)[ii] for x in S],0)
print(f"这批里判官三遍全漏的 bad:{blind.sum()} 条;判官三票在这批上 br@80={br_at(flash,yy):.4f} AUC={auc(flash,yy):.4f}")
res=[]
for j,k in enumerate(K):
    v=M[:,j]; ok=np.isfinite(v)
    if ok.sum()<len(v)*0.8: continue
    v=np.where(ok,v,np.nanmedian(v))
    for sgn in (1,-1):
        a=auc(sgn*v,yy)
        if a>=0.5:
            s=rp(sgn*v); top=s>=np.quantile(s,0.80)
            res.append((br_at(s,yy),a,k,sgn,(top&blind).sum()/max(1,blind.sum())))
            break
res.sort(reverse=True)
print(f"\n{'列':16s} {'br@80':>7s} {'AUC':>7s}  盲区召回  方向")
for br,a,k,sgn,hit in res[:16]:
    print(f"{k:16s} {br:7.4f} {a:7.4f}  {hit*100:6.1f}%  {'低分为坏' if sgn<0 else '高分为坏'}")
print(f"\n随机水平:盲区召回 20%")
