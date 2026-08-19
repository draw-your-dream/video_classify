#!/usr/bin/env python3
"""newref v3:形象还原轴加深(唯一奏效的信号族)。
32 帧 × SigLIP2-so400m,对本款官方视角图的相似度曲线 + 与其它款的判别余量 + 首帧漂移。
输出 data/pbase/out/newref3_1233.csv,并缓存帧嵌入 newref3_emb.npy 供后续复用。"""
import csv, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path.home()/"tutu-video-eval"; D=ROOT/"data"; OUT=D/"pbase/out"
import os
MODEL=os.environ.get("NR3_MODEL","google/siglip2-so400m-patch16-512"); NF=32; BATCH=int(os.environ.get("NR3_BATCH","8"))
TAG=os.environ.get("NR3_TAG","")
proc=AutoImageProcessor.from_pretrained(MODEL)
_m=AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16)
model=(_m.vision_model if hasattr(_m,"vision_model") else _m).to("cuda").eval()

@torch.no_grad()
def embed(pils):
    out=[]
    for i in range(0,len(pils),BATCH):
        x=proc(images=pils[i:i+BATCH], return_tensors="pt")["pixel_values"].half().to("cuda")
        o=model(pixel_values=x)
        f=o.pooler_output if getattr(o,"pooler_output",None) is not None else o.last_hidden_state.mean(1)
        out.append(f.float().cpu().numpy())
    E=np.concatenate(out); return E/np.linalg.norm(E,axis=1,keepdims=True)

views=sorted((D/"sku_ref_v2/views").glob("*.jpg"))
vsku=[p.stem.rsplit("_v",1)[0] for p in views]
VE=embed([Image.open(p).convert("RGB") for p in views])
print(f"官方视角图 {len(views)} 张 / {len(set(vsku))} 款", flush=True)

vids=[v if v.endswith(".mp4") else v+".mp4" for v in json.load(open(OUT/"X303_vids.json"))]
def frames(path,n=NF):
    cap=cv2.VideoCapture(str(path)); tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ids=np.linspace(0,max(tot-1,0),n).astype(int); got=[]; want=set(ids.tolist()); i=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        if i in want: got.append(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))
        i+=1
    cap.release()
    while len(got)<n and got: got.append(got[-1])
    return got

def stats(c,p):
    d={f"{p}_mean":c.mean(),f"{p}_min":c.min(),f"{p}_max":c.max(),f"{p}_first":c[0],
       f"{p}_last":c[-1],f"{p}_std":c.std(),f"{p}_drop":c[0]-c.min(),f"{p}_end_drop":c[0]-c[-1]}
    t=np.arange(len(c)); d[f"{p}_slope"]=np.polyfit(t,c,1)[0]
    d[f"{p}_lowfrac"]=float((c<c.mean()-c.std()).mean())
    return d

rows=[]; t0=time.time(); EMB=np.zeros((len(vids),NF,VE.shape[1]),np.float16)
for k,v in enumerate(vids):
    fp=D/"videos"/v
    if not fp.exists(): rows.append({"filename":v}); continue
    F=embed(frames(fp)); EMB[k]=F.astype(np.float16)
    sku=v.split("__")[2]
    own=np.array([s==sku for s in vsku]); oth=~own
    C=F@VE.T
    r={"filename":v}
    if own.sum():
        co=C[:,own].max(1); r.update(stats(co,"own"))
        idx=C[:,own].argmax(1); r["own_switch"]=float((np.diff(idx)!=0).mean())
        srt=np.sort(C[:,own],1); r["own_gap12"]=float((srt[:,-1]-srt[:,-2]).mean()) if own.sum()>1 else 0.0
        if oth.sum():
            ce=C[:,oth].max(1); m=co-ce
            r.update(stats(m,"marg")); r["oth_mean"]=float(ce.mean())
    s0=F@F[0]; r.update(stats(s0,"src"))
    adj=(F[1:]*F[:-1]).sum(1); r["adj_mean"]=float(adj.mean()); r["adj_min"]=float(adj.min())
    rows.append(r)
    if (k+1)%100==0: print(f"{k+1}/{len(vids)} ({time.time()-t0:.0f}s)", flush=True)
keys=["filename"]+sorted({k for r in rows for k in r if k!="filename"})
with open(OUT/f"newref3{TAG}_1233.csv","w",newline="") as f:
    w=csv.DictWriter(f,keys); w.writeheader()
    for r in rows: w.writerow({k:r.get(k,"") for k in keys})
np.save(OUT/f"newref3{TAG}_emb.npy",EMB)
print("DONE", len(rows), len(keys)-1, "列", flush=True)
