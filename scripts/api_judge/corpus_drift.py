#!/usr/bin/env python3
"""旧 0430 语料上抽 DINOv2 漂移特征(全局 src + patch 同位置/软对应),一次前向两族全出。
用法: python corpus_drift.py <视频根目录> <文件清单jsonl(含 video/abs_path)> <输出csv>"""
import csv, json, os, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
VROOT=Path(sys.argv[1]); LIST=Path(sys.argv[2]); OUTF=Path(sys.argv[3])
MODEL=os.environ.get("DRIFT_MODEL","facebook/dinov2-base"); NF=int(os.environ.get("DRIFT_NF","24")); B=16
proc=AutoImageProcessor.from_pretrained(MODEL)
model=AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
@torch.no_grad()
def enc(pils):
    P=[]; G=[]
    for i in range(0,len(pils),B):
        x=proc(images=pils[i:i+B], return_tensors="pt")["pixel_values"].half().to("cuda")
        h=model(pixel_values=x).last_hidden_state
        g=h[:,0]; g=g/g.norm(dim=-1,keepdim=True)
        p=h[:,1:]; p=p/p.norm(dim=-1,keepdim=True)
        G.append(g.float().cpu().numpy()); P.append(p.float().cpu().numpy())
    return np.concatenate(G), np.concatenate(P)
def frames(path,n=NF):
    cap=cv2.VideoCapture(str(path)); tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    want=set(np.linspace(0,max(tot-1,0),n).astype(int).tolist()); got=[]; i=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        if i in want: got.append(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))
        i+=1
    cap.release()
    while len(got)<n and got: got.append(got[-1])
    return got
def stats(c,p):
    t=np.arange(len(c))
    return {f"{p}_mean":c.mean(),f"{p}_min":c.min(),f"{p}_max":c.max(),f"{p}_first":c[0],f"{p}_last":c[-1],
            f"{p}_std":c.std(),f"{p}_drop":c[0]-c.min(),f"{p}_end_drop":c[0]-c[-1],
            f"{p}_slope":np.polyfit(t,c,1)[0],f"{p}_lowfrac":float((c<c.mean()-c.std()).mean()),
            f"{p}_p10":np.percentile(c,10)}
rows=[]; t0=time.time()
items=[json.loads(l) for l in open(LIST)]
for k,e in enumerate(items):
    rel=e.get("abs_path","").split("/data/s3/")[-1] or e["video"]
    fp=VROOT/rel
    if not fp.exists(): fp=VROOT/e["video"]
    r={"video":e["video"],"label":e.get("label","")}
    if fp.exists():
        G,P=enc(frames(fp)); g0=G[0]; P0=P[0]
        src=G@g0; adj=(G[1:]*G[:-1]).sum(1)
        al=(P[1:]*P0[None]).sum(-1); sf=np.einsum('tpd,qd->tpq',P[1:],P0).max(-1)
        r.update(stats(src,"src")); r["adj_mean"]=float(adj.mean()); r["adj_min"]=float(adj.min())
        r.update(stats(al.mean(1),"al")); r.update(stats(np.percentile(al,10,axis=1),"al10"))
        r.update(stats(sf.mean(1),"sf")); r.update(stats(np.percentile(sf,10,axis=1),"sf10"))
        r["mot_mean"]=float((sf.mean(1)-al.mean(1)).mean()); r["worst_patch"]=float(sf.min())
    rows.append(r)
    if (k+1)%100==0: print(f"{k+1}/{len(items)} ({time.time()-t0:.0f}s)",flush=True)
keys=["video","label"]+sorted({k for r in rows for k in r if k not in ("video","label")})
with open(OUTF,"w",newline="") as f:
    w=csv.DictWriter(f,keys); w.writeheader()
    for r in rows: w.writerow({k:r.get(k,"") for k in keys})
print("DRIFT_DONE",len(rows),len(keys)-2,"列",flush=True)
