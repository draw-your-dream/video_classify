#!/usr/bin/env python3
"""newref4:DINOv2 patch 级结构漂移(全局池化看不见的局部形变)。
每帧取 patch token,与首帧做两种对照:同位置余弦(空间对齐漂移)与软对应余弦(每个 patch 对首帧所有
patch 取最大,容忍位移)。二者的差 = 位移量;软对应低 = 真的变形了。输出 newref4_1233.csv"""
import csv, json, os, time
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
ROOT=Path.home()/"tutu-video-eval"; D=ROOT/"data"; OUT=D/"pbase/out"
MODEL=os.environ.get("NR4_MODEL",str(ROOT/".hf_cache/dinov2-base-ms")); NF=int(os.environ.get("NR4_NF","24")); B=16
proc=AutoImageProcessor.from_pretrained(MODEL)
model=AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
@torch.no_grad()
def patches(pils):
    out=[]
    for i in range(0,len(pils),B):
        x=proc(images=pils[i:i+B], return_tensors="pt")["pixel_values"].half().to("cuda")
        h=model(pixel_values=x).last_hidden_state[:,1:]          # 去掉 CLS
        h=h/h.norm(dim=-1,keepdim=True)
        out.append(h.float().cpu().numpy())
    return np.concatenate(out)
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
    return {f"{p}_mean":c.mean(),f"{p}_min":c.min(),f"{p}_last":c[-1],f"{p}_std":c.std(),
            f"{p}_drop":c[0]-c.min(),f"{p}_slope":np.polyfit(t,c,1)[0],f"{p}_p10":np.percentile(c,10)}
vids=[v if v.endswith(".mp4") else v+".mp4" for v in json.load(open(OUT/"X303_vids.json"))]
rows=[]; t0=time.time()
for k,v in enumerate(vids):
    fp=D/"videos"/v
    if not fp.exists(): rows.append({"filename":v}); continue
    P=patches(frames(fp)); P0=P[0]
    align=(P[1:]*P0[None]).sum(-1)                     # (T-1,Np) 同位置
    soft=np.einsum('tpd,qd->tpq',P[1:],P0).max(-1)     # (T-1,Np) 软对应
    r={"filename":v}
    r.update(stats(align.mean(1),"al")); r.update(stats(soft.mean(1),"sf"))
    r.update(stats(np.percentile(align,10,axis=1),"al10")); r.update(stats(np.percentile(soft,10,axis=1),"sf10"))
    r["mot_mean"]=float((soft.mean(1)-align.mean(1)).mean())    # 位移量
    r["mot_max"]=float((soft.mean(1)-align.mean(1)).max())
    r["worst_patch"]=float(soft.min())
    rows.append(r)
    if (k+1)%100==0: print(f"{k+1}/{len(vids)} ({time.time()-t0:.0f}s)",flush=True)
keys=["filename"]+sorted({k for r in rows for k in r if k!="filename"})
with open(OUT/"newref4_1233.csv","w",newline="") as f:
    w=csv.DictWriter(f,keys); w.writeheader()
    for r in rows: w.writerow({k:r.get(k,"") for k in keys})
print("DONE",len(rows),len(keys)-1,"列",flush=True)
