#!/usr/bin/env python3
"""newref5:角色裁切放大后的 DINOv2 形象漂移。
动机(R7/E-CEIL-3):判官三遍全漏的 bad 里 78% 是「还原度」,即五官/衣物细节;
整帧 224 下角色只占 7-51% 画面,细节只剩一两个 patch。用 grounding-dino 逐帧框出角色、
裁切后再喂 224,给这个信号 1.4-3.8 倍线性分辨率。
另附框几何列:框面积/中心/长宽比轨迹 + 检测分数轨迹(角色是否还认得出是蘑菇)。
环境:~/.venvs/tutu-ex/bin/python,必走 gpu run。"""
import os, sys, csv, cv2, json, time
import numpy as np, torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, AutoModel
D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
HF=Path.home()/"tutu-video-eval/.hf_cache"
NF=int(os.environ.get("NR5_FRAMES","32")); LIMIT=int(os.environ.get("NR5_LIMIT","0"))
TAG=os.environ.get("NR5_TAG","5"); TEXT="a mushroom character."
dev="cuda"
GD=str(next((HF/"models--IDEA-Research--grounding-dino-base/snapshots").glob("*")))
gproc=AutoProcessor.from_pretrained(GD); gmdl=AutoModelForZeroShotObjectDetection.from_pretrained(GD).to(dev).eval()
DV=str(HF/"dinov2-base-ms")
dproc=AutoImageProcessor.from_pretrained(DV); dmdl=AutoModel.from_pretrained(DV).to(dev).eval().half()
def frames(vp,n):
    cap=cv2.VideoCapture(str(vp)); T=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if T<=0:
        fr=[]
        while True:
            ok,f=cap.read()
            if not ok: break
            fr.append(f)
        cap.release()
        if not fr: return []
        ix=np.linspace(0,len(fr)-1,min(n,len(fr))).astype(int); return [fr[i] for i in ix]
    ix=np.unique(np.linspace(0,T-1,n).astype(int)); out=[]
    for i in ix:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,f=cap.read()
        if ok: out.append(f)
    cap.release(); return out
@torch.no_grad()
def boxes(imgs):
    """批量检测,返回 [(x0,y0,x1,y1,score)];失败给 (nan,)*4,score 0"""
    res=[]
    for i in range(0,len(imgs),8):
        ch=imgs[i:i+8]; H,W=ch[0].shape[:2]
        inp=gproc(images=ch,text=[TEXT]*len(ch),return_tensors="pt").to(dev)
        o=gmdl(**inp)
        pp=gproc.post_process_grounded_object_detection(o,inp.input_ids,threshold=0.20,text_threshold=0.20,
                                                       target_sizes=[(H,W)]*len(ch))
        for r in pp:
            s=r["scores"].float().cpu().numpy(); b=r["boxes"].float().cpu().numpy()
            if len(s)==0: res.append((np.nan,)*4+(0.0,))
            else:
                j=int(s.argmax()); res.append((*b[j],float(s[j])))
    return res
@torch.no_grad()
def embed(crops):
    inp=dproc(images=crops,return_tensors="pt").to(dev); inp["pixel_values"]=inp["pixel_values"].half()
    o=dmdl(**inp)
    h=o.last_hidden_state.float()
    g=o.pooler_output.float() if getattr(o,"pooler_output",None) is not None else h.mean(1)
    g=torch.nn.functional.normalize(g,dim=-1)
    p=torch.nn.functional.normalize(h[:,1:],dim=-1)
    return g.cpu().numpy(), p.cpu().numpy()
def stats(c):
    """c: 长度 T-1 的相似度序列(对 frame0)"""
    c=np.asarray(c,float)
    if len(c)==0: return dict.fromkeys(["mean","min","last","first","drop","std","slope","lowfrac"],0.0)
    x=np.arange(len(c))
    return {"mean":c.mean(),"min":c.min(),"last":c[-1],"first":c[0],"drop":c[0]-c[-1],
            "std":c.std(),"slope":float(np.polyfit(x,c,1)[0]) if len(c)>1 else 0.0,
            "lowfrac":float((c<c.mean()-c.std()).mean())}
vids=sorted(p.name for p in (D/"videos").glob("*.mp4"))
if LIMIT: vids=vids[:LIMIT]
print(f"视频 {len(vids)} 条,每条 {NF} 帧",flush=True)
rows=[]; t0=time.time()
for k,vn in enumerate(vids):
    fr=frames(D/"videos"/vn,NF)
    if len(fr)<4: continue
    rgb=[cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in fr]
    bx=boxes(rgb)
    H,W=rgb[0].shape[:2]
    crops=[]; geo=[]
    for img,(x0,y0,x1,y1,sc) in zip(rgb,bx):
        if not np.isfinite(x0):
            x0,y0,x1,y1=0,0,W,H
        cx,cy=(x0+x1)/2,(y0+y1)/2; w,h=max(8.0,x1-x0),max(8.0,y1-y0)
        m=0.18; w*=1+m; h*=1+m
        a0=int(max(0,cx-w/2)); a1=int(min(W,cx+w/2)); b0=int(max(0,cy-h/2)); b1=int(min(H,cy+h/2))
        if a1-a0<8 or b1-b0<8: a0,b0,a1,b1=0,0,W,H
        crops.append(img[b0:b1,a0:a1])
        geo.append((w*h/(W*H), cx/W, cy/H, w/max(1e-6,h), sc))
    G,P=embed(crops)
    cg=(G[1:]*G[0][None]).sum(-1)
    al=(P[1:]*P[0][None]).sum(-1)                      # 同位置 patch 对齐
    adj=(G[1:]*G[:-1]).sum(-1)
    geo=np.array(geo,float)
    r={"filename":vn}
    for k2,v in stats(cg).items(): r["g_"+k2]=v
    r["g_adjmin"]=float(adj.min()) if len(adj) else 0.0
    for k2,v in stats(al.mean(1)).items(): r["pa_"+k2]=v
    for k2,v in stats(np.percentile(al,10,axis=1)).items(): r["p10_"+k2]=v
    r["pa_worstmin"]=float(al.min())
    r["box_area_mean"]=geo[:,0].mean(); r["box_area_drop"]=geo[0,0]-geo[-1,0]
    r["box_area_std"]=geo[:,0].std(); r["box_area_min"]=geo[:,0].min()
    r["box_cx_std"]=geo[:,1].std(); r["box_cy_std"]=geo[:,2].std(); r["box_ar_std"]=geo[:,3].std()
    r["det_mean"]=geo[:,4].mean(); r["det_min"]=geo[:,4].min(); r["det_fail"]=float((geo[:,4]<=0).mean())
    rows.append(r)
    if (k+1)%50==0: print(f"{k+1}/{len(vids)} ({time.time()-t0:.0f}s)",flush=True)
K=[k for k in rows[0] if k!="filename"]
with open(OUT/f"newref{TAG}_1233.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,["filename"]+K); w.writeheader(); w.writerows(rows)
print(f"DONE {len(rows)} 条 {len(K)} 列 → newref{TAG}_1233.csv  用时 {time.time()-t0:.0f}s",flush=True)
