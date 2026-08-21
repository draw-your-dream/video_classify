#!/usr/bin/env python3
"""newref6:部件级形象还原特征(本地,无 API)。
来源:draw-your-dream/image-dataset-curation-filtering @ef0c450 的图片质检思路——
定位后裁切、按部件分开判、伞盖用几何判据("宽>高的扁盖"退化成"高≈宽的圆球"或"缩到与身体同宽")。
本脚本把那套判据做成可度量的时序特征:
  A 伞盖几何:cap_w/cap_h、cap_w/body_w、cap_h/body_h、伞盖中心相对本体中心偏移
  B 裁切后自参照漂移:DINOv2(本体特写_t vs 本体特写_0)、(伞盖特写_t vs 伞盖特写_0)
  C 裁切后绝对还原:DINOv2 特写_t 对该款 5 张官方视角的 max/mean 余弦(本体 + 伞盖各一组)
  D 检出健康度:本体/伞盖检测分轨迹(伞盖认不出本身就是缺陷信号)
环境:~/.venvs/tutu-ex/bin/python;必走 gpu run。"""
import os, csv, cv2, time, glob
import numpy as np, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, AutoModel
D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"; HF=Path.home()/"tutu-video-eval/.hf_cache"
NF=int(os.environ.get("NR6_FRAMES","16")); LIMIT=int(os.environ.get("NR6_LIMIT","0"))
TAG=os.environ.get("NR6_TAG","6"); DET_PX=int(os.environ.get("NR6_DETPX","640"))
BODYQ="a mushroom character."; CAPQ="the round cap on top of a mushroom."
dev="cuda"
GD=str(next((HF/"models--IDEA-Research--grounding-dino-base/snapshots").glob("*")))
gproc=AutoProcessor.from_pretrained(GD); gmdl=AutoModelForZeroShotObjectDetection.from_pretrained(GD).to(dev).eval()
DV=str(HF/"dinov2-base-ms")
dproc=AutoImageProcessor.from_pretrained(DV); dmdl=AutoModel.from_pretrained(DV).to(dev).eval()

def shrink(img,px):
    h,w=img.shape[:2]; s=px/max(h,w)
    return (cv2.resize(img,(int(w*s),int(h*s))) if s<1 else img), (1/s if s<1 else 1.0)
@torch.no_grad()
def detect(imgs,query,thr=0.20):
    """批量检测,返回每图最高分框(原图坐标)与分数;无框给 None"""
    res=[]
    for i in range(0,len(imgs),8):
        ch=imgs[i:i+8]; sm=[]; sc=[]
        for im in ch:
            s,f=shrink(im,DET_PX); sm.append(s); sc.append(f)
        H,W=sm[0].shape[:2]
        inp=gproc(images=sm,text=[query]*len(sm),return_tensors="pt").to(dev)
        with torch.autocast("cuda",dtype=torch.float16):
            o=gmdl(**inp)
        pp=gproc.post_process_grounded_object_detection(o,inp.input_ids,threshold=thr,text_threshold=thr,
                                                       target_sizes=[(H,W)]*len(sm))
        for r,f in zip(pp,sc):
            s=r["scores"].float().cpu().numpy(); b=r["boxes"].float().cpu().numpy()
            if len(s)==0: res.append((None,0.0))
            else:
                j=int(s.argmax()); res.append((tuple(float(x)*f for x in b[j]),float(s[j])))
    return res
def sqcrop(img,box,margin=0.10,canvas=224):
    H,W=img.shape[:2]
    if box is None: box=(0,0,W,H)
    x0,y0,x1,y1=box; w=max(8.0,x1-x0); h=max(8.0,y1-y0); cx,cy=(x0+x1)/2,(y0+y1)/2
    w*=1+margin; h*=1+margin; side=max(w,h)
    a0=int(max(0,cx-side/2)); a1=int(min(W,cx+side/2)); b0=int(max(0,cy-side/2)); b1=int(min(H,cy+side/2))
    if a1-a0<8 or b1-b0<8: a0,b0,a1,b1=0,0,W,H
    c=img[b0:b1,a0:a1]
    return cv2.resize(c,(canvas,canvas),interpolation=cv2.INTER_AREA)
@torch.no_grad()
def emb(crops):
    inp=dproc(images=[Image.fromarray(c) for c in crops],return_tensors="pt").to(dev)
    with torch.autocast("cuda",dtype=torch.float16):
        o=dmdl(**inp)
    h=o.last_hidden_state.float()
    g=o.pooler_output.float() if getattr(o,"pooler_output",None) is not None else h.mean(1)
    return torch.nn.functional.normalize(g,dim=-1).cpu().numpy()
def frames(vp,n):
    cap=cv2.VideoCapture(str(vp)); T=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out=[]
    if T>0:
        for i in np.unique(np.linspace(0,T-1,n).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,f=cap.read()
            if ok: out.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    cap.release(); return out
def st(v,pre,d):
    v=np.asarray([x for x in v if np.isfinite(x)],float)
    if len(v)==0:
        for k in ("mean","min","max","last","first","drop","std","slope"): d[pre+k]=0.0
        return
    x=np.arange(len(v))
    d[pre+"mean"]=v.mean(); d[pre+"min"]=v.min(); d[pre+"max"]=v.max()
    d[pre+"last"]=v[-1]; d[pre+"first"]=v[0]; d[pre+"drop"]=v[0]-v[-1]
    d[pre+"std"]=v.std(); d[pre+"slope"]=float(np.polyfit(x,v,1)[0]) if len(v)>1 else 0.0

# ---- 官方视角:同样做本体/伞盖裁切并嵌入 ----
views=sorted(glob.glob(str(D/"sku_ref_v2/views/*.jpg")))
VS={}
vimgs=[cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB) for p in views]
vb=detect(vimgs,BODYQ); vc=detect(vimgs,CAPQ)
bodyc=[sqcrop(im,b) for im,(b,_) in zip(vimgs,vb)]
capc=[sqcrop(im,c) for im,(c,_) in zip(vimgs,vc)]
EB=np.concatenate([emb(bodyc[i:i+16]) for i in range(0,len(bodyc),16)])
EC=np.concatenate([emb(capc[i:i+16]) for i in range(0,len(capc),16)])
for i,p in enumerate(views):
    sku=Path(p).stem.rsplit("_v",1)[0]
    VS.setdefault(sku,{"body":[],"cap":[]})
    VS[sku]["body"].append(EB[i]); VS[sku]["cap"].append(EC[i])
for k in VS: VS[k]={a:np.stack(b) for a,b in VS[k].items()}
print("官方视角:",{k:len(v["body"]) for k,v in VS.items()},flush=True)

vids=sorted(p.name for p in (D/"videos").glob("*.mp4"))
if LIMIT: vids=vids[:LIMIT]
rows=[]; t0=time.time()
for k,vn in enumerate(vids):
    fr=frames(D/"videos"/vn,NF)
    if len(fr)<4: continue
    sku=vn.split("__")[2] if len(vn.split("__"))>2 else ""
    bb=detect(fr,BODYQ); cc=detect(fr,CAPQ)
    bcrops=[sqcrop(im,b) for im,(b,_) in zip(fr,bb)]
    ccrops=[sqcrop(im,c) for im,(c,_) in zip(fr,cc)]
    GB=emb(bcrops); GC=emb(ccrops)
    d={"filename":vn}
    # A 伞盖几何
    ar=[];cbw=[];cbh=[];off=[]
    for (b,_),(c,_) in zip(bb,cc):
        if b is None or c is None: continue
        bw=max(1e-6,b[2]-b[0]); bh=max(1e-6,b[3]-b[1])
        cw=max(1e-6,c[2]-c[0]); chh=max(1e-6,c[3]-c[1])
        ar.append(cw/chh); cbw.append(cw/bw); cbh.append(chh/bh)
        off.append(abs(((c[0]+c[2])/2-(b[0]+b[2])/2))/bw)
    st(ar,"capar_",d); st(cbw,"capbw_",d); st(cbh,"capbh_",d); st(off,"capoff_",d)
    # B 裁切后自参照
    st((GB[1:]*GB[0][None]).sum(-1),"selfb_",d)
    st((GC[1:]*GC[0][None]).sum(-1),"selfc_",d)
    # C 裁切后绝对还原(对该款官方视角)
    if sku in VS:
        st((GB@VS[sku]["body"].T).max(1),"absb_",d)
        st((GC@VS[sku]["cap"].T).max(1),"absc_",d)
        st((GB@VS[sku]["body"].T).mean(1),"absbm_",d)
    else:
        for pre in ("absb_","absc_","absbm_"): st([],pre,d)
    # D 检出健康度
    st([s for _,s in bb],"detb_",d); st([s for _,s in cc],"detc_",d)
    d["detb_fail"]=float(np.mean([b is None for b,_ in bb])); d["detc_fail"]=float(np.mean([c is None for c,_ in cc]))
    rows.append(d)
    if (k+1)%50==0: print(f"{k+1}/{len(vids)} ({time.time()-t0:.0f}s)",flush=True)
K=[k for k in rows[0] if k!="filename"]
with open(OUT/f"newref{TAG}_1233.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,["filename"]+K); w.writeheader(); w.writerows(rows)
print(f"DONE {len(rows)} 条 {len(K)} 列 → newref{TAG}_1233.csv  {time.time()-t0:.0f}s",flush=True)
