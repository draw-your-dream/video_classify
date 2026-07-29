#!/usr/bin/env python3
"""角色检测+裁剪归一化 -> 身份/尺寸时序特征。GroundingDINO 文本提示检测 TUTU。"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
ROOT=Path("/root/tutu-video-eval"); sys.path.insert(0,str(ROOT/"src"))
from tutu_eval.io.video_loader import load_video
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModel, AutoImageProcessor

HC=str(ROOT/".hf_cache")
dproc=AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base",cache_dir=HC)
dmodel=AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base",cache_dir=HC).to("cuda").eval()
sproc=AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384",cache_dir=HC)
smodel=AutoModel.from_pretrained("google/siglip2-so400m-patch14-384",cache_dir=HC,torch_dtype=torch.float16).to("cuda").eval()
PROMPT="a mushroom character. a plush toy. a mushroom."
NF=16
def detect(imgs):
    boxes=[]
    for j in range(0,len(imgs),8):
        chunk=imgs[j:j+8]
        inp=dproc(images=chunk,text=[PROMPT]*len(chunk),return_tensors="pt").to("cuda")
        with torch.inference_mode(): out=dmodel(**inp)
        res=dproc.post_process_grounded_object_detection(out,inp.input_ids,threshold=0.25,
              text_threshold=0.25,target_sizes=[im.size[::-1] for im in chunk])
        for r,im in zip(res,chunk):
            if len(r["boxes"])==0: boxes.append(None)
            else: boxes.append(r["boxes"][int(r["scores"].argmax())].cpu().numpy())
    return boxes
def crop(im,b,marg=0.25):
    W,H=im.size
    if b is None: s=min(W,H); return im.crop(((W-s)//2,(H-s)//2,(W+s)//2,(H+s)//2))
    x0,y0,x1,y1=b; cx,cy=(x0+x1)/2,(y0+y1)/2; s=max(x1-x0,y1-y0)*(1+marg)
    s=max(s,32)
    return im.crop((max(0,cx-s/2),max(0,cy-s/2),min(W,cx+s/2),min(H,cy+s/2)))
@torch.inference_mode()
def embed(crops):
    out=[]
    for j in range(0,len(crops),16):
        inp=sproc(images=crops[j:j+16],return_tensors="pt").to("cuda")
        inp["pixel_values"]=inp["pixel_values"].half()
        out.append(smodel.vision_model(**inp).pooler_output.float().cpu().numpy())
    return np.concatenate(out,0)

def run(vdir, stems, outp):
    recs=[]; t0=time.time()
    for i,s in enumerate(stems,1):
        v=load_video(str(Path(vdir)/f"{s}.mp4"),sparse_count=NF,sparse_short_side=720)
        imgs=[Image.fromarray(f) for f in v.frames_sparse]
        bs=detect(imgs)
        det=[b for b in bs if b is not None]
        areas=np.array([ (b[2]-b[0])*(b[3]-b[1]) for b in det]) if det else np.array([0.0])
        W,H=imgs[0].size; areas=areas/(W*H)
        E=embed([crop(im,b) for im,b in zip(imgs,bs)])
        N=E/np.linalg.norm(E,axis=1,keepdims=True).clip(1e-9)
        cons=(N[:-1]*N[1:]).sum(1); c=N.mean(0); c/=np.linalg.norm(c).clip(1e-9); ss=N@c
        recs.append(dict(stem=s, det_rate=len(det)/len(bs),
            area_mean=float(areas.mean()), area_cv=float(areas.std()/(areas.mean()+1e-9)),
            area_max_ratio=float(areas.max()/(areas.min()+1e-9)) if len(areas)>1 else 1.0,
            c_cons_min=float(cons.min()), c_cons_bot3=float(np.sort(cons)[:3].mean()),
            c_cons_mean=float(cons.mean()), c_cons_std=float(cons.std()),
            c_self_min=float(ss.min()), c_self_std=float(ss.std()),
            c_first_last=float(N[0]@N[-1])))
        np.save(Path(outp).parent/f"cropemb_{s}.npy", E) if False else None
        if i%50==0: print(f"  {i}/{len(stems)} {time.time()-t0:.0f}s",flush=True)
    pd.DataFrame(recs).to_csv(outp,index=False); print("写出",outp,flush=True)

P=pd.read_csv(ROOT/"data/prod500/prod500.csv")
run(ROOT/"data/prod500/videos", list(P.stem), "/root/prod_crop.csv")
