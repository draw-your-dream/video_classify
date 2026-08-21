#!/usr/bin/env python3
"""ipfid:把对方 repo 的「形象还原度」打分逐帧跑成时序特征。
每帧:OWLv2 定框 → DINOv2 粗筛 top3 → SAM 出掩膜 → 掩膜外涂白裁剪 → 对该款官方视角取最大余弦。
产出列:
  fid_*    掩膜裁剪对官方参考的余弦(还原度主分)轨迹统计
  raw_*    未抠图框的余弦轨迹(对照:抠图带来多少增益)
  gap_*    fid - raw(定位/抠图健康度)
  self_*   掩膜裁剪_t 对掩膜裁剪_0 的余弦(裁切后自参照漂移)
  cap*     winner 掩膜剪影的伞盖几何(宽/高、伞盖宽/身体宽、张开度)——对方判据的可度量版
  box*     本体框面积/长宽比/中心轨迹
环境:~/.venvs/tutu-ex/bin/python;必走 gpu run。"""
import os, csv, cv2, time
import numpy as np
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent))
import vendor_ipcheck as V
D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
NF=int(os.environ.get("IPF_FRAMES","6")); LIMIT=int(os.environ.get("IPF_LIMIT","0"))
TAG=os.environ.get("IPF_TAG",""); START=int(os.environ.get("IPF_START","0"))
V.load(); print("模型就绪",flush=True)
REF=V.ref_embeddings(D/"sku_ref_v2/views")
print("参考嵌入:",{k:v.shape[0] for k,v in REF.items()},flush=True)

def frames(vp,n):
    cap=cv2.VideoCapture(str(vp)); T=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out=[]
    if T>0:
        for i in np.unique(np.linspace(0,T-1,n).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,f=cap.read()
            if ok: out.append(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)))
    cap.release(); return out
def st(v,pre,d):
    v=np.asarray([x for x in v if x is not None and np.isfinite(x)],float)
    if len(v)==0:
        for k in ("mean","min","max","first","last","drop","std","slope"): d[pre+k]=np.nan
        return
    x=np.arange(len(v))
    d[pre+"mean"]=v.mean(); d[pre+"min"]=v.min(); d[pre+"max"]=v.max()
    d[pre+"first"]=v[0]; d[pre+"last"]=v[-1]; d[pre+"drop"]=v[0]-v[-1]
    d[pre+"std"]=v.std(); d[pre+"slope"]=float(np.polyfit(x,v,1)[0]) if len(v)>1 else 0.0
def cap_geom(m):
    """winner 掩膜剪影 → 伞盖几何(对方的 cap_ok 判据:宽>高的扁盖 vs 高≈宽的圆球/缩到与身体同宽)"""
    if m is None: return None
    mm=V.fill_holes(np.asarray(m,bool)); ys,xs=np.where(mm)
    if len(xs)<50: return None
    t,b=int(ys.min()),int(ys.max()); H=b-t+1
    if H<8: return None
    w=mm[t:b+1].sum(1).astype(float)
    lo,hi=int(0.20*H),max(int(0.75*H),int(0.20*H)+1)
    nk=lo+int(np.argmin(w[lo:hi])); nw=max(1.0,w[nk])
    cw=w[:nk+1].max(); ch=nk+1; bw=w[nk:].max() if nk<len(w) else cw
    return dict(ar=cw/max(1.0,ch), cb=cw/max(1.0,bw), hf=ch/H, flare=cw/nw,
                sol=mm.sum()/max(1.0,(xs.max()-xs.min()+1)*H))
vids=sorted(p.name for p in (D/"videos").glob("*.mp4"))
if START: vids=vids[START:]
if LIMIT: vids=vids[:LIMIT]
outp=OUT/f"ipfid{TAG}_1233.csv"
done=set()
if outp.exists():
    for r in csv.DictReader(open(outp,encoding="utf-8")): done.add(r["filename"])
    print(f"续跑:已有 {len(done)} 条",flush=True)
rows=[]; t0=time.time(); nsk=0
for k,vn in enumerate(vids):
    if vn in done: continue
    sku=vn.split("__")[2] if len(vn.split("__"))>2 else ""
    R=REF.get(sku)
    if R is None: nsk+=1; continue
    fr=frames(D/"videos"/vn,NF)
    if len(fr)<3: nsk+=1; continue
    res=[V.best_subject(f,R) for f in fr]
    ok=[r for r in res if r is not None]
    if not ok: nsk+=1; continue
    d={"filename":vn,"n_ok":len(ok),"n_frames":len(fr)}
    st([r["dino"] for r in ok],"fid_",d)
    st([r["u_max"] for r in ok],"raw_",d)
    st([r["dino"]-r["u_max"] for r in ok],"gap_",d)
    crops=[r["crop"] for r in ok if r["crop"] is not None]
    if len(crops)>=2:
        E=V.demb(crops); st((E[1:]*E[0][None]).sum(-1),"self_",d)
    else: st([],"self_",d)
    g=[cap_geom(r["mask"]) for r in ok]; g=[x for x in g if x]
    for key,pre in (("ar","capar_"),("cb","capcb_"),("hf","caphf_"),("flare","capfl_"),("sol","capsol_")):
        st([x[key] for x in g],pre,d)
    W,H=fr[0].size
    st([( (r["body"][2]-r["body"][0])*(r["body"][3]-r["body"][1]) )/(W*H) for r in ok],"boxarea_",d)
    st([ (r["body"][2]-r["body"][0])/max(1,(r["body"][3]-r["body"][1])) for r in ok],"boxar_",d)
    st([ ((r["body"][0]+r["body"][2])/2)/W for r in ok],"boxcx_",d)
    rows.append(d)
    if len(rows)%25==0:
        hdr=not outp.exists()
        with open(outp,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,list(rows[0].keys()))
            if hdr: w.writeheader()
            w.writerows(rows)
        el=time.time()-t0
        print(f"{len(done)+len(rows)}/{len(vids)+len(done)} 用时{el:.0f}s ({el/max(1,len(rows)):.2f}s/条) 跳过{nsk}",flush=True)
        done|={r["filename"] for r in rows}; rows=[]
if rows:
    hdr=not outp.exists()
    with open(outp,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(rows[0].keys()))
        if hdr: w.writeheader()
        w.writerows(rows)
print(f"DONE 写入 {outp} 跳过{nsk} 总用时{time.time()-t0:.0f}s",flush=True)
