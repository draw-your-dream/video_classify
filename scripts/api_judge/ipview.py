#!/usr/bin/env python3
"""首帧裸框裁剪对该款每一张官方视角的余弦(逐张存),用于验证参考集的视角取舍。
动机:对方 _collect_refs 有排除约定(排除 _2/_5、含 _6),说明参考集里掺进无用视角会拉低 max。"""
import os, csv, cv2, sys, time
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import vendor_ipcheck as V
D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
V.load(); REF=V.ref_embeddings(D/"sku_ref_v2/views")
NV=max(v.shape[0] for v in REF.values())
vids=sorted(p.name for p in (D/"videos").glob("*.mp4"))
outp=OUT/"ipview_1233.csv"; done=set()
if outp.exists():
    for r in csv.DictReader(open(outp,encoding="utf-8")): done.add(r["filename"])
rows=[]; t0=time.time()
for k,vn in enumerate(vids):
    if vn in done: continue
    sku=vn.split("__")[2] if len(vn.split("__"))>2 else ""
    R=REF.get(sku)
    if R is None: continue
    cap=cv2.VideoCapture(str(D/"videos"/vn)); ok,f=cap.read(); cap.release()
    if not ok: continue
    full=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    try: boxes=[b for b in V.owl_boxes(full) if (b[2]-b[0])>2 and (b[3]-b[1])>2]
    except Exception: boxes=[]
    if not boxes: continue
    E=V.demb([full.crop(b) for b in boxes])
    S=E@R.T                       # (框数, 视角数)
    i=int(S.max(1).argmax())      # winner 框 = 对任一视角最像的那个
    d={"filename":vn,"sku":sku}
    for j in range(NV): d[f"bv{j}"]=float(S[i,j]) if j<S.shape[1] else np.nan
    rows.append(d)
    if len(rows)%50==0:
        hdr=not outp.exists()
        with open(outp,"a",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,list(rows[0].keys()))
            if hdr: w.writeheader()
            w.writerows(rows)
        print(f"{len(done)+len(rows)} 条 {time.time()-t0:.0f}s",flush=True)
        done|={r["filename"] for r in rows}; rows=[]
if rows:
    hdr=not outp.exists()
    with open(outp,"a",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,list(rows[0].keys()))
        if hdr: w.writeheader()
        w.writerows(rows)
print(f"DONE {time.time()-t0:.0f}s",flush=True)
