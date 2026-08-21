#!/usr/bin/env python3
"""首帧上的变体对照:把对方 ip_score 的两路候选、局部放大、白底方形 pad 逐一补齐,
看哪一处是关键。只算首帧(1233 帧),每个变体几分钟。
列:
  v_box      = OWLv2 winner 框裸裁剪(= 现在的 raw_first,基线复现)
  v_pad      = 同框但白底方形 pad 到 768 再嵌入(补齐 _square_pad)
  v_local    = _owl_local_cands:外扩35%+局部放大到512+SAM+掩膜裁剪(补齐"专治小主体")
  v_seg      = _seg_cands:SAM 自动掩膜生成的候选(补齐另一路候选)
  v_ipscore  = 两路并集取最大(= 对方完整 ip_score)
  n_cand     = 候选数
  per-view   = 对该款 5 张官方视角逐一的余弦(用于分析哪些视角有用)
"""
import os, csv, cv2, sys, time
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import vendor_ipcheck as V
D=Path.home()/"tutu-video-eval/data"; OUT=D/"pbase/out"
LIMIT=int(os.environ.get("VAR_LIMIT","0")); TAG=os.environ.get("VAR_TAG","")
V.load(); REF=V.ref_embeddings(D/"sku_ref_v2/views")
print("参考:",{k:v.shape[0] for k,v in REF.items()},flush=True)
vids=sorted(p.name for p in (D/"videos").glob("*.mp4"))
if LIMIT: vids=vids[:LIMIT]
outp=OUT/f"ipvar{TAG}_1233.csv"
done=set()
if outp.exists():
    for r in csv.DictReader(open(outp,encoding="utf-8")): done.add(r["filename"])
    print(f"续跑 已有{len(done)}",flush=True)
rows=[]; t0=time.time()
for k,vn in enumerate(vids):
    if vn in done: continue
    sku=vn.split("__")[2] if len(vn.split("__"))>2 else ""
    R=REF.get(sku)
    if R is None: continue
    cap=cv2.VideoCapture(str(D/"videos"/vn)); ok,f=cap.read(); cap.release()
    if not ok: continue
    full=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    d={"filename":vn}
    try: boxes=V.owl_boxes(full)
    except Exception: boxes=[]
    boxes=[b for b in boxes if (b[2]-b[0])>2 and (b[3]-b[1])>2]
    if boxes:
        E=V.demb([full.crop(b) for b in boxes]); u=(E@R.T).max(1)
        i=int(np.argmax(u)); d["v_box"]=float(u.max())
        Ep=V.demb([V.square_pad(full,boxes[i])]); d["v_pad"]=float((Ep@R.T).max())
        pv=(Ep@R.T)[0]
        for j in range(min(5,len(pv))): d[f"view{j}"]=float(pv[j])
    else:
        d["v_box"]=np.nan; d["v_pad"]=np.nan
    try:
        lc=V.owl_local_cands(full)
        d["v_local"]=float((V.demb(lc)@R.T).max()) if lc else np.nan
        d["n_local"]=len(lc)
    except Exception: d["v_local"]=np.nan; d["n_local"]=0
    try:
        sc=V.seg_cands(V._small(full))
        d["v_seg"]=float((V.demb(sc)@R.T).max()) if sc else np.nan
        d["n_seg"]=len(sc)
    except Exception: d["v_seg"]=np.nan; d["n_seg"]=0
    d["v_ipscore"]=float(np.nanmax([d.get("v_local",np.nan), d.get("v_seg",np.nan)]))
    rows.append(d)
    if len(rows)%25==0:
        hdr=not outp.exists()
        with open(outp,"a",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,list(rows[0].keys()))
            if hdr: w.writeheader()
            w.writerows(rows)
        el=time.time()-t0
        print(f"{len(done)+len(rows)} 条 {el:.0f}s ({el/len(rows):.2f}s/条)",flush=True)
        done|={r["filename"] for r in rows}; rows=[]
if rows:
    hdr=not outp.exists()
    with open(outp,"a",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,list(rows[0].keys()))
        if hdr: w.writeheader()
        w.writerows(rows)
print(f"DONE {time.time()-t0:.0f}s",flush=True)
