#!/usr/bin/env python3
"""v8 = 冠军配置的精确复刻(依据 out_holdout_full.jsonl / out_run2_s*.jsonl 的 token 画像反推):
  MEDIA_RESOLUTION_LOW + rubric_v2_withsku + 3 张官方形象图 + 参考图(有真源图用真源图,否则视频首帧)
目标 token 画像:total 3325-3412 / IMAGE 1044-1065 / TEXT 574-580 / VIDEO 1700-1775
用法: python run_flash_v8.py <out.jsonl> [workers]"""
import json, os, sys, subprocess, time, threading, csv, shutil
from pathlib import Path
from multiprocessing.dummy import Pool
sys.path.insert(0, str(Path(__file__).parent))
import run_pilot as RP
ROOT=Path(os.environ.get("V8_ROOT","/workspace/r2")); D=ROOT/"data"
OUTF=Path(sys.argv[1]); W=int(sys.argv[2]) if len(sys.argv)>2 else 16
VID=D/"videos"; SRC=D/"srcimgs"; REFS=D/"refs_exact"; REFS.mkdir(exist_ok=True)
SKU=str(D/"sku_ref_v2/views")
RP.MEDIA_RES="MEDIA_RESOLUTION_LOW"; RP.THINK="high"
RP.RUBRIC=open(Path(__file__).parent/"rubric_v2_withsku.txt",encoding="utf-8").read()
key=os.environ["GOOGLE_API_KEY"]; model="gemini-3.6-flash"
mapr={r["filename"]:r for r in csv.DictReader(open(D/"api_judge_video_image_map.csv",encoding="utf-8-sig"))}
rows=[r["filename"] for r in csv.DictReader(open(D/"tutu_task1_annotations_1233.csv",encoding="utf-8-sig"))]
done=set()
if OUTF.exists():
    for l in open(OUTF):
        try: done.add(json.loads(l)["filename"])
        except Exception: pass
todo=[f for f in rows if f not in done]
print(f"待跑 {len(todo)}/{len(rows)},并发 {W},官方形象图目录 {SKU}",flush=True)
lock=threading.Lock(); fh=open(OUTF,"a"); t0=time.time(); n=[0]
def work(fn):
    vp=VID/fn; row={"filename":fn,"model":model,"rubric":"v2_withsku","media":"LOW"}
    sid=(mapr.get(fn,{}) or {}).get("image_sample_id") or ""
    sp=SRC/(sid+".png") if sid else None
    rp=REFS/(fn[:-4]+".png")
    if sp is not None and sp.exists():
        if not rp.exists():
            try: shutil.copy(sp,rp)
            except Exception: pass
        row["ref"]="exact"
    else:
        if not rp.exists():
            subprocess.run(["ffmpeg","-loglevel","error","-y","-i",str(vp),"-frames:v","1",str(rp)],
                           check=False,capture_output=True)
        row["ref"]="frame0"
    if not vp.exists(): row["error"]="missing_video"
    else:
        for a in range(3):
            try:
                j,mv,us=RP.call_gemini(vp,model,key,5,rp if rp.exists() else None,SKU)
                row.update(result=j,model_version=mv,usage=us); break
            except Exception as e:
                if a==2: row["error"]=repr(e)[:200]
                else: time.sleep(5*(a+1))
    with lock:
        fh.write(json.dumps(row,ensure_ascii=False)+"\n"); fh.flush(); n[0]+=1
        if n[0]%50==0: print(f"[{n[0]}/{len(todo)}] {(time.time()-t0)/n[0]:.2f}s/条",flush=True)
Pool(W).map(work,todo)
print("V8_DONE",flush=True)
