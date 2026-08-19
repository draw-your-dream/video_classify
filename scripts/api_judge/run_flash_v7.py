#!/usr/bin/env python3
"""v7:完整复刻历史配置 —— 旧 rubric 原文 + **真源图**作参考(缺图才退回视频首帧)+ LOW 分辨率 + 无官方形象图。
参考图来源:data/srcimgs/<image_sample_id>.png,映射表 data/api_judge_video_image_map.csv。
用法: python run_flash_v7.py <out.jsonl> [workers]"""
import json, os, sys, subprocess, time, threading, csv, shutil
from pathlib import Path
from multiprocessing.dummy import Pool
sys.path.insert(0, str(Path(__file__).parent))
import run_pilot as RP
ROOT=Path(os.environ.get("V7_ROOT","/workspace/r2")); D=ROOT/"data"
OUTF=Path(sys.argv[1]); W=int(sys.argv[2]) if len(sys.argv)>2 else 16
VID=D/"videos"; SRC=D/"srcimgs"; REFS=D/"refs_exact"; REFS.mkdir(exist_ok=True)
RP.MEDIA_RES="MEDIA_RESOLUTION_LOW"; RP.THINK="high"   # rubric 用 run_pilot 内置原文,不改
key=os.environ["GOOGLE_API_KEY"]; model="gemini-3.6-flash"
mapr={r["filename"]:r for r in csv.DictReader(open(D/"api_judge_video_image_map.csv",encoding="utf-8-sig"))}
rows=[r["filename"] for r in csv.DictReader(open(D/"tutu_task1_annotations_1233.csv",encoding="utf-8-sig"))]
done=set()
if OUTF.exists():
    for l in open(OUTF):
        try: done.add(json.loads(l)["filename"])
        except Exception: pass
todo=[f for f in rows if f not in done]
nex=sum(1 for f in rows if (SRC/((mapr.get(f,{}).get("image_sample_id") or "x")+".png")).exists())
print(f"待跑 {len(todo)}/{len(rows)},其中可用真源图 {nex} 条,并发 {W}",flush=True)
lock=threading.Lock(); fh=open(OUTF,"a"); t0=time.time(); n=[0]
def work(fn):
    vp=VID/fn; row={"filename":fn,"model":model,"rubric":"v1_builtin"}
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
                j,mv,us=RP.call_gemini(vp,model,key,5,rp if rp.exists() else None,None)
                row.update(result=j,model_version=mv,usage=us); break
            except Exception as e:
                if a==2: row["error"]=repr(e)[:200]
                else: time.sleep(5*(a+1))
    with lock:
        fh.write(json.dumps(row,ensure_ascii=False)+"\n"); fh.flush(); n[0]+=1
        if n[0]%50==0: print(f"[{n[0]}/{len(todo)}] {(time.time()-t0)/n[0]:.2f}s/条",flush=True)
Pool(W).map(work,todo)
print("V7_DONE",flush=True)
