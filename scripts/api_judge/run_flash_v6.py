#!/usr/bin/env python3
"""flash v5 问法:注入完整分级标准(硬规则/排查顺序/不扣分清单/标签判据/风格区分)+ 五维度严重度输出。
配置与前两遍对齐:MEDIA_RESOLUTION_LOW、仅首帧参考图、不带官方形象图。
用法: python run_flash_v5.py <out.jsonl> [workers]"""
import json, os, sys, subprocess, time, threading
from pathlib import Path
from multiprocessing.dummy import Pool
sys.path.insert(0, str(Path(__file__).parent))
import run_pilot as RP
ROOT=Path(os.environ.get("V5_ROOT","/workspace/r2")); D=ROOT/"data"
OUTF=Path(sys.argv[1]); W=int(sys.argv[2]) if len(sys.argv)>2 else 16
VID=D/"videos"; REFS=D/"frames0"; REFS.mkdir(exist_ok=True)
RP.MEDIA_RES="MEDIA_RESOLUTION_LOW"; RP.THINK="high"
RP.RUBRIC=open(Path(__file__).parent/"rubric_v6.txt",encoding="utf-8").read()
key=os.environ["GOOGLE_API_KEY"]; model="gemini-3.6-flash"
import csv
rows=[r["filename"] for r in csv.DictReader(open(D/"tutu_task1_annotations_1233.csv",encoding="utf-8-sig"))]
done=set()
if OUTF.exists():
    for l in open(OUTF):
        try: done.add(json.loads(l)["filename"])
        except Exception: pass
todo=[f for f in rows if f not in done]
print(f"待跑 {len(todo)}/{len(rows)},并发 {W}",flush=True)
lock=threading.Lock(); fh=open(OUTF,"a"); t0=time.time(); n=[0]
def work(fn):
    vp=VID/fn; row={"filename":fn,"model":model,"rubric":"v6"}
    rp=REFS/(fn[:-4]+".png")
    if not rp.exists():
        subprocess.run(["ffmpeg","-loglevel","error","-y","-i",str(vp),"-frames:v","1",str(rp)],
                       check=False,capture_output=True)
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
print("V6_DONE",flush=True)
