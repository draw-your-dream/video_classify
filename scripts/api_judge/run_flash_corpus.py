#!/usr/bin/env python3
"""旧 0430 语料 eval 968 的 flash 单遍(只带首帧参考图,老款式没有官方形象图)。
用法: python run_flash_corpus.py <out.jsonl> [workers]"""
import json, os, sys, subprocess, time, threading
from pathlib import Path
from multiprocessing.dummy import Pool
sys.path.insert(0, str(Path(__file__).parent))
import run_pilot as RP
ROOT=Path("/workspace/r2"); D=ROOT/"data"
OUTF=Path(sys.argv[1]); W=int(sys.argv[2]) if len(sys.argv)>2 else 12
VR=D/"corpus_eval"; REFS=D/"corpus_frames0"; REFS.mkdir(exist_ok=True)
RP.MEDIA_RES="MEDIA_RESOLUTION_HIGH"; RP.THINK="high"
key=os.environ["GOOGLE_API_KEY"]; model="gemini-3.6-flash"
items=[json.loads(l) for l in open(ROOT/"splits/eval_v3.jsonl")]
done=set()
if OUTF.exists():
    for l in open(OUTF):
        try: done.add(json.loads(l)["video"])
        except Exception: pass
todo=[e for e in items if e["video"] not in done]
print(f"待跑 {len(todo)}/{len(items)},并发 {W}",flush=True)
lock=threading.Lock(); fh=open(OUTF,"a"); t0=time.time(); n=[0]
def work(e):
    rel=e["abs_path"].split("/data/s3/")[-1]; vp=VR/rel
    row={"video":e["video"],"rel":rel,"label":e.get("label",""),"model":model}
    rp=REFS/(e["video"][:-4]+".png")
    if not rp.exists():
        subprocess.run(["ffmpeg","-loglevel","error","-y","-i",str(vp),"-frames:v","1",str(rp)],
                       check=False,capture_output=True)
    if not vp.exists(): row["error"]="missing_video"
    else:
        for a in range(3):
            try:
                j,mv,us=RP.call_gemini(vp,model,key,5,rp if rp.exists() else None,None)
                row.update(result=j,model_version=mv,usage=us); break
            except Exception as ex:
                if a==2: row["error"]=repr(ex)[:200]
                else: time.sleep(5*(a+1))
    with lock:
        fh.write(json.dumps(row,ensure_ascii=False)+"\n"); fh.flush(); n[0]+=1
        if n[0]%50==0: print(f"[{n[0]}/{len(todo)}] {(time.time()-t0)/n[0]:.2f}s/条",flush=True)
Pool(W).map(work,todo)
print("CORPUS_FLASH_DONE",flush=True)
