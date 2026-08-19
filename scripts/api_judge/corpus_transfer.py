#!/usr/bin/env python3
"""把 2.0 上的融合配方搬到旧 0430 语料:在 eval 968 内部做交叉验证,
基线 = [p_base(train上训的LGBM分) ⊕ 15专家OOF],逐个加旧语料既有信号块,看 br@80 与 ev@95 的增量。
所有 oof15/p_base 都由 train 侧模型产出,对 eval 无泄漏;比较在同一折下进行。"""
import csv, json, pickle, sys
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
R=Path.home()/"tutu-video-eval"; S3=R/"data/s3"
def br_at(p,y,rel=0.8):
    gn=np.sort(p[y==0]); b=p[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return float(((b>t).sum()+(b==t).sum()*(1-fr))/len(b))
def ev_at(p,y,rec=0.95):
    b=np.sort(p[y==1]); T=b[int(np.floor((1-rec)*len(b)))]
    return float((p[y==0]<T).mean())
def auc(p,y):
    r=rankdata(p); pos=r[y==1]; return float((pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum()))
def rp(x):
    x=np.asarray(x,float); x=np.nan_to_num(x,nan=np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0,
                                            posinf=0.0,neginf=0.0)
    return rankdata(x)/len(x)
ev=[json.loads(l) for l in open(R/"splits/eval_v3.jsonl")]
def relkey(a):
    p=a.split("/data/s3/")[-1]
    return p.replace("v-0430-skus/","").replace("v-0430-ti2i2v/","ti2i2v/").replace("v-0430-rlhf/","rlhf/").replace("rlhf-0430/","rlhf/")
rels=[relkey(e["abs_path"]) for e in ev]
y=np.array([1 if e["label"]=="bad" else 0 for e in ev]); N=len(y)
o_tr,o_ev,ytr,yev,_=pickle.load(open(R/"upstream/cache_v3/_stack_15expert.pkl","rb"))
assert (yev==y).all()
pb=np.load(R/"data/pbase/out/corpus_base_eval_p.npy")
def load(fn,sep=","):
    f=S3/fn
    if not f.exists(): return None,None
    rows=list(csv.DictReader(open(f),delimiter=sep)); key=rows[0].get("rel") and "rel" or list(rows[0])[0]
    cols=[k for k in rows[0] if k!=key and not any(x in (k or "") for x in ("frames","list","ids"))]
    idx={r[key]:r for r in rows}
    M=np.zeros((N,len(cols))); hit=0
    for i,rl in enumerate(rels):
        r=idx.get(rl)
        if r:
            hit+=1
            vals=[]
            for c in cols:
                try: vals.append(float(r[c]))
                except Exception: vals.append(0.0)
            M[i]=vals
    return (M if hit>N*0.5 else None), (f"{fn} {len(cols)}列 覆盖{hit}/{N}")
BLOCKS={}
for fn,sep in (("e22_drift.csv",","),("corpus_new_feats.tsv","\t"),("w5_dover.csv",","),("w2_cotracker.csv",","),
               ("w3_raft.csv",","),("w1_qalign.csv",","),("w4_pyiqa.csv",","),("brow_scan.csv",","),
               ("face_act_full.csv",","),("bbox_h.csv",","),("tail_pink.csv",","),("e21_bitstream.csv",","),
               ("e23_depth.csv",",")):
    M,info=load(fn,sep)
    print(("  ✓ " if M is not None else "  ✗ ")+str(info),flush=True)
    if M is not None: BLOCKS[fn.split(".")[0]]=M
base=[rp(pb)]+[rp(o_ev[:,j]) for j in range(o_ev.shape[1])]
def run(extra,seeds=range(42,50)):
    X=np.stack(base+[rp(c) for c in extra],1); oofs=[]
    for sd in seeds:
        oof=np.zeros(N)
        for tr,te in StratifiedKFold(10,shuffle=True,random_state=sd).split(X,y):
            sc=StandardScaler().fit(X[tr])
            m=LogisticRegression(C=100,max_iter=6000).fit(sc.transform(X[tr]),y[tr])
            oof[te]=m.predict_proba(sc.transform(X[te]))[:,1]
        oofs.append(oof)
    per=[br_at(o,y) for o in oofs]; bag=np.mean([rp(o) for o in oofs],0)
    return np.mean(per),br_at(bag,y),ev_at(bag,y),auc(bag,y)
print("\n=== eval968 内部 10 折 × 8 种子 ===")
m,b,e,a=run([]); print(f"  {'基线(p_base⊕oof15)':26s} 单{m:.4f} 装袋 br@80={b:.4f} ev@95={e:.4f} AUC={a:.4f}",flush=True)
best=[]
for nm,M in BLOCKS.items():
    cols=[M[:,j] for j in range(M.shape[1])]
    m2,b2,e2,a2=run(cols)
    print(f"  +{nm:25s} 单{m2:.4f} 装袋 br@80={b2:.4f} ev@95={e2:.4f} AUC={a2:.4f}  Δ单{m2-m:+.4f}",flush=True)
    if m2>m: best.append((m2-m,nm,M))
best.sort(reverse=True)
if best:
    print("\n=== 累加过线块 ===")
    acc=[]; cur=m
    for d,nm,M in best:
        trial=acc+[M[:,j] for j in range(M.shape[1])]
        m2,b2,e2,a2=run(trial)
        print(f"  +{nm:25s} → 单{m2:.4f} 装袋 br@80={b2:.4f} ev@95={e2:.4f} {'收' if m2>cur else '弃'}",flush=True)
        if m2>cur: acc=trial; cur=m2
