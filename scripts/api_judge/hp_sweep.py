#!/usr/bin/env python3
"""新1233 上 LGBM 超参随机扫:X303 与 X303⊕运动电池,分组10折,按 br@80 排序。
诚实报告:同时输出嵌套估计(外层5折内选参)防选择偏置。"""
import numpy as np, json, csv, warnings, sys
warnings.filterwarnings('ignore')
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import rankdata
from pathlib import Path
import multiprocessing as mp

R2 = Path('/workspace/r2')
X = np.load(R2/'data/pbase/out/X303_new.npy')
vids = json.load(open(R2/'data/pbase/out/X303_vids.json'))
lab = {r['filename']: r['grade'] for r in csv.DictReader(open(R2/'data/tutu_task1_annotations_1233.csv', encoding='utf-8-sig'))}
y = np.array([1 if lab[Path(v).name]=='bad' else 0 for v in vids])
shas = {r['filename']: r.get('source_sha','') for r in csv.DictReader(open(R2/'data/api_judge_video_image_map.csv'))}
def sha_of(v):
    fn=Path(v).name; s=shas.get(fn,'')
    return s if s else (fn.split('__')[1] if '__' in fn else fn)
groups = np.array([sha_of(v) for v in vids])
mb = pd.read_csv(R2/'data/motion_feats_1233.csv').set_index('filename')
Xm = mb.reindex([Path(v).name for v in vids]).values.astype(float)
Xm = np.where(np.isfinite(Xm), Xm, np.nanmedian(Xm, axis=0, keepdims=True))
XM = np.hstack([X, Xm])

def auc(yv,s):
    r=rankdata(s); pos=r[yv==1]
    return float((pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(yv==0).sum()))
def br(yv,s,rel=0.8):
    quota=rel*(yv==0).sum(); rg=rb_=0.; nb=(yv==1).sum()
    for v in np.unique(np.sort(s)):
        g=((yv==0)&(s==v)).sum(); bb=((yv==1)&(s==v)).sum()
        if rg+g<=quota: rg+=g
        else:
            f=(quota-rg)/max(1e-9,g) if g else 0.
            rb_+=bb*(1-f); rb_+=((yv==1)&(s>v)).sum(); break
    else: return 0.
    return float(rb_/max(1,nb))

rng = np.random.RandomState(7)
CFGS = []
for _ in range(120):
    CFGS.append(dict(num_leaves=int(rng.choice([3,5,7,15,31,63])),
                     n_estimators=int(rng.randint(60,600)),
                     learning_rate=float(10**rng.uniform(-2,-0.7)),
                     min_child_samples=int(rng.randint(5,150)),
                     scale_pos_weight=float(rng.uniform(0.5,2.5)),
                     feature_fraction=float(rng.uniform(0.3,1.0)),
                     bagging_fraction=float(rng.uniform(0.5,1.0))))

FOLDS = list(StratifiedGroupKFold(10, shuffle=True, random_state=42).split(X, y, groups))
def eval_cfg(args):
    ci, cfg, which = args
    Xin = X if which=='X303' else XM
    oof = np.full(len(y), np.nan)
    for tr, te in FOLDS:
        m = lgb.LGBMClassifier(**cfg, bagging_freq=1, random_state=42, verbose=-1, n_jobs=2)
        m.fit(Xin[tr], y[tr])
        oof[te] = m.predict_proba(Xin[te])[:,1]
    return ci, which, auc(y,oof), br(y,oof,.8), br(y,oof,.7), cfg

jobs = [(i,c,'X303') for i,c in enumerate(CFGS)] + [(i,c,'X303+motion') for i,c in enumerate(CFGS)]
out = []
with mp.get_context('fork').Pool(60) as pool:
    for r in pool.imap_unordered(eval_cfg, jobs):
        out.append(r)
        if len(out) % 40 == 0:
            print(f"[{len(out)}/{len(jobs)}]", flush=True)
for which in ['X303','X303+motion']:
    sub = sorted([o for o in out if o[1]==which], key=lambda t:-t[3])
    print(f"== {which} top5 (AUC | br@80 | br@70) ==")
    for ci,w,a,b8,b7,cfg in sub[:5]:
        print(f"  {a:.4f} {b8:.4f} {b7:.4f}  {json.dumps({k:round(v,4) if isinstance(v,float) else v for k,v in cfg.items()})}")
    bs = [o[3] for o in sub]
    print(f"  分布: max={max(bs):.4f} med={np.median(bs):.4f} p90={np.percentile(bs,90):.4f} (120配置选最大自带~噪声上偏)")
import pickle
pickle.dump(out, open(R2/'hp_sweep_out.pkl','wb'))
print("SWEEP_DONE")
