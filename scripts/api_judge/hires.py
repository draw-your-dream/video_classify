import csv,json,collections,numpy as np,itertools
from pathlib import Path
from scipy.stats import rankdata,spearmanr
D=Path.home()/'tutu-video-eval/data'; OUT=D/'pbase/out'
def br_at(s,y,rel=0.8):
    gn=np.sort(s[y==0]); b=s[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return float(((b>t).sum()+(b==t).sum()*(1-fr))/len(b))
def auc(s,y):
    r=rankdata(s); pos=r[y==1]; return float((pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(y==0).sum()))
def rp(x): return rankdata(x)/len(x)
vids=[v if v.endswith('.mp4') else v+'.mp4' for v in json.load(open(OUT/'X303_vids.json'))]
mapr={r['filename']:r for r in csv.DictReader(open(D/'api_judge_video_image_map.csv',encoding='utf-8-sig'))}
y=np.array([1 if mapr[v]['grade']=='bad' else 0 for v in vids])
def raw(f):
    m={}
    for l in open(f,encoding='utf-8'):
        try:o=json.loads(l)
        except:continue
        r=o.get('result') or {}
        if 'bad_score' in r: m[o['filename']]=r['bad_score']
    return np.array([m.get(v,np.nan) for v in vids],float)
s1=json.load(open(OUT/'flash_full_1233.json')); s2=json.load(open(OUT/'flash_run2_1233.json'))
S={'run1(LOW,官方图)':np.array([s1.get(v,np.nan) for v in vids],float),
   'run2(LOW,官方图)':np.array([s2.get(v,np.nan) for v in vids],float),
   'v8(LOW,官方图)':raw(OUT/'flash_v8_raw.jsonl'),
   'HIGH+官方图(原判作废)':raw(OUT/'flash_run3_raw.jsonl'),
   'LOW无官方图(原判作废)':raw(OUT/'flash_run3b_raw.jsonl')}
for k in S: S[k]=np.where(np.isnan(S[k]),np.nanmedian(S[k]),S[k])
z0=np.all(np.stack([S[k]==0 for k in ['run1(LOW,官方图)','run2(LOW,官方图)','v8(LOW,官方图)']],1),1)
huan=np.array(['还原度' in (mapr[v]['reasons'] or '') for v in vids])
print(f"{'':24s} {'br@80':>7s} {'AUC':>7s}  0分块n  三遍全漏208条中它抓到(>70)  还原度bad召回(>70)")
for k,v in S.items():
    tgt=(y==1)&z0
    rec=int(((v>70)&tgt).sum())
    hr=((v>70)&(y==1)&huan).sum()/max(1,((y==1)&huan).sum())
    print(f'{k:24s} {br_at(rp(v),y):7.4f} {auc(v,y):7.4f}  {int((v==0).sum()):5d}   {rec:4d}/208 ({rec/208*100:3.0f}%)      {hr*100:4.1f}%')
print()
for k in ['HIGH+官方图(原判作废)','LOW无官方图(原判作废)']:
    print(f'  {k} 与 run1 秩相关 {spearmanr(S[k],S["run1(LOW,官方图)"]).statistic:.3f}')
# 加进三票均值看看
base=[S['run1(LOW,官方图)'],S['run2(LOW,官方图)'],S['v8(LOW,官方图)']]
A3=np.mean([rp(x) for x in base],0)
print(f'\n  三票均值 br@80 = {br_at(A3,y):.4f}')
for k in ['HIGH+官方图(原判作废)','LOW无官方图(原判作废)']:
    A4=np.mean([rp(x) for x in base]+[rp(S[k])],0)
    print(f'  + {k} 作第四票 → {br_at(A4,y):.4f}')
