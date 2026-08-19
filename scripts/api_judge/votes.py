import csv,json,numpy as np,itertools
from pathlib import Path
from scipy.stats import rankdata
D=Path.home()/'tutu-video-eval/data'; OUT=D/'pbase/out'
def br_at(s,y,rel=0.8):
    gn=np.sort(s[y==0]); b=s[y==1]; k=int(np.floor(rel*len(gn)))
    t=gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return float(((b>t).sum()+(b==t).sum()*(1-fr))/len(b))
def rp(x): return rankdata(x)/len(x)
vids=[v if v.endswith('.mp4') else v+'.mp4' for v in json.load(open(OUT/'X303_vids.json'))]
mapr={r['filename']:r for r in csv.DictReader(open(D/'api_judge_video_image_map.csv',encoding='utf-8-sig'))}
y=np.array([1 if mapr[v]['grade']=='bad' else 0 for v in vids])
s1=json.load(open(OUT/'flash_full_1233.json')); s2=json.load(open(OUT/'flash_run2_1233.json'))
v8={}
for l in open(OUT/'flash_v8_raw.jsonl'):
    o=json.loads(l); r=o.get('result') or {}
    if 'bad_score' in r: v8[o['filename']]=r['bad_score']
R={'run1':np.array([s1.get(v,np.nan) for v in vids],float),
   'run2':np.array([s2.get(v,np.nan) for v in vids],float),
   'run3v8':np.array([v8.get(v,np.nan) for v in vids],float)}
for k in R: R[k]=np.where(np.isnan(R[k]),np.nanmedian(R[k]),R[k])
print('单遍 br@80:', {k:round(br_at(rp(v),y),4) for k,v in R.items()})
for k,v in R.items():
    m=(v==0)&(y==1)
    print(f'  {k}: 真bad打0分 {m.sum()}/{int(y.sum())} ({m.mean()*100:.0f}%) ; 真bad>70分 {int(((v>70)&(y==1)).sum())}')
hi=np.stack([(v>70).astype(int) for v in R.values()],1); cnt=hi[y==1].sum(1)
print('  真bad 被>70抓到的遍数分布:', {int(i):int((cnt==i).sum()) for i in range(4)})
h0=np.stack([(v==0).astype(int) for v in R.values()],1); c0=h0[y==1].sum(1)
print('  真bad 被打0分的遍数分布:', {int(i):int((c0==i).sum()) for i in range(4)})
gn=np.stack([(v>70).astype(int) for v in R.values()],1); cg=gn[y==0].sum(1)
print('  合格视频 被>70误判的遍数分布:', {int(i):int((cg==i).sum()) for i in range(4)})
ks=list(R.values())
for k in (1,2,3):
    vals=[br_at(np.mean([rp(ks[i]) for i in c],0),y) for c in itertools.combinations(range(3),k)]
    print(f'  {k} 票秩均值 br@80 均值 {np.mean(vals):.4f}  各组合 {[round(x,4) for x in vals]}')
