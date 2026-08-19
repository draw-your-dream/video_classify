import csv,json,collections,numpy as np
from pathlib import Path
D=Path.home()/'tutu-video-eval/data'; OUT=D/'pbase/out'
vids=[v if v.endswith('.mp4') else v+'.mp4' for v in json.load(open(OUT/'X303_vids.json'))]
mapr={r['filename']:r for r in csv.DictReader(open(D/'api_judge_video_image_map.csv',encoding='utf-8-sig'))}
y=np.array([1 if mapr[v]['grade']=='bad' else 0 for v in vids])
s1=json.load(open(OUT/'flash_full_1233.json')); s2=json.load(open(OUT/'flash_run2_1233.json'))
v8={}
for l in open(OUT/'flash_v8_raw.jsonl'):
    o=json.loads(l); r=o.get('result') or {}
    if 'bad_score' in r: v8[o['filename']]=r['bad_score']
S=[np.array([d.get(v,np.nan) for v in vids],float) for d in (s1,s2,v8)]
S=[np.where(np.isnan(x),np.nanmedian(x),x) for x in S]
z0=np.all(np.stack([x==0 for x in S],1),1)          # 三遍全打 0
never=np.all(np.stack([x<=70 for x in S],1),1)      # 三遍都没超过 70
def dist(mask,name):
    c=collections.Counter()
    n=0
    for i,v in enumerate(vids):
        if not mask[i]: continue
        n+=1
        rs=[r.strip() for r in (mapr[v]['reasons'] or '').split(';') if r.strip()]
        if not rs: c['(无标注理由)']+=1
        for r in rs: c[r]+=1
    print(f'--- {name} n={n} ---')
    for k,val in c.most_common(12): print(f'   {val:4d} ({val/max(n,1)*100:4.0f}%)  {k}')
bad=y==1
dist(bad,'全部真 bad')
dist(bad&z0,'三遍全打0分的 bad(判官完全没看见)')
dist(bad&never,'三遍都没超过70的 bad(放行线以下)')
dist(bad&~never,'至少一遍抓到的 bad(对照)')
