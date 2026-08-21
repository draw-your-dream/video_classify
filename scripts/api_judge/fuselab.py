#!/usr/bin/env python3
"""融合层本地实验室(零 API 成本)。SEL 种子 42-44 选,CONF 种子 47-49 确认,8seed 42-49 汇报。
基线 全1233/20折: 8seed 单 0.4846 装袋 0.4982 | dev867/10折: 单 0.4730 装袋 0.4889"""
import sys, numpy as np
sys.path.insert(0,str(__import__("pathlib").Path(__file__).parent))
import _base43 as B
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
rp=B.rp; br_at=B.br_at; auc=B.auc; y=B.y; groups=B.groups; N=B.N
def run(X,mask,nf,seeds,C=100,w=None,qt=False):
    Xd=X[mask]; yd=y[mask]; gd=groups[mask]; oofs=[]
    for sd in seeds:
        oof=np.zeros(len(yd))
        for tr,te in StratifiedGroupKFold(nf,shuffle=True,random_state=sd).split(Xd,yd,gd):
            if qt:
                sc=QuantileTransformer(n_quantiles=min(1000,len(tr)),output_distribution="normal",random_state=0).fit(Xd[tr])
            else:
                sc=StandardScaler().fit(Xd[tr])
            m=LogisticRegression(C=C,max_iter=8000)
            m.fit(sc.transform(Xd[tr]),yd[tr],sample_weight=None if w is None else w[mask][tr])
            oof[te]=m.predict_proba(sc.transform(Xd[te]))[:,1]
        oofs.append(oof)
    per=[br_at(o,yd) for o in oofs]; bag=np.mean([rp(o) for o in oofs],0)
    return np.mean(per),np.std(per),br_at(bag,yd),auc(bag,yd)
def report(nm,X,**kw):
    for tag,mask,nf in (("dev867/10折",B.dm,10),("全1233/20折",B.ALL,20)):
        m,s,b,a=run(X,mask,nf,tuple(range(42,50)),**kw)
        m3,_,b3,_=run(X,mask,nf,(42,43,44),**kw)
        m2,_,b2,_=run(X,mask,nf,(47,48,49),**kw)
        print(f"  {nm:26s} {tag:12s} 8seed 单{m:.4f}±{s:.4f} 装袋{b:.4f} AUC{a:.4f} | SEL 单{m3:.4f} 装袋{b3:.4f} | CONF 单{m2:.4f} 装袋{b2:.4f}",flush=True)
X=B.X43
print("=== 0 基线 ===",flush=True); report("43列 C=100",X)
print("=== 1 全列秩变换(等价于对每列做分位归一)===",flush=True)
Xr=np.stack([rp(X[:,j]) if len(np.unique(X[:,j]))>2 else X[:,j] for j in range(X.shape[1])],1)
report("秩变换 C=100",Xr)
report("秩变换 QuantileTransformer",Xr,qt=True)
print("=== 2 C 扫描(原始列)===",flush=True)
for C in (0.03,0.1,0.3,1,3,10):
    report(f"C={C}",X,C=C)
print("=== 3 C 扫描(秩变换列)===",flush=True)
for C in (0.1,1,10):
    report(f"秩变换 C={C}",Xr,C=C)
