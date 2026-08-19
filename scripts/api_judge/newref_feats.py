#!/usr/bin/env python3
"""新参照池特征:43 张 SKU2.0 官方立绘 × 两嵌入空间(与视频缓存同模型),
每视频对自己款式算相似特征 → /workspace/r2/newref_feats.csv + AUC/br 体检。"""
import json, csv, sys
from pathlib import Path
import numpy as np
import torch

R2 = Path('/workspace/r2')
CACHE = R2 / 'pbase/upstream/data/cache'
VIEWS = R2 / 'data/sku_ref_v2/views'

# ---- 嵌入 43 张立绘 ----
from PIL import Image
refs = sorted(VIEWS.glob('*.jpg'))
skus = {}
for p in refs:
    sku = p.stem.rsplit('_v', 1)[0]
    skus.setdefault(sku, []).append(p)
print("SKU refs:", {k: len(v) for k, v in skus.items()})

import open_clip
mc, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device="cuda")
mc.eval()
from transformers import AutoImageProcessor, AutoModel
mn = "google/siglip2-so400m-patch14-384"
proc = AutoImageProcessor.from_pretrained(mn)
ms = AutoModel.from_pretrained(mn, torch_dtype=torch.float16).to("cuda").eval()

def emb_clip(paths):
    ims = torch.stack([prep(Image.open(p).convert('RGB')) for p in paths]).to("cuda")
    with torch.no_grad():
        f = mc.encode_image(ims).float().cpu().numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)

def emb_sig(paths):
    ims = [Image.open(p).convert('RGB') for p in paths]
    inputs = proc(images=ims, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = ms.vision_model(pixel_values=inputs["pixel_values"].half())
        f = out.pooler_output.float().cpu().numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)

REF = {sku: (emb_clip(ps), emb_sig(ps)) for sku, ps in skus.items()}
print("ref embedded")

# ---- 逐视频特征 ----
targets = [json.loads(l) for l in open(R2 / 'pbase/upstream/splits/train_v2.jsonl')]
def stats(F, R):  # F:(8,d) frames normed, R:(k,d) refs normed
    C = (F @ R.T).max(1)  # 每帧对最像视角
    return [float(C.mean()), float(C.min()), float(C[0]), float(C[-1]),
            float(C[0] - C.min()), float(C.std())]
cols = [f'{sp}_{st}' for sp in ('clip', 'sig') for st in ('mean','min','first','last','drop','std')]
rows = []
for e in targets:
    fn = Path(e['video']).name
    stem = Path(e['video']).stem
    sku = fn.split('__')[2]
    Rc, Rs = REF[sku]
    fc = np.load(CACHE / 'clip_emb_v2' / e['label'] / f'{stem}.npy')
    fs = np.load(CACHE / 'siglip2_so400m_v2' / e['label'] / f'{stem}.npy')
    fc = fc / np.linalg.norm(fc, axis=1, keepdims=True)
    fs = fs / np.linalg.norm(fs, axis=1, keepdims=True)
    rows.append([fn, e['label']] + stats(fc, Rc) + stats(fs, Rs))
with open(R2 / 'newref_feats.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['filename', 'label'] + cols); w.writerows(rows)
print("features written", len(rows))

# ---- 体检 ----
from scipy.stats import rankdata
lab = {r['filename']: (r['grade'], r.get('reasons','')) for r in csv.DictReader(open(R2/'data/tutu_task1_annotations_1233.csv', encoding='utf-8-sig'))}
y = np.array([1 if lab[r[0]][0]=='bad' else 0 for r in rows])
yfid = np.array([1 if (lab[r[0]][0]=='bad' and '还原度' in lab[r[0]][1]) else (0 if lab[r[0]][0]!='bad' else -1) for r in rows])
X = np.array([r[2:] for r in rows], dtype=float)
def auc(yv, s):
    m = yv >= 0; yy, ss = yv[m], s[m]
    r = rankdata(ss); pos = r[yy==1]
    return float((pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(yy==0).sum()))
print("== 新参照特征单列体检(全badAUC | 还原度族AUC)==")
for i, c in enumerate(cols):
    a, af = auc(y, X[:,i]), auc(yfid, X[:,i])
    print(f"  {c:12s} {max(a,1-a):.3f}  {max(af,1-af):.3f}")
print("NEWREF_DONE")
