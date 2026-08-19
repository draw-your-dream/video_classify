#!/usr/bin/env python3
"""角色裁剪版还原度特征:OWLv2 框抠角色 → CLIP/SigLIP 嵌入 →
① vs 自己款式官方立绘(max-cos 统计) ② vs 自己首帧裁剪(漂移统计)。
输出 /workspace/r2/cropfid_feats.csv + 体检。"""
import json, csv
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

R2 = Path('/workspace/r2')
VIEWS = R2 / 'data/sku_ref_v2/views'

boxes = {}
for l in open(R2 / 'apijudge/out/motion_feats.csv.boxes.jsonl'):
    d = json.loads(l)
    boxes[d['filename']] = (d['frames'], d['boxes'])

import open_clip
mc, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device="cuda")
mc.eval()
from transformers import AutoImageProcessor, AutoModel
mn = "google/siglip2-so400m-patch14-384"
proc = AutoImageProcessor.from_pretrained(mn)
ms = AutoModel.from_pretrained(mn, torch_dtype=torch.float16).to("cuda").eval()

def emb_clip(pils):
    ims = torch.stack([prep(p) for p in pils]).to("cuda")
    with torch.no_grad():
        f = mc.encode_image(ims).float().cpu().numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)

def emb_sig(pils):
    inputs = proc(images=pils, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = ms.vision_model(pixel_values=inputs["pixel_values"].half())
        f = out.pooler_output.float().cpu().numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)

# 官方立绘参照(白底孤立角色,不需再裁)
refs = sorted(VIEWS.glob('*.jpg'))
skus = {}
for p in refs:
    skus.setdefault(p.stem.rsplit('_v', 1)[0], []).append(p)
REF = {}
for sku, ps in skus.items():
    pils = [Image.open(p).convert('RGB') for p in ps]
    REF[sku] = (emb_clip(pils), emb_sig(pils))
print("refs embedded", flush=True)

def interp_box(frames, bxs, t, W, H):
    fr = np.array(frames, float)
    b = np.array(bxs, float)
    x0 = np.interp(t, fr, b[:, 0]); y0 = np.interp(t, fr, b[:, 1])
    x1 = np.interp(t, fr, b[:, 2]); y1 = np.interp(t, fr, b[:, 3])
    # 15% pad
    w, h = x1 - x0, y1 - y0
    x0 = max(0, x0 - .15 * w); y0 = max(0, y0 - .15 * h)
    x1 = min(W, x1 + .15 * w); y1 = min(H, y1 + .15 * h)
    return int(x0), int(y0), int(x1), int(y1)

targets = [json.loads(l) for l in open(R2 / 'pbase/upstream/splits/train_v2.jsonl')]
def stats(F, R):
    C = (F @ R.T).max(1)
    return [float(C.mean()), float(C.min()), float(C[0]), float(C[0] - C.min()), float(C.std())]
def drift(F):
    C = F @ F[0]
    return [float(C[1:].mean()), float(C.min()), float(C[-1]), float(1 - C.min()), float(C.std())]
cols = ([f'r_{sp}_{st}' for sp in ('clip','sig') for st in ('mean','min','first','drop','std')] +
        [f'd_{sp}_{st}' for sp in ('clip','sig') for st in ('mean','min','last','maxdrop','std')])
rows, errs = [], 0
import time
t0 = time.time()
for i, e in enumerate(targets):
    fn = Path(e['video']).name
    try:
        cap = cv2.VideoCapture(f'/workspace/r2/videos/{fn}')
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(3)); H = int(cap.get(4))
        idxs = set(np.linspace(0, max(0, N - 1), 8).astype(int).tolist())
        fr_b, bx = boxes[fn]
        pils = []
        t = 0
        while True:  # 顺序读,免随机 seek
            ok, im = cap.read()
            if not ok: break
            if t in idxs:
                x0, y0, x1, y1 = interp_box(fr_b, bx, t, W, H)
                crop = im[y0:y1, x0:x1]
                if crop.size == 0: crop = im
                pils.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            t += 1
        cap.release()
        if len(pils) < 4: raise RuntimeError('too few frames')
        fc, fs = emb_clip(pils), emb_sig(pils)
        sku = fn.split('__')[2]
        Rc, Rs = REF[sku]
        rows.append([fn, e['label']] + stats(fc, Rc) + stats(fs, Rs) + drift(fc) + drift(fs))
    except Exception as ex:
        errs += 1
        if errs < 5: print('ERR', fn[:40], repr(ex)[:80], flush=True)
    if (i + 1) % 200 == 0:
        print(f'[{i+1}/{len(targets)}] {(time.time()-t0)/(i+1):.2f}s/条', flush=True)
with open(R2 / 'cropfid_feats.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['filename', 'label'] + cols); w.writerows(rows)
print(f'written {len(rows)} errs {errs}', flush=True)

# 体检
from scipy.stats import rankdata
lab = {r['filename']: (r['grade'], r.get('reasons','')) for r in csv.DictReader(open(R2/'data/tutu_task1_annotations_1233.csv', encoding='utf-8-sig'))}
y = np.array([1 if lab[r[0]][0]=='bad' else 0 for r in rows])
def fam(keys):
    return np.array([1 if (lab[r[0]][0]=='bad' and any(k in lab[r[0]][1] for k in keys)) else (0 if lab[r[0]][0]!='bad' else -1) for r in rows])
yfid = fam(('还原度',)); ycloth = fam(('衣服/身体的时间一致性',))
X = np.array([r[2:] for r in rows], dtype=float)
def auc(yv, s):
    m = yv >= 0; yy, ss = yv[m], s[m]
    r = rankdata(ss); pos = r[yy==1]
    return float((pos.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*(yy==0).sum()))
print("== 角色裁剪版特征(全bad | 还原度族 | 衣物一致族)==")
for i, c in enumerate(cols):
    a, af, ac = auc(y, X[:,i]), auc(yfid, X[:,i]), auc(ycloth, X[:,i])
    print(f"  {c:14s} {max(a,1-a):.3f}  {max(af,1-af):.3f}  {max(ac,1-ac):.3f}")
print("CROPFID_DONE", flush=True)
