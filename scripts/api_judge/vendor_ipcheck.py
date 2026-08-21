# -*- coding: utf-8 -*-
"""从 draw-your-dream/image-dataset-curation-filtering @ef0c450 移植的形象还原度定位/打分组件。
原文件:skus_insert/ip_check.py + skus_insert/qc_zoom.py(best_subject_crop)。
改动仅限:模型改从本机快照加载;去掉道具分支与 Gemini 调用;保留阈值与选主体逻辑不变。
栈:OWLv2 开放词定框 → DINOv2 粗筛 → 对候选跑 SAM 出掩膜 → 掩膜外涂白裁剪 → DINOv2 对参考图取最大余弦。"""
import glob, os
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from transformers import (AutoImageProcessor, AutoModel,
                          Owlv2Processor, Owlv2ForObjectDetection, SamModel, SamProcessor)
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAXSIDE = 768; TOPK = 6; SHORTLIST = 3
DINO_MERGE_GAP = 0.04          # 原值
OWL_THR = 0.04                 # 原值
QUERIES = ["a mushroom plush toy", "a small cartoon mushroom character",
           "a cute plush toy figurine", "a small stuffed toy"]
MS = Path.home()/"tutu-video-eval/.hf_cache/ms"
def _snap(pat):
    c = sorted(glob.glob(str(MS/pat)))
    if not c: raise FileNotFoundError(f"未找到快照 {pat}")
    return c[-1]
_M = {}
def load():
    if _M: return
    sam = _snap("facebook/sam-vit-base"); owl = _snap("google/owlv2-base-patch16-ensemble")
    din = os.environ.get("IPF_DINO") or _snap("AI-ModelScope/dinov2-large")
    _M["sam_p"] = SamProcessor.from_pretrained(sam)
    _M["sam"] = SamModel.from_pretrained(sam).to(DEVICE).eval()
    _M["owl_p"] = Owlv2Processor.from_pretrained(owl)
    _M["owl"] = Owlv2ForObjectDetection.from_pretrained(owl).to(DEVICE).eval()
    _M["din_p"] = AutoImageProcessor.from_pretrained(din)
    _M["din"] = AutoModel.from_pretrained(din).to(DEVICE).eval()

def _small(im, ms=MAXSIDE):
    im = im.convert("RGB"); w, h = im.size; s = min(1.0, ms/max(w, h))
    return im.resize((int(w*s), int(h*s)), Image.LANCZOS) if s < 1 else im

@torch.no_grad()
def demb(crops):
    if not crops: return np.zeros((0, 1024), np.float32)
    x = _M["din_p"](images=crops, return_tensors="pt").to(DEVICE)
    e = _M["din"](**x).pooler_output.float().cpu().numpy()
    return e/(np.linalg.norm(e, axis=1, keepdims=True)+1e-8)

def ref_crop(im):
    """参考图去纯色边:按四角中位色差裁到物体外接框 + 6% padding(原样)。"""
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[-1]); im = bg
    else:
        im = im.convert("RGB")
    a = np.asarray(im).astype(np.int16)
    cn = np.concatenate([a[:3, :3].reshape(-1, 3), a[:3, -3:].reshape(-1, 3),
                         a[-3:, :3].reshape(-1, 3), a[-3:, -3:].reshape(-1, 3)])
    df = np.abs(a-np.median(cn, 0)).sum(2); ys, xs = np.where(df > 40)
    if len(xs):
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
        p = int(0.06*max(x1-x0, y1-y0))
        im = im.crop((max(0, x0-p), max(0, y0-p), min(im.width, x1+p), min(im.height, y1+p)))
    return im

def fill_holes(m):
    m = ndimage.binary_closing(m, iterations=3); m = ndimage.binary_fill_holes(m)
    lbl, n = ndimage.label(m)
    if n > 1:
        cnt = np.bincount(lbl.ravel()); cnt[0] = 0
        keep = np.where(cnt >= 0.15*cnt.max())[0]
        m = np.isin(lbl, keep); m = ndimage.binary_fill_holes(m)
    return m

def mcrop(im, m, p=0.12):
    """掩膜外涂白 + 裁到掩膜外接框 + 12% padding(原样)。这一步让生成图与白底参考图可比。"""
    m = fill_holes(m); ys, xs = np.where(m)
    if len(xs) == 0: return im.convert("RGB")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    arr = np.asarray(im.convert("RGB")).copy(); arr[~m] = 255; W, H = im.size
    pw = int((x1-x0)*p); ph = int((y1-y0)*p)
    return Image.fromarray(arr).crop((max(0, x0-pw), max(0, y0-ph), min(W, x1+pw), min(H, y1+ph)))

@torch.no_grad()
def owl_boxes(im, topk=TOPK, extra_queries=None):
    q = QUERIES+list(extra_queries or [])
    inp = _M["owl_p"](text=[q], images=im, return_tensors="pt").to(DEVICE)
    out = _M["owl"](**inp); ts = torch.tensor([im.size[::-1]]).to(DEVICE)
    res = _M["owl_p"].post_process_object_detection(out, threshold=OWL_THR, target_sizes=ts)[0]
    b = res["boxes"].cpu().numpy(); sc = res["scores"].cpu().numpy()
    return [tuple(map(int, b[i])) for i in sc.argsort()[::-1][:topk]]

@torch.no_grad()
def sam_boxes_multi(im, boxes):
    """一次图像编码 + 多框提示(原实现每框重跑一次编码器,这里只改效率不改结果口径)。"""
    inp = _M["sam_p"](im, input_boxes=[[list(b) for b in boxes]], return_tensors="pt").to(DEVICE)
    out = _M["sam"](**inp)
    mks = _M["sam_p"].image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0].numpy().astype(bool)
    res = []
    for q in range(mks.shape[0]):
        mm = mks[q]; H, W = mm.shape[-2:]
        ar = [m.sum() for m in mm]; order = np.argsort(ar)[::-1]; pick = mm[order[-1]]
        for i in order:
            if ar[i] < 0.9*H*W: pick = mm[i]; break
        res.append(pick)
    return res

@torch.no_grad()
def sam_box(im, box):
    inp = _M["sam_p"](im, input_boxes=[[list(box)]], return_tensors="pt").to(DEVICE)
    out = _M["sam"](**inp)
    mks = _M["sam_p"].image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0].numpy().astype(bool)
    H, W = mks.shape[-2:]; ar = [m.sum() for m in mks]; order = np.argsort(ar)[::-1]
    for i in order:
        if ar[i] < 0.9*H*W: return mks[i]
    return mks[order[-1]]

def _overlap(a, b): return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
def _union(a, b):
    if a is None: return b
    if b is None: return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
def mask_bbox(m):
    if m is None: return None
    mm = fill_holes(np.asarray(m, bool)); ys, xs = np.where(mm)
    if len(xs) == 0: return None
    return (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
def _bleed(box, W, H, frac=0.90):
    return (box[2]-box[0]) >= frac*W and (box[3]-box[1]) >= frac*H

def best_subject(full: Image.Image, R: np.ndarray):
    """原 best_subject_crop 的本体分支。返回 dict:
       dino=掩膜裁剪对参考图的最大余弦(形象还原度分)、u_max=未抠图框的最大余弦、
       body=并集后的本体框、mask=winner 掩膜、crop=白底掩膜裁剪。"""
    W, H = full.size
    try: boxes = owl_boxes(full)
    except Exception: boxes = []
    boxes = [b for b in boxes if (b[2]-b[0]) > 2 and (b[3]-b[1]) > 2]
    if not boxes: return None
    E = demb([full.crop(b) for b in boxes]); u = (E@R.T).max(1)
    cand = list(np.argsort(u)[::-1][:SHORTLIST])
    try:
        masks = sam_boxes_multi(full, [boxes[j] for j in cand])
        crops = [mcrop(full, mm) for mm in masks]
        sims = (demb(crops)@R.T).max(1)
    except Exception:
        masks = [None]*len(cand); crops = [None]*len(cand); sims = np.full(len(cand), -1.0)
    t = int(np.argmax(sims))
    i, s, m, c = cand[t], float(sims[t]), masks[t], crops[t]
    bx0, by0, bx1, by1 = boxes[i]
    thr = float(u.max())-DINO_MERGE_GAP
    for k, b in enumerate(boxes):
        if k != i and u[k] >= thr and _overlap(boxes[i], b):
            bx0, by0, bx1, by1 = min(bx0, b[0]), min(by0, b[1]), max(bx1, b[2]), max(by1, b[3])
    body = (bx0, by0, bx1, by1)
    mb = mask_bbox(m)
    if mb is not None and not _bleed(mb, W, H): body = _union(body, mb)
    return {"dino": float(s), "u_max": float(u.max()), "body": body, "mask": m, "crop": c}

def ref_embeddings(view_dir, skus=None):
    """官方视角图 → 每款一组参考嵌入(先 ref_crop 去底再嵌入,与原实现一致)。"""
    out = {}
    for p in sorted(glob.glob(str(Path(view_dir)/"*.jpg"))+glob.glob(str(Path(view_dir)/"*.png"))):
        sku = Path(p).stem.rsplit("_v", 1)[0]
        if skus and sku not in skus: continue
        out.setdefault(sku, []).append(ref_crop(Image.open(p)))
    return {k: demb(v) for k, v in out.items()}
