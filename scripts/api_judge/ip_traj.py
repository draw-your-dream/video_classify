#!/usr/bin/env python3
"""ip_dino 视频轨迹:逐帧算「主体 vs 该款官方参考」DINOv2 相似度,输出轨迹特征。

移植自参考仓库 skus_insert/ip_check.py(SAM 全分割 + OWLv2 局部放大双路候选 →
DINOv2 与参考多视图余弦取 max)。视频版:每条均匀采 8 帧,得相似度轨迹;
还原度走样(尤其中段短暂走样)应表现为轨迹凹陷。

用法:
  python ip_traj.py --videos videos --manifest dev150.csv --sku-refs sku_refs --out out/ip_traj.jsonl
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PPS = 16
MAXSIDE = 768
TOPK = 20
SAM_MODEL = "facebook/sam-vit-base"
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
DINO_MODEL = "facebook/dinov2-large"
QUERIES = ["a mushroom plush toy", "a small cartoon mushroom character",
           "a cute plush toy figurine", "a small stuffed toy"]

_M = {}


def _load():
    if _M:
        return
    from transformers import (AutoImageProcessor, AutoModel, pipeline,
                              Owlv2Processor, Owlv2ForObjectDetection, SamModel, SamProcessor)
    _M["seg"] = pipeline("mask-generation", model=SAM_MODEL, device=0 if DEVICE == "cuda" else -1)
    _M["sam_p"] = SamProcessor.from_pretrained(SAM_MODEL)
    _M["sam"] = SamModel.from_pretrained(SAM_MODEL).to(DEVICE).eval()
    _M["owl_p"] = Owlv2Processor.from_pretrained(OWL_MODEL)
    _M["owl"] = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(DEVICE).eval()
    _M["din_p"] = AutoImageProcessor.from_pretrained(DINO_MODEL)
    _M["din"] = AutoModel.from_pretrained(DINO_MODEL).to(DEVICE).eval()


def _small(im, ms=MAXSIDE):
    im = im.convert("RGB")
    w, h = im.size
    s = min(1.0, ms / max(w, h))
    return im.resize((int(w * s), int(h * s)), Image.LANCZOS) if s < 1 else im


@torch.no_grad()
def _demb(crops):
    if not crops:
        return np.zeros((0, 1024), np.float32)
    x = _M["din_p"](images=crops, return_tensors="pt").to(DEVICE)
    e = _M["din"](**x).pooler_output.float().cpu().numpy()
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def _ref_crop(im):
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    a = np.asarray(im).astype(np.int16)
    cn = np.concatenate([a[:3, :3].reshape(-1, 3), a[:3, -3:].reshape(-1, 3),
                         a[-3:, :3].reshape(-1, 3), a[-3:, -3:].reshape(-1, 3)])
    df = np.abs(a - np.median(cn, 0)).sum(2)
    ys, xs = np.where(df > 40)
    if len(xs):
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        p = int(0.06 * max(x1 - x0, y1 - y0))
        im = im.crop((max(0, x0 - p), max(0, y0 - p), min(im.width, x1 + p), min(im.height, y1 + p)))
    return im


def _fill_holes(m):
    m = ndimage.binary_closing(m, iterations=3)
    m = ndimage.binary_fill_holes(m)
    lbl, n = ndimage.label(m)
    if n > 1:
        cnt = np.bincount(lbl.ravel())
        cnt[0] = 0
        keep = np.where(cnt >= 0.15 * cnt.max())[0]
        m = np.isin(lbl, keep)
        m = ndimage.binary_fill_holes(m)
    return m


def _mcrop(im, m, p=0.12):
    m = _fill_holes(m)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return im.convert("RGB")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    arr = np.asarray(im.convert("RGB")).copy()
    arr[~m] = 255
    W, H = im.size
    pw = int((x1 - x0) * p)
    ph = int((y1 - y0) * p)
    return Image.fromarray(arr).crop((max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)))


def _seg_cands(im):
    W, H = im.size
    out = _M["seg"](im, points_per_side=PPS, pred_iou_thresh=0.85,
                    stability_score_thresh=0.9, points_per_batch=64)
    ms = [np.asarray(m, bool) for m in out["masks"]]
    ms = [m for m in ms if 0.004 <= m.sum() / (W * H) <= 0.6]
    ms.sort(key=lambda m: -m.sum())
    return [_mcrop(im, m) for m in ms[:TOPK]]


@torch.no_grad()
def _owl_boxes(im, topk=6):
    inp = _M["owl_p"](text=[QUERIES], images=im, return_tensors="pt").to(DEVICE)
    out = _M["owl"](**inp)
    ts = torch.tensor([im.size[::-1]]).to(DEVICE)
    pp = (getattr(_M["owl_p"], "post_process_object_detection", None)
          or getattr(_M["owl_p"].image_processor, "post_process_object_detection", None)
          or getattr(_M["owl_p"], "post_process_grounded_object_detection"))
    res = pp(out, threshold=0.04, target_sizes=ts)[0]
    b = res["boxes"].cpu().numpy()
    sc = res["scores"].cpu().numpy()
    return [tuple(map(int, b[i])) for i in sc.argsort()[::-1][:topk]]


@torch.no_grad()
def _sam_box(im, box):
    inp = _M["sam_p"](im, input_boxes=[[list(box)]], return_tensors="pt").to(DEVICE)
    out = _M["sam"](**inp)
    mks = _M["sam_p"].image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(),
        inp["reshaped_input_sizes"].cpu())[0][0].numpy().astype(bool)
    H, W = mks.shape[-2:]
    ar = [m.sum() for m in mks]
    order = np.argsort(ar)[::-1]
    for i in order:
        if ar[i] < 0.9 * H * W:
            return mks[i]
    return mks[order[-1]]


def _owl_local_cands(full):
    W, H = full.size
    crops = []
    for b in _owl_boxes(full):
        x0, y0, x1, y1 = b
        bw = x1 - x0
        bh = y1 - y0
        if bw <= 2 or bh <= 2:
            continue
        ex = (max(0, x0 - int(bw * 0.35)), max(0, y0 - int(bh * 0.35)),
              min(W, x1 + int(bw * 0.35)), min(H, y1 + int(bh * 0.35)))
        loc = full.crop(ex)
        lw, lh = loc.size
        sc = min(4.0, 512 / max(lw, lh))
        loc = loc.resize((max(1, int(lw * sc)), max(1, int(lh * sc))), Image.LANCZOS)
        bl = ((x0 - ex[0]) * sc, (y0 - ex[1]) * sc, (x1 - ex[0]) * sc, (y1 - ex[1]) * sc)
        try:
            crops.append(_mcrop(loc, _sam_box(loc, bl)))
        except Exception:
            pass
    return crops


def ip_score(full, ref_embs):
    cand = _seg_cands(_small(full)) + _owl_local_cands(full)
    if not cand:
        return None
    E = _demb(cand)                       # (N,1024)
    return round(float((E @ ref_embs.T).max(1).max()), 4)


def sku_of(filename):
    return filename[:-4].split("__")[2]


def build_ref_embs(sku_dir, skus):
    refs = {}
    for sku in skus:
        pats = sorted(glob.glob(os.path.join(sku_dir, f"{sku}*_[1-6].png")))
        if not pats:
            base = sku.split("（")[0].split("(")[0]
            pats = sorted(glob.glob(os.path.join(sku_dir, f"{base}*_[1-6].png")))
        crops = [_ref_crop(Image.open(p)) for p in pats[:6]]
        refs[sku] = _demb(crops) if crops else None
        print(f"ref {sku}: {len(crops)} views", flush=True)
    return refs


def frames_of(video_path, n=8):
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video_path),
                        "-vf", f"fps={n}/5", "-frames:v", str(n), f"{td}/f%02d.png"],
                       check=False, capture_output=True)
        fs = sorted(Path(td).glob("f*.png"))
        return [Image.open(f).convert("RGB") for f in fs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sku-refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nframes", type=int, default=8)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8-sig")))
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["filename"])
            except Exception:
                pass
    _load()
    refs = build_ref_embs(args.sku_refs, sorted({sku_of(r["filename"]) for r in rows}))
    f = open(args.out, "a")
    t0, n = time.time(), 0
    for r in rows:
        fn = r["filename"]
        if fn in done:
            continue
        row = {"filename": fn}
        try:
            R = refs.get(sku_of(fn))
            if R is None or not len(R):
                row["error"] = "no_ref"
            else:
                ims = frames_of(Path(args.videos) / fn, args.nframes)
                sims = [ip_score(im, R) for im in ims]
                sims_v = [s for s in sims if s is not None]
                if not sims_v:
                    row["error"] = "no_subject"
                else:
                    row["sims"] = sims
                    row["ip_min"] = min(sims_v)
                    row["ip_mean"] = round(float(np.mean(sims_v)), 4)
                    row["ip_first"] = sims_v[0]
                    row["ip_drop"] = round(sims_v[0] - min(sims_v), 4)
        except Exception as e:
            row["error"] = repr(e)[:150]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        n += 1
        if n % 5 == 0:
            print(f"[{n}] {(time.time()-t0)/n:.1f}s/条", flush=True)
    print("IPTRAJ_DONE", flush=True)


if __name__ == "__main__":
    main()
