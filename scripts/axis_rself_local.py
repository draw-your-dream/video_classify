#!/usr/bin/env python
"""R-self 自参照轴(2026-08-01 预注册,c_first_last 的语料对应物,本地跑)。

每视频 f0 抠像为自锚,so400m-512 嵌入,drift_t = 1 - cos(f_t, f0);
视频分 = drift 的 max / p75 / mean / 末段均值(last4) / 斜率。
划分与 patch_bank_eval.split_goods 完全同源(md5 种子,BANK_FRAC=300/455),
此处 numpy 独立实现以脱离盒侧依赖;AUC 用秩实现。
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BANK_FRAC = 300 / 455


def stem_seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def split_goods(goods_by_style: dict[str, list[str]]):
    bank, ev = [], []
    for style in sorted(goods_by_style):
        rels = sorted(goods_by_style[style])
        rng = np.random.default_rng(stem_seed("split:" + style))
        rng.shuffle(rels)
        nb = int(round(len(rels) * BANK_FRAC))
        bank += rels[:nb]
        ev += rels[nb:]
    return bank, ev


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # 并列取平均秩
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


class So400m:
    def __init__(self, device="cuda"):
        from transformers import AutoModel, AutoProcessor
        mid = "google/siglip2-so400m-patch16-512"
        self.p = AutoProcessor.from_pretrained(mid)
        self.m = AutoModel.from_pretrained(mid, torch_dtype=torch.float16).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def embed(self, ims: list[Image.Image]) -> np.ndarray:
        inp = self.p(images=ims, return_tensors="pt").to(self.device)
        f = self.m.get_image_features(pixel_values=inp["pixel_values"].half())
        if not torch.is_tensor(f):
            f = f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/prod500/mech_subset.tsv"))
    ap.add_argument("--cut-dir", default=str(ROOT / "data/sam3_cutouts"))
    ap.add_argument("--out", default=str(ROOT / "data/prod500/axis_rself_scores.csv"))
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()

    rows = [l.split("\t") for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    goods_by_style = defaultdict(list)
    for rel, label in rows:
        if label == "good":
            goods_by_style[rel.split("/")[0]].append(rel)
    bank_rels, eval_rels = split_goods(goods_by_style)
    bank_set, eval_set = set(bank_rels), set(eval_rels)

    emb = So400m()
    print("model ready", flush=True)

    recs = []
    batch_ims, batch_meta = [], []

    def flush():
        nonlocal batch_ims, batch_meta
        if not batch_ims:
            return
        es = emb.embed(batch_ims)
        for (ridx, fidx), e in zip(batch_meta, es):
            recs[ridx]["embs"][fidx] = e
        batch_ims, batch_meta = [], []

    for rel, label in rows:
        cdir = Path(args.cut_dir) / rel.replace(".mp4", "")
        jpgs = sorted(cdir.glob("f*.jpg"))
        group = ("bad" if label != "good"
                 else "eval_good" if rel in eval_set else "bank_good")
        recs.append({"rel": rel, "group": group, "embs": {}})
        ridx = len(recs) - 1
        for j in jpgs:
            batch_ims.append(Image.open(j).convert("RGB"))
            batch_meta.append((ridx, int(j.stem[1:])))
            if len(batch_ims) >= args.batch:
                flush()
        if ridx % 100 == 99:
            flush()
            print(f"  {ridx+1}/{len(rows)}", flush=True)
    flush()

    out = []
    for r in recs:
        embs = r["embs"]
        rec = {"rel": r["rel"], "group": r["group"], "n_frames": len(embs)}
        if 0 in embs and len(embs) >= 5:
            e0 = embs[0]
            ks = sorted(k for k in embs if k > 0)
            d = np.array([1.0 - float(embs[k] @ e0) for k in ks])
            rec["rs_max"] = float(d.max())
            rec["rs_p75"] = float(np.percentile(d, 75))
            rec["rs_mean"] = float(d.mean())
            rec["rs_last4"] = float(d[-4:].mean())
            rec["rs_slope"] = float(np.polyfit(ks, d, 1)[0])
            # 相邻帧跳变口径(帧间突变,补充轴)
            es = [embs[k] for k in sorted(embs)]
            jd = np.array([1.0 - float(es[i + 1] @ es[i]) for i in range(len(es) - 1)])
            rec["rs_jump_max"] = float(jd.max())
            rec["rs_jump_p90"] = float(np.percentile(jd, 90))
        out.append(rec)

    import csv
    cols = ["rel", "group", "n_frames", "rs_max", "rs_p75", "rs_mean",
            "rs_last4", "rs_slope", "rs_jump_max", "rs_jump_p90"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c) for c in cols})
    print(f"写出 {args.out} ({len(out)} rows)")

    pos = [r for r in out if r["group"] == "bad"]
    neg = [r for r in out if r["group"] == "eval_good"]
    print(f"\n== AUC: bad({len(pos)}) vs eval_good({len(neg)}) ==")
    res = []
    for c in cols[3:]:
        p = np.array([r.get(c, np.nan) if r.get(c) is not None else np.nan for r in pos], float)
        n = np.array([r.get(c, np.nan) if r.get(c) is not None else np.nan for r in neg], float)
        res.append((auc(p, n), c))
    for a, c in sorted(res, reverse=True):
        print(f"  {c:14s} AUC={a:.3f}")
    print("RSELF_DONE")


if __name__ == "__main__":
    main()
