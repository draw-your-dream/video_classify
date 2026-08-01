#!/usr/bin/env python
"""P3 三轴评估(R 参照相似 / G 几何 / C 计数),语料 464 bad vs 156 eval-good。

R:帧抠像嵌入(so400m-512 / DreamSim)对本款参照池(饰品款9张+基础款98张并集,
  基础款=98张)max 相似;帧异常 = 1-maxsim(so400m)或 min 余弦距(DreamSim);
  视频分 = 帧异常 p75 / mean(持续不像才算,抗单帧遮挡)。
G:身高/伞宽轨迹。静态 = |log(vid中位 / 款good中位)|(款统计只用 bank 侧,防泄漏);
  漂移 = |log(末4有效帧均值 / 首4有效帧均值)|;附 bbox 高宽比同口径。
C:实例数轨迹:frac(n>=2)、最长连续 n>=2、检出后消失率 frac(n==0 | 已出现过)。
划分沿用 patch_bank_eval.split_goods(款分层,确定性种子)。输出 CSV + AUC 表。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_bank_eval import auc, load_manifest, split_goods  # noqa: E402
from prep_ref_embeds import Embedders  # noqa: E402

MIN_VALID = 4


def geo_series(geo_json: str):
    geo = json.loads(geo_json)
    r, asp, frames = [], [], []
    for g in geo:
        if g["cap_width"] >= 8 and g["height"] >= 16:
            r.append(g["height"] / g["cap_width"])
            x0, y0, x1, y1 = g["bbox"]
            asp.append((y1 - y0 + 1) / max(1, x1 - x0 + 1))
            frames.append(g["frame"])
    return np.array(r), np.array(asp), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/root/mech/data/prod500/mech_subset.tsv")
    ap.add_argument("--feat-dir", default="/root/mech/data/sam3_feat")
    ap.add_argument("--cut-dir", default="/root/mech/data/sam3_cutouts")
    ap.add_argument("--refs", default="/root/mech/ref_embeds.npz")
    ap.add_argument("--out", default="/root/mech/axis_rgc_scores.csv")
    ap.add_argument("--no-ref", action="store_true", help="跳过 R 轴(嵌入未就绪时)")
    args = ap.parse_args()

    man = load_manifest(Path(args.manifest))
    goods = man[man.label == "good"]
    bank_rels, eval_rels = split_goods(goods)
    bank_set = set(bank_rels)
    bads = man[man.label != "good"].rel.tolist()

    emb = None
    ref_so, ref_ds = {}, {}
    if not args.no_ref:
        z = np.load(args.refs, allow_pickle=True)
        styles = z["style"]
        base_so = z["so400m"][styles == "基础款"]
        base_ds = z["dreamsim"][styles == "基础款"]
        for st in np.unique(styles):
            sel_so, sel_ds = z["so400m"][styles == st], z["dreamsim"][styles == st]
            if st != "基础款":
                sel_so = np.concatenate([sel_so, base_so])
                sel_ds = np.concatenate([sel_ds, base_ds])
            ref_so[st] = torch.tensor(sel_so).cuda()
            ref_ds[st] = torch.tensor(sel_ds).cuda()
        emb = Embedders()
        print("refs ready:", {k: len(v) for k, v in ref_so.items()}, flush=True)

    rows = []
    all_rels = ([(r, "bank_good") for r in bank_rels]
                + [(r, "eval_good") for r in eval_rels]
                + [(r, "bad") for r in bads])
    for i, (rel, group) in enumerate(all_rels):
        p = Path(args.feat_dir) / rel.replace(".mp4", ".npz")
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        style = rel.split("/")[0]
        rec = {"rel": rel, "style": style, "group": group}

        n = z["n_inst"].astype(int)
        n = n[n >= 0]
        if len(n):
            seen = np.maximum.accumulate(n > 0)
            rec["c_multi_frac"] = float((n >= 2).mean())
            runs, cur = 0, 0
            for v in n:
                cur = cur + 1 if v >= 2 else 0
                runs = max(runs, cur)
            rec["c_multi_run"] = float(runs)
            rec["c_vanish_frac"] = float(((n == 0) & seen).mean())

        r, asp, frames = geo_series(str(z["geo"]))
        if len(r) >= MIN_VALID:
            rec["g_ratio_med"] = float(np.median(r))
            rec["g_asp_med"] = float(np.median(asp))
            k = max(2, min(4, len(r) // 2))
            rec["g_drift"] = float(abs(np.log(np.mean(r[-k:]) / np.mean(r[:k]))))
            rec["g_asp_drift"] = float(abs(np.log(np.mean(asp[-k:]) / np.mean(asp[:k]))))

        if emb is not None and style in ref_so:
            cdir = Path(args.cut_dir) / rel.replace(".mp4", "")
            jpgs = sorted(cdir.glob("f*.jpg"))
            if jpgs:
                ims = [Image.open(j).convert("RGB") for j in jpgs]
                so = torch.tensor(emb.so400m(ims)).cuda()
                ds = torch.tensor(emb.dreamsim(ims)).cuda()
                d_so = (1.0 - (so @ ref_so[style].T).amax(1)).cpu().numpy()
                d_ds = (1.0 - (ds @ ref_ds[style].T).amax(1)).cpu().numpy()
                for nme, d in (("so", d_so), ("ds", d_ds)):
                    rec[f"r_{nme}_p75"] = float(np.percentile(d, 75))
                    rec[f"r_{nme}_mean"] = float(d.mean())
                    rec[f"r_{nme}_max"] = float(d.max())
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_rels)}", flush=True)

    df = pd.DataFrame(rows)

    # G 静态口径:款 good 中位数只用 bank 侧
    for col in ("g_ratio_med", "g_asp_med"):
        if col not in df:
            continue
        stat = df[(df.group == "bank_good")].groupby("style")[col].median()
        df[col + "_dev"] = df.apply(
            lambda x: abs(np.log(x[col] / stat[x.style]))
            if pd.notna(x.get(col)) and x.style in stat and stat[x.style] > 0 else np.nan,
            axis=1)

    df.to_csv(args.out, index=False)
    print(f"写出 {args.out} ({len(df)} rows)")
    pos = df[df.group == "bad"]
    neg = df[df.group == "eval_good"]
    sc = [c for c in df.columns if c[:2] in ("r_", "g_", "c_") and df[c].dtype != object]
    print("\n== AUC: bad vs eval_good ==")
    res = []
    for c in sc:
        a = auc(pos[c].dropna().values, neg[c].dropna().values)
        res.append((a, c, pos[c].notna().sum(), neg[c].notna().sum()))
    for a, c, nb, ng in sorted(res, reverse=True):
        print(f"  {c:22s} AUC={a:.3f}  (n={nb}/{ng})")
    print("AXIS_RGC_DONE")


if __name__ == "__main__":
    main()
