#!/usr/bin/env python
"""P3 三轴 prod500 打分(2026-08-01)。

与 axis_rgc_eval.py 同口径,差异:
  - R 轴参照池 = 全部 152 张(prod 无款标签,max-over-all 免款分配错误)
  - 不算 AUC,只出逐视频分数表,27/27 全召回集成分析回本地做
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
from axis_rgc_eval import geo_series, MIN_VALID  # noqa: E402
from prep_ref_embeds import Embedders  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/root/mech/data/prod500/prod_manifest.tsv")
    ap.add_argument("--feat-dir", default="/root/mech/data/sam3_feat_prod")
    ap.add_argument("--cut-dir", default="/root/mech/data/sam3_cutouts_prod")
    ap.add_argument("--refs", default="/root/mech/ref_embeds.npz")
    ap.add_argument("--out", default="/root/mech/axis_rgc_prod.csv")
    args = ap.parse_args()

    z = np.load(args.refs, allow_pickle=True)
    ref_so = torch.tensor(z["so400m"]).cuda()
    ref_ds = torch.tensor(z["dreamsim"]).cuda()
    emb = Embedders()
    print(f"refs: {ref_so.shape[0]} (all styles)", flush=True)

    rows = []
    lines = [l.split("\t") for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    for i, (rel, label) in enumerate(lines):
        p = Path(args.feat_dir) / rel.replace(".mp4", ".npz")
        if not p.exists():
            continue
        zz = np.load(p, allow_pickle=True)
        rec = {"rel": rel, "label": label}
        n = zz["n_inst"].astype(int)
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
        r, asp, _ = geo_series(str(zz["geo"]))
        if len(r) >= MIN_VALID:
            rec["g_ratio_med"] = float(np.median(r))
            rec["g_asp_med"] = float(np.median(asp))
            k = max(2, min(4, len(r) // 2))
            rec["g_drift"] = float(abs(np.log(np.mean(r[-k:]) / np.mean(r[:k]))))
            rec["g_asp_drift"] = float(abs(np.log(np.mean(asp[-k:]) / np.mean(asp[:k]))))
        cdir = Path(args.cut_dir) / rel.replace(".mp4", "")
        jpgs = sorted(cdir.glob("f*.jpg"))
        if jpgs:
            ims = [Image.open(j).convert("RGB") for j in jpgs]
            so = torch.tensor(emb.so400m(ims)).cuda()
            ds = torch.tensor(emb.dreamsim(ims)).cuda()
            d_so = (1.0 - (so @ ref_so.T).amax(1)).cpu().numpy()
            d_ds = (1.0 - (ds @ ref_ds.T).amax(1)).cpu().numpy()
            for nme, d in (("so", d_so), ("ds", d_ds)):
                rec[f"r_{nme}_p75"] = float(np.percentile(d, 75))
                rec[f"r_{nme}_mean"] = float(d.mean())
                rec[f"r_{nme}_max"] = float(d.max())
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(lines)}", flush=True)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"写出 {args.out} ({len(rows)} rows)")
    print("AXIS_PROD_DONE")


if __name__ == "__main__":
    main()
