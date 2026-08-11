#!/usr/bin/env python
"""E50 最终:规则否决门 eval 单发(第 11 次 eval 接触,2026-08-11)。

门的构成在运行前由 --axes 冻结(brow / brow+size2),此后不再改动。
记账:被否决的 bad 计入召回;阈值只补剩余缺口;被否决的可放行样本从分子扣除。
输出三口径 + 明细(拦了谁、代价是谁)。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e50_hybrid import gn_hybrid, gn_plain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="brow", help="逗号分隔:brow,size2")
    args = ap.parse_args()
    axes = set(args.axes.split(","))

    ev = [json.loads(l) for l in open("splits/eval_v3.jsonl")]
    tier = {r["video"]: r["label"] for r in ev}
    rows = list(csv.DictReader(open("data/s3/predictions_e18_eval.csv")))
    p = np.array([float(r["p_e18"]) for r in rows])
    vids = [r["video"] for r in rows]
    t = np.array([tier[v] for v in vids])
    y = (t == "bad").astype(int)

    veto_set = set()
    prov = {}
    if "brow" in axes:
        for src in ("data/s3/brow_v6_hits.json",):
            for d in json.load(open(src)):
                v = os.path.basename(d["rel"])
                veto_set.add(v)
                prov.setdefault(v, "brow_hist")
        for l in open("data/s3/brow_confirm_full.jsonl"):
            d = json.loads(l)
            if d.get("eyebrows"):
                v = os.path.basename(d["rel"])
                veto_set.add(v)
                prov.setdefault(v, "brow_new")
    if "size2" in axes:
        for l in open("data/s3/size_pair_confirm.jsonl"):
            d = json.loads(l)
            if d.get("size_change"):
                v = os.path.basename(d["rel"])
                veto_set.add(v)
                prov.setdefault(v, "size2")

    veto = np.array([v in veto_set for v in vids])
    print(f"门构成 axes={sorted(axes)} | 否决集全库 {len(veto_set)} | eval 内 {int(veto.sum())}")
    base = gn_plain(p, y)
    h, det = gn_hybrid(p, y, veto, detail=True)
    print(f"\n===== E50 eval 单发 =====")
    print(f"基准 ev@95 = {base:.4f}")
    print(f"混合 ev@95 = {h:.4f}   Δ = {h-base:+.4f}")
    print(f"细节: {det}")
    for rec, name in ((0.90, "ev@90"), (1.00, "ev@100")):
        b0 = gn_plain(p, y, rec)
        h2 = gn_hybrid(p, y, veto, rec)
        print(f"{name}: 基准 {b0:.4f} → 混合 {h2:.4f} (Δ{h2-b0:+.4f})")

    # 明细:拦下的尾部 bad 与付出的代价
    b = np.sort(p[y == 1])
    T0 = b[len(b) - int(np.ceil(0.95 * len(b)))]
    print(f"\n拦下的尾部 bad(p<T0={T0:.4f}):")
    for i, v in enumerate(vids):
        if veto[i] and y[i] == 1 and p[i] < T0:
            print(f"  {v:12s} p={p[i]:.4f}  [{prov.get(v)}]")
    Tn = det["T"]
    print(f"代价(被否决且 p<新T={Tn:.4f} 的可放行样本):")
    for i, v in enumerate(vids):
        if veto[i] and y[i] == 0 and p[i] < Tn:
            print(f"  {v:12s} p={p[i]:.4f} {t[i]}  [{prov.get(v)}]")
    print("E50_EVAL_SHOT_DONE")


if __name__ == "__main__":
    main()
