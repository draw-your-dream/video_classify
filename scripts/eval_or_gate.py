#!/usr/bin/env python
"""OR 门评估(FACTOR_PREREG.md 协议):域内分位 + 全召回阈值 + LOO + bootstrap。

阈值规则:susp_i = max over 轴 of 分位;t = min over bads of susp;
good 放行 = susp < t(NaN 视为不放行,保守)。
LOO:逐 bad 剔除后重算 t,看该 bad 是否仍被拦。
bootstrap:重采样 27 bads 定阈,good 放行率的分布(CI 2.5/97.5)。

用法:
  eval_or_gate.py                      # 基线(dim_fidelity + c_first_last)
  eval_or_gate.py --add f1_cap_cos_fl  # 基线 + 新因子(可多个)
  eval_or_gate.py --sweep              # 每个新因子单独并门 + 全组合,输出对比表
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# 因子方向:+1 = 数值越大越可疑;-1 = 越小越可疑
DIRECTIONS = {
    "dim_fidelity": +1,   # 线上 rubric 的劣化分:越高越差(bad均值0.097>good 0.091,已数据裁决)
    "c_first_last": -1,
    "f1_cap_cos_fl": -1,
    "f1_cap_hist_bhat": +1,
    "f1_cap_cos_min": -1,
    "f2_anchor_min": -1,
    "f3_face_cos_fl": -1,
    "f3_face_cos_min": -1,
    "f4_logh_range": +1,
    "f5_res_p95_mean": +1,
    "f5_res_p95_max": +1,
}
BASELINE = ["dim_fidelity", "c_first_last"]


def load_table() -> pd.DataFrame:
    gt = pd.read_csv(ROOT / "data/prod500/prod500.csv")[["stem", "y_bad", "human_label",
                                                          "dim_fidelity"]]
    crop = pd.read_csv(ROOT / "data/prod500/prod_crop.csv")[["stem", "c_first_last"]]
    df = gt.merge(crop, on="stem", how="left")
    fpath = ROOT / "data/prod500/factors_f1f5.jsonl"
    if fpath.exists():
        recs = [json.loads(l) for l in fpath.open()]
        fdf = pd.DataFrame(recs)
        fdf["stem"] = fdf["stem"].str[:32]
        df = df.merge(fdf, on="stem", how="left")
    return df


def susp(df: pd.DataFrame, axes: list[str]) -> np.ndarray:
    s = np.full(len(df), np.nan)
    for a in axes:
        d = DIRECTIONS[a]
        r = (d * df[a]).rank(pct=True, na_option="keep").values
        s = np.fmax(s, r)
    return s


def evaluate(df: pd.DataFrame, axes: list[str], n_boot: int = 2000, seed: int = 0):
    s = susp(df, axes)
    bad = df["y_bad"].values == 1
    sb, sg = s[bad], s[~bad]
    t = np.nanmin(sb)
    if np.isnan(sb).any():          # 有 bad 全轴 NaN → 无法设阈
        return {"error": "bad with all-NaN axes", "n_nan_bad": int(np.isnan(sb).sum())}
    passed = (sg < t) & ~np.isnan(sg)
    # LOO
    loo_miss = []
    for i in np.where(bad)[0]:
        others = np.delete(sb, np.where(np.where(bad)[0] == i)[0][0])
        if s[i] < np.nanmin(others):
            loo_miss.append(df.iloc[i]["stem"][:8])
    # bootstrap
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        tb = np.min(rng.choice(sb, size=len(sb), replace=True))
        rates.append(np.mean((sg < tb) & ~np.isnan(sg)))
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return {
        "axes": axes, "threshold": float(t),
        "recall": f"{int(bad.sum())}/{int(bad.sum())}",
        "good_pass": float(passed.mean()),
        "loo_miss": loo_miss,
        "boot_ci": (float(lo), float(hi)),
    }


def report(r: dict):
    if "error" in r:
        print(f"  ERROR: {r}")
        return
    print(f"  axes={'+'.join(r['axes'])}")
    print(f"  good放行 {r['good_pass']*100:.1f}%  (阈值 {r['threshold']:.3f}, "
          f"CI [{r['boot_ci'][0]*100:.1f}%, {r['boot_ci'][1]*100:.1f}%])  "
          f"LOO漏 {len(r['loo_miss'])}/27 {r['loo_miss']}")


def hard_bad_table(df: pd.DataFrame, axes: list[str]):
    """逐 bad 的各轴分位表,按最难排序。"""
    bad = df[df["y_bad"] == 1].copy()
    full = df.copy()
    for a in axes:
        d = DIRECTIONS[a]
        full["ax_" + a] = (d * full[a]).rank(pct=True, na_option="keep")
    cols = ["ax_" + a for a in axes]
    bt = full[full["y_bad"] == 1][["stem"] + cols].copy()
    bt["stem"] = bt["stem"].str[:8]
    bt["best"] = bt[cols].max(axis=1, skipna=True)
    print(bt.sort_values("best").to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--table", action="store_true", help="逐bad分位表")
    args = ap.parse_args()

    df = load_table()
    have = [c for c in DIRECTIONS if c in df.columns and df[c].notna().sum() > 400
            or c in BASELINE]
    print(f"n={len(df)} bad={int(df.y_bad.sum())} 可用轴: "
          f"{[c for c in DIRECTIONS if c in df.columns]}")

    print("\n== 基线 ==")
    report(evaluate(df, BASELINE))

    if args.add:
        print(f"\n== 基线 + {args.add} ==")
        report(evaluate(df, BASELINE + args.add))
        if args.table:
            hard_bad_table(df, BASELINE + args.add)
    elif args.sweep:
        news = [c for c in DIRECTIONS if c not in BASELINE and c in df.columns
                and df[c].notna().sum() > 300]
        for f in news:
            print(f"\n== 基线 + {f} ==")
            report(evaluate(df, BASELINE + [f]))
        if len(news) >= 2:
            print(f"\n== 基线 + 全部新因子 ==")
            report(evaluate(df, BASELINE + news))
    elif args.table:
        hard_bad_table(df, BASELINE)


if __name__ == "__main__":
    main()
