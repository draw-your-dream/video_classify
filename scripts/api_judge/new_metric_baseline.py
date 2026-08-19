#!/usr/bin/env python3
"""新口径基线测算:固定放行 80% 的 good+normal,报 bad 去除率。

br@R = 在恰好放行 R 比例可放行样本(good+normal)时,被拦下的 bad 比例。
labels 用官方修正版(15174/6113/7219 → bad)。同时报 70%/90% 口径与眉毛门混合制。
"""
import csv
import json
import os
import numpy as np

RELABEL_BAD = {"15174.mp4", "6113.mp4", "7219.mp4"}


def load(path, col):
    rows = list(csv.DictReader(open(path)))
    y, p, vids = [], [], []
    for r in rows:
        lab = "bad" if (r["video"] in RELABEL_BAD or r["label"] == "bad") else r["label"]
        y.append(1 if lab == "bad" else 0)
        p.append(float(r[col]))
        vids.append(r["video"])
    return np.array(y), np.array(p), vids


def br_at(y, p, veto=None, rel=0.80):
    """放行率恰为 rel 时的 bad 去除率。veto: bool 数组,被否决样本直接拦截。"""
    if veto is None:
        veto = np.zeros(len(y), bool)
    gn = y == 0
    n_gn = gn.sum()
    need = int(np.floor(rel * n_gn))          # 需要放行的 gn 数
    freed = p[gn & ~veto]                      # 可被放行的 gn 分数
    if len(freed) < need:
        return None                            # 否决太多,放行率达不到
    T = np.sort(freed)[need - 1] if need > 0 else -np.inf  # 第 need 小分数为界(p<=T 放行)
    nb = y.sum()
    removed = int((y == 1)[veto].sum() if veto.any() else 0)
    removed += int(((y == 1) & ~veto & (p > T)).sum())
    return removed / nb, float(T)


def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    methods = [
        ("前人 p_base(可部署基线)", "inference/artifacts_v3/predictions_v3.csv", "p_base"),
        ("前人 p_final(级联,样本内过拟合)", "inference/artifacts_v3/predictions_v3.csv", "p_final"),
        ("C5(线性meta+raw)", "data/s3/predictions_c5_eval.csv", "p_c5"),
        ("E18 冠军", "data/s3/predictions_e18_eval.csv", "p_e18"),
        ("E43(交互约束)", "data/s3/predictions_e43_eval.csv", "p_e43"),
        ("M1", "data/s3/predictions_m1_eval.csv", "p_m1"),
    ]
    # 前人文件含全部视频,过滤到 eval_v3
    ev = {json.loads(l)["video"] for l in open("splits/eval_v3.jsonl")}

    print(f"{'方法':<28} {'br@70%':>8} {'br@80%':>8} {'br@90%':>8}")
    results = {}
    for name, path, col in methods:
        y, p, vids = load(path, col)
        keep = np.array([v in ev for v in vids])
        y, p, vids = y[keep], p[keep], [v for v, k in zip(vids, keep) if k]
        out = []
        for r in (0.70, 0.80, 0.90):
            res = br_at(y, p, rel=r)
            out.append(f"{res[0]:.3f}" if res else "n/a")
        results[name] = (y, p, vids)
        print(f"{name:<28} {out[0]:>8} {out[1]:>8} {out[2]:>8}")

    # E18 + 眉毛门 v6.1(VLM确认 ∧ CNN bmax>=0.05)
    y, p, vids = results["E18 冠军"]
    bmax = {}
    for r in csv.DictReader(open("data/s3/brow_scan.csv")):
        bmax[os.path.basename(r["rel"])] = float(r["bmax"])
    veto_set = set()
    for d in json.load(open("data/s3/brow_v6_hits.json")):
        veto_set.add(os.path.basename(d["rel"]))
    for l in open("data/s3/brow_confirm_full.jsonl"):
        d = json.loads(l)
        if d.get("eyebrows"):
            veto_set.add(os.path.basename(d["rel"]))
    veto_set = {v for v in veto_set if bmax.get(v, 1.0) >= 0.05}
    veto = np.array([v in veto_set for v in vids])
    out = []
    for r in (0.70, 0.80, 0.90):
        res = br_at(y, p, veto=veto, rel=r)
        out.append(f"{res[0]:.3f}" if res else "n/a")
    print(f"{'E18 + 眉毛门v6.1(混合制)':<28} {out[0]:>8} {out[1]:>8} {out[2]:>8}")
    print(f"\neval_v3 (修正标签): {int(y.sum())} bad / {int((y==0).sum())} good+normal")


if __name__ == "__main__":
    main()
