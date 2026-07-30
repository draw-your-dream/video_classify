#!/usr/bin/env python
"""对照实验:按款分库(参照系收紧假设检验,2026-07-30)。

全局库结果:eval_good P50=0.42 vs bank_good P50=0.19——新视频的新道具/场景/配饰
把"正常流形"撑得远超 299 条库的覆盖,novelty 噪声地板淹没缺陷信号(AUC 0.543)。
分款 AUC 异质(冰晶 0.767 / 甜品 0.444)提示参照系越紧机制越有效。
本实验:每款只用同款 bank-good 建库与直方图集,款内打分,
汇报每款 AUC + 加权池化 AUC(权 = n_bad*n_good),与全局库对照。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_bank_eval import (ROOT, assign, auc, build_bank, frame_hist,  # noqa: E402
                             load_manifest, score_video, spherical_kmeans,
                             split_goods, video_frames_tensors)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = ROOT / "data/corpus_patch_feat"
    man = load_manifest(ROOT / "data/prod500/mech_subset.tsv")
    goods = man[man.label == "good"]
    bank_rels, eval_rels = split_goods(goods)
    bank_set, eval_set = set(bank_rels), set(eval_rels)

    rows = []
    for style, grp in man.groupby("style"):
        b_rels = [r for r in grp.rel if r in bank_set]
        e_rels = [r for r in grp.rel if r in eval_set]
        bad_rels = grp[grp.label != "good"].rel.tolist()
        if len(b_rels) < 5 or len(e_rels) < 3 or len(bad_rels) < 5:
            print(f"{style}: 样本不足跳过 (bank {len(b_rels)} eval {len(e_rels)} bad {len(bad_rels)})")
            continue
        print(f"== {style}: bank {len(b_rels)} / eval {len(e_rels)} / bad {len(bad_rels)}",
              flush=True)
        bank = build_bank(feat_dir, b_rels, device)
        cents = spherical_kmeans(bank)
        hists = []
        for rel in b_rels:
            p = feat_dir / rel.replace(".mp4", ".npz")
            if p.exists():
                for x, m in video_frames_tensors(p, device):
                    if m.any():
                        hists.append(frame_hist(assign(x[m], cents)))
        hists = np.stack(hists)
        for group, rels in (("eval_good", e_rels), ("bad", bad_rels)):
            for rel in rels:
                p = feat_dir / rel.replace(".mp4", ".npz")
                if not p.exists():
                    continue
                rec = {"rel": rel, "style": style, "group": group}
                rec.update(score_video(p, bank, cents, hists, device))
                rows.append(rec)
        del bank
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/prod500/patch_bank_scores_style.csv", index=False)
    sc = [c for c in df.columns if c.startswith("s_")]
    print("\n== 按款分库 AUC(款内)==")
    pooled = {c: [0.0, 0.0] for c in sc}
    for style, grp in df.groupby("style"):
        b = grp[grp.group == "bad"]
        g = grp[grp.group == "eval_good"]
        w = len(b) * len(g)
        line = f" {style} ({len(b)},{len(g)}):"
        for c in ("s_hist_mean", "s_top5_fg_mean", "s_top1_fg_mean"):
            if c in grp:
                a = auc(b[c].values, g[c].values)
                line += f" {c.replace('s_','').replace('_mean','')}={a:.3f}"
        print(line)
        for c in sc:
            a = auc(b[c].values, g[c].values)
            if not np.isnan(a):
                pooled[c][0] += w * a
                pooled[c][1] += w
    print("\n== 加权池化 AUC ==")
    res = sorted(((v[0] / v[1], c) for c, v in pooled.items() if v[1] > 0), reverse=True)
    for a, c in res:
        print(f"  {c:22s} AUC={a:.3f}")
    print("STYLE_EVAL_DONE")


if __name__ == "__main__":
    main()
