#!/usr/bin/env python
"""线A 对照:训好的头在 held-out 合成缺陷上的判别力(仪器自检)。

eval_good 帧(训练从未见过)各生成 缺陷版 与 干净版,帧级 AUC。
高(>0.85)=> 头有效,真 bad 失败在标签侧;低 => 头/合成本身失败。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axis_rself_local import split_goods, stem_seed, auc  # noqa: E402
from synth_defects import DEFECTS, synthesize  # noqa: E402
from train_synth_head import Head, load_backbone, patch_feats, to_tensor, frame_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    device = "cuda"
    rows = [l.split("\t") for l in
            (ROOT / "data/prod500/mech_subset.tsv").read_text().splitlines() if l.strip()]
    goods_by_style = defaultdict(list)
    for rel, label in rows:
        if label == "good":
            goods_by_style[rel.split("/")[0]].append(rel)
    _, eval_rels = split_goods(goods_by_style)

    frames = []
    for rel in eval_rels:
        frames += sorted((ROOT / "data/sam3_cutouts" / rel.replace(".mp4", "")).glob("f*.jpg"))
    rng = np.random.default_rng(stem_seed("sanity"))
    picks = [frames[i] for i in rng.choice(len(frames), 240, replace=False)]

    backbone = load_backbone(device)
    head = Head().to(device)
    head.load_state_dict(torch.load(ROOT / "data/prod500/synth_head_v0.pt"))
    head.eval()

    kinds = list(DEFECTS)
    per_kind = {k: ([], []) for k in kinds}
    with torch.no_grad():
        for p in picks:
            im = cv2.imread(str(p))
            if im is None:
                continue
            k = kinds[int(rng.integers(len(kinds)))]
            got = synthesize(im, k, rng)
            if got is None:
                continue
            xs = torch.stack([to_tensor(got[0]), to_tensor(im)]).to(device)
            fs = frame_score(head(patch_feats(backbone, xs))).cpu().numpy()
            per_kind[k][0].append(fs[0])
            per_kind[k][1].append(fs[1])

    allp = np.array(sum((v[0] for v in per_kind.values()), []))
    alln = np.array(sum((v[1] for v in per_kind.values()), []))
    print(f"held-out 合成帧级 AUC = {auc(allp, alln):.3f}  (n={len(allp)})")
    for k, (p_, n_) in per_kind.items():
        if len(p_) >= 10:
            print(f"  {k:12s} AUC={auc(np.array(p_), np.array(n_)):.3f} (n={len(p_)})")
    print("SANITY_DONE")


if __name__ == "__main__":
    main()
