#!/usr/bin/env python
"""W5a:DOVER 全语料打分(technical/aesthetic/fused 三列)。批量版 evaluate_one_video,模型只装载一次。"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/mech/DOVER")

import numpy as np
import torch
import yaml

from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from dover.models import DOVER

MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
STD = torch.FloatTensor([58.395, 57.12, 57.375])


def fuse(results):
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (results[1] + 0.08285) / 0.03774 * 0.3896
    return 1 / (1 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/root/mech/data/corpus_videos")
    ap.add_argument("--manifest", default="/root/mech/manifest_all.tsv")
    ap.add_argument("--out", default="/root/mech/data/w5_dover.csv")
    ap.add_argument("--opt", default="/root/mech/DOVER/dover.yml")
    ap.add_argument("--ckpt", default="/root/mech/models/dover/DOVER.pth")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rels = [l.split("\t")[0] for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    done = set()
    out_p = Path(args.out)
    if out_p.exists():
        done = {r[0] for r in csv.reader(open(out_p))}
    else:
        csv.writer(open(out_p, "w", newline="")).writerow(["rel", "dover_tech", "dover_aes", "dover_fused"])
    todo = [r for r in rels if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"todo {len(todo)}", flush=True)

    opt = yaml.safe_load(open(args.opt))
    evaluator = DOVER(**opt["model"]["args"]).to("cuda")
    evaluator.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    evaluator.eval()
    dopt = opt["data"]["val-l1080p"]["args"]
    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
        else:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"], sopt["t_frag"],
                sopt["frame_interval"], sopt["num_clips"])
    print("dover loaded", flush=True)

    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    for k, rel in enumerate(todo):
        row = None
        try:
            views, _ = spatial_temporal_view_decomposition(
                str(Path(args.videos_dir) / rel), dopt["sample_types"], temporal_samplers)
            for key, v in views.items():
                num_clips = dopt["sample_types"][key].get("num_clips", 1)
                views[key] = (((v.permute(1, 2, 3, 0) - MEAN) / STD)
                              .permute(3, 0, 1, 2)
                              .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                              .transpose(0, 1).to("cuda"))
            with torch.inference_mode():
                results = [r.mean().item() for r in evaluator(views)]
            row = [f"{results[0]:.5f}", f"{results[1]:.5f}", f"{fuse(results):.5f}"]
        except Exception as e:
            if k < 3:
                print("ERR", rel, repr(e)[:100], flush=True)
        w.writerow([rel] + (row if row else ["nan"] * 3))
        if (k + 1) % 200 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    f.close()
    print("W5A_DONE", flush=True)


if __name__ == "__main__":
    main()
