#!/usr/bin/env python
"""每视频角色 bbox 高度(384 坐标系,中位数)——运动规则 v2 的速度归一化分母。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/mech")
SIDE = 384


def main():
    rels = [l.split("\t")[0] for l in (ROOT / "manifest_all.tsv").read_text().splitlines() if l.strip()]
    out = open(ROOT / "data/bbox_h.csv", "w", newline="")
    w = csv.writer(out)
    w.writerow(["rel", "bbox_h384", "bbox_w384", "frame_maxside"])
    for k, rel in enumerate(rels):
        p = ROOT / "data/sam3_feat" / rel.replace(".mp4", ".npz")
        row = ["nan", "nan", "nan"]
        if p.exists():
            try:
                z = np.load(p, allow_pickle=True)
                geo = json.loads(str(z["geo"]))
                if geo:
                    cap = cv2.VideoCapture(str(ROOT / "data/corpus_videos" / rel))
                    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH); H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    cap.release()
                    ms = max(H, W)
                    if ms > 0:
                        s = SIDE / ms
                        hs = [(g["bbox"][3] - g["bbox"][1]) * s for g in geo]
                        ws = [(g["bbox"][2] - g["bbox"][0]) * s for g in geo]
                        row = [f"{np.median(hs):.2f}", f"{np.median(ws):.2f}", f"{ms:.0f}"]
            except Exception:
                pass
        w.writerow([rel] + row)
        if (k + 1) % 1000 == 0:
            print(k + 1, flush=True)
    out.close()
    print("BBOXH_DONE", flush=True)


if __name__ == "__main__":
    main()
