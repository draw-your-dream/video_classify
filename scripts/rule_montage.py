#!/usr/bin/env python
"""规则审核 gallery 素材:每条候选规则命中的视频 → 4x4 帧拼图(旗标帧红框)。
盒侧运行,读 e12_flags.jsonl + sam3_cutouts,输出 review_montage/<rule>/<label>_<vid>.jpg + index.json"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/mech")
OUT = ROOT / "review_montage"
CATS = ["eyebrows", "tail", "extra_limb", "missing_limb", "eye_anomaly", "mouth_anomaly"]
CELL = 200


def montage(rel, flag_frames):
    d = ROOT / "data/sam3_cutouts" / rel.replace(".mp4", "")
    jpgs = sorted(d.glob("f*.jpg"))[:16]
    if len(jpgs) < 4:
        return None
    cells = []
    for i in range(16):
        if i < len(jpgs):
            im = cv2.imread(str(jpgs[i]))
            im = cv2.resize(im, (CELL, CELL)) if im is not None else np.full((CELL, CELL, 3), 230, np.uint8)
        else:
            im = np.full((CELL, CELL, 3), 230, np.uint8)
        idx = int(jpgs[i].stem[1:]) if i < len(jpgs) else -1
        color = (0, 0, 220) if idx in flag_frames else (200, 200, 200)
        th = 6 if idx in flag_frames else 1
        cv2.rectangle(im, (0, 0), (CELL - 1, CELL - 1), color, th)
        cv2.putText(im, f"f{idx}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
        cv2.putText(im, f"f{idx}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
        cells.append(im)
    rows = [np.concatenate(cells[r * 4:(r + 1) * 4], 1) for r in range(4)]
    return np.concatenate(rows, 0)


def main():
    labels = {}
    for f in ("corpus_full.tsv", "manifest_rlhf.tsv"):
        for l in (ROOT / f).read_text().splitlines():
            if l.strip():
                rel, lab = l.split("\t")
                labels[rel] = lab
    flag = {}
    for l in open(ROOT / "data/e12_flags.jsonl"):
        j = json.loads(l)
        if "parse_error" in j or "error" in j:
            continue
        flag[j["rel"]] = j
    rng = random.Random(3)
    plan = {}
    for c, cap_bad, cap_good in (("eyebrows", 40, 40), ("tail", 30, 30), ("eye_anomaly", 30, 30),
                                 ("missing_limb", 15, 15), ("mouth_anomaly", 15, 15)):
        hits_b = [r for r, j in flag.items() if len(j.get(c) or []) >= 1 and labels.get(r, "?") == "bad"]
        hits_g = [r for r, j in flag.items() if len(j.get(c) or []) >= 1 and labels.get(r, "?") != "bad"]
        rng.shuffle(hits_b); rng.shuffle(hits_g)
        plan[c] = [(r, "bad") for r in hits_b[:cap_bad]] + [(r, labels.get(r, "good")) for r in hits_g[:cap_good]]
    index = {}
    for c, items in plan.items():
        d = OUT / c
        d.mkdir(parents=True, exist_ok=True)
        index[c] = []
        for rel, lab in items:
            m = montage(rel, set(flag[rel].get(c) or []))
            if m is None:
                continue
            name = f"{lab}_{rel.replace('/', '_').replace('.mp4', '')}.jpg"
            cv2.imwrite(str(d / name), m, [cv2.IMWRITE_JPEG_QUALITY, 88])
            index[c].append({"rel": rel, "label": lab, "img": f"{c}/{name}",
                             "frames": flag[rel].get(c) or [], "note": flag[rel].get("note", "")})
        print(c, len(index[c]), flush=True)
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    print("MONTAGE_DONE")


if __name__ == "__main__":
    main()
