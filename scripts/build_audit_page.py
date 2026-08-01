#!/usr/bin/env python
"""盲审页构建(2026-08-01 预注册裁决实验)。

组成:VLM旗标bad 7 + 随机未旗标bad 23 + 随机eval_good 10,乱序匿名 case01..case40。
每 case = 16 帧抠像 4x4 网格图。产出 audit_page/(index.html + imgs/)+ 私密答案表。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axis_rself_local import split_goods, stem_seed  # noqa: E402
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
CUT = ROOT / "data/sam3_cutouts"
OUT = ROOT / "data/prod500/audit_page"
FLAGS = ["black_spots", "mouth_abnormal", "limb_count_abnormal",
         "tail", "body_elongated", "other_defect"]


def grid(rel: str, cell: int = 220) -> Image.Image | None:
    d = CUT / rel.replace(".mp4", "")
    jpgs = sorted(d.glob("f*.jpg"))
    if len(jpgs) < 4:
        return None
    canvas = Image.new("RGB", (cell * 4, cell * 4), (255, 255, 255))
    for i, j in enumerate(jpgs[:16]):
        im = Image.open(j).convert("RGB").resize((cell, cell))
        canvas.paste(im, ((i % 4) * cell, (i // 4) * cell))
    return canvas


def main():
    recs = [json.loads(l) for l in
            (ROOT / "data/prod500/vlm_v2.jsonl").read_text().splitlines() if l.strip()]
    flagged = [r["rel"] for r in recs if r.get("group") == "bad"
               and any(r.get(f) for f in FLAGS)]
    judged_bads = [r["rel"] for r in recs if r.get("group") == "bad"]
    unflagged = sorted(set(judged_bads) - set(flagged))

    rows = [l.split("\t") for l in
            (ROOT / "data/prod500/mech_subset.tsv").read_text().splitlines() if l.strip()]
    goods_by_style = defaultdict(list)
    for rel, label in rows:
        if label == "good":
            goods_by_style[rel.split("/")[0]].append(rel)
    _, eval_rels = split_goods(goods_by_style)

    rng = np.random.default_rng(stem_seed("blind-audit"))
    sel_unflag = [unflagged[i] for i in rng.choice(len(unflagged), 23, replace=False)]
    sel_good = [eval_rels[i] for i in rng.choice(len(eval_rels), 10, replace=False)]

    cases = ([(r, "bad_flagged") for r in flagged]
             + [(r, "bad_unflagged") for r in sel_unflag]
             + [(r, "good") for r in sel_good])
    order = rng.permutation(len(cases))
    cases = [cases[i] for i in order]

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "imgs").mkdir(parents=True)
    key, items = {}, []
    n = 0
    for rel, tag in cases:
        g = grid(rel)
        if g is None:
            continue
        n += 1
        cid = f"case{n:02d}"
        g.save(OUT / "imgs" / f"{cid}.jpg", quality=88)
        key[cid] = {"rel": rel, "tag": tag}
        items.append(cid)

    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>TUTU 盲审</title>",
            "<style>body{font-family:sans-serif;background:#222;color:#eee;margin:20px}",
            "img{width:100%;max-width:880px;border:1px solid #555}",
            "h3{margin:30px 0 6px}</style></head><body>",
            "<h2>TUTU 静帧盲审(40 例)</h2>",
            "<p>每例是同一条视频的 16 帧角色抠像(时间顺序)。请逐例判断:"
            "<b>仅凭这些静帧,角色形象是否有明确缺陷</b>(黑点/嘴/肢体数/尾巴/比例/五官崩坏等;"
            "姿态、视角、饰品、遮挡不算)。把「有缺陷」的 case 编号记下来回复即可,"
            "拿不准可标「?」。</p>"]
    for cid in items:
        html.append(f"<h3>{cid}</h3><img src='imgs/{cid}.jpg' loading='lazy'>")
    html.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    json.dump(key, open(ROOT / "data/prod500/audit_key.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"cases: {n} (flagged {len(flagged)} / unflag {len(sel_unflag)} / good {len(sel_good)})")
    print("page ->", OUT / "index.html")


if __name__ == "__main__":
    main()
