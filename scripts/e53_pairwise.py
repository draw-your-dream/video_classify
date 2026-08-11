#!/usr/bin/env python
"""E53:VLM 成对活物感比较试点(2026-08-11 预注册,自主新想法批)。

与 E4(绝对打分判官,midrank 0.546 已死)的本质区别:成对比较("哪边更像活物")
远易于绝对判断("这条像不像活物")——人类感知实验的标准结论,本项目从未试过。
设计:候选(train 运动尾部bad 40 + train good 40,盲序)× 3 条固定参照 good,
每次比较给两张 2×4 八帧拼图,位置交替防顺序偏置。
判准(发车前冻结):运动bad 的平均"输给参照"票数 − good 的 ≥ 1.0(满分3),
且 good 的高输票率(≥2票)≤ 20%,才谈扩展。
输出 data/pairwise_pilot.jsonl。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")

Q = """图A与图B各是一条AI生成视频的8帧拼图(每张图内按时间顺序排列,2行4列)。两条视频里都有毛绒蘑菇角色「蘑菇TUTU」。
请比较:哪条视频里角色的动作更自然、更像有生命的活物?(判断依据:身体柔软有弹性而非僵硬刚体、动作连贯不卡顿、有表情或姿态变化、与环境互动自然)
输出一行JSON:{"more_alive": "A" 或 "B", "note": "一句话"}"""

OUT = ROOT / "data/pairwise_pilot.jsonl"


def montage(rel):
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    jp = sorted(d.glob("f*.jpg"))
    if len(jp) < 8:
        return None
    idx = np.linspace(0, len(jp) - 1, 8).round().astype(int)
    tiles = []
    for i in idx:
        im = cv2.imread(str(jp[i]))
        if im is None:
            return None
        tiles.append(cv2.resize(im, (224, 224)))
    rows = [np.hstack(tiles[:4]), np.hstack(tiles[4:])]
    return Image.fromarray(cv2.cvtColor(np.vstack(rows), cv2.COLOR_BGR2RGB))


def main():
    tr = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    rel_of = {}
    for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
        if l.strip():
            rel = l.split("\t")[0]
            rel_of[rel.split("/")[-1]] = rel
    reasons = {r["path"]: r.get("reasons", "") for r in csv.DictReader(
        open(ROOT / "data/s3/merged_labels.csv", encoding="utf-8-sig"))}
    e18 = np.load(ROOT / "data/s3/e18_train_oof.npy")
    KW = ("僵硬", "卡顿/少活人感", "四肢不动", "静止不动")
    rig = {}
    for r in csv.DictReader(open(ROOT / "data/rigid_feats.csv")):
        try:
            rig[r["rel"].split("/")[-1]] = float(r["m_mean"])
        except Exception:
            pass

    bad_i = [i for i, r in enumerate(tr) if r["label"] == "bad" and r["video"] in rel_of]
    bs = np.array([e18[i] for i in bad_i])
    cut = np.quantile(bs, 0.30)
    MT = [tr[i]["video"] for i in bad_i if e18[i] <= cut
          and any(k in reasons.get(tr[i]["video"], "") for k in KW)]
    G_all = [(e18[i], tr[i]["video"]) for i, r in enumerate(tr)
             if r["label"] == "good" and r["video"] in rel_of]
    G_all.sort()
    # 参照:模型最有信心的 good 里,运动量(m_mean)最高的 3 条(确定性)
    refs = sorted(G_all[:200], key=lambda x: -rig.get(x[1], 0))[:3]
    refs = [v for _s, v in refs]
    print("参照 good:", refs, flush=True)
    rng = np.random.RandomState(11)
    cand_b = list(rng.choice(MT, min(40, len(MT)), replace=False))
    goods = [v for _s, v in G_all if v not in refs]
    cand_g = list(rng.choice(goods, len(cand_b), replace=False))
    pool = [(v, "MT") for v in cand_b] + [(v, "G") for v in cand_g]
    rng.shuffle(pool)
    print(f"候选 {len(pool)}(运动bad {len(cand_b)} / good {len(cand_g)})× 3 参照", flush=True)

    done = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                d = json.loads(l)
                done.add((d["cand"], d["ref"]))
            except Exception:
                pass

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-32B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)

    def parse(t):
        m = re.search(r"\{.*\}", t, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None

    mcache = {}

    def M(v):
        if v not in mcache:
            mcache[v] = montage(rel_of[v])
        return mcache[v]

    f = open(OUT, "a")
    t0 = time.time()
    n = 0
    for ci, (cand, grp) in enumerate(pool):
        for ri, ref in enumerate(refs):
            if (cand, ref) in done:
                continue
            mc, mr = M(cand), M(ref)
            row = {"cand": cand, "grp": grp, "ref": ref}
            if mc is None or mr is None:
                row["error"] = "no_montage"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            # 位置交替:偶数次候选=A,奇数次候选=B
            cand_is_A = (ci + ri) % 2 == 0
            ims = [mc, mr] if cand_is_A else [mr, mc]
            row["cand_pos"] = "A" if cand_is_A else "B"
            try:
                content = ([{"type": "text", "text": "图A:"}, {"type": "image", "image": ims[0]},
                            {"type": "text", "text": "图B:"}, {"type": "image", "image": ims[1]},
                            {"type": "text", "text": Q}])
                inputs = proc.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=90, do_sample=False)
                text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
                j = parse(text)
                if j and j.get("more_alive") in ("A", "B"):
                    row["winner"] = j["more_alive"]
                    row["cand_wins"] = int(j["more_alive"] == row["cand_pos"])
                    row["note"] = j.get("note", "")
                else:
                    row["parse_error"] = text[:80]
            except Exception as e:
                row["error"] = repr(e)[:100]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0:
                f.flush()
                print(f"[{n}] {(time.time()-t0)/n:.1f}s/对", flush=True)
    f.close()

    # 即时判读
    votes = {}
    for l in open(OUT):
        d = json.loads(l)
        if "cand_wins" in d:
            votes.setdefault((d["cand"], d["grp"]), []).append(1 - d["cand_wins"])
    lo = {"MT": [], "G": []}
    for (c, g), vs in votes.items():
        lo[g].append(sum(vs))
    for g in ("MT", "G"):
        a = np.array(lo[g])
        if len(a):
            print(f"[{g}] n={len(a)} 平均输票 {a.mean():.2f}/3  高输票(>=2)率 {(a>=2).mean():.1%}",
                  flush=True)
    print("E53_PAIRWISE_DONE", flush=True)


if __name__ == "__main__":
    main()
