#!/usr/bin/env python
"""E61b:眉毛二段式复核(2026-08-12)。v2 整体重写判定失败(keep 2.8%),改为:
v1 命中不动,针对命中视频追加一个聚焦问题——"被认作眉毛的线条是否其实是眯起的眼睛"。
排除条款增量添加(E25 定律),v1 的判定框架零改动。
验证集同 E61:must-keep(三锚+train命中)应回答 is_eye=false;must-drop(11 误报)应 is_eye=true。
判准(冻结):keep 保留率(is_eye=false)≥90% 且 drop 删除率(is_eye=true)≥60%。
输出 data/brow_recheck.jsonl。
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

Q3 = """以下最多16张图是同一条AI生成视频的逐帧画面,已放大到毛绒蘑菇角色「蘑菇TUTU」的头部区域(按时间顺序)。
背景:此前的检测认为画面中角色眼睛上方存在"眉毛状线条"。请你复核一个具体问题:
TUTU 的官方设定是两颗实心黑豆眼、无眉毛。TUTU 眯眼或闭眼时,黑豆眼本身会变成一条黑色横线或弧线。
问题:被认作"眉毛"的那条线,它更可能是什么?
- 如果在任意帧里,该线条**下方或同位置**能看到另外的圆点状眼睛(即线条与眼睛同时存在),那它是眉毛 → is_eye=false
- 如果该线条出现的帧里都看不到另外的眼睛(线条本身就位于眼睛的位置),那它是眯起的眼睛 → is_eye=true
输出一行JSON:{"is_eye": true/false, "note": "一句话"}"""

OUT = ROOT / "data/brow_recheck.jsonl"

MUST_KEEP_EVAL = ["5719.mp4", "6565.mp4", "6228.mp4"]


def frames_of(rel):
    for cdir in (ROOT / "data/crops_geo", ROOT / "data/crops_v3"):
        d = cdir / rel.replace(".mp4", "")
        jp = sorted(d.glob("f*.jpg"))[:16]
        if len(jp) >= 8:
            ims = []
            for p in jp:
                im = cv2.imread(str(p))
                if im is None:
                    break
                H, W = im.shape[:2]
                u = im[0:int(H * 0.62), :]
                u = cv2.resize(u, (W * 2, int(H * 0.62 * 2)), interpolation=cv2.INTER_CUBIC)
                ims.append(Image.fromarray(cv2.cvtColor(u, cv2.COLOR_BGR2RGB)))
            if len(ims) >= 8:
                return ims
    return None


def main():
    rel_of = {}
    for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
        if l.strip():
            rel = l.split("\t")[0]
            rel_of[rel.split("/")[-1]] = rel
    cl = json.load(open(ROOT / "data/s3/relabel_candidates_v3.json"))
    md = [x["video"] for x in cl if x["axis"] == "眉毛" and x.get("user_verdict") == "keep"]
    tr_v = {json.loads(l)["video"] for l in open(ROOT / "splits/train_v3.jsonl")}
    mk = list(MUST_KEEP_EVAL)
    for l in open(ROOT / "data/s3/brow_confirm_full.jsonl"):
        d = json.loads(l)
        v = os.path.basename(d["rel"])
        if d.get("eyebrows") and v in tr_v:
            mk.append(v)
    mk = mk[:43]
    todo = [(v, "KEEP") for v in mk] + [(v, "DROP") for v in md]
    print(f"must-keep {len(mk)} | must-drop {len(md)}", flush=True)

    done = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                done.add(json.loads(l)["video"])
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

    f = open(OUT, "a")
    t0 = time.time()
    n = 0
    for v, grp in todo:
        if v in done or v not in rel_of:
            continue
        row = {"video": v, "grp": grp}
        try:
            ims = frames_of(rel_of[v])
            if ims is None:
                row["error"] = "no_frames"
            else:
                content = [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": Q3}]
                inputs = proc.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=110, do_sample=False)
                text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
                j = parse(text)
                if j is not None and isinstance(j.get("is_eye"), bool):
                    row.update(is_eye=j["is_eye"], note=j.get("note", "")[:70])
                else:
                    row["parse_error"] = text[:70]
        except Exception as e:
            row["error"] = repr(e)[:100]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        if n % 15 == 0:
            f.flush()
            print(f"[{n}/{len(todo)}] {(time.time()-t0)/n:.1f}s/条", flush=True)
    f.close()

    kk = kt = dk = dt = 0
    anchors = {}
    for l in open(OUT):
        d = json.loads(l)
        if "is_eye" not in d:
            continue
        if d["grp"] == "KEEP":
            kt += 1
            kk += int(not d["is_eye"])
            if d["video"] in MUST_KEEP_EVAL:
                anchors[d["video"]] = ("保留" if not d["is_eye"] else "误删")
        else:
            dt += 1
            dk += int(d["is_eye"])
    print(f"must-keep 保留 {kk}/{kt} = {kk/max(1,kt):.1%} (判准>=90%)", flush=True)
    print(f"must-drop 删除 {dk}/{dt} = {dk/max(1,dt):.1%} (判准>=60%)", flush=True)
    print(f"三锚: {anchors}", flush=True)
    print("E61B_DONE", flush=True)


if __name__ == "__main__":
    main()
