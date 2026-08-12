#!/usr/bin/env python
"""E60:悬空规则重建 + eval zone 扫描(2026-08-12 预注册)。

历史悬空规则(twokey_v3,3% 盲测误报,31+确认)脚本随旧机器丢失,从未扫过 eval。
重建纪律:①prompt 按历史确认 note 的语义忠实重建(悬空=身体下方无支撑面、未被手持、
非跳跃瞬间;绳吊算悬空——历史 15999 计命中);②先在历史 238 候选上重放,
与历史命中集(33)一致率 ≥ 80% 才算重建成功;③过门才扫 eval zone(e18<0.30,label-blind)。
输入:crops_v3 16 帧(全身可见,判支撑需要脚下区域)。
输出 data/floating_confirm.jsonl。
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

Q = """以下最多16张图是同一条AI生成视频的逐帧画面(按时间顺序,第0张起)。只判断毛绒蘑菇角色「蘑菇TUTU」本体。
官方物理设定:TUTU 是有重量的毛绒玩偶,正常情况下应有支撑——站/坐/趴在某个表面上,或被手拿着/举着。
问题:是否看到角色**持续悬浮在空中**——身体下方没有任何可见支撑面、没有被手持或倚靠,且不是跳跃的瞬间动作?
被绳子吊在空中算悬空;跳跃/被抛起的短暂瞬间不算;身体部分搭在物体上(扒住栏杆、趴在边沿)不算。
输出一行JSON:{"floating": [帧号], "note": "一句话"}"""

OUT = ROOT / "data/floating_confirm.jsonl"


def frames_of(rel, n=16):
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    jp = sorted(d.glob("f*.jpg"))[:n]
    if len(jp) < 8:
        return None
    ims = []
    for p in jp:
        im = cv2.imread(str(p))
        if im is None:
            return None
        H, W = im.shape[:2]
        s = 448 / max(H, W)
        ims.append(Image.fromarray(cv2.cvtColor(
            cv2.resize(im, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB)))
    return ims


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["replay", "eval"], default="replay")
    args = ap.parse_args()

    if args.stage == "replay":
        hist = [json.loads(l) for l in open(ROOT / "data/s3/twokey_confirm_v3.jsonl")]
        todo = [(d["rel"], "H") for d in hist if d.get("rule") == "floating"]
    else:
        ev = [json.loads(l) for l in open(ROOT / "splits/eval_v3.jsonl")]
        rel_of = {}
        for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
            if l.strip():
                rel = l.split("\t")[0]
                rel_of[rel.split("/")[-1]] = rel
        pr = {r["video"]: float(r["p_e18"]) for r in csv.DictReader(
            open(ROOT / "data/s3/predictions_e18_eval.csv"))}
        todo = [(rel_of[r["video"]], "EV") for r in ev
                if r["video"] in rel_of and pr.get(r["video"], 1) < 0.30]
    print(f"stage={args.stage} 待跑 {len(todo)}", flush=True)

    done = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                d = json.loads(l)
                done.add((d["rel"], d["stage"]))
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
    for rel, grp in todo:
        if (rel, args.stage) in done:
            continue
        row = {"rel": rel, "grp": grp, "stage": args.stage}
        try:
            ims = frames_of(rel)
            if ims is None:
                row["error"] = "no_frames"
            else:
                content = [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": Q}]
                inputs = proc.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=120, do_sample=False)
                text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
                j = parse(text)
                if j is not None:
                    row.update(floating=j.get("floating", []), note=j.get("note", "")[:80])
                else:
                    row["parse_error"] = text[:80]
        except Exception as e:
            row["error"] = repr(e)[:100]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        if n % 25 == 0:
            f.flush()
            print(f"[{n}/{len(todo)}] {(time.time()-t0)/n:.1f}s/条", flush=True)
    f.close()

    if args.stage == "replay":
        hist = {d["rel"]: bool(d.get("hit")) for d in
                (json.loads(l) for l in open(ROOT / "data/s3/twokey_confirm_v3.jsonl"))
                if d.get("rule") == "floating"}
        agree = tot = hit_new = hit_old = 0
        for l in open(OUT):
            d = json.loads(l)
            if d.get("stage") != "replay" or "floating" not in d:
                continue
            new = bool(d["floating"])
            old = hist.get(d["rel"])
            if old is None:
                continue
            tot += 1
            agree += int(new == old)
            hit_new += int(new)
            hit_old += int(old)
        if tot:
            print(f"重放一致率 {agree}/{tot} = {agree/tot:.1%}  (新命中 {hit_new} vs 历史 {hit_old})",
                  flush=True)
    print("E60_DONE", flush=True)


if __name__ == "__main__":
    main()
