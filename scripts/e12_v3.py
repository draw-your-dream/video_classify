#!/usr/bin/env python
"""E12-v3:精修prompt + 宽松原图裁剪(2026-08-04 预注册,方案A)。

只对 v1 曾旗标(eyebrows/tail/eye_anomaly/mouth_anomaly 任一)的视频重问。
修订点:眉毛排除眯眼;尾巴排除穿戴物;眼睛排除高光/遮挡;嘴巴区分说话开合与崩坏。"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

PROMPT = """以下最多16张图是同一条AI生成视频的逐帧画面裁剪(以角色为中心,含背景与周边物体),按时间顺序编号(第0张、第1张……)。\n画面中可能出现其他玩偶或物体:只判断毛绒蘑菇角色「蘑菇TUTU」本体(浅黄色身体、红色伞盖),其他一概忽略。\n角色手持或佩戴的物品(望远镜、放大镜、背包等)属于合理设定,注意不要把持物遮挡当成五官异常。\n总原则:凡被物体、手、视角遮挡或因画质看不清的部位,一律默认正常,不报任何异常;只报画面中清晰可见、确定存在的异常,绝不从「看不到某部位」推断出问题。
官方设定:TUTU 无眉毛、无尾巴;两颗实心黑豆眼;一张小嘴,说话或表情变化时嘴会开合。
逐帧检查下列异常。只报确定看到的,并严格遵守每条的排除规则:
1. eyebrows(眉毛):必须是位于眼睛上方、与眼睛明显分离的独立弧线或条状结构。
   眯起的窄眼、眼睛本身形状的变化,不算眉毛。
2. tail(多余附属结构):只报从身体长出来的、非穿戴的结构(如尾巴、多余肢芽)。
   穿戴或携带的物品(背包、围巾、帽饰、挂件、手持道具)是合理设定,不算。
3. eye_anomaly(眼睛异常):只报确定的眼睛数量错误、位置错乱、形状崩坏。
   眼内高光反光、画质模糊、被手/道具/角度遮挡导致看不清的,一律不算。
4. mouth_broken(嘴部崩坏):只报嘴部撕裂、错位、出现两张嘴等结构性崩坏。
   说话引起的张嘴闭嘴、嘴形大小变化是正常动画,不算。
输出一行JSON,异常帧号放对应数组,没有则空数组:
{"eyebrows": [], "tail": [], "eye_anomaly": [], "mouth_broken": [], "note": "一句话"}"""


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut-dir", default="/root/mech/data/crops_v3")
    ap.add_argument("--v1", default="/root/mech/data/e12_flags.jsonl")
    ap.add_argument("--out", default="/root/mech/data/e12_v3.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    CATS = ["eyebrows", "tail", "eye_anomaly", "mouth_anomaly"]
    todo = []
    for l in open(args.v1):
        j = json.loads(l)
        if any(len(j.get(c) or []) >= 1 for c in CATS):
            todo.append(j["rel"])
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["rel"] for l in Path(args.out).read_text().splitlines() if l.strip()}
    todo = [r for r in todo if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"v1旗标候选待重问 {len(todo)}", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)

    out_f = open(args.out, "a")
    t0 = time.time()
    for k, rel in enumerate(todo):
        d = Path(args.cut_dir) / rel.replace(".mp4", "")
        qcp = d / "qc.json"
        if qcp.exists():
            import json as _j
            if _j.loads(qcp.read_text()).get("failed"):
                out_f.write(json.dumps({"rel": rel, "error": "qc_failed"}) + "\n")
                continue
        jpgs = sorted(d.glob("f*.jpg"))[:16]
        if len(jpgs) < 4:
            out_f.write(json.dumps({"rel": rel, "error": "no_cutouts"}) + "\n")
            continue
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in jpgs]
        content.append({"type": "text", "text": PROMPT})
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=180, do_sample=False)
        text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        j = parse_json(text) or {"parse_error": text[:150]}
        j["rel"] = rel
        out_f.write(json.dumps(j, ensure_ascii=False) + "\n")
        if (k + 1) % 25 == 0:
            out_f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.1f}s/vid", flush=True)
    out_f.close()
    print("E12V3_DONE", flush=True)


if __name__ == "__main__":
    main()
