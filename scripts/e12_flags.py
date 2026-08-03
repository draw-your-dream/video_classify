#!/usr/bin/env python
"""E12:VLM 事实提取器(Qwen3-VL-32B,2026-08-03 预注册,prompt 冻结)。

输入 = 每视频 16 帧白底抠像;只问客观事实(六类异常的确定帧号),不问质量。
输出 jsonl 断点续跑。"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

PROMPT = """以下最多16张图是同一条AI生成视频中角色「蘑菇TUTU」的逐帧白底抠像,按时间顺序编号(第0张、第1张……)。
官方设定:TUTU 无眉毛、无牙齿、无舌头、无尾巴;两只手臂两条腿,短圆无手指;两颗实心黑豆眼;一张小嘴。
任务:逐帧检查是否出现下列客观异常。只在确定看到时报告;画质模糊、遮挡、姿态导致的看不清一律不报。
1. eyebrows:出现眉毛的帧号
2. tail:出现尾巴或多余附属结构的帧号
3. extra_limb:可见手臂与腿总数超过4、或出现明显多余肢体的帧号
4. missing_limb:身体正面可见却缺失手臂/腿的帧号(被遮挡不算)
5. eye_anomaly:眼睛异常(多于两只/形状崩坏/位置错乱)的帧号
6. mouth_anomaly:嘴巴撕裂、扭曲、位置错误的帧号
只输出一行JSON,异常帧号放进对应数组,没有则为空数组:
{"eyebrows": [], "tail": [], "extra_limb": [], "missing_limb": [], "eye_anomaly": [], "mouth_anomaly": [], "note": "一句话"}"""


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
    ap.add_argument("--cut-dir", default="/root/mech/data/sam3_cutouts")
    ap.add_argument("--manifest", default="/root/mech/manifest_all.tsv")
    ap.add_argument("--out", default="/root/mech/data/e12_flags.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rels = [l.split("\t")[0] for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["rel"] for l in Path(args.out).read_text().splitlines() if l.strip()}
    todo = [r for r in rels if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"total {len(rels)} done {len(done)} todo {len(todo)}", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)

    out_f = open(args.out, "a")
    t0 = time.time()
    for k, rel in enumerate(todo):
        d = Path(args.cut_dir) / rel.replace(".mp4", "")
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
        j["n_imgs"] = len(jpgs)
        out_f.write(json.dumps(j, ensure_ascii=False) + "\n")
        if (k + 1) % 25 == 0:
            out_f.flush()
            el = time.time() - t0
            print(f"[{k+1}/{len(todo)}] {el/(k+1):.1f}s/vid elapsed {el/60:.1f}m", flush=True)
    out_f.close()
    print("E12_DONE", flush=True)


if __name__ == "__main__":
    main()
