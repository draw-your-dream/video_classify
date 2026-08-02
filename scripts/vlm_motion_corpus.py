#!/usr/bin/env python
"""E4 运动判官(Qwen3-VL-32B,2026-08-02 预注册,prompt/采样/用法冻结)。

输入 = 裸视频帧 12 张等间隔(max边448,不抠像——运动判断需要背景参照);
输出 = 每条一行 JSON {stiff,jerky,limbs_frozen,no_motion,motion_bad,reason}。
覆盖 corpus_full + rlhf 全部 4845;断点续跑;主分 motion_bad,副分 max(四子项)。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

PROMPT = """以下12张图按时间顺序等间隔取自同一条约5秒的AI生成短视频。主角是毛绒蘑菇角色「蘑菇TUTU」。
你的任务是仅根据这些帧判断主角的运动质量,不评价形象、画质或背景内容。
四类运动缺陷定义:
1. stiff 僵硬:身体像一块硬物被整体翻动或平移,四肢与身体没有自然弯曲和缓冲。
2. jerky 卡顿/不连贯:相邻时刻姿态或位置跳跃,动作一顿一顿,衔接断裂。
3. limbs_frozen 四肢不动:身体在移动或转向,但四肢完全固定不摆动,呈摆件式平移。
4. no_motion 无有效动态:主角全程几乎静止,或只有镜头推拉/背景在动、主角没有自主动作。
注意:动作幅度小但流畅自然不算缺陷;判断连贯性要看相邻帧姿态差是否平滑过渡。
每类给0-100分(0=完全没有该缺陷,100=确定有且严重),再给 motion_bad 总分0-100:
该视频因运动质量问题应被判废的置信度。
只输出一行JSON,不要输出其他内容:
{"stiff": 0, "jerky": 0, "limbs_frozen": 0, "no_motion": 0, "motion_bad": 0, "reason": "简短一句"}"""

N_FRAMES = 12
MAX_SIDE = 448


def read_frames(vp: Path):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return []
    idxs = [int(round(i * (n - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)]
    out, want = [], set(idxs)
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            h, w = fr.shape[:2]
            s = MAX_SIDE / max(h, w)
            fr = cv2.resize(fr, (int(w * s), int(h * s)))
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        k += 1
    cap.release()
    return out


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--out", default=str(ROOT / "data/vlm_motion.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for mf in (ROOT / "corpus_full.tsv", ROOT / "manifest_rlhf.tsv"):
        rows += [l.split("\t")[0] for l in mf.read_text().splitlines() if l.strip()]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["rel"] for l in Path(args.out).read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"total {len(rows)} done {len(done)} todo {len(todo)}", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)

    out_f = open(args.out, "a")
    t0 = time.time()
    for k, rel in enumerate(todo):
        frames = read_frames(Path(args.videos_dir) / rel)
        if len(frames) < 6:
            out_f.write(json.dumps({"rel": rel, "error": "no_frames"}, ensure_ascii=False) + "\n")
            out_f.flush()
            continue
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": PROMPT})
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0]
        j = parse_json(text) or {"parse_error": text[:150]}
        j["rel"] = rel
        out_f.write(json.dumps(j, ensure_ascii=False) + "\n")
        if (k + 1) % 25 == 0:
            out_f.flush()
            el = time.time() - t0
            print(f"[{k+1}/{len(todo)}] {el/(k+1):.1f}s/vid elapsed {el/60:.1f}m", flush=True)
    out_f.close()
    print("VLM_MOTION_DONE", flush=True)


if __name__ == "__main__":
    main()
