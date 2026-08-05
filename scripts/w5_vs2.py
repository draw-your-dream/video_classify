#!/usr/bin/env python
"""W5b:VideoScore2 全语料打分(visual/alignment/physical 三维,CoT后正则解析)。
i2v 无生成 prompt,喂按风格区分的占位描述;alignment 维如实标注为不可信,主用 visual+physical。"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

TPL = """You are an expert for evaluating AI-generated videos from three dimensions: (1) visual quality - clarity, smoothness, artifacts; (2) text-to-video alignment - fidelity to the prompt; (3) physical/common-sense consistency - naturalness and physics plausibility.
The text prompt of this video is: {t2v_prompt}
Watch the video carefully, reason step by step, then conclude with your scores in exactly this format:
visual quality: <score 1-10>
text-to-video alignment: <score 1-10>
physical/common-sense consistency: <score 1-10>"""

PAT = re.compile(r"visual quality:\s*(\d+).*?alignment:\s*(\d+).*?consistency:\s*(\d+)", re.S | re.I)


def read_frames(vp, fps=2.0, max_n=16):
    cap = cv2.VideoCapture(str(vp))
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 24
    if tot <= 1:
        cap.release(); return []
    step = max(1, int(round(vfps / fps)))
    idxs = list(range(0, tot, step))[:max_n]
    want = set(idxs); out = {}; k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            H, W = fr.shape[:2]
            s = 448 / max(H, W)
            out[k] = Image.fromarray(cv2.cvtColor(
                cv2.resize(fr, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB))
        k += 1
    cap.release()
    return [out[i] for i in sorted(out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/root/mech/data/corpus_videos")
    ap.add_argument("--manifest", default="/root/mech/manifest_all.tsv")
    ap.add_argument("--out", default="/root/mech/data/w5_vs2.csv")
    ap.add_argument("--model", default="/root/mech/models/videoscore2")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rels = [l.split("\t")[0] for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    done = set()
    out_p = Path(args.out)
    if out_p.exists():
        done = {r[0] for r in csv.reader(open(out_p))}
    else:
        csv.writer(open(out_p, "w", newline="")).writerow(["rel", "vs2_visual", "vs2_align", "vs2_phys"])
    todo = [r for r in rels if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"todo {len(todo)}", flush=True)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print("vs2 loaded", flush=True)

    def prompt_for(rel):
        if "ti2i2v" in rel:
            desc = "The plush mushroom character TUTU, a small living creature, moves naturally in a CG animated scene."
        else:
            desc = "A plush mushroom toy figurine named TUTU comes to life and moves naturally in a real photographed scene."
        return TPL.format(t2v_prompt=desc)

    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    for k, rel in enumerate(todo):
        row = None
        try:
            ims = read_frames(Path(args.videos_dir) / rel)
            if len(ims) >= 4:
                content = [{"type": "video", "video": ims}]
                content.append({"type": "text", "text": prompt_for(rel)})
                msgs = [{"role": "user", "content": content}]
                inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                m = PAT.search(text)
                if m:
                    row = [m.group(1), m.group(2), m.group(3)]
                elif k < 3:
                    print("PARSE_FAIL", rel, text[-200:], flush=True)
        except Exception as e:
            if k < 3:
                print("ERR", rel, repr(e)[:120], flush=True)
        w.writerow([rel] + (row if row else ["nan"] * 3))
        if (k + 1) % 100 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    f.close()
    print("W5B_DONE", flush=True)


if __name__ == "__main__":
    main()
