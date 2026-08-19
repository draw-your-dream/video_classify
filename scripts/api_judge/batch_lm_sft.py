#!/usr/bin/env python3
"""批量 LoRA SFT 预测:模型加载一次,循环全部视频(移植 lora_sft_v3 predict_one_video)。
适配器用 HF v1(lora_sft_v1_qwen25vl)——v3 适配器已随旧机器丢失,注明轻微版本错位。
输出 data/cache/lm_sft_v3_pred/<label>/<stem>.json(与训练缓存同 schema)。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/pbase/upstream")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from tutu_eval.io.video_loader import load_video  # noqa: E402
from lora_sft_v3_qwen25vl import SYSTEM, USER_BINARY  # noqa: E402

ADAPTER = "/workspace/pbase/hfassets/lora_sft_v1_qwen25vl"
BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT = ROOT / "data/cache/lm_sft_v3_pred"


def main():
    targets = [json.loads(l) for l in open(ROOT / "splits/train_v2.jsonl")]
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    good_tok = proc.tokenizer("好", add_special_tokens=False)["input_ids"][0]
    bad_tok = proc.tokenizer("坏", add_special_tokens=False)["input_ids"][0]
    t0, n = time.time(), 0
    for e in targets:
        out_p = OUT / e["label"] / (Path(e["video"]).stem + ".json")
        if out_p.exists():
            continue
        try:
            v = load_video(e["abs_path"], sparse_count=8, sparse_short_side=192)
            pil = [Image.fromarray(f) for f in v.frames_sparse]
            content = [{"type": "image", "image": img} for img in pil]
            content.append({"type": "text", "text": USER_BINARY})
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": content}]
            try:
                pt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
            except TypeError:
                pt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[pt], images=pil, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1]
                lg = float(logits[good_tok].item())
                lb = float(logits[bad_tok].item())
                p = float(torch.softmax(torch.tensor([lg, lb]), dim=0)[1])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps({"label": e["label"], "video": e["video"],
                                         "p_bad": p, "good_logit": lg, "bad_logit": lb},
                                        ensure_ascii=False))
        except Exception as ex:
            print("ERR", e["video"], repr(ex)[:100], flush=True)
        n += 1
        if n % 50 == 0:
            print(f"[sft {n}/{len(targets)}] {(time.time()-t0)/n:.1f}s/条", flush=True)
    print("SFT_BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
