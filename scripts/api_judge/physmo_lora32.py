#!/usr/bin/env python3
"""物理/运动专项 LoRA @ Qwen2.5-VL-32B(fold0 GO/NO-GO 门)。

监督 = 物理/运动族 bad → 坏,good/normal → 好(其他 bad 剔除);输入 = 8 帧(无参照图)。
bf16 + LoRA + gradient checkpointing,batch 1。
输出 /workspace/r2/pred_physmo32_fold0/<label>/<stem>.json
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

R2 = Path("/workspace/r2")
sys.path.insert(0, str(R2 / "up/src"))
from tutu_eval.io.video_loader import load_video  # noqa: E402

SYSTEM = "你是 AI 生成视频缺陷检测专家。简短回答。"
USER = ("以下是同一条AI生成视频的逐帧画面。只关注两类问题:"
        "①物理与物体:角色或物体是否漂浮/穿模/凭空出现消失/不符合重力支撑;"
        "②运动:角色动作是否僵硬呆板/卡顿/四肢锁死不动/整段静止缺乏生命感。"
        "外观像不像、画风、颜色一律不管。"
        "该视频在物理或运动上是否有明显问题?只回答一个字:好 或 坏。")

BASE = "Qwen/Qwen2.5-VL-32B-Instruct"
KEYS = ("物理规律", "不合理的物体", "僵硬", "卡顿", "四肢不动", "静止不动", "运动主体")

lab = {r["filename"]: (r["grade"], r.get("reasons", "")) for r in csv.DictReader(
    open(R2 / "data/tutu_task1_annotations_1233.csv", encoding="utf-8-sig"))}


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="sdpa")
    model.gradient_checkpointing_enable()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    model.enable_input_require_grads()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    train_all = [json.loads(l) for l in open(R2 / "up/splits/fold0_train.jsonl")]
    eval_all = [json.loads(l) for l in open(R2 / "up/splits/fold0_eval.jsonl")]

    def is_pm(e):
        g, r = lab[Path(e["video"]).name]
        return g == "bad" and any(k in r for k in KEYS)

    train_targets = [e for e in train_all if e["label"] != "bad" or is_pm(e)]
    for e in train_targets:
        e["_y_bad"] = is_pm(e)
    print(f"train {len(train_targets)} (pm-bad {sum(e['_y_bad'] for e in train_targets)}) "
          f"eval {len(eval_all)}", flush=True)

    def build(e, with_answer=True):
        v = load_video(e["abs_path"], sparse_count=8, sparse_short_side=192)
        pil = [Image.fromarray(f) for f in v.frames_sparse]
        content = [{"type": "image", "image": img} for img in pil]
        content.append({"type": "text", "text": USER})
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": content}]
        try:
            pt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            pt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if with_answer:
            target = "坏" if e["_y_bad"] else "好"
            full = pt + target + proc.tokenizer.eos_token
            pi = proc(text=[pt], images=pil, return_tensors="pt", padding=True)
            fi = proc(text=[full], images=pil, return_tensors="pt", padding=True)
            labels = fi["input_ids"].clone()
            labels[:, :pi["input_ids"].shape[-1]] = -100
            pad = proc.tokenizer.pad_token_id
            if pad is not None:
                labels[fi["input_ids"] == pad] = -100
            sample = {k: v2.to("cuda") for k, v2 in fi.items() if isinstance(v2, torch.Tensor)}
            sample["labels"] = labels.to("cuda")
            return sample
        return proc(text=[pt], images=pil, return_tensors="pt", padding=True).to("cuda")

    random.seed(42)
    random.shuffle(train_targets)
    total = len(train_targets)
    model.train()
    losses, t0 = [], time.time()
    for step in range(total):
        e = train_targets[step]
        try:
            sample = build(e)
            out = model(**sample)
            loss = out.loss
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            del sample, out, loss
        except Exception as ex:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"step {step} err {type(ex).__name__} {str(ex)[:80]}", file=sys.stderr)
            continue
        if step % 25 == 0:
            rec = sum(losses[-25:]) / max(1, len(losses[-25:]))
            eta = (time.time() - t0) / max(1, step + 1) * (total - step - 1) / 60
            print(f"  step {step}/{total} loss={rec:.4f} eta={eta:.1f}min", flush=True)

    outdir = R2 / "pred_physmo32_fold0"
    model.eval()
    gt = proc.tokenizer("好", add_special_tokens=False)["input_ids"][0]
    bt = proc.tokenizer("坏", add_special_tokens=False)["input_ids"][0]
    with torch.inference_mode():
        for i, e in enumerate(eval_all):
            op = outdir / e["label"] / (Path(e["video"]).stem + ".json")
            op.parent.mkdir(parents=True, exist_ok=True)
            if op.exists():
                continue
            try:
                inputs = build(e, with_answer=False)
                logits = model(**inputs).logits[0, -1]
                lg, lb2 = float(logits[gt]), float(logits[bt])
                p = float(torch.softmax(torch.tensor([lg, lb2]), 0)[1])
                op.write_text(json.dumps({"label": e["label"], "video": e["video"],
                                          "p_bad": p}, ensure_ascii=False))
            except Exception as ex:
                print(f"FAIL {e['video'][:40]} {ex}", file=sys.stderr)
                torch.cuda.empty_cache()
            if i % 100 == 0:
                print(f"  pred {i}/{len(eval_all)}", flush=True)
    print("PHYSMO32_FOLD0_DONE", flush=True)


if __name__ == "__main__":
    main()
