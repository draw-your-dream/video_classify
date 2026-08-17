#!/usr/bin/env python3
"""参考图条件化的还原度专项 LoRA(新子标签监督,3 折分组,顺序跑完三折)。

输入 = [官方立绘 v0] + 8 视频帧;监督 = 还原度bad→坏,good/normal→好(其他 bad 不入训练)。
预测 = 各折 eval 全量(含非还原 bad,便于全口径评估)。
输出 /workspace/r2/pred_fidlora_fold{k}/<label>/<stem>.json
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
USER = ("第一张图是该款式蘑菇TUTU的官方标准形象(3D渲染,画风差异不算缺陷,只比对结构与配色)。"
        "之后是同一条AI生成视频的逐帧画面。"
        "视频中角色的外观还原度是否有问题(伞盖/配件/围巾/五官/配色与官方形象不符、形状走样、部件消失变形)?"
        "只回答一个字:好 或 坏。")

BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
VIEWS = R2 / "data/sku_ref_v2/views"

lab = {r["filename"]: (r["grade"], r.get("reasons", "")) for r in csv.DictReader(
    open(R2 / "data/tutu_task1_annotations_1233.csv", encoding="utf-8-sig"))}


def ref_of(fn):
    sku = fn.split("__")[2]
    return VIEWS / f"{sku}_v0.jpg"


def run_fold(k, proc, model_ctor):
    train_all = [json.loads(l) for l in open(R2 / f"up/splits/fold{k}_train.jsonl")]
    eval_all = [json.loads(l) for l in open(R2 / f"up/splits/fold{k}_eval.jsonl")]

    def is_fid(e):
        g, r = lab[Path(e["video"]).name]
        return g == "bad" and "还原度" in r

    train_targets = [e for e in train_all if e["label"] != "bad" or is_fid(e)]
    for e in train_targets:
        e["_y_bad"] = is_fid(e)
    n_bad = sum(e["_y_bad"] for e in train_targets)
    print(f"fold{k}: train {len(train_targets)} (fid-bad {n_bad}) eval {len(eval_all)}", flush=True)

    model = model_ctor()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    def build(e, with_answer=True):
        fn = Path(e["video"]).name
        ref = Image.open(ref_of(fn)).convert("RGB")
        ref.thumbnail((448, 448))
        v = load_video(e["abs_path"], sparse_count=8, sparse_short_side=192)
        pil = [ref] + [Image.fromarray(f) for f in v.frames_sparse]
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
            target = "坏" if e.get("_y_bad", e["label"] == "bad") else "好"
            full = pt + target + proc.tokenizer.eos_token
            pi = proc(text=[pt], images=pil, return_tensors="pt", padding=True)
            fi = proc(text=[full], images=pil, return_tensors="pt", padding=True)
            labels = fi["input_ids"].clone()
            labels[:, :pi["input_ids"].shape[-1]] = -100
            pad = proc.tokenizer.pad_token_id
            if pad is not None:
                labels[fi["input_ids"] == pad] = -100
            sample = {kk: vv.to("cuda") for kk, vv in fi.items() if isinstance(vv, torch.Tensor)}
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
        if step % 50 == 0:
            rec = sum(losses[-50:]) / max(1, len(losses[-50:]))
            eta = (time.time() - t0) / max(1, step + 1) * (total - step - 1) / 60
            print(f"  f{k} step {step}/{total} loss={rec:.4f} eta={eta:.1f}min", flush=True)

    outdir = R2 / f"pred_fidlora_fold{k}"
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
                lg, lb = float(logits[gt]), float(logits[bt])
                p = float(torch.softmax(torch.tensor([lg, lb]), 0)[1])
                op.write_text(json.dumps({"label": e["label"], "video": e["video"],
                                          "p_bad": p}, ensure_ascii=False))
            except Exception as ex:
                print(f"FAIL {e['video'][:40]} {ex}", file=sys.stderr)
                torch.cuda.empty_cache()
            if i % 100 == 0:
                print(f"  f{k} pred {i}/{len(eval_all)}", flush=True)
    del model
    torch.cuda.empty_cache()
    print(f"FOLD{k}_DONE", flush=True)


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, use_fast=False)

    def ctor():
        m = AutoModelForImageTextToText.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True, attn_implementation="sdpa")
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        return get_peft_model(m, cfg)

    for k in range(3):
        run_fold(k, proc, ctor)
    print("FIDLORA_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
