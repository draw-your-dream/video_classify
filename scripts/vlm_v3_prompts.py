#!/usr/bin/env python
"""V3:同模型(Qwen3-VL-32B)prompt 三变体对照(2026-08-01 预注册)。

a 先描述后判定(CoT) / b 基率校准+逐帧0-100概率 / c 找不同式逐条置信。
同 120 试点集(prompt 开发集);胜出者须新抽 120 确认才晋级。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_bank_eval import auc, load_manifest, split_goods, stem_seed  # noqa: E402
from vlm_v_axis import pick_ref  # noqa: E402

RULES = """判定背景:
- 姿态、视角、表情、手持道具、身上饰品差异、遮挡导致的部分不可见,都属于正常变化,不是缺陷。
- 形象缺陷指角色本体的问题,例如:脸/身上出现参考图没有的黑点污斑;嘴巴撕裂扭曲错位;
  可见手脚数量异常或多余肢体;长出尾巴等多余附属结构;身体比例明显拉长或压扁;
  五官消失/融化/裂开等崩坏。"""

PROMPT_A = """第一张图是毛毡蘑菇角色「蘑菇TUTU」的官方标准形象参考图,后面4张是同一条AI生成视频不同时刻的角色抠像。

""" + RULES + """

请分两步:
第一步:逐帧(第2到第5张图)与参考图对比,简要列出你观察到的所有差异(每帧1-2行)。
第二步:基于以上观察,最后单独一行输出 JSON(不要包在代码块里):
{"black_spots": bool, "mouth_abnormal": bool, "limb_count_abnormal": bool, "tail": bool, "body_elongated": bool, "other_defect": bool, "fidelity_score": 0-10整数}"""

PROMPT_B = """第一张图是毛毡蘑菇角色「蘑菇TUTU」的官方标准形象参考图,后面4张是同一条AI生成视频不同时刻的角色抠像。这批视频约有一半存在形象缺陷,请仔细检查,不要轻易给零分。

""" + RULES + """

给每一帧打一个"形象缺陷概率"(0-100 整数):完全一致=0;有点可疑但可能是正常变化=30;
比较明显的缺陷=70;确定无疑的崩坏=95。只输出一行 JSON:
{"frame_probs": [p1, p2, p3, p4], "video_defect_prob": 整数, "main_issue": "最主要的问题,一句话"}"""

PROMPT_C = """第一张图是毛毡蘑菇角色「蘑菇TUTU」的官方标准形象参考图,后面4张是同一条AI生成视频不同时刻的角色抠像。

我们来玩找不同:仔细对比角色本体与参考图,列出所有你能发现的不同点(忽略姿态/视角/表情/饰品/遮挡)。

""" + RULES + """

只输出一行 JSON:
{"differences": [{"desc": "一句话", "is_defect": bool, "confidence": 0-100}], "overall_defect_prob": 0-100}"""

VARIANTS = {"a": (PROMPT_A, 500), "b": (PROMPT_B, 220), "c": (PROMPT_C, 500)}


def parse_last_json(text: str):
    cands = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.S)
    for c in reversed(cands):
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def score_of(variant: str, js: dict) -> float | None:
    try:
        if variant == "a":
            s = 10 - int(js["fidelity_score"])
            flags = ["black_spots", "mouth_abnormal", "limb_count_abnormal",
                     "tail", "body_elongated", "other_defect"]
            return float(s + 10 * any(bool(js.get(f)) for f in flags))
        if variant == "b":
            ps = [float(p) for p in js.get("frame_probs", [])]
            ps.append(float(js.get("video_defect_prob", 0)))
            return max(ps) if ps else None
        if variant == "c":
            confs = [float(d.get("confidence", 0)) for d in js.get("differences", [])
                     if d.get("is_defect")]
            base = float(js.get("overall_defect_prob", 0))
            return max(confs + [base]) if (confs or base is not None) else None
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/root/mech/data/prod500/mech_subset.tsv")
    ap.add_argument("--cut-dir", default="/root/mech/data/sam3_cutouts")
    ap.add_argument("--renders", default="/root/mech/renders")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--variants", default="b,a,c")
    ap.add_argument("--n-bad", type=int, default=60)
    ap.add_argument("--n-good", type=int, default=60)
    args = ap.parse_args()

    man = load_manifest(Path(args.manifest))
    goods = man[man.label == "good"]
    _, eval_rels = split_goods(goods)
    bads = sorted(man[man.label != "good"].rel.tolist())
    rng = np.random.default_rng(stem_seed("vlm-pilot"))
    bads = [bads[i] for i in rng.choice(len(bads), 120, replace=False)][:args.n_bad]
    rng2 = np.random.default_rng(stem_seed("vlm-pilot-good"))
    eval_sel = [eval_rels[i] for i in rng2.choice(len(eval_rels), args.n_good, replace=False)]
    todo = [(r, "bad") for r in bads] + [(r, "eval_good") for r in eval_sel]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)
    refs = {st: pick_ref(Path(args.renders), st) for st in man["style"].unique()
            if (Path(args.renders) / st).exists()}

    for variant in args.variants.split(","):
        prompt, max_tok = VARIANTS[variant]
        out_p = Path(f"/root/mech/vlm_v3{variant}.jsonl")
        done = set()
        if out_p.exists():
            done = {json.loads(l)["rel"] for l in out_p.read_text().splitlines() if l.strip()}
        f = out_p.open("a")
        t0, n = time.time(), 0
        for rel, group in todo:
            if rel in done:
                continue
            style = rel.split("/")[0]
            cdir = Path(args.cut_dir) / rel.replace(".mp4", "")
            frames = [Image.open(cdir / f"f{i:02d}.jpg").convert("RGB")
                      for i in (0, 5, 10, 15) if (cdir / f"f{i:02d}.jpg").exists()]
            if len(frames) < 2 or style not in refs:
                f.write(json.dumps({"rel": rel, "group": group, "error": "no_frames"}) + "\n")
                continue
            content = ([{"type": "image", "image": refs[style]}]
                       + [{"type": "image", "image": fr} for fr in frames]
                       + [{"type": "text", "text": prompt}])
            inputs = proc.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
            text = proc.decode(gen[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
            js = parse_last_json(text)
            sc = score_of(variant, js) if js else None
            f.write(json.dumps({"rel": rel, "group": group, "score": sc,
                                "raw": text[-800:]}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"[{variant}:{n}] {(time.time()-t0)/n:.1f}s/vid", flush=True)
        f.close()
        recs = [json.loads(l) for l in out_p.read_text().splitlines() if l.strip()]
        pos = np.array([r["score"] for r in recs
                        if r["group"] == "bad" and r.get("score") is not None], float)
        neg = np.array([r["score"] for r in recs
                        if r["group"] == "eval_good" and r.get("score") is not None], float)
        print(f"== V3{variant} AUC = {auc(pos, neg):.3f} "
              f"(bad {len(pos)} / good {len(neg)}, parsed {len(pos)+len(neg)}/{len(recs)})",
              flush=True)
    print("V3_ALL_DONE")


if __name__ == "__main__":
    main()
