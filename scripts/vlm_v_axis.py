#!/usr/bin/env python
"""V 轴试点:VLM 判官(Qwen3-VL-8B,参考图对照,2026-08-01 预注册)。

输入 = 本款参考图(白底)+ 视频抠像 4 帧(f0/f5/f10/f15);中文结构化问答输出 JSON。
试点集:语料 bad 抽 120(种子确定)+ eval_good 全量;主轴 = 10 - 整体还原分。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_bank_eval import load_manifest, split_goods, stem_seed  # noqa: E402
from prep_ref_embeds import white_canvas  # noqa: E402

PROMPT = """第一张图是毛毡蘑菇角色「蘑菇TUTU」的官方标准形象参考图。后面4张图是同一条AI生成视频在不同时刻的角色抠像(白底)。

判定规则:
- 姿态、视角、表情、手持道具、身上饰品的差异,以及因遮挡导致的部分身体不可见,都属于正常变化,一律不算缺陷。
- 只判断角色本体是否出现下列明确缺陷:
  1. black_spots: 脸部或身体出现参考图没有的黑点/污斑
  2. mouth_abnormal: 嘴巴形状明显不合理(位置错误/撕裂/扭曲变形)
  3. limb_count_abnormal: 可见手脚数量异常(多于2手2脚,或出现明显多余肢体)
  4. tail: 长出尾巴或参考图没有的多余附属结构
  5. body_elongated: 身体比例明显拉长/矮胖变形(与参考图对比)
  6. other_defect: 其他明显的形象崩坏(五官消失/融化/裂开等)
- fidelity_score: 角色本体与参考图的一致性 0-10 整数(10=完全一致,忽略姿态视角饰品差异)

只输出一行 JSON,不要输出其他内容:
{"black_spots": false, "mouth_abnormal": false, "limb_count_abnormal": false, "tail": false, "body_elongated": false, "other_defect": false, "fidelity_score": 9, "reason": "简短一句"}"""

N_BAD = 120


def pick_ref(renders: Path, style: str) -> Image.Image:
    d = renders / style
    cands = sorted(d.glob("sku-*1.png")) or sorted(d.glob("kf-101.png")) \
        or [p for p in sorted(d.glob("*.png")) if not p.name.endswith("_mask.png")]
    p = cands[0]
    mp = p.with_name(p.stem + "_mask.png")
    return white_canvas(Image.open(p), Image.open(mp) if mp.exists() else None)


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/root/mech/data/prod500/mech_subset.tsv")
    ap.add_argument("--cut-dir", default="/root/mech/data/sam3_cutouts")
    ap.add_argument("--renders", default="/root/mech/renders")
    ap.add_argument("--out", default="/root/mech/vlm_v_axis.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()

    man = load_manifest(Path(args.manifest))
    goods = man[man.label == "good"]
    _, eval_rels = split_goods(goods)
    bads = sorted(man[man.label != "good"].rel.tolist())
    rng = np.random.default_rng(stem_seed("vlm-pilot"))
    bads = [bads[i] for i in rng.choice(len(bads), N_BAD, replace=False)]
    todo = [(r, "bad") for r in bads] + [(r, "eval_good") for r in eval_rels]
    print(f"pilot: {len(bads)} bad + {len(eval_rels)} eval_good", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)

    refs = {st: pick_ref(Path(args.renders), st) for st in man["style"].unique()
            if (Path(args.renders) / st).exists()}

    out_f = open(args.out, "a")
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["rel"] for l in Path(args.out).read_text().splitlines() if l.strip()}
    t0 = time.time()
    n = 0
    for rel, group in todo:
        if rel in done:
            continue
        style = rel.split("/")[0]
        cdir = Path(args.cut_dir) / rel.replace(".mp4", "")
        frames = []
        for i in (0, 5, 10, 15):
            p = cdir / f"f{i:02d}.jpg"
            if p.exists():
                frames.append(Image.open(p).convert("RGB"))
        if len(frames) < 2 or style not in refs:
            out_f.write(json.dumps({"rel": rel, "group": group, "error": "no_frames"},
                                   ensure_ascii=False) + "\n")
            continue
        content = [{"type": "image", "image": refs[style]}]
        content += [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": PROMPT})
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        text = proc.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rec = {"rel": rel, "group": group, "raw": text}
        js = parse_json(text)
        if js:
            rec.update({k: js.get(k) for k in
                        ("black_spots", "mouth_abnormal", "limb_count_abnormal",
                         "tail", "body_elongated", "other_defect",
                         "fidelity_score", "reason")})
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        n += 1
        if n % 20 == 0:
            print(f"[{n}] {(time.time()-t0)/n:.1f}s/vid", flush=True)

    out_f.close()
    # AUC
    import pandas as pd
    recs = [json.loads(l) for l in Path(args.out).read_text().splitlines() if l.strip()]
    df = pd.DataFrame([r for r in recs if r.get("fidelity_score") is not None])
    if len(df):
        df["v_main"] = 10 - pd.to_numeric(df.fidelity_score, errors="coerce")
        flags = ["black_spots", "mouth_abnormal", "limb_count_abnormal",
                 "tail", "body_elongated", "other_defect"]
        for f in flags:
            df[f] = df[f].astype(bool)
        df["v_flag_any"] = df[flags].any(axis=1).astype(float)
        pos = df[df.group == "bad"]
        neg = df[df.group == "eval_good"]
        from patch_bank_eval import auc
        print(f"\nparsed {len(df)}/{len(recs)};bad {len(pos)} / good {len(neg)}")
        print(f"v_main(10-还原分) AUC = {auc(pos.v_main.values, neg.v_main.values):.3f}")
        print(f"v_flag_any        AUC = {auc(pos.v_flag_any.values, neg.v_flag_any.values):.3f}")
        for f in flags:
            print(f"  {f:20s} bad {pos[f].mean():.2f} vs good {neg[f].mean():.2f}")
    print("VLM_V_DONE")


if __name__ == "__main__":
    main()
