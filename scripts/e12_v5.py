#!/usr/bin/env python
"""E12-v5(2026-08-04 预注册):v4基础上新增三类(手指/无脚/多余肢体,用户35条判读);嘴部回退v3严格版;候选=名单文件(v5半量)。原v4说明:
眉毛线加固——窄眼/眯成一条线不是眉毛,须同帧同时看到眼睛+其上方分离线条才算;
尾巴/眼睛/嘴巴三线适当放宽(明显即可报,不再要求绝对确定),排除规则保留。
输入与 v3 相同:crops_v3 宽松原图裁剪,v1 旗标候选全量重问。"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

PROMPT = """以下最多16张图是同一条AI生成视频的逐帧画面裁剪(以角色为中心,含背景与周边物体),按时间顺序编号(第0张、第1张……)。\n画面中可能出现其他玩偶或物体:只判断毛绒蘑菇角色「蘑菇TUTU」本体(浅黄色身体、红色伞盖),其他一概忽略。\n角色手持或佩戴的物品(望远镜、放大镜、背包等)属于合理设定,注意不要把持物遮挡当成五官异常。\n总原则:凡被物体、手、视角遮挡或因画质看不清的部位,一律默认正常,不报任何异常;只报画面中清晰可见的异常,绝不从「看不到某部位」推断出问题。
官方设定:TUTU 无眉毛、无尾巴;两颗实心黑豆眼;一张小嘴,说话或表情变化时嘴会开合。
逐帧检查下列异常,遵守每条的判定与排除规则:
1. eyebrows(眉毛):必须在同一帧里同时看到「眼睛」和「眼睛上方与之分离的弧线/条状结构」,二者都清晰可见才算眉毛。
   最重要的排除:TUTU 眯眼/闭眼时,眼睛本身会变窄、变成一条横线——此时那条线就在眼睛的位置上、下方没有另一只眼睛,这是眼睛不是眉毛,绝对不要报。只看到一条线而看不到它下方的眼睛时,一律按眯眼处理,不报。
2. tail(多余附属结构):从身体长出来的尾巴、肢芽等多余结构,看起来较明显即可报,不必绝对确定。
   排除:穿戴或携带的物品(背包、围巾、帽饰、挂件、手持道具)是合理设定,不算。
3. eye_anomaly(眼睛异常):眼睛数量不对、位置错乱、形状明显崩坏、两眼明显不对称(一大一小/一高一低),较明显即可报。
   排除:眼内高光反光、画质模糊、被手/道具/角度遮挡导致看不清的,不算。
4. mouth_broken(嘴部崩坏):只报嘴部撕裂、错位、出现两张嘴等结构性崩坏,只报确定看到的。
   排除:说话引起的张嘴闭嘴、嘴形大小变化是正常动画,不算。
5. fingers(手指分明):官方设定无手指。只报可见手部出现根根分明的手指(如五指张开)的帧;手部模糊、被遮挡、看不清的一律不报。
6. no_feet(无脚缺肢):仅当全身清晰可见、角色处于站立或行走姿态,而腿的末端明显没有脚时才报;坐姿、趴姿、下半身被遮挡或画面裁切的一律不报。
7. extra_limb(多余肢体):只报同一帧内清晰可见的四肢总数超过4,或出现来历不明的多余手/腿(如从身后伸出的第三只手)。被抱着的物品、影子、其他玩偶的肢体不算。
输出一行JSON,异常帧号放对应数组,没有则空数组:
{"eyebrows": [], "tail": [], "eye_anomaly": [], "mouth_broken": [], "fingers": [], "no_feet": [], "extra_limb": [], "note": "一句话"}"""


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
    ap.add_argument("--list", default="/root/mech/v5_list.txt")
    ap.add_argument("--out", default="/root/mech/data/e12_v5.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    todo = [l.strip() for l in open(args.list) if l.strip()]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["rel"] for l in Path(args.out).read_text().splitlines()
                if l.strip() and "error" not in json.loads(l)}
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
            if json.loads(qcp.read_text()).get("median_ref_sim", 1.0) < 0.20:
                out_f.write(json.dumps({"rel": rel, "error": "qc_failed"}) + "\n")
                continue
        jpgs = sorted(d.glob("f*.jpg"))[:16]
        if len(jpgs) < 4:
            out_f.write(json.dumps({"rel": rel, "error": "no_cutouts"}) + "\n")
            continue
        try:
            content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in jpgs]
            content.append({"type": "text", "text": PROMPT})
            msgs = [{"role": "user", "content": content}]
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=220, do_sample=False)
            text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            j = parse_json(text) or {"parse_error": text[:150]}
        except Exception as e:
            j = {"error": repr(e)[:120]}
        j["rel"] = rel
        out_f.write(json.dumps(j, ensure_ascii=False) + "\n")
        if (k + 1) % 25 == 0:
            out_f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.1f}s/vid", flush=True)
    out_f.close()
    print("E12V4_DONE", flush=True)


if __name__ == "__main__":
    main()
