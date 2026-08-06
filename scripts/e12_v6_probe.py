#!/usr/bin/env python
"""E12-v6 探针(2026-08-06):4 目标 + 3 对照,逐条问三问。
Q1 眉毛 v3 措辞(宽) | Q2 眉毛 v4 措辞(窄眼排除) | Q3 尾巴粉色锚定(用户建议)。
判准:目标 H086/H089/H101 至少 Q1 命中眉毛,H099 命中尾巴;对照- 两条 Q2 不误报。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/root/mech")

HEAD = """以下最多16张图是同一条AI生成视频的逐帧画面裁剪(以角色为中心,含背景与周边物体),按时间顺序编号(第0张、第1张……)。
画面中可能出现其他玩偶或物体:只判断毛绒蘑菇角色「蘑菇TUTU」本体(浅黄色身体、红色伞盖),其他一概忽略。
角色手持或佩戴的物品属于合理设定;凡被物体、手、视角遮挡或因画质看不清的部位,一律默认正常,绝不从「看不到」推断问题。
官方设定:TUTU 无眉毛、无尾巴;两颗实心黑豆眼。
"""

Q1 = HEAD + """问题:是否看到眉毛?眉毛指位于眼睛上方、与眼睛明显分离的独立弧线或条状结构(深色短线、弯眉等)。
眯起的窄眼、眼睛本身形状的变化,不算眉毛。
输出一行JSON:{"eyebrows": [看到眉毛的帧号], "note": "一句话"}"""

Q2 = HEAD + """问题:是否看到眉毛?必须在同一帧里同时看到「眼睛」和「眼睛上方与之分离的弧线/条状结构」,二者都清晰可见才算眉毛。
最重要的排除:TUTU 眯眼/闭眼时,眼睛本身会变窄、变成一条横线——此时那条线就在眼睛的位置上、下方没有另一只眼睛,这是眼睛不是眉毛,绝对不要报。只看到一条线而看不到它下方的眼睛时,一律按眯眼处理,不报。
输出一行JSON:{"eyebrows": [看到眉毛的帧号], "note": "一句话"}"""

Q3 = HEAD + """问题:TUTU 的臀部、背后或身体后方,是否长出粉色(或浅色)的凸起、球状物或尾巴状附属物?
官方设定 TUTU 没有尾巴,身体后方不应有任何长出来的结构。
排除:背包、挂件、衣物下摆等穿戴物;其他玩偶的身体部位;被遮挡看不清的一律不报。
输出一行JSON:{"tail": [看到该结构的帧号], "color": "颜色", "note": "一句话"}"""


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
    ap.add_argument("--cut-dir", default=str(ROOT / "data/crops_v3"))
    ap.add_argument("--probe", default=str(ROOT / "v6_probe.json"))
    ap.add_argument("--out", default=str(ROOT / "data/e12_v6_probe.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    args = ap.parse_args()
    probe = json.load(open(args.probe))

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)

    def ask(ims, q):
        content = [{"type": "image", "image": im} for im in ims]
        content.append({"type": "text", "text": q})
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=140, do_sample=False)
        return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    f = open(args.out, "w")
    for p in probe:
        d = Path(args.cut_dir) / p["rel"].replace(".mp4", "")
        jpgs = sorted(d.glob("f*.jpg"))[:16]
        row = dict(p)
        if len(jpgs) < 4:
            row["error"] = f"no_crops({len(jpgs)})"
        else:
            ims = [Image.open(x).convert("RGB") for x in jpgs]
            for name, q in (("q1_v3宽", Q1), ("q2_v4窄", Q2), ("q3_尾粉锚", Q3)):
                j = parse_json(ask(ims, q)) or {"parse_error": 1}
                row[name] = j
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{p['tag']}:", flush=True)
        for k in ("q1_v3宽", "q2_v4窄", "q3_尾粉锚"):
            if k in row:
                print(f"  {k}: {json.dumps(row[k], ensure_ascii=False)[:160]}", flush=True)
        if "error" in row:
            print("  ", row["error"], flush=True)
    f.close()
    print("V6_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
