#!/usr/bin/env python
"""E50-C:尾巴粉色物体 VLM 确认(2026-08-11 预注册)。

预筛(冻结):tail_pink.csv 上 persist>=0.9 & hue_med∈[140,180] & ratio_med∈[0.002,0.06]
& maxblob<=0.08 → 全语料 77 候选(锚 5822 在内,eval 侧 15 条)。
第二把钥匙:VLM 下半身放大确认(与眉毛链同款客观事实提问纪律,不诱导)。
输出 data/tail_confirm_full.jsonl。
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path("/root/mech")

Q = """以下最多16张图是同一条AI生成视频的逐帧画面,已放大到角色身体中下部区域(按时间顺序,第0张起)。只判断毛绒蘑菇角色「蘑菇TUTU」本体。
官方设定:TUTU 没有尾巴,身体后方/臀部没有任何附属凸起物。被遮挡或看不清的部位一律默认正常。
问题:是否看到附着在角色臀部、身体后方或体侧的小型粉色/品红色凸起物(尾巴状、球状或条状,与身体相连)?
角色手里拿着的道具、背景里的粉色物体、以及角色本身服装上原有的花纹都不算。
输出一行JSON:{"tail": [帧号], "note": "一句话"}"""

OUT = ROOT / "data/tail_confirm_full.jsonl"


def main():
    cands = []
    for r in csv.DictReader(open(ROOT / "data/tail_pink.csv")):
        try:
            pers = float(r["pk_persist"]); ratio = float(r["pk_ratio_med"])
            hue = float(r["pk_hue_med"]); mb = float(r["pk_maxblob"])
        except Exception:
            continue
        if pers >= 0.9 and 140 <= hue <= 180 and 0.002 <= ratio <= 0.06 and mb <= 0.08:
            cands.append(r["rel"])
    done = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                done.add(json.loads(l)["rel"])
            except Exception:
                pass
    todo = [r for r in cands if r not in done]
    print(f"候选 {len(cands)} | 待跑 {len(todo)}", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-32B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)

    def parse(t):
        m = re.search(r"\{.*\}", t, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None

    f = open(OUT, "a")
    t0 = time.time()
    for k, rel in enumerate(todo):
        d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
        jpgs = sorted(d.glob("f*.jpg"))[:16]
        row = {"rel": rel}
        try:
            if len(jpgs) < 4:
                row["error"] = "no_crops"
            else:
                ims = []
                for p in jpgs:
                    im = cv2.imread(str(p))
                    H, W = im.shape[:2]
                    u = im[int(H * 0.30):, :]          # 下部 70%(臀部/体侧)
                    u = cv2.resize(u, (W * 2, int(u.shape[0] * 2)), interpolation=cv2.INTER_CUBIC)
                    ims.append(Image.fromarray(cv2.cvtColor(u, cv2.COLOR_BGR2RGB)))
                content = [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": Q}]
                inputs = proc.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=140, do_sample=False)
                text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
                j = parse(text)
                if j:
                    row.update(tail=j.get("tail", []), note=j.get("note", ""))
                else:
                    row["parse_error"] = text[:100]
        except Exception as e:
            row["error"] = repr(e)[:110]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (k + 1) % 10 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.1f}s/条", flush=True)
    f.close()
    print("E50C_TAIL_DONE", flush=True)


if __name__ == "__main__":
    main()
