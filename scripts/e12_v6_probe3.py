#!/usr/bin/env python
"""v6 二探:放大视图重问。眉毛=裁剪上部60%放大;尾巴=全帧原图(不裁剪,防止取景切掉背后)。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path("/root/mech")

HEAD = """以下最多16张图是同一条AI生成视频的逐帧画面(按时间顺序)。只判断毛绒蘑菇角色「蘑菇TUTU」本体。
官方设定:TUTU 无眉毛、无尾巴;两颗实心黑豆眼。被遮挡或看不清的部位一律默认正常。
"""
QE = HEAD + """这些图已放大到角色头部区域。问题:是否看到眉毛?眉毛指位于眼睛上方、与眼睛明显分离的独立弧线、斜线或条状痕迹(颜色可深可浅)。
最重要的排除:TUTU 眯眼/闭眼时,眼睛本身会变窄、变成一条横线或弧线——此时那条线就在眼睛的位置上、下方没有另一只眼睛,这是眼睛不是眉毛,绝对不要报。
判定标准:同一帧里必须既能看到眼睛(圆点或窄线均可),又能在其上方看到与之分离的另一条线/痕迹,二者同时存在才算眉毛;只有一条线时一律按眯眼处理,不报。
输出一行JSON:{"eyebrows": [帧号], "note": "一句话"}"""
QT = HEAD + """问题:TUTU 的臀部、背后、身体下后方,是否长出粉色(或其他颜色)的凸起、球状物或尾巴状附属物?
排除:穿戴物、手持物、其他玩偶、被遮挡看不清的。
输出一行JSON:{"tail": [帧号], "color": "颜色", "note": "一句话"}"""


def read_full_frames(vp, n=16):
    cap = cv2.VideoCapture(str(vp))
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if tot <= 1:
        cap.release(); return []
    idxs = [int(round(i * (tot - 1) / (n - 1))) for i in range(n)]
    want = set(idxs); out = {}; k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            out[k] = fr
        k += 1
    cap.release()
    ims = []
    for i in sorted(out):
        fr = out[i]
        H, W = fr.shape[:2]
        s = 560 / max(H, W)
        ims.append(Image.fromarray(cv2.cvtColor(cv2.resize(fr, (int(W*s), int(H*s))), cv2.COLOR_BGR2RGB)))
    return ims


def upper_zoom(jpgs):
    ims = []
    for p in jpgs:
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        crop = im[0:int(H*0.62), :]
        crop = cv2.resize(crop, (W*2, int(H*0.62*2)), interpolation=cv2.INTER_CUBIC)
        ims.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    return ims


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    probe = json.load(open(ROOT / "v6_probe.json"))
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-32B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
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

    f = open(ROOT / "data/e12_v6_probe3.jsonl", "w")
    for p in probe:
        d = ROOT / "data/crops_v3" / p["rel"].replace(".mp4", "")
        jpgs = sorted(d.glob("f*.jpg"))[:16]
        row = dict(p)
        if len(jpgs) >= 4:
            j = parse_json(ask(upper_zoom(jpgs), QE)) or {"parse_error": 1}
            row["eyebrow_zoom"] = j
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{p['tag']}:", flush=True)
        for k in ("eyebrow_zoom", "tail_full"):
            if k in row:
                print(f"  {k}: {json.dumps(row[k], ensure_ascii=False)[:170]}", flush=True)
    f.close()
    print("V6_PROBE3_DONE", flush=True)


if __name__ == "__main__":
    main()
