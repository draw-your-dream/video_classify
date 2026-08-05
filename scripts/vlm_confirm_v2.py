#!/usr/bin/env python
"""两把钥匙·第二把 v2(2026-08-04 用户审核修正):悬空排除攀附支撑;突变限定凭空从无到有。
floating(悬空无支撑)/ scenejump(背景突变·物体凭空出现消失·镜头切换)。
输入=全帧12张;名单含盲样对照(grp字段),VLM 不知情。"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path("/root/mech")

Q = {
    "floating": """以下24张图是同一条约5秒的AI生成视频按时间顺序抽取的帧,主角是毛绒蘑菇角色「蘑菇TUTU」。
请只回答一个客观问题:TUTU 在站立、坐、停留状态时,身体下方是否有真实的支撑面(地面、桌面、物体表面)?
只有当 TUTU 的整个身体没有与任何可支撑物体接触时才算悬空:脚下无支撑、身体没有倚靠任何东西、手臂也没有扒着/攀着/抓着/挂在任何物体上。
排除(以下一律算有支撑,不是悬空):脚站在或身体坐趴在任何表面上;手臂扒着、攀着、抓着、挂在物体上(树枝、桌沿、杯沿、绳索等);倚靠在物体上;跳跃腾空的过程;被手拿起;下半身或接触点被遮挡看不清。
只输出一行JSON:{"hit": 1或0, "frames": [悬空的帧号], "note": "一句话"}""",
    "scenejump": """以下24张图是同一条约5秒的AI生成视频按时间顺序抽取的帧,主角是毛绒蘑菇角色「蘑菇TUTU」。
请只回答一个客观问题:画面中是否发生了「凭空跳变」——同一位置的物体/文字/建筑在相邻帧之间凭空出现、凭空消失或被替换成别的东西;或整个场景像剪辑切换了镜头。
关键排除:镜头正常平移/拉近拉远时,新内容从画面边缘逐渐进入视野——这是拍到了新东西,不是凭空出现,一律不算。TUTU自身动作、光影渐变也不算。只有「原本视野内的位置上突然从无到有/从有到无」才算。
只输出一行JSON:{"hit": 1或0, "frames": [突变发生的帧号], "note": "一句话"}""",
}


def read_frames(vp, n=24):
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
        s = 448 / max(H, W)
        ims.append(Image.fromarray(cv2.cvtColor(cv2.resize(fr, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB)))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--items", default=str(ROOT / "twokey_items.json"))
    ap.add_argument("--out", default=str(ROOT / "data/twokey_confirm_v2.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    items = json.load(open(args.items))
    done = set()
    if Path(args.out).exists():
        for l in Path(args.out).read_text().splitlines():
            if l.strip():
                j = json.loads(l)
                if "error" not in j:
                    done.add((j["rel"], j["rule"]))
    todo = [it for it in items if (it["rel"], it["rule"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"todo {len(todo)}", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)

    f = open(args.out, "a"); t0 = time.time()
    for k, it in enumerate(todo):
        row = dict(it)
        try:
            ims = read_frames(Path(args.videos_dir) / it["rel"])
            if len(ims) < 6:
                raise RuntimeError("too few frames")
            content = [{"type": "image", "image": im} for im in ims]
            content.append({"type": "text", "text": Q[it["rule"]]})
            msgs = [{"role": "user", "content": content}]
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=96, do_sample=False)
            text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            j = parse_json(text)
            if j is None:
                row["parse_error"] = text[:100]
            else:
                row.update(hit=int(j.get("hit", 0)), frames=j.get("frames", []), note=j.get("note", ""))
        except Exception as e:
            row["error"] = repr(e)[:120]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (k + 1) % 40 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.1f}s/条", flush=True)
    f.close()
    print("TWOKEY_DONE", flush=True)


if __name__ == "__main__":
    main()
