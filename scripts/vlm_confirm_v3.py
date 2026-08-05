#!/usr/bin/env python
"""悬空 v3(2026-08-04):v1 原措辞 + 仅加攀附排除一句(v2 重排除塌零,回退)。
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
    "floating": """以下12张图是同一条约5秒的AI生成视频按时间顺序抽取的帧,主角是毛绒蘑菇角色「蘑菇TUTU」。
请只回答一个客观问题:TUTU 在站立、坐、停留状态时,身体下方是否有真实的支撑面(地面、桌面、物体表面)?
若它持续悬浮在空中、脚下无任何支撑、且不是跳跃或被举起的瞬间动作,记为悬空。
排除:跳跃腾空的过程、被手拿起、坐在或趴在可见物体上、手臂扒着/攀着/抓着/挂在物体上(树枝、桌沿、绳索等,这也算有支撑)、下半身被遮挡看不清——这些一律不算悬空。
只输出一行JSON:{"hit": 1或0, "frames": [悬空的帧号], "note": "一句话"}""",
    "scenejump": """以下12张图是同一条约5秒的AI生成视频按时间顺序抽取的帧,主角是毛绒蘑菇角色「蘑菇TUTU」。
请只回答一个客观问题:背景或场景内容是否发生了「突变」——具体指:背景中的物体/文字/建筑在相邻帧之间瞬间出现、瞬间消失或被替换;或场景像切换了镜头、整体环境突然改变;或画面中凭空出现新物体。
排除:正常的镜头平移/拉近拉远、TUTU自身的动作变化、光影渐变——这些一律不算。
只输出一行JSON:{"hit": 1或0, "frames": [突变发生的帧号], "note": "一句话"}""",
}


def read_frames(vp, n=12):
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
    ap.add_argument("--out", default=str(ROOT / "data/twokey_confirm_v3.jsonl"))
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
    todo = [it for it in items if it["rule"] == "floating" and (it["rel"], it["rule"]) not in done]
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
