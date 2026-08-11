#!/usr/bin/env python
"""E50-B2:大小变化第二把钥匙 v2——首末帧并排对比(2026-08-11 预注册)。

v1(12 帧序列提问)失败原因判读:渐变式尺寸变化摊到 12 帧每帧仅 ~2%,
跨帧比较是 VLM 盲区,三条锚全部漏判且 note 自信("比例保持稳定")。
v2 改呈现:只给第一帧与最后一帧两张图,把渐变压成一次直接对比。
开发集=train 侧候选(有 reasons 可验证),冻结后才看 eval 侧候选。
判准(发车前冻结):train 候选中「大小变化/还原类」bad 的旗标率 与 good 候选旗标率
之差 ≥ 40pt 且 good 旗标率 ≤ 15%,才把该轴并入否决门。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")

Q = """这是同一条AI生成视频的两张画面:第一张是视频的第一帧,第二张是视频的最后一帧(相隔约5秒)。画面里有毛绒蘑菇角色「蘑菇TUTU」。
请对比两张图,回答一个客观问题:第二张图里,角色相对于**背景参照物**(桌面、家具、建筑等)的大小,与第一张相比是否发生了明显变化(明显更大或明显更小,一眼可见的程度)?
判别要点:
- 若是镜头整体推近/拉远,背景参照物也会同步放大缩小——这种情况角色相对背景的比例不变,回答 false;
- 只有角色相对背景的比例本身明显变了,才回答 true。
输出一行JSON:{"size_change": true/false, "direction": "grow/shrink/none", "note": "一句话"}"""

OUT = ROOT / "data/size_pair_confirm.jsonl"


def cands():
    import glob
    out = []
    for p in glob.glob(str(ROOT / "data/sam3_feat/*/*.npz")):
        z = np.load(p, allow_pickle=True)
        try:
            g = json.loads(str(z["geo"]))
        except Exception:
            continue
        if not isinstance(g, list) or len(g) < 8:
            continue
        cw = np.array([float(x.get("cap_width", 0)) for x in g])
        cy = np.array([(x["bbox"][1] + x["bbox"][3]) / 2 for x in g])
        if (cw <= 0).any():
            continue
        k = np.ones(3) / 3
        s = np.convolve(cw, k, mode="valid")
        yv = np.convolve(cy, k, mode="valid")
        ratio = float(s.max() / max(1e-6, s.min()))
        d = np.diff(s)
        dy = np.diff(yv)
        mono = float(abs(d.sum()) / (np.abs(d).sum() + 1e-6))
        corr = float(np.corrcoef(d, dy)[0, 1]) if d.std() > 0 and dy.std() > 0 else 0.0
        if mono >= 0.8 and corr <= -0.5 and ratio >= 1.2:
            rel = "/".join(Path(p).parts[-2:]).replace(".npz", ".mp4")
            out.append(rel)
    return sorted(out)


def first_last(rel):
    vp = ROOT / "data/corpus_videos" / rel
    if not vp.exists():
        return None
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 4:
        cap.release()
        return None
    frames = {}
    want = {0, total - 1}
    k = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k in want:
            H, W = im.shape[:2]
            s = 768 / max(H, W)
            frames[k] = Image.fromarray(cv2.cvtColor(
                cv2.resize(im, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB))
        k += 1
    cap.release()
    if 0 in frames and (total - 1) in frames:
        return [frames[0], frames[total - 1]]
    return None


def main():
    todo_all = cands()
    done = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                done.add(json.loads(l)["rel"])
            except Exception:
                pass
    todo = [r for r in todo_all if r not in done]
    print(f"候选 {len(todo_all)} | 待跑 {len(todo)}", flush=True)

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
        row = {"rel": rel}
        try:
            ims = first_last(rel)
            if ims is None:
                row["error"] = "no_frames"
            else:
                content = [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": Q}]
                inputs = proc.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=120, do_sample=False)
                text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
                j = parse(text)
                if j is not None:
                    row.update(size_change=j.get("size_change"),
                               direction=j.get("direction"), note=j.get("note", ""))
                else:
                    row["parse_error"] = text[:100]
        except Exception as e:
            row["error"] = repr(e)[:110]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (k + 1) % 25 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.1f}s/条", flush=True)
    f.close()
    print("E50B2_SIZE_PAIR_DONE", flush=True)


if __name__ == "__main__":
    main()
