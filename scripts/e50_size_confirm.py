#!/usr/bin/env python
"""E50-B:大小变化(违反透视)VLM 确认(2026-08-11 预注册)。

预筛(冻结,来自 sam3_feat geo 轨迹):平滑 cap_width 的 mono>=0.8 & 透视corr<=-0.5 & ratio>=1.2
→ 全语料 200 候选(train 147 @ 40% 精度平台,特征无法再提纯 → 第二把钥匙)。
输入用**原始整帧**(crops 已长边归一化,尺寸信息被销毁,不可用)。
提问纪律:中性事实比较,不告知预筛结论(防 E25 式诱导假确认);
以"镜头推拉(背景同步缩放)不算"做排除条款。
输出 data/size_confirm_full.jsonl。先看 train 侧精度再决定是否入否决门。
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

Q = """以下最多12张图是同一条AI生成视频的逐帧画面(按时间顺序,第0张起,时长约5秒)。画面里有毛绒蘑菇角色「蘑菇TUTU」。
请做一个客观的尺寸对比:观察角色相对于其周围背景物体(桌面、家具、房间等)的大小,从头到尾是否发生了明显变化(明显变大或明显变小,肉眼一眼可见的程度)?
注意两种情况的区分:
- 若是镜头推近/拉远,背景物体会与角色**同步**放大缩小,这种不算角色大小变化;
- 若角色相对背景的比例本身在变(角色变大而背景不变,或反之),才算。
角色靠近/远离镜头造成的透视变化:若变化幅度与其移动距离相称(近大远小、合乎常理)也不算。
输出一行JSON:{"size_change": true/false, "direction": "grow/shrink/none", "note": "一句话"}"""

OUT = ROOT / "data/size_confirm_full.jsonl"


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


def frames_of(rel, n=12):
    vp = ROOT / "data/corpus_videos" / rel
    if not vp.exists():
        return None
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 4:
        cap.release()
        return None
    want = set(np.linspace(0, total - 1, n).round().astype(int).tolist())
    ims, k = [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k in want:
            H, W = im.shape[:2]
            s = 640 / max(H, W)
            ims.append(Image.fromarray(cv2.cvtColor(
                cv2.resize(im, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB)))
        k += 1
    cap.release()
    return ims if len(ims) >= 4 else None


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
            ims = frames_of(rel)
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
        if (k + 1) % 20 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.1f}s/条", flush=True)
    f.close()
    print("E50B_SIZE_DONE", flush=True)


if __name__ == "__main__":
    main()
