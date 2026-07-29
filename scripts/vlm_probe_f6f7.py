#!/usr/bin/env python
"""F6a-d + F7:Qwen3-VL-30B 定向问题 logit 轴(FACTOR_PREREG.md 预注册)。

每问强制"只答是/否",因子值 = logprob(是) - logprob(否)(首生成 token 处)。
对照假设:此前笼统问"是否违反物理规律"AUC 0.588,定向问题应显著更强。

跑在 H100(93GB):bf16 全量加载,16 帧/视频。输出 JSONL 可断点续跑。
用法: python vlm_probe_f6f7.py --videos-dir /root/videos --out /root/f6f7.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
N_FRAMES = 16

QUESTIONS = {
    # B 类:悬空/无支撑
    "f6a_levitate": "视频里的毛毡蘑菇玩偶是否在某些时刻悬在空中、没有站在或贴在任何表面上?只回答是或否。",
    # I 类:物体自主运动
    "f6b_selfmove": "画面里是否有玩偶以外的物体(如布、纸、盖子、餐具)在没有人或外力接触的情况下自行移动、变形或变大?只回答是或否。",
    # D 类:静态不可能(时间信号原理性盲区,唯一出路)
    "f6c_static_impossible": "画面中是否存在物理上不可能的静态情形:例如镜子里的倒影与实物不符、平面图画呈现出真实的立体深度、物体以不可能维持的方式保持平衡?只回答是或否。",
    # F/G 类:凭空出现/消失
    "f6d_appear": "视频中是否有物体凭空出现或凭空消失?只回答是或否。",
    # F7:IP 模板符合度(把 IP 规范写进 prompt,自比对盲区的绝对参照)
    "f7_ip_conform": ("这个 IP 角色的标准形象是:矮胖圆润的身体、短腿、圆形头部、头顶戴一顶带白色斑点的红色蘑菇菌盖、"
                      "毛毡布偶质感。视频中的角色是否在体型比例、菌盖、五官或质感上明显偏离了这个标准形象?只回答是或否。"),
}


def read_frames(path: Path, n: int = N_FRAMES) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sorted(set(np.linspace(0, max(0, total - 1), n).round().astype(int).tolist()))
    out, idx, wi = [], 0, 0
    while wi < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if idx == want[wi]:
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            wi += 1
        idx += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()

    tok = proc.tokenizer
    yes_ids = [tok.encode("是", add_special_tokens=False)[0]]
    no_ids = [tok.encode("否", add_special_tokens=False)[0]]
    print("yes/no token ids:", yes_ids, no_ids, flush=True)

    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["stem"])
            except Exception:
                pass
    vids = sorted(Path(args.videos_dir).glob("*.mp4"))
    vids = [v for v in vids if v.stem not in done]
    if args.limit:
        vids = vids[: args.limit]
    print(f"todo {len(vids)} (skip {len(done)})", flush=True)

    import time
    t0 = time.time()
    with out.open("a") as f:
        for i, v in enumerate(vids):
            rec = {"stem": v.stem}
            try:
                frames = read_frames(v)
                for key, q in QUESTIONS.items():
                    messages = [{
                        "role": "user",
                        "content": ([{"type": "image", "image": im} for im in frames]
                                    + [{"type": "text", "text": q}]),
                    }]
                    inputs = proc.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        logits = model(**inputs).logits[0, -1]
                    lp = torch.log_softmax(logits.float(), dim=-1)
                    rec[key] = float(lp[yes_ids[0]] - lp[no_ids[0]])
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:200]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(vids)}] {el/(i+1):.1f}s/vid", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
