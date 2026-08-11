#!/usr/bin/env python
"""E50-A:眉毛 VLM 确认全量补齐(2026-08-11 预注册)。

历史缺口:brow_confirm 仅覆盖 850 候选中 top-400(CNN 饱和截断),用户审计的
5719/6565/6228(眉毛)与 5822(尾巴)全部未进过 VLM 确认。
本脚本:①eval 全量 968 条(label-blind,脱离 CNN 依赖)②全语料 btop3≥0.4 未确认候选。
prompt / 放大方式(上部62%×2)/ 模型(Qwen3-VL-32B bf16)全部冻结自 v6 已验证版本,零改动。
eval 无 crops 的少数视频回退原始 mp4 整帧(记 fallback 标志)。
输出 data/brow_confirm_full.jsonl,断点续跑。
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")

Q = """以下最多16张图是同一条AI生成视频的逐帧画面,已放大到角色头部区域(按时间顺序,第0张起)。只判断毛绒蘑菇角色「蘑菇TUTU」本体。
官方设定:TUTU 无眉毛;两颗实心黑豆眼。被遮挡或看不清的部位一律默认正常。
问题:是否看到眉毛?眉毛指位于眼睛上方、与眼睛明显分离的独立弧线、斜线或条状痕迹(颜色可深可浅)。
眯起的窄眼本身不算眉毛;但如果同一帧里既有眼睛、眼睛上方又另有分离的线条/痕迹,那就是眉毛。
输出一行JSON:{"eyebrows": [帧号], "note": "一句话"}"""

OUT = ROOT / "data/brow_confirm_full.jsonl"


def frames_of(rel):
    """优先 crops(与 v6 同分布);无 crops 回退原始视频整帧。返回 (PIL列表, fallback标志)。"""
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    jpgs = sorted(d.glob("f*.jpg"))[:16]
    if len(jpgs) >= 4:
        ims = []
        for p in jpgs:
            im = cv2.imread(str(p))
            if im is None:
                return None, False
            H, W = im.shape[:2]
            u = im[0:int(H * 0.62), :]
            u = cv2.resize(u, (W * 2, int(H * 0.62 * 2)), interpolation=cv2.INTER_CUBIC)
            ims.append(Image.fromarray(cv2.cvtColor(u, cv2.COLOR_BGR2RGB)))
        return ims, False
    vp = ROOT / "data/corpus_videos" / rel
    if not vp.exists():
        return None, True
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 4:
        cap.release()
        return None, True
    want = set(np.linspace(0, total - 1, 16).round().astype(int).tolist())
    ims, k = [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k in want:
            H, W = im.shape[:2]
            u = im[0:int(H * 0.62), :]
            s = 900 / max(u.shape[:2])
            u = cv2.resize(u, (int(W * s), int(u.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
            ims.append(Image.fromarray(cv2.cvtColor(u, cv2.COLOR_BGR2RGB)))
        k += 1
    cap.release()
    return (ims if len(ims) >= 4 else None), True


def main():
    rel_of = {}
    for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
        if l.strip():
            rel = l.split("\t")[0]
            rel_of[rel.split("/")[-1]] = rel

    # 队列①:eval 全量(label-blind,不看标签只取视频名)
    ev = [json.loads(l) for l in open(ROOT / "splits/eval_v3.jsonl")]
    q_eval = [rel_of[r["video"]] for r in ev if r["video"] in rel_of]

    # 队列②:全语料 btop3>=0.4 候选
    q_cand = []
    for r in csv.DictReader(open(ROOT / "data/brow_scan.csv")):
        try:
            if float(r["btop3"]) >= 0.4:
                q_cand.append((float(r["btop3"]), r["rel"]))
        except Exception:
            pass
    q_cand.sort(reverse=True)

    done = set()
    for src in (OUT, ROOT / "data/brow_confirm.jsonl"):
        if src.exists():
            for l in open(src):
                try:
                    done.add(json.loads(l)["rel"])
                except Exception:
                    pass
    # eval 优先(决定否决门),再补语料候选;跳过历史已确认
    todo = [r for r in q_eval if r not in done]
    seen = set(todo) | done
    todo += [r for _s, r in q_cand if r not in seen]
    print(f"eval全量 {len(q_eval)} | 候选≥0.4 {len(q_cand)} | 历史已确认 {len(done)} | 本次待跑 {len(todo)}",
          flush=True)

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
            ims, fb = frames_of(rel)
            if fb:
                row["fallback"] = 1
            if ims is None:
                row["error"] = "no_frames"
            else:
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
                    row.update(eyebrows=j.get("eyebrows", []), note=j.get("note", ""))
                else:
                    row["parse_error"] = text[:100]
        except Exception as e:
            row["error"] = repr(e)[:110]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (k + 1) % 25 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.1f}s/条", flush=True)
    f.close()
    print("E50A_BROW_FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
