#!/usr/bin/env python
"""E59:衣服/外观改变 VLM 首末对比(2026-08-12 预注册)。

目标:尾部 15051(衣服/身体的时间一致性)。E55 dinov2 距离失败(嵌入对换色不敏感,
15051 仅 33 分位);但这是**类别型**视觉差异(衣服变没变),VLM 擅长——区别于
大小变化那种渐变量(VLM 盲区,E50-B 已死)。
呈现:crops_v3 第一帧与最后一帧的角色紧裁剪并排两图。
两阶段(label-blind 队列,冻结判准):
  ①train 验证:衣服类 bad(reasons 含 衣服/身体的时间一致性)∪ train zone good 采样;
    判准:衣服bad 旗标率 − good 旗标率 ≥ 40pt 且 good ≤ 10%;
  ②过门才扫 eval zone(e18<0.30 的全部 eval 视频,label-blind)。
输出 data/clothes_confirm.jsonl。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")

Q = """两张图分别是同一条AI生成视频里毛绒蘑菇角色「蘑菇TUTU」的第一帧裁剪和最后一帧裁剪(相隔约5秒)。
请对比两张图,回答一个客观问题:角色的**服装、配饰或身体颜色**是否发生了明显改变?
算改变的例子:衣服换了款式或颜色、帽子/配饰消失或凭空出现、身体或伞盖颜色明显变了。
不算改变:姿态/动作/朝向不同、光照阴影变化、轻微模糊、因角度导致的部分遮挡。
输出一行JSON:{"changed": true/false, "what": "变了什么(没变则空)", "note": "一句话"}"""

OUT = ROOT / "data/clothes_confirm.jsonl"


def pair_of(rel):
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    jp = sorted(d.glob("f*.jpg"))
    if len(jp) < 8:
        return None
    ims = []
    for p in (jp[0], jp[-1]):
        im = cv2.imread(str(p))
        if im is None:
            return None
        H, W = im.shape[:2]
        s = 448 / max(H, W)
        ims.append(Image.fromarray(cv2.cvtColor(
            cv2.resize(im, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB)))
    return ims


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "eval"], default="train")
    args = ap.parse_args()

    tr = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    ev = [json.loads(l) for l in open(ROOT / "splits/eval_v3.jsonl")]
    rel_of = {}
    for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
        if l.strip():
            rel = l.split("\t")[0]
            rel_of[rel.split("/")[-1]] = rel
    reasons = {r["path"]: r.get("reasons", "") for r in csv.DictReader(
        open(ROOT / "data/s3/merged_labels.csv", encoding="utf-8-sig"))}
    e18 = np.load(ROOT / "data/s3/e18_train_oof.npy")

    if args.stage == "train":
        cb = [r["video"] for r in tr if r["label"] == "bad"
              and "衣服/身体的时间一致性" in reasons.get(r["video"], "")]
        goods = [(e18[i], r["video"]) for i, r in enumerate(tr)
                 if r["label"] == "good" and e18[i] < 0.30]
        rng = np.random.RandomState(3)
        gs = [v for _s, v in goods]
        gs = list(rng.choice(gs, min(120, len(gs)), replace=False))
        todo = [(v, "CB") for v in cb] + [(v, "G") for v in gs]
        rng.shuffle(todo)
    else:
        # eval zone label-blind:用 e18 的 eval 预测(单发已存)选 p<0.30
        pr = {r["video"]: float(r["p_e18"]) for r in csv.DictReader(
            open(ROOT / "data/s3/predictions_e18_eval.csv"))}
        todo = [(r["video"], "EV") for r in ev if pr.get(r["video"], 1) < 0.30]
    print(f"stage={args.stage} 待跑 {len(todo)}", flush=True)

    done = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                done.add(json.loads(l)["video"])
            except Exception:
                pass

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
    n = 0
    for v, grp in todo:
        if v in done or v not in rel_of:
            continue
        row = {"video": v, "grp": grp, "stage": args.stage}
        try:
            ims = pair_of(rel_of[v])
            if ims is None:
                row["error"] = "no_frames"
            else:
                content = ([{"type": "text", "text": "第一帧:"}, {"type": "image", "image": ims[0]},
                            {"type": "text", "text": "最后一帧:"}, {"type": "image", "image": ims[1]},
                            {"type": "text", "text": Q}])
                inputs = proc.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=110, do_sample=False)
                text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
                j = parse(text)
                if j is not None:
                    row.update(changed=bool(j.get("changed")), what=j.get("what", ""),
                               note=j.get("note", "")[:80])
                else:
                    row["parse_error"] = text[:80]
        except Exception as e:
            row["error"] = repr(e)[:100]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        if n % 25 == 0:
            f.flush()
            print(f"[{n}/{len(todo)}] {(time.time()-t0)/n:.1f}s/条", flush=True)
    f.close()

    if args.stage == "train":
        import collections
        st = collections.defaultdict(collections.Counter)
        for l in open(OUT):
            d = json.loads(l)
            if d.get("stage") == "train" and "changed" in d:
                st[d["grp"]]["tot"] += 1
                if d["changed"]:
                    st[d["grp"]]["chg"] += 1
        for g in ("CB", "G"):
            c = st[g]
            if c["tot"]:
                print(f"[{g}] n={c['tot']} 旗标率 {c['chg']/c['tot']:.1%}", flush=True)
    print("E59_DONE", flush=True)


if __name__ == "__main__":
    main()
