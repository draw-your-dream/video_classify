#!/usr/bin/env python
"""E50-D:僵硬/四肢不动的 VLM 事实提取试点(2026-08-11 预注册,train 侧盲测)。

与 E4(失败的零样本运动判官,midrank AUC 0.546)的本质区别:E4 问整体运动质量(主观评价),
本试点问**具体可核对的事实**——四肢相对躯干的姿态是否变化、角色是否有整体位移;
事实提取范式已在眉毛/悬空上验证(81-84% 精度)。
与 E30 rigid 特征(数值统计,已失败)的区别:VLM 看的是空间结构(哪里在动),非全局统计量。

盲测设计(train 侧,不碰 eval):
  实验组 = train bad ∩ reasons含[僵硬|四肢不动|静止不动] ∩ E18-OOF 处于 bad 内最低 30%(尾部同类)
  对照组 = train good 等量随机(同源混合)
  打乱顺序,VLM 不知标签。
判准(发车前冻结):实验组「四肢无变化」旗标率 与 对照组 之差 ≥ 40 个百分点
(如 70% vs 30%),才谈扩展到全量与否决门;否则运动族 VLM 事实路线收档。
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

Q = """以下最多16张图是同一条AI生成视频的逐帧画面(按时间顺序,第0张起,时长约5秒)。只观察毛绒蘑菇角色「蘑菇TUTU」本体(短粗四肢、蘑菇伞盖)。
请回答两个客观问题:
1. limbs_move:对比各帧,角色的手臂或腿相对躯干的姿态是否发生了明显变化(挥手、抬腿、摆臂等,任意一处即算)?true/false
2. body_moves:角色整体(躯干)在画面中是否有明显的位置移动或转身?true/false
注意:摄像机推拉/画面缩放导致的大小变化不算角色自身运动;衣物或背景的变化不算。
输出一行JSON:{"limbs_move": true/false, "body_moves": true/false, "note": "一句话"}"""

OUT = ROOT / "data/motion_pilot.jsonl"


def main():
    # 组装盲测名单(确定性,seed 固定)
    tr = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    rel_of = {}
    for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
        if l.strip():
            rel = l.split("\t")[0]
            rel_of[rel.split("/")[-1]] = rel
    reasons = {r["path"]: r.get("reasons", "") for r in csv.DictReader(
        open(ROOT / "data/s3/merged_labels.csv", encoding="utf-8-sig"))}
    e18 = np.load(ROOT / "data/s3/e18_train_oof.npy")
    assert len(e18) == len(tr)
    KW = ("僵硬", "四肢不动", "静止不动")
    bad_i = [i for i, r in enumerate(tr) if r["label"] == "bad" and r["video"] in rel_of]
    bad_scores = np.array([e18[i] for i in bad_i])
    cut = np.quantile(bad_scores, 0.30)
    exp = [i for i in bad_i if e18[i] <= cut
           and any(k in reasons.get(tr[i]["video"], "") for k in KW)]
    goods = [i for i, r in enumerate(tr) if r["label"] == "good" and r["video"] in rel_of]
    rng = np.random.RandomState(7)
    exp = list(rng.choice(exp, min(30, len(exp)), replace=False))
    ctl = list(rng.choice(goods, len(exp), replace=False))
    pool = [(int(i), "exp") for i in exp] + [(int(i), "ctl") for i in ctl]
    rng.shuffle(pool)
    print(f"实验组 {len(exp)}(僵硬类尾部bad) 对照组 {len(ctl)}(随机good),盲序混合", flush=True)

    done = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                done.add(json.loads(l)["rel"])
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
    for i, grp in pool:
        rel = rel_of[tr[i]["video"]]
        if rel in done:
            continue
        d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
        jpgs = sorted(d.glob("f*.jpg"))[:16]
        row = {"rel": rel, "grp": grp}
        try:
            if len(jpgs) < 4:
                row["error"] = "no_crops"
            else:
                ims = []
                for p in jpgs:
                    im = cv2.imread(str(p))
                    ims.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
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
                    row.update(limbs_move=j.get("limbs_move"), body_moves=j.get("body_moves"),
                               note=j.get("note", ""))
                else:
                    row["parse_error"] = text[:100]
        except Exception as e:
            row["error"] = repr(e)[:110]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        if n % 10 == 0:
            f.flush()
            print(f"[{n}/{len(pool)}] {(time.time()-t0)/n:.1f}s/条", flush=True)
    f.close()

    # 即时判读
    import collections
    stat = collections.defaultdict(collections.Counter)
    for l in open(OUT):
        d = json.loads(l)
        if "limbs_move" in d and d["limbs_move"] is not None:
            stat[d["grp"]]["no_limb" if not d["limbs_move"] else "limb"] += 1
            if not d["limbs_move"] and not d.get("body_moves"):
                stat[d["grp"]]["frozen"] += 1
    for g in ("exp", "ctl"):
        c = stat[g]
        tot = c["no_limb"] + c["limb"]
        if tot:
            print(f"[{g}] n={tot} 四肢无变化率={c['no_limb']/tot:.2%} 完全冻结率={c['frozen']/tot:.2%}",
                  flush=True)
    print("E50D_MOTION_PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
