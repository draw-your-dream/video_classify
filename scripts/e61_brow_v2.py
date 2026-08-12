#!/usr/bin/env python
"""E61:眉毛 prompt v2——窄眼防护(2026-08-12,用户诊断:错判主要来自眯起的窄眼)。

v2 修改(仅加强排除条款,判断框架不动——E25 教训:重写框架会翻转崩溃方向):
显式描述"眯眼/闭眼时黑豆眼本身呈横线/弧线"的形态,并要求**同帧同时可见眼睛与其上方分离线条**
才算眉毛;只见一条线时按眯眼处理。
验证集(全部有裁决依据):
  must-keep = 5719/6565/6228(用户亲验真眉毛)+ train 新确认命中 40 条(80% 精度基础);
  must-drop = 用户裁决维持原标注的 11 条 eval 误报(R17-R26 中除 R27/R28)。
判准(冻结):must-keep 保留率 ≥ 90% 且 must-drop 删除率 ≥ 60%,才替换生产 prompt 并重算门。
输出 data/brow_v2_confirm.jsonl。
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

Q2 = """以下最多16张图是同一条AI生成视频的逐帧画面,已放大到角色头部区域(按时间顺序,第0张起)。只判断毛绒蘑菇角色「蘑菇TUTU」本体。
官方设定:TUTU 无眉毛;两颗实心黑豆眼。被遮挡或看不清的部位一律默认正常。
【重要:眯眼不是眉毛】TUTU 眯眼或闭眼时,黑豆眼本身会变成一条黑色横线或弧线——那是眼睛,不是眉毛。
判定规则:只有当**同一帧里同时看到**眼睛(圆点状或眯成的线状)**和**位于其上方、与之明显分离的另一条弧线/斜线/条状痕迹时,才算看到眉毛;
如果只看到一条线而它上方没有另一条分离的线,一律按眯眼处理,不算眉毛。
问题:按上述规则,是否看到眉毛?
输出一行JSON:{"eyebrows": [同时满足规则的帧号], "note": "一句话"}"""

OUT = ROOT / "data/brow_v2_confirm.jsonl"

MUST_KEEP_EVAL = ["5719.mp4", "6565.mp4", "6228.mp4"]
MUST_DROP = ["20074.mp4", "5805.mp4", "6817.mp4", "7254.mp4", "6937.mp4",
             "5684.mp4"]  # 前6条;其余5条从清单文件读


def frames_of(rel, cdir):
    d = Path(cdir) / rel.replace(".mp4", "")
    jp = sorted(d.glob("f*.jpg"))[:16]
    if len(jp) < 8:
        return None
    ims = []
    for p in jp:
        im = cv2.imread(str(p))
        if im is None:
            return None
        H, W = im.shape[:2]
        u = im[0:int(H * 0.62), :]
        u = cv2.resize(u, (W * 2, int(H * 0.62 * 2)), interpolation=cv2.INTER_CUBIC)
        ims.append(Image.fromarray(cv2.cvtColor(u, cv2.COLOR_BGR2RGB)))
    return ims


def main():
    rel_of = {}
    for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
        if l.strip():
            rel = l.split("\t")[0]
            rel_of[rel.split("/")[-1]] = rel

    # must-drop 全 11 条:清单里 axis=眉毛 且 user_verdict=keep
    md = set(MUST_DROP)
    cl = json.load(open(ROOT / "data/s3/relabel_candidates_v3.json"))
    for x in cl:
        if x["axis"] == "眉毛" and x.get("user_verdict") == "keep":
            md.add(x["video"])
    # must-keep:3 锚 + train 新确认命中(brow_confirm_full 中 train 侧 hits)
    tr_v = {json.loads(l)["video"] for l in open(ROOT / "splits/train_v3.jsonl")}
    mk = list(MUST_KEEP_EVAL)
    for l in open(ROOT / "data/s3/brow_confirm_full.jsonl"):
        d = json.loads(l)
        v = os.path.basename(d["rel"])
        if d.get("eyebrows") and v in tr_v:
            mk.append(v)
    mk = mk[:43]
    todo = [(v, "KEEP") for v in mk] + [(v, "DROP") for v in sorted(md)]
    print(f"must-keep {len(mk)} | must-drop {len(md)}", flush=True)

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
        row = {"video": v, "grp": grp}
        try:
            ims = None
            for cdir in (ROOT / "data/crops_geo", ROOT / "data/crops_v3"):
                ims = frames_of(rel_of[v], cdir)
                if ims:
                    break
            if ims is None:
                row["error"] = "no_frames"
            else:
                content = [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": Q2}]
                inputs = proc.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=130, do_sample=False)
                text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
                j = parse(text)
                if j is not None:
                    row.update(eyebrows=j.get("eyebrows", []), note=j.get("note", "")[:70])
                else:
                    row["parse_error"] = text[:70]
        except Exception as e:
            row["error"] = repr(e)[:100]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        if n % 15 == 0:
            f.flush()
            print(f"[{n}/{len(todo)}] {(time.time()-t0)/n:.1f}s/条", flush=True)
    f.close()

    keep_ok = keep_t = drop_ok = drop_t = 0
    anchors = {}
    for l in open(OUT):
        d = json.loads(l)
        if "eyebrows" not in d:
            continue
        hit = bool(d["eyebrows"])
        if d["grp"] == "KEEP":
            keep_t += 1
            keep_ok += int(hit)
            if d["video"] in MUST_KEEP_EVAL:
                anchors[d["video"]] = hit
        else:
            drop_t += 1
            drop_ok += int(not hit)
    print(f"must-keep 保留 {keep_ok}/{keep_t} = {keep_ok/max(1,keep_t):.1%} (判准>=90%)", flush=True)
    print(f"must-drop 删除 {drop_ok}/{drop_t} = {drop_ok/max(1,drop_t):.1%} (判准>=60%)", flush=True)
    print(f"三锚: {anchors}", flush=True)
    print("E61_DONE", flush=True)


if __name__ == "__main__":
    main()
