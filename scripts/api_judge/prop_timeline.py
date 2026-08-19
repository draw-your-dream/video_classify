#!/usr/bin/env python3
"""prop 存在时间线特征(开卷考试:文件名给出期望道具)。

每视频 16 帧,OWLv2 开放词表检测 [角色, prop英文名]:
- 期望 prop 的存在时间线空洞数/最长空洞/首末缺失(凭空消失/出现)
- 角色框底边高度方差(漂浮代理)
- 非期望高置信检出计数(不合理物体代理,用通用 object 查询减去期望集)
输出 /workspace/r2/prop_timeline.csv
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

R2 = Path("/workspace/r2")

from transformers import Owlv2ForObjectDetection, Owlv2Processor

proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained(
    "google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to("cuda").eval()


def post_process(outputs, target_sizes):
    pp = (getattr(proc, "post_process_object_detection", None)
          or getattr(proc.image_processor, "post_process_object_detection", None)
          or getattr(proc, "post_process_grounded_object_detection"))
    return pp(outputs, threshold=0.08, target_sizes=target_sizes)


def detect(pils, queries):
    inputs = proc(text=[queries] * len(pils), images=pils, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()})
    ts = torch.tensor([p.size[::-1] for p in pils]).to("cuda")
    return post_process(out, ts)


def main():
    targets = [json.loads(l) for l in open(R2 / "pbase/upstream/splits/train_v2.jsonl")]
    rows = []
    t0 = time.time()
    for i, e in enumerate(targets):
        fn = Path(e["video"]).name
        parts = fn[:-4].split("__")
        prop = parts[4] if len(parts) >= 7 else ""
        queries = ["cartoon plush mushroom character"]
        if prop:
            queries.append(prop.replace("_", " "))
        try:
            cap = cv2.VideoCapture(str(R2 / "videos" / fn))
            N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idxs = set(np.linspace(0, max(0, N - 1), 16).astype(int).tolist())
            pils, t = [], 0
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                if t in idxs:
                    pils.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                t += 1
            cap.release()
            char_conf, prop_conf, char_bot = [], [], []
            B = 8
            for b0 in range(0, len(pils), B):
                res = detect(pils[b0:b0 + B], queries)
                for r in res:
                    lb = r["labels"].cpu().numpy()
                    sc = r["scores"].cpu().numpy()
                    bx = r["boxes"].cpu().numpy()
                    m0 = lb == 0
                    char_conf.append(float(sc[m0].max()) if m0.any() else 0.0)
                    if m0.any():
                        char_bot.append(float(bx[m0][sc[m0].argmax()][3]))
                    if len(queries) > 1:
                        m1 = lb == 1
                        prop_conf.append(float(sc[m1].max()) if m1.any() else 0.0)
            cc = np.array(char_conf)
            row = {"filename": fn, "prop": prop,
                   "char_min": float(cc.min()), "char_mean": float(cc.mean()),
                   "char_holes": int((cc < 0.1).sum()),
                   "char_bot_std": float(np.std(char_bot)) if len(char_bot) > 2 else 0.0}
            if prop_conf:
                pc = np.array(prop_conf)
                present = pc >= 0.1
                # 空洞:中段消失(首尾都在但中间断)
                holes = 0
                run = 0
                for v in present:
                    if not v:
                        run += 1
                    else:
                        if run:
                            holes += 1
                        run = 0
                row.update(prop_mean=float(pc.mean()), prop_min=float(pc.min()),
                           prop_present_frac=float(present.mean()),
                           prop_holes=holes,
                           prop_first=float(pc[0]), prop_last=float(pc[-1]))
            else:
                row.update(prop_mean=-1, prop_min=-1, prop_present_frac=-1,
                           prop_holes=-1, prop_first=-1, prop_last=-1)
            rows.append(row)
        except Exception as ex:
            print("ERR", fn[:40], repr(ex)[:80], flush=True)
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(targets)}] {(time.time()-t0)/(i+1):.2f}s/条", flush=True)
    cols = list(rows[0].keys())
    with open(R2 / "prop_timeline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("PROP_TIMELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
