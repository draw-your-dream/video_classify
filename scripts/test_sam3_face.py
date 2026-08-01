#!/usr/bin/env python
"""线F 前置:SAM3 能否定位 TUTU 的脸(prompt 试验 + 蒙太奇核验)。

对 12 条随机视频的第 0 帧跑三个 face prompt,输出:脸框裁剪蒙太奇
(每行一个 prompt,行内 12 例,分数标注),供人工核验后选定 prompt。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROMPTS = [
    "the face of the mushroom character",
    "face",
    "the eyes and mouth area of the plush toy",
]


def read_frame0(p: Path):
    cap = cv2.VideoCapture(str(p))
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def main():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path="/root/mech/checkpoints/sam3.pt",
                                   load_from_HF=False)
    proc = Sam3Processor(model)
    vids = sorted(Path("/root/mech/data/corpus_videos").glob("*/*.mp4"))
    rng = np.random.default_rng(7)
    picks = [vids[i] for i in rng.choice(len(vids), 12, replace=False)]

    rows = []
    for prompt in PROMPTS:
        cells = []
        for vp in picks:
            fr = read_frame0(vp)
            if fr is None:
                cells.append(np.zeros((200, 200, 3), np.uint8))
                continue
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                st = proc.set_image(im)
                out = proc.set_text_prompt(state=st, prompt=prompt)
            sc = out["scores"].detach().float().cpu().numpy().reshape(-1)
            if len(sc) == 0 or sc.max() < 0.05:
                c = cv2.resize(fr, (200, 200))
                cv2.putText(c, "MISS", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cells.append(c)
                continue
            j = int(sc.argmax())
            m = out["masks"][j].detach().cpu().numpy().astype(bool)
            m = m[0] if m.ndim == 3 else m
            ys, xs = np.where(m)
            if len(ys) == 0:
                cells.append(np.zeros((200, 200, 3), np.uint8))
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            s = int(max(y1 - y0, x1 - x0) * 1.4 / 2) + 8
            ya, yb = max(0, cy - s), min(fr.shape[0], cy + s)
            xa, xb = max(0, cx - s), min(fr.shape[1], cx + s)
            crop = fr[ya:yb, xa:xb].copy()
            c = cv2.resize(crop, (200, 200))
            cv2.putText(c, f"{sc[j]:.2f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cells.append(c)
        strip = np.concatenate(cells, 1)
        strip = cv2.copyMakeBorder(strip, 26, 2, 2, 2, cv2.BORDER_CONSTANT,
                                   value=(40, 40, 40))
        cv2.putText(strip, prompt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        rows.append(strip)
    cv2.imwrite("/root/mech/face_prompt_montage.jpg", np.concatenate(rows, 0),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("FACE_TEST_DONE")


if __name__ == "__main__":
    main()
