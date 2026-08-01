#!/usr/bin/env python
"""线F:SAM3 脸部裁剪提取(2026-08-01 预注册)。

每视频 16 帧原生分辨率 -> SAM3 "the face of the mushroom character"
-> 最高分实例掩码 bbox ×1.4 方形裁剪 -> 384px jpg + 分数记录。
score < 0.5 记"脸不可见",跳过不罚(遮挡安全)。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROMPT = "the face of the mushroom character"
SCORE_THR = 0.5
OUT_PX = 384


def read_frames(path: Path, n: int = 16):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    want = sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))
    frames, idx, wi = [], 0, 0
    while wi < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if idx == want[wi]:
            frames.append(fr)
            wi += 1
        idx += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/root/mech/data/prod500/mech_subset.tsv")
    ap.add_argument("--videos-dir", default="/root/mech/data/corpus_videos")
    ap.add_argument("--out-dir", default="/root/mech/data/face_crops")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path="/root/mech/checkpoints/sam3.pt",
                                   load_from_HF=False)
    proc = Sam3Processor(model)
    print("sam3 loaded", flush=True)

    rows = [l.split("\t") for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    t0, done = time.time(), 0
    for rel, label in rows:
        vdir = Path(args.out_dir) / rel.replace(".mp4", "")
        meta_p = vdir / "meta.json"
        if meta_p.exists():
            done += 1
            continue
        vdir.mkdir(parents=True, exist_ok=True)
        frames = read_frames(Path(args.videos_dir) / rel)
        meta = {"label": label, "scores": []}
        for i, fr in enumerate(frames):
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    st = proc.set_image(im)
                    out = proc.set_text_prompt(state=st, prompt=PROMPT)
                sc = out["scores"].detach().float().cpu().numpy().reshape(-1)
                if len(sc) == 0 or sc.max() < SCORE_THR:
                    meta["scores"].append(float(sc.max()) if len(sc) else 0.0)
                    continue
                j = int(sc.argmax())
                m = out["masks"][j].detach().cpu().numpy().astype(bool)
                m = m[0] if m.ndim == 3 else m
                ys, xs = np.where(m)
                if len(ys) < 50:
                    meta["scores"].append(0.0)
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                s = int(max(y1 - y0, x1 - x0) * 1.4 / 2) + 8
                ya, yb = max(0, cy - s), min(fr.shape[0], cy + s)
                xa, xb = max(0, cx - s), min(fr.shape[1], cx + s)
                crop = fr[ya:yb, xa:xb]
                cv2.imwrite(str(vdir / f"f{i:02d}.jpg"),
                            cv2.resize(crop, (OUT_PX, OUT_PX)),
                            [cv2.IMWRITE_JPEG_QUALITY, 93])
                meta["scores"].append(float(sc[j]))
            except Exception as e:
                print(f"ERR {rel} f{i}: {e}", flush=True)
                meta["scores"].append(-1.0)
        meta_p.write_text(json.dumps(meta))
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"[{done}/{len(rows)}] {el/max(1,done):.1f}s/vid elapsed {el/60:.1f}m",
                  flush=True)
    print("FACE_EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
