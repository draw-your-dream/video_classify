#!/usr/bin/env python
"""SAM3 掩码底座提取(P3,2026-08-01 预注册)。

每视频 16 均匀帧 -> SAM3 文本 prompt 全实例 -> 最高分实例白底抠像 448 jpg
+ 逐帧几何/计数记录(npz)。为 R(参照相似)/G(几何)/C(计数)三轴与 VLM 轴共用。
盒侧布局 /root/mech;python=/venv/main/bin/python。
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

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "A fluffy mushroom-like creature with a light-yellow body and a red cap"
N_FRAMES = 16
CANVAS = 448
SCORE_THR = 0.5          # 实例计数口径:score >= thr 才算一个实例
CAP_BAND = 0.40          # 伞宽 = 掩码上部 40% 内最大行宽


def read_frames(path: Path, n: int = N_FRAMES):
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


def mask_geometry(m: np.ndarray):
    """m: bool (H,W)。返回 height, cap_width, area, bbox。"""
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h = int(y1 - y0 + 1)
    band = m[y0:y0 + max(1, int(round(h * CAP_BAND)))]
    cap_w = int(band.sum(1).max()) if band.size else 0
    return {"height": h, "cap_width": cap_w, "area": int(m.sum()),
            "bbox": [int(x0), int(y0), int(x1), int(y1)]}


def white_cutout(fr_bgr: np.ndarray, m: np.ndarray, canvas: int = CANVAS):
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = fr_bgr[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    mm = m[y0:y1 + 1, x0:x1 + 1].astype(np.float32)[..., None]
    cut = (crop * mm + 255.0 * (1 - mm)).astype(np.uint8)
    h, w = cut.shape[:2]
    s = canvas / max(h, w)
    cut = cv2.resize(cut, (max(1, int(w * s)), max(1, int(h * s))))
    out = np.full((canvas, canvas, 3), 255, np.uint8)
    yo, xo = (canvas - cut.shape[0]) // 2, (canvas - cut.shape[1]) // 2
    out[yo:yo + cut.shape[0], xo:xo + cut.shape[1]] = cut
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/prod500/mech_subset.tsv"))
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/sam3_feat"))
    ap.add_argument("--cutout-dir", default=str(ROOT / "data/sam3_cutouts"))
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/sam3.pt"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=args.ckpt, load_from_HF=False)
    proc = Sam3Processor(model)
    print("sam3 loaded", flush=True)

    out_dir, cut_dir = Path(args.out_dir), Path(args.cutout_dir)
    rows = [l.split("\t") for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    t0 = time.time()
    done = 0
    for rel, label in rows:
        npz_p = out_dir / rel.replace(".mp4", ".npz")
        if npz_p.exists():
            done += 1
            continue
        npz_p.parent.mkdir(parents=True, exist_ok=True)
        vdir = cut_dir / rel.replace(".mp4", "")
        vdir.mkdir(parents=True, exist_ok=True)
        frames = read_frames(Path(args.videos_dir) / rel)
        n_inst, top_score, geo = [], [], []
        for i, fr in enumerate(frames):
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    st = proc.set_image(im)
                    outp = proc.set_text_prompt(state=st, prompt=PROMPT)
                masks, scores = outp["masks"], outp["scores"]
                sc = scores.detach().float().cpu().numpy().reshape(-1)
                keep = sc >= SCORE_THR
                n_inst.append(int(keep.sum()))
                if len(sc) and sc.max() > 0.05:
                    j = int(sc.argmax())
                    m = masks[j].detach().cpu().numpy().astype(bool)
                    m = m[0] if m.ndim == 3 else m
                    g = mask_geometry(m)
                    top_score.append(float(sc[j]))
                    if g is not None:
                        geo.append({"frame": i, **g})
                        cv2.imwrite(str(vdir / f"f{i:02d}.jpg"),
                                    white_cutout(fr, m),
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    top_score.append(0.0)
            except Exception as e:
                print(f"ERR {rel} f{i}: {e}", flush=True)
                n_inst.append(-1)
                top_score.append(-1.0)
        np.savez_compressed(
            npz_p, n_inst=np.array(n_inst, np.int16),
            top_score=np.array(top_score, np.float32),
            geo=json.dumps(geo), label=label)
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f"[{done}/{len(rows)}] {el/max(1,done):.1f}s/vid elapsed {el/60:.1f}m",
                  flush=True)
    print("SAM3_EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
