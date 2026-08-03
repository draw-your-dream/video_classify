#!/usr/bin/env python
"""E11:V-JEPA2 ViT-giant 视频级嵌入(盒侧,2026-08-03 预注册)。

每条视频均匀取 64 帧(模型口径 fpc64,短边 256/384 由 processor 处理),
输出 pooler/均值池化嵌入一行,npz 汇总 {video: 名, emb: (N,D)}。断点续跑(jsonl 落盘)。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] if "mech" in str(Path(__file__).resolve()) else Path("/root/mech")


def read_clip(vp, n_frames):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        cap.release(); return None
    idxs = [int(round(i * (n - 1) / (n_frames - 1))) for i in range(n_frames)]
    want = {}
    for i in idxs:
        want.setdefault(i, 0)
    out, k = [], 0
    grab = sorted(want)
    gi = 0
    frames_by_idx = {}
    while gi < len(grab):
        ok, fr = cap.read()
        if not ok:
            break
        if k == grab[gi]:
            frames_by_idx[k] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            gi += 1
        k += 1
    cap.release()
    if not frames_by_idx:
        return None
    last = None
    clip = []
    for i in idxs:
        if i in frames_by_idx:
            last = frames_by_idx[i]
        clip.append(last if last is not None else list(frames_by_idx.values())[0])
    return np.stack(clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--manifest", default=str(ROOT / "manifest_all.tsv"))
    ap.add_argument("--out", default=str(ROOT / "data/vjepa2g.jsonl"))
    ap.add_argument("--model", default="facebook/vjepa2-vitg-fpc64-256")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rels = [l.split("\t")[0] for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    done = set()
    out_p = Path(args.out)
    if out_p.exists():
        done = {json.loads(l)["rel"] for l in out_p.read_text().splitlines() if l.strip()}
    todo = [r for r in rels if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"total {len(rels)} done {len(done)} todo {len(todo)}", flush=True)

    from transformers import AutoModel, AutoVideoProcessor
    proc = AutoVideoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    n_frames = getattr(model.config, "frames_per_clip", 64)
    print("model loaded, frames_per_clip:", n_frames, flush=True)

    f = open(out_p, "a")
    t0 = time.time()
    n_err = 0
    for k, rel in enumerate(todo):
        try:
            clip = read_clip(Path(args.videos_dir) / rel, n_frames)
            if clip is None:
                raise RuntimeError("no_frames")
            with torch.inference_mode():
                inp = proc(videos=list(clip), return_tensors="pt").to("cuda")
                out = model(**inp.to(torch.bfloat16))
                h = out.last_hidden_state  # (1, T*, D)
                emb = h.float().mean(dim=1).squeeze(0).cpu().numpy()
            f.write(json.dumps({"rel": rel, "emb": [round(float(x), 5) for x in emb]}) + "\n")
        except Exception as e:
            n_err += 1
            if n_err <= 5 or n_err % 100 == 0:
                print(f"ERR[{n_err}] {rel}: {repr(e)[:100]}", flush=True)
            f.write(json.dumps({"rel": rel, "error": repr(e)[:80]}) + "\n")
        if (k + 1) % 50 == 0:
            f.flush()
            el = time.time() - t0
            print(f"[{k+1}/{len(todo)}] {el/(k+1):.2f}s/vid elapsed {el/60:.1f}m", flush=True)
    f.close()
    print("VJEPA2_DONE", flush=True)


if __name__ == "__main__":
    main()
