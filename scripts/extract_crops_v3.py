#!/usr/bin/env python
"""v3 提取:参考图锚定实例选择 + 帧连续 + QC + 宽松原图裁剪(2026-08-04 预注册)。

每帧:SAM3 全候选实例(score>=0.3)-> 每个候选 bbox 裁剪 -> so400m 嵌入
-> 选择分 = 对152参考池最大cos + 0.15*与上一帧选中框的IoU -> 取最高。
输出:crops_v3/<rel>/fNN.jpg(bbox外扩25%原画面裁剪,448)+ qc.json(逐帧 ref_sim/bbox)。
QC:视频级 median ref_sim < 0.45 记 failed。默认只处理 --rels-file 列表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")
N_FRAMES = 16
PROMPT = "A fluffy mushroom-like creature with a light-yellow body and a red cap"


def read_frames(vp):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        cap.release(); return []
    idxs = [int(round(i * (n - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)]
    want = set(idxs); out = {}; k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            out[k] = fr
        k += 1
    cap.release()
    return [(i, out[i]) for i in sorted(out)]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rels-file", required=True)
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/crops_v3"))
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/sam3.pt"))
    ap.add_argument("--ref", default=str(ROOT / "ref_embeds.npz"))
    args = ap.parse_args()

    rels = [l.strip() for l in open(args.rels_file) if l.strip()]
    out_dir = Path(args.out_dir)
    done = {p.parent.name + "/" + p.name for p in []}
    todo = [r for r in rels if not (out_dir / r.replace(".mp4", "") / "qc.json").exists()]
    print(f"todo {len(todo)}/{len(rels)}", flush=True)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=args.ckpt, load_from_HF=False)
    proc = Sam3Processor(model)
    from transformers import AutoModel, AutoProcessor
    mid = "google/siglip2-so400m-patch16-512"
    sp = AutoProcessor.from_pretrained(mid)
    sm = AutoModel.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
    R = np.load(args.ref, allow_pickle=True)["so400m"].astype(np.float32)
    print("models loaded", flush=True)

    @torch.inference_mode()
    def embed(ims):
        inp = sp(images=ims, return_tensors="pt").to("cuda")
        f = sm.get_image_features(pixel_values=inp["pixel_values"].half())
        if not torch.is_tensor(f):
            f = f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    import time
    t0 = time.time()
    for k, rel in enumerate(todo):
        vdir = out_dir / rel.replace(".mp4", "")
        vdir.mkdir(parents=True, exist_ok=True)
        qc = []
        prev_bb = None
        try:
            for fi, fr in read_frames(Path(args.videos_dir) / rel):
                H, W = fr.shape[:2]
                im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    st = proc.set_image(im)
                    outp = proc.set_text_prompt(state=st, prompt=PROMPT)
                masks, scores = outp["masks"], outp["scores"]
                sc = scores.detach().float().cpu().numpy().reshape(-1)
                cand = [i for i in range(len(sc)) if sc[i] >= 0.3][:5]
                if not cand:
                    qc.append({"frame": fi, "ok": False})
                    continue
                bbs, crops = [], []
                for i in cand:
                    m = masks[i].detach().cpu().numpy().astype(bool)
                    m = m[0] if m.ndim == 3 else m
                    ys, xs = np.where(m)
                    if len(ys) < 50:
                        bbs.append(None); crops.append(None); continue
                    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                    bbs.append((int(x0), int(y0), int(x1), int(y1)))
                    mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
                    cx0, cy0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
                    cx1, cy1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
                    crops.append(fr[cy0:cy1, cx0:cx1])
                valid = [i for i in range(len(cand)) if crops[i] is not None]
                if not valid:
                    qc.append({"frame": fi, "ok": False})
                    continue
                pil = [Image.fromarray(cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB)) for i in valid]
                E = embed(pil)
                sims = (E @ R.T).max(1)
                best, best_s = None, -9
                for j, i in enumerate(valid):
                    s = float(sims[j]) + 0.15 * iou(bbs[i], prev_bb)
                    if s > best_s:
                        best, best_s, best_sim = i, s, float(sims[j])
                prev_bb = bbs[best]
                crop = crops[best]
                h, w = crop.shape[:2]
                s448 = 448 / max(h, w)
                crop = cv2.resize(crop, (int(w * s448), int(h * s448)))
                cv2.imwrite(str(vdir / f"f{fi:02d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                qc.append({"frame": fi, "ok": True, "ref_sim": round(best_sim, 4),
                           "bbox": [int(v) for v in bbs[best]]})
        except Exception as e:
            qc.append({"error": repr(e)[:120]})
        sims = [q["ref_sim"] for q in qc if q.get("ok")]
        (vdir / "qc.json").write_text(json.dumps(
            {"rel": rel, "median_ref_sim": round(float(np.median(sims)), 4) if sims else 0.0,
             "failed": (not sims) or float(np.median(sims)) < 0.45, "frames": qc}, ensure_ascii=False))
        if (k + 1) % 50 == 0:
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    print("CROPS_V3_DONE", flush=True)


if __name__ == "__main__":
    main()
