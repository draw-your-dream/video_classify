#!/usr/bin/env python
"""E22 首帧漂移专家(2026-08-04 预注册):每帧对首帧的 DINOv2 相似度序列。

两路:全帧 与 角色框内(bbox 来自 sam3_feat)。
特征 12 列:g_end g_slope g_maxjump g_jumpt g_auc c_end c_slope c_maxjump c_jumpt c_auc
           g_curv(二阶) c_curv"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")
N_FRAMES = 16
COLS = "g_end g_slope g_maxjump g_jumpt g_auc g_curv c_end c_slope c_maxjump c_jumpt c_auc c_curv".split()


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


def curve_feats(sims, prefix):
    d = 1.0 - np.array(sims)
    t = np.arange(len(d))
    out = {}
    out[f"{prefix}_end"] = float(d[-1])
    out[f"{prefix}_slope"] = float(np.polyfit(t, d, 1)[0]) if len(d) > 3 else np.nan
    dj = np.abs(np.diff(d))
    out[f"{prefix}_maxjump"] = float(dj.max()) if len(dj) else np.nan
    out[f"{prefix}_jumpt"] = float(np.argmax(dj) / max(1, len(dj))) if len(dj) else np.nan
    out[f"{prefix}_auc"] = float(d.mean())
    out[f"{prefix}_curv"] = float(np.abs(np.diff(d, 2)).mean()) if len(d) > 4 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--feat-dir", default=str(ROOT / "data/sam3_feat"))
    ap.add_argument("--manifest", default=str(ROOT / "manifest_all.tsv"))
    ap.add_argument("--out", default=str(ROOT / "data/e22_drift.csv"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rels = [l.split("\t")[0] for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    done = set()
    out_p = Path(args.out)
    if out_p.exists():
        done = {r[0] for r in csv.reader(open(out_p))}
    else:
        csv.writer(open(out_p, "w", newline="")).writerow(["rel"] + COLS)
    todo = [r for r in rels if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"todo {len(todo)}", flush=True)

    from transformers import AutoModel, AutoImageProcessor
    mid = "/root/mech/models/dinov2-base"
    sp = AutoImageProcessor.from_pretrained(mid)
    sm = AutoModel.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()

    @torch.inference_mode()
    def embed(ims):
        inp = sp(images=ims, return_tensors="pt").to("cuda")
        f = sm(pixel_values=inp["pixel_values"].half()).last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    import time
    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    for k, rel in enumerate(todo):
        row = None
        try:
            frames = read_frames(Path(args.videos_dir) / rel)
            if len(frames) >= 8:
                # bbox 序列
                bbs = {}
                p = Path(args.feat_dir) / rel.replace(".mp4", ".npz")
                if p.exists():
                    z = np.load(p, allow_pickle=True)
                    for g in json.loads(str(z["geo"])):
                        bbs[int(g["frame"])] = g["bbox"]
                g_ims, c_ims = [], []
                for j, (fi, fr) in enumerate(frames):
                    H, W = fr.shape[:2]
                    s = 336 / max(H, W)
                    g_ims.append(Image.fromarray(cv2.cvtColor(cv2.resize(fr, (int(W*s), int(H*s))), cv2.COLOR_BGR2RGB)))
                    bb = bbs.get(j) or bbs.get(min(bbs, key=lambda kk: abs(kk - j)) if bbs else None)
                    if bb:
                        x0, y0, x1, y1 = [int(v) for v in bb]
                        crop = fr[max(0,y0):min(H,y1), max(0,x0):min(W,x1)]
                        if crop.size > 900:
                            c_ims.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                        else:
                            c_ims.append(None)
                    else:
                        c_ims.append(None)
                Eg = embed(g_ims)
                gs = (Eg[1:] @ Eg[0]).tolist()
                ft = curve_feats([1.0] + gs, "g")
                cc = [im for im in c_ims if im is not None]
                if len(cc) >= 8 and c_ims[0] is not None:
                    Ec = embed(cc)
                    cs = (Ec[1:] @ Ec[0]).tolist()
                    ft.update(curve_feats([1.0] + cs, "c"))
                else:
                    ft.update({f"c_{k2}": np.nan for k2 in ("end","slope","maxjump","jumpt","auc","curv")})
                row = [f"{ft[c]:.5g}" for c in COLS]
        except Exception as e:
            if k < 3:
                print("ERR", rel, repr(e)[:100], flush=True)
        w.writerow([rel] + (row if row else ["nan"] * len(COLS)))
        if (k + 1) % 200 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    f.close()
    print("E22_DONE", flush=True)


if __name__ == "__main__":
    main()
