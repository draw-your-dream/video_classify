#!/usr/bin/env python3
"""轻量运动电池(新1233可移植版):直读 mp4,OWLv2 定角色框 → e52 帧差 + e58 光流特征。

阶段A(GPU):每视频取 5 关键帧跑 OWLv2,得角色 bbox 轨迹 → boxes.jsonl;
阶段B(CPU多进程):全帧率 128px 帧差电池(全帧+角色区双通道,e52 移植)
                + 角色区 Farneback 光流节律/重力/过零(e58 移植)。
输出 out/motion_feats.csv。特征提取不读任何标签。

用法: python motion_battery.py --videos videos --manifest all1233.csv --out out/motion_feats.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
import cv2
import numpy as np

cv2.setNumThreads(1)

QUERIES = ["a mushroom plush toy", "a small cartoon mushroom character",
           "a cute plush toy figurine", "a small stuffed toy"]
BASE = "still_frac spike_ratio jump_cnt bimod asym acf1 hf_ratio".split()
FLOWK = "rhy_pow rhy_peak micro_rhy grav_r2 fallrise vzc vmag".split()
COLS = ["filename"] + BASE + ["c_" + k for k in BASE] + FLOWK + ["n_fr"]


def series_feats(d):
    d = np.asarray(d, float)
    if len(d) < 30:
        return {}
    med = float(np.median(d[d > 0])) if (d > 0).any() else 1e-6
    out = {}
    out["still_frac"] = float((d < 0.15 * med).mean())
    out["spike_ratio"] = float(np.percentile(d, 99) / max(1e-9, np.percentile(d, 50)))
    out["jump_cnt"] = float((d > 4 * med).sum())
    h, _ = np.histogram(np.log(d + 1e-6), bins=24)
    h = h / max(1, h.sum())
    pk = [i for i in range(1, 23) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.03]
    bim = 0.0
    if len(pk) >= 2:
        bim = float(min(h[pk[0]], h[pk[-1]]) - h[pk[0]:pk[-1] + 1].min())
    out["bimod"] = bim
    s = np.convolve(d, np.ones(5) / 5, mode="same")
    pk2 = [i for i in range(3, len(s) - 3)
           if s[i] == s[max(0, i - 3):i + 4].max() and s[i] > 2 * med]
    asyms = []
    for i in pk2[:8]:
        l = r = 0
        while i - l - 1 >= 0 and s[i - l - 1] < s[i - l]:
            l += 1
        while i + r + 1 < len(s) and s[i + r + 1] < s[i + r]:
            r += 1
        if l + r > 2:
            asyms.append((r - l) / (l + r))
    out["asym"] = float(np.mean(np.abs(asyms))) if asyms else 0.0
    z = d - d.mean()
    ac = np.correlate(z, z, "full")[len(d) - 1:]
    out["acf1"] = float(ac[1] / max(1e-9, ac[0]))
    f = np.abs(np.fft.rfft(z)) ** 2
    out["hf_ratio"] = float(f[len(f) // 2:].sum() / max(1e-9, f[1:].sum()))
    return out


def flow_feats(vp, fr, B, W0, H0):
    """角色区 Farneback 光流:节律/重力弹道/过零(e58 移植)。"""
    cap = cv2.VideoCapture(vp)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    prev = None
    vy, vm = [], []
    k = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        b = [float(np.interp(k, fr, B[:, j])) for j in range(4)]
        x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
        x1, y1 = min(W0, int(b[2])), min(H0, int(b[3]))
        if x1 - x0 < 16 or y1 - y0 < 16:
            prev = None
            k += 1
            continue
        g = cv2.cvtColor(cv2.resize(im[y0:y1, x0:x1], (96, 96)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 11, 2, 5, 1.1, 0)
            vy.append(float(fl[..., 1].mean()))
            vm.append(float(np.hypot(fl[..., 0], fl[..., 1]).mean()))
        prev = g
        k += 1
    cap.release()
    vy = np.asarray(vy, float)
    vm = np.asarray(vm, float)
    out = {}
    if len(vy) < 40:
        return out
    n = len(vy)
    z = vy - vy.mean()
    f = np.abs(np.fft.rfft(z)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= 0.5) & (freqs <= 4.0)
    out["rhy_pow"] = float(f[band].sum() / max(1e-9, f[1:].sum()))
    ac = np.correlate(z, z, "full")[n - 1:]
    ac /= max(1e-9, ac[0])
    l0, l1 = int(0.25 * fps), min(n - 1, int(2.0 * fps))
    out["rhy_peak"] = float(ac[l0:l1].max()) if l1 > l0 else 0.0
    lo = vm < np.median(vm)
    if lo.sum() >= 32:
        z2 = vy[lo] - vy[lo].mean()
        f2 = np.abs(np.fft.rfft(z2)) ** 2
        fq2 = np.fft.rfftfreq(len(z2), d=1.0 / fps)
        b2 = (fq2 >= 0.5) & (fq2 <= 4.0)
        out["micro_rhy"] = float(f2[b2].sum() / max(1e-9, f2[1:].sum()))
    cy = np.cumsum(vy)
    Wn = max(8, int(0.5 * fps))
    r2s = []
    for s0 in range(0, n - Wn, Wn // 2):
        seg = cy[s0:s0 + Wn]
        if vm[s0:s0 + Wn].mean() < np.percentile(vm, 40):
            continue
        t = np.arange(len(seg))
        co = np.polyfit(t, seg, 2)
        pred = np.polyval(co, t)
        ss = ((seg - seg.mean()) ** 2).sum()
        r2s.append(1 - ((seg - pred) ** 2).sum() / max(1e-9, ss))
    out["grav_r2"] = float(np.median(r2s)) if r2s else np.nan
    up = vy[vy < -1e-4]
    dn = vy[vy > 1e-4]
    if len(up) >= 5 and len(dn) >= 5:
        out["fallrise"] = float(np.abs(dn).mean() / max(1e-6, np.abs(up).mean()))
    out["vzc"] = float((np.diff(np.sign(z)) != 0).mean())
    out["vmag"] = float(np.median(vm))
    return out


def diff_feats(vp, fr, B, W0, H0):
    cap = cv2.VideoCapture(vp)
    prev = None
    dfull, dchar = [], []
    k = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(im, (160, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            dfull.append(float(np.abs(g - prev).mean()))
            if B is not None:
                b = [float(np.interp(k, fr, B[:, j])) for j in range(4)]
                x0 = max(0, min(155, int(b[0] / W0 * 160)))
                y0 = max(0, min(91, int(b[1] / H0 * 96)))
                x1 = min(160, max(x0 + 4, int(b[2] / W0 * 160)))
                y1 = min(96, max(y0 + 4, int(b[3] / H0 * 96)))
                dchar.append(float(np.abs(g[y0:y1, x0:x1] - prev[y0:y1, x0:x1]).mean()))
        prev = g
        k += 1
    cap.release()
    row = series_feats(dfull)
    row.update({"c_" + k2: v for k2, v in series_feats(dchar).items()})
    row["n_fr"] = len(dfull)
    return row


def one(args):
    fn, vdir, box = args
    row = {"filename": fn}
    vp = os.path.join(vdir, fn)
    if not os.path.exists(vp) or box is None:
        return row
    try:
        fr = np.array(box["frames"], float)
        B = np.array(box["boxes"], float)
        W0, H0 = box["size"]
        row.update(diff_feats(vp, fr, B, W0, H0))
        row.update(flow_feats(vp, fr, B, W0, H0))
    except Exception:
        pass
    return row


# ---------- 阶段A:OWLv2 定框 ----------
def stage_boxes(videos, manifest, out_boxes):
    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    pp = (getattr(proc, "post_process_object_detection", None)
          or getattr(proc.image_processor, "post_process_object_detection", None)
          or getattr(proc, "post_process_grounded_object_detection"))
    done = set()
    if os.path.exists(out_boxes):
        for l in open(out_boxes):
            try:
                done.add(json.loads(l)["filename"])
            except Exception:
                pass
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    f = open(out_boxes, "a")
    import time
    t0, n = time.time(), 0
    for r in rows:
        fn = r["filename"]
        if fn in done:
            continue
        vp = os.path.join(videos, fn)
        rec = {"filename": fn}
        try:
            cap = cv2.VideoCapture(vp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idxs = [int(x) for x in np.linspace(0, max(0, total - 2), 5)]
            frames, keep = [], []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, im = cap.read()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                    keep.append(i)
            cap.release()
            if frames:
                W0, H0 = frames[0].size
                boxes = []
                with torch.no_grad():
                    inp = proc(text=[QUERIES] * len(frames), images=frames,
                               return_tensors="pt").to(DEVICE)
                    out = model(**inp)
                    ts = torch.tensor([[H0, W0]] * len(frames)).to(DEVICE)
                    res = pp(out, threshold=0.04, target_sizes=ts)
                for rr in res:
                    if len(rr["scores"]):
                        j = int(rr["scores"].argmax())
                        boxes.append([float(x) for x in rr["boxes"][j].tolist()])
                    else:
                        boxes.append(None)
                fr_k = [k for k, b in zip(keep, boxes) if b]
                bx = [b for b in boxes if b]
                if bx:
                    rec.update(frames=fr_k, boxes=bx, size=[W0, H0])
        except Exception as e:
            rec["error"] = repr(e)[:120]
        f.write(json.dumps(rec) + "\n")
        f.flush()
        n += 1
        if n % 50 == 0:
            print(f"[boxes {n}] {(time.time()-t0)/n:.2f}s/条", flush=True)
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", default=None)
    ap.add_argument("--procs", type=int, default=48)
    args = ap.parse_args()
    boxes_path = args.boxes or (args.out + ".boxes.jsonl")
    stage_boxes(args.videos, args.manifest, boxes_path)
    print("STAGE_A_DONE", flush=True)

    boxmap = {}
    for l in open(boxes_path):
        d = json.loads(l)
        if "boxes" in d:
            boxmap[d["filename"]] = d
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8-sig")))
    jobs = [(r["filename"], args.videos, boxmap.get(r["filename"])) for r in rows]
    import multiprocessing as mp
    import time
    t0 = time.time()
    with mp.Pool(args.procs) as pool, open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            w.writerow(row)
            if (i + 1) % 100 == 0:
                f.flush()
                print(f"[feats {i+1}/{len(jobs)}] {(time.time()-t0)/(i+1):.2f}s/条", flush=True)
    print("MOTION_BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
