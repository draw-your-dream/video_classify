#!/usr/bin/env python
"""僵硬特征电池(2026-08-06 预注册):四族"运动-形变耦合"特征,全语料。
F1 刚体拟合残差(RAFT 16对,角色框内相似变换拟合): rr_mean rr_p25 rr_lowfrac quad_dis
F2 事件对齐形变(bbox面积/长宽比变化 × 运动能量耦合): def_corr_area def_corr_asp def_amp_ratio
F3 节奏(角色区逐帧差能量全帧率序列): onset_rise prof_peak hf_static en_acf1 en_cv
F4 跟随滞后(上1/3条带 vs 下2/3条带能量互相关): lag_ul lag_gain
另: m_mean(运动量) n_ok。共15列。"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/root/mech")
COLS = ("rr_mean rr_p25 rr_lowfrac quad_dis def_corr_area def_corr_asp def_amp_ratio "
        "onset_rise prof_peak hf_static en_acf1 en_cv lag_ul lag_gain m_mean").split()
N_ANCH = 16


def read_all(vp, small=256, big=384):
    cap = cv2.VideoCapture(str(vp))
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if tot <= 8:
        cap.release(); return None
    idxs = [int(round(i * (tot - 1) / (N_ANCH - 1))) for i in range(N_ANCH)]
    want = set(idxs)
    smalls, bigs = [], {}
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        H, W = fr.shape[:2]
        s = small / max(H, W)
        smalls.append(cv2.cvtColor(cv2.resize(fr, (int(W*s), int(H*s))), cv2.COLOR_BGR2GRAY))
        if k in want:
            b = big / max(H, W)
            bigs[k] = cv2.resize(fr, (int(W*b), int(H*b)))
        k += 1
    cap.release()
    return smalls, [bigs[i] for i in sorted(bigs)], idxs, tot, max(H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--feat-dir", default=str(ROOT / "data/sam3_feat"))
    ap.add_argument("--manifest", default=str(ROOT / "manifest_all.tsv"))
    ap.add_argument("--out", default=str(ROOT / "data/rigid_feats.csv"))
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

    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to("cuda").eval()

    @torch.inference_mode()
    def flow_pairs(bigs):
        t = torch.stack([torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                         for b in bigs]).float().to("cuda") / 255.0
        t = t * 2 - 1
        H, W = t.shape[-2:]
        H8, W8 = H - H % 8, W - W % 8
        t = t[..., :H8, :W8]
        fl = raft(t[:-1], t[1:])[-1]
        return fl.cpu().numpy()  # (P,2,h,w)

    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    for k, rel in enumerate(todo):
        row = None
        try:
            rd = read_all(Path(args.videos_dir) / rel)
            bbs = {}
            p = Path(args.feat_dir) / rel.replace(".mp4", ".npz")
            if p.exists():
                z = np.load(p, allow_pickle=True)
                for g in json.loads(str(z["geo"])):
                    bbs[int(g["frame"])] = g["bbox"]
            if rd is not None and len(bbs) >= 6:
                smalls, bigs, idxs, tot, M0 = rd
                Hs, Ws = smalls[0].shape[:2]
                # bbox 锚点(384系)→ 逐帧线性插值(256系)
                def bb_at(j):  # j: 抽帧序号 0..15 → 原始坐标
                    key = min(bbs, key=lambda kk: abs(kk - j))
                    return bbs[key]
                boxes = np.array([bb_at(j) for j in range(N_ANCH)], float)  # 原始系
                sc_small = 256.0 / M0
                sc_big = 384.0 / M0
                dense_boxes = np.zeros((len(smalls), 4))
                for d in range(4):
                    dense_boxes[:, d] = np.interp(np.arange(len(smalls)), idxs, boxes[:, d]) * sc_small
                # F3/F4: 角色区逐帧差能量 + 上下条带
                en, en_u, en_l, areas, asps = [], [], [], [], []
                for t_i in range(1, len(smalls)):
                    x0, y0, x1, y1 = [int(max(0, v)) for v in dense_boxes[t_i]]
                    x1 = min(Ws, max(x1, x0 + 4)); y1 = min(Hs, max(y1, y0 + 4))
                    a = smalls[t_i][y0:y1, x0:x1].astype(np.float32)
                    b = smalls[t_i - 1][y0:y1, x0:x1].astype(np.float32)
                    if a.shape != b.shape or a.size < 16:
                        en.append(np.nan); en_u.append(np.nan); en_l.append(np.nan)
                        areas.append(np.nan); asps.append(np.nan); continue
                    d = np.abs(a - b)
                    en.append(float(d.mean()))
                    h3 = max(1, (y1 - y0) // 3)
                    en_u.append(float(d[:h3].mean())); en_l.append(float(d[h3:].mean()))
                    areas.append(float((x1 - x0) * (y1 - y0)))
                    asps.append(float((y1 - y0) / max(1, x1 - x0)))
                en = np.array(en); en_u = np.array(en_u); en_l = np.array(en_l)
                areas = np.array(areas); asps = np.array(asps)
                ok = np.isfinite(en)
                if ok.sum() < 20:
                    raise RuntimeError("too few dense frames")
                en, en_u, en_l, areas, asps = en[ok], en_u[ok], en_l[ok], areas[ok], asps[ok]
                # F1: RAFT 相似变换残差
                fl = flow_pairs(bigs)
                rrs, qds, mags = [], [], []
                for pi in range(fl.shape[0]):
                    j = pi  # 对 idxs[pi]->idxs[pi+1]
                    x0, y0, x1, y1 = [int(v * sc_big) for v in bb_at(j)]
                    fh, fw = fl.shape[-2:]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(fw, x1), min(fh, y1)
                    if x1 - x0 < 16 or y1 - y0 < 16:
                        continue
                    u = fl[pi, 0, y0:y1, x0:x1]; v = fl[pi, 1, y0:y1, x0:x1]
                    mag = float(np.sqrt(u**2 + v**2).mean())
                    mags.append(mag)
                    if mag < 0.15:
                        continue  # 静止对不计残差
                    ys, xs = np.mgrid[y0:y1:8, x0:x1:8]
                    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
                    du = u[::8, ::8].ravel(); dv = v[::8, ::8].ravel()
                    dst = pts + np.stack([du, dv], 1)
                    M, _ = cv2.estimateAffinePartial2D(pts, dst, method=cv2.RANSAC,
                                                       ransacReprojThreshold=1.5)
                    if M is None:
                        continue
                    pred = pts @ M[:, :2].T + M[:, 2] - pts
                    res = np.sqrt(((np.stack([du, dv], 1) - pred) ** 2).sum(1))
                    rrs.append(float(res.mean() / (mag + 1e-6)))
                    # 象限分歧
                    h2, w2 = (y1 - y0) // 2, (x1 - x0) // 2
                    qv = []
                    for qy in (0, 1):
                        for qx in (0, 1):
                            qu = u[qy*h2:(qy+1)*h2, qx*w2:(qx+1)*w2].mean()
                            qw = v[qy*h2:(qy+1)*h2, qx*w2:(qx+1)*w2].mean()
                            qv.append((qu, qw))
                    qv = np.array(qv)
                    qds.append(float(np.linalg.norm(qv - qv.mean(0), axis=1).mean() / (mag + 1e-6)))
                # F2: 事件对齐形变
                dA = np.abs(np.diff(areas)) / (areas[:-1] + 1e-6)
                dS = np.abs(np.diff(asps))
                e2 = en[1:]
                def corr(a, b):
                    if len(a) < 10 or a.std() < 1e-9 or b.std() < 1e-9:
                        return np.nan
                    return float(np.corrcoef(a, b)[0, 1])
                hi = e2 >= np.quantile(e2, 0.8); lo = e2 <= np.quantile(e2, 0.2)
                # F3: 节奏
                thr = np.quantile(en, 0.5)
                onsets = []
                t_i = 1
                while t_i < len(en):
                    if en[t_i] > thr * 1.5 and en[t_i - 1] <= thr:
                        j2 = t_i
                        while j2 < len(en) and en[j2] < en.max() * 0.8:
                            j2 += 1
                        onsets.append(j2 - t_i)
                        t_i = j2
                    t_i += 1
                static = en <= np.quantile(en, 0.3)
                if static.sum() >= 16:
                    seg = en[static] - en[static].mean()
                    sp = np.abs(np.fft.rfft(seg)) ** 2
                    hf = float(sp[len(sp)//2:].sum() / (sp[1:].sum() + 1e-9))
                else:
                    hf = np.nan
                # F4: 跟随滞后
                eu = en_u - en_u.mean(); el = en_l - en_l.mean()
                xc = np.correlate(eu, el, "full")
                mid = len(el) - 1
                win = 6
                seg2 = xc[mid - win: mid + win + 1]
                lag = int(np.argmax(seg2)) - win
                gain = float(seg2.max() / (abs(xc[mid]) + 1e-9))
                ft = dict(
                    rr_mean=float(np.mean(rrs)) if rrs else np.nan,
                    rr_p25=float(np.quantile(rrs, 0.25)) if len(rrs) > 3 else np.nan,
                    rr_lowfrac=float(np.mean(np.array(rrs) < 0.35)) if rrs else np.nan,
                    quad_dis=float(np.mean(qds)) if qds else np.nan,
                    def_corr_area=corr(dA, e2), def_corr_asp=corr(dS, e2),
                    def_amp_ratio=float((dA[hi].mean() + 1e-9) / (dA[lo].mean() + 1e-9)) if hi.sum() > 3 and lo.sum() > 3 else np.nan,
                    onset_rise=float(np.mean(onsets)) if onsets else np.nan,
                    prof_peak=float(np.quantile(en, 0.95) / (en.mean() + 1e-9)),
                    hf_static=hf,
                    en_acf1=corr(en[:-1], en[1:]),
                    en_cv=float(en.std() / (en.mean() + 1e-9)),
                    lag_ul=float(lag), lag_gain=gain,
                    m_mean=float(np.mean(mags)) if mags else np.nan)
                row = [f"{ft[c]:.5g}" for c in COLS]
        except Exception as e:
            if k < 5:
                print("ERR", rel, repr(e)[:110], flush=True)
        w.writerow([rel] + (row if row else ["nan"] * len(COLS)))
        if (k + 1) % 200 == 0:
            f.flush()
            print(f"[{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.2f}s/vid", flush=True)
    f.close()
    print("RIGID_DONE", flush=True)


if __name__ == "__main__":
    main()
