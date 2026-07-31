#!/usr/bin/env python
"""第0步诊断:逐簇 AUC 分解——缺陷信号集中在哪些身体区域?(2026-07-31 预注册)

64 个球面 k-means 簇 = 免费无监督部件分割,逐簇看 bad-vs-good 判别力:
  patch 异常 d = 1 - max cos to bank(与 P1 同口径),按簇归组
  视频×簇分 = 该簇 patch 异常 top-20% 均值(池化全帧,簇内尾部口径)
  簇占比 occ = 该簇 patch 数 / 前景 patch 总数(逻辑异常口径,双向)
  逐簇 AUC 只在该簇出现(>=5 patch)的视频上算,另报双侧覆盖数
产物:cluster_auc.csv + cluster_montage_parts.jpg(每簇 12 个样本 patch,
      行按 AUC 降序,供人工辨认簇=脸/菇伞/四肢/背景衣物)
判读门(预注册,先登记后评估):
  存在簇 AUC>=0.65 且蒙太奇显示其对应面部/肢体区域 -> 部件线(线A/B)立项;
  全簇 <0.60 -> DINOv2+bank 对此类缺陷不敏感,部件线改走测量/合成监督路线。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_bank_eval import (ROOT, KMEANS_K, assign, auc, build_bank,  # noqa: E402
                             load_manifest, patch_dists, split_goods,
                             spherical_kmeans, video_frames_tensors)

MIN_PATCH = 5
TOPQ = 0.2
M_PER_CLUSTER = 12
SCAN_BANK_VIDEOS = 80
GRID = 37
PATCH_PX = 14


def read_frames(path, n=16):
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


def sq_crop_arr(fr, b):
    H, W = fr.shape[:2]
    if b is None or np.isnan(np.asarray(b, dtype=float)).any():
        s = min(W, H)
        x0, y0 = (W - s) // 2, (H - s) // 2
        return fr[y0:y0 + s, x0:x0 + s]
    x0, y0, x1, y1 = b
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(max(x1 - x0, y1 - y0) * 1.25, 32)
    xa, ya = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    xb, yb = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    return fr[ya:yb, xa:xb]


def per_video_cluster(npz_path, bank, cents, device):
    """返回 ({k: top20%均值}, occ[64])。"""
    dv = {k: [] for k in range(KMEANS_K)}
    for x, m in video_frames_tensors(npz_path, device):
        if not m.any():
            continue
        xm = x[m]
        d = patch_dists(xm, bank).cpu().numpy()
        a = assign(xm, cents).cpu().numpy()
        for k in np.unique(a):
            dv[k].append(d[a == k])
    out, occ, tot = {}, np.zeros(KMEANS_K), 0
    for k in range(KMEANS_K):
        v = np.concatenate(dv[k]) if dv[k] else np.empty(0)
        occ[k] = len(v)
        tot += len(v)
        if len(v) >= MIN_PATCH:
            kk = max(1, int(round(TOPQ * len(v))))
            out[k] = float(np.sort(v)[-kk:].mean())
    return out, occ / max(1, tot)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = ROOT / "data/corpus_patch_feat"
    vid_dir = ROOT / "data/corpus_videos"
    man = load_manifest(ROOT / "data/prod500/mech_subset.tsv")
    goods = man[man.label == "good"]
    bank_rels, eval_rels = split_goods(goods)
    bads = man[man.label != "good"].rel.tolist()

    bank = build_bank(feat_dir, bank_rels, device)
    print("kmeans...", flush=True)
    cents = spherical_kmeans(bank)

    recs = []
    for group, rels in (("eval_good", eval_rels), ("bad", bads)):
        for i, rel in enumerate(rels):
            p = feat_dir / rel.replace(".mp4", ".npz")
            if not p.exists():
                continue
            sc, occ = per_video_cluster(p, bank, cents, device)
            recs.append({"rel": rel, "group": group, "sc": sc, "occ": occ})
            if (i + 1) % 100 == 0:
                print(f"  {group} {i+1}/{len(rels)}", flush=True)

    rows = []
    for k in range(KMEANS_K):
        pos = np.array([r["sc"][k] for r in recs if r["group"] == "bad" and k in r["sc"]])
        neg = np.array([r["sc"][k] for r in recs if r["group"] == "eval_good" and k in r["sc"]])
        opos = np.array([r["occ"][k] for r in recs if r["group"] == "bad"])
        oneg = np.array([r["occ"][k] for r in recs if r["group"] == "eval_good"])
        rows.append({"cluster": k, "auc_top20": auc(pos, neg),
                     "n_bad": len(pos), "n_good": len(neg),
                     "auc_occ": auc(opos, oneg),
                     "occ_bad": float(opos.mean()), "occ_good": float(oneg.mean())})
    df = pd.DataFrame(rows).sort_values("auc_top20", ascending=False)
    df.to_csv(ROOT / "data/prod500/cluster_auc.csv", index=False)
    print("\n== 逐簇 AUC(top-20% 口径,降序前 15)==")
    print(df.head(15).to_string(index=False))
    hi = df[(df.auc_top20 >= 0.65) & (df.n_bad >= 50) & (df.n_good >= 30)]
    print(f"\n达标簇(AUC>=0.65 且覆盖充分): {len(hi)} 个 -> {hi.cluster.tolist()}")

    # ---- 蒙太奇:每簇 12 个样本 patch(bank 视频),行按 AUC 降序 ----
    print("采样蒙太奇 patch ...", flush=True)
    pool = {k: [] for k in range(KMEANS_K)}
    for rel in bank_rels[:SCAN_BANK_VIDEOS]:
        p = feat_dir / rel.replace(".mp4", ".npz")
        if not p.exists():
            continue
        z = np.load(p)
        det, fg, feat = z["det"], z["fg"], z["feat"]
        for i in np.where(det)[0]:
            m = fg[i]
            if not m.any():
                continue
            x = torch.from_numpy(feat[i][m]).to(device, torch.float16)
            x = torch.nn.functional.normalize(x.float(), dim=-1).half()
            a = assign(x, cents).cpu().numpy()
            gidx = np.where(m)[0]
            for j, k in enumerate(a):
                pool[int(k)].append((rel, int(i), int(gidx[j])))

    chosen = {}
    for k in range(KMEANS_K):
        if pool[k]:
            rng = np.random.default_rng(k)
            idx = rng.choice(len(pool[k]), min(M_PER_CLUSTER, len(pool[k])),
                             replace=False)
            chosen[k] = [pool[k][i] for i in idx]

    by_rel = {}
    for k, lst in chosen.items():
        for rel, i, pidx in lst:
            by_rel.setdefault(rel, []).append((k, i, pidx))
    thumbs = {k: {} for k in range(KMEANS_K)}
    for rel, items in by_rel.items():
        frames = read_frames(vid_dir / rel)
        if not frames:
            continue
        z = np.load(feat_dir / rel.replace(".mp4", ".npz"))
        for k, i, pidx in items:
            if i >= len(frames):
                continue
            b = z["boxes"][i] if z["det"][i] else None
            crop = cv2.resize(sq_crop_arr(frames[i], b), (518, 518))
            r, c = pidx // GRID, pidx % GRID
            cy, cx = r * PATCH_PX + 7, c * PATCH_PX + 7
            y0, x0 = max(0, cy - 24), max(0, cx - 24)
            t = crop[y0:y0 + 48, x0:x0 + 48]
            if t.size:
                thumbs[k][(rel, i, pidx)] = cv2.resize(t, (64, 64))

    rows_img = []
    for _, row in df.iterrows():
        k = int(row.cluster)
        cells = list(thumbs[k].values())[:M_PER_CLUSTER]
        if not cells:
            continue
        while len(cells) < M_PER_CLUSTER:
            cells.append(np.zeros((64, 64, 3), np.uint8))
        label = np.zeros((64, 170, 3), np.uint8)
        a_str = "nan" if np.isnan(row.auc_top20) else f"{row.auc_top20:.2f}"
        cv2.putText(label, f"c{k:02d} A={a_str}", (4, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(label, f"O={row.auc_occ:.2f} n={row.n_bad}/{row.n_good}",
                    (4, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        rows_img.append(np.concatenate([label] + cells, 1))
    if rows_img:
        out = ROOT / "data/prod500/cluster_montage_parts.jpg"
        cv2.imwrite(str(out), np.concatenate(rows_img, 0),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"montage -> {out}")
    print("CLUSTER_DECOMP_DONE")


if __name__ == "__main__":
    main()
