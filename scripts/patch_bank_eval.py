#!/usr/bin/env python
"""单类 patch 异常检测——建库与打分(机制验证,语料 919 子集)。

协议 v2(2026-07-30 修订,依据业界调研 + 用户要求,先登记后评估):
  划分:语料 good 455 按「款」分层确定性划分(seed=0),约 300 建库 / 155 评估,
        视频级不相交,每款在两侧都有覆盖(防止某款缺席导致整款误报);bad 464 全评。
  结构异常分支(AnomalyDINO/PatchCore 范式):
    patch 异常 = 1 - max cos sim to bank(1-NN);
    帧分 = 前景 patch 异常 top-1% / top-5% 均值(AnomalyDINO 即 top-1% 均值口径);
    视频分 = 帧分 max / p75 / mean(p75 = 持续性口径,防姿态转换/遮挡单帧假警)。
  逻辑异常分支(CSAD patch-histogram 简化版,靶多余肢体/部件占比失衡):
    bank patch 球面 k-means(k=64)-> 每检出帧前景 patch 的聚类占比直方图
    -> 帧分 = 到 bank 帧直方图集的 1-NN L1 距离 -> 视频分同上三口径。
  主指标:bad vs eval-good AUC(fg 口径);bg 口径仅对照。
晋级关:AUC >= 0.70 且 fg > bg -> 允许迁移 prod500(库/直方图集改用 prod good 重建,
阈值走分位协议,n=27 上不调任何超参)。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
BANK_FRAC = 300 / 455
PATCHES_PER_VIDEO = 1300
TOPQ = {"top1": 0.01, "top5": 0.05}
KMEANS_K = 64


def stem_seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def load_manifest(p: Path) -> pd.DataFrame:
    rows = [l.split("\t") for l in p.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows, columns=["rel", "label"])
    df["style"] = df.rel.str.split("/").str[0]
    return df


def split_goods(goods: pd.DataFrame):
    """款分层确定性划分:每款按 BANK_FRAC 进库,余下评估。"""
    bank, ev = [], []
    for style, grp in goods.groupby("style"):
        rels = sorted(grp.rel.tolist())
        rng = np.random.default_rng(stem_seed("split:" + style))
        rng.shuffle(rels)
        nb = int(round(len(rels) * BANK_FRAC))
        bank += rels[:nb]
        ev += rels[nb:]
        print(f"  {style}: bank {nb} / eval {len(rels) - nb}")
    return bank, ev


def video_patches(npz_path: Path):
    z = np.load(npz_path)
    det = z["det"]
    if not det.any():
        return [], []
    idx = np.where(det)[0]
    return [z["feat"][i] for i in idx], [z["fg"][i] for i in idx]


def build_bank(feat_dir: Path, rels: list[str], device) -> torch.Tensor:
    chunks = []
    for rel in rels:
        p = feat_dir / rel.replace(".mp4", ".npz")
        if not p.exists():
            continue
        feats, fgs = video_patches(p)
        if not feats:
            continue
        allp = np.concatenate([f[m] for f, m in zip(feats, fgs)], 0)
        rng = np.random.default_rng(stem_seed(rel))
        if len(allp) > PATCHES_PER_VIDEO:
            allp = allp[rng.choice(len(allp), PATCHES_PER_VIDEO, replace=False)]
        chunks.append(allp)
    bank = torch.from_numpy(np.concatenate(chunks, 0)).to(device, torch.float16)
    bank = torch.nn.functional.normalize(bank.float(), dim=-1).half()
    print(f"bank: {bank.shape[0]} patches from {len(chunks)} videos", flush=True)
    return bank


@torch.inference_mode()
def spherical_kmeans(x: torch.Tensor, k: int = KMEANS_K, iters: int = 25,
                     seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k]].float()
    c = torch.nn.functional.normalize(c, dim=-1)
    for _ in range(iters):
        a = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        for j in range(0, x.shape[0], 262144):
            a[j:j + 262144] = (x[j:j + 262144].float() @ c.T).argmax(1)
        for ki in range(k):
            sel = x[a == ki].float()
            if len(sel):
                c[ki] = torch.nn.functional.normalize(sel.mean(0), dim=-1)
    return c.half()


@torch.inference_mode()
def assign(x: torch.Tensor, cents: torch.Tensor) -> torch.Tensor:
    return (x.float() @ cents.float().T).argmax(1)


@torch.inference_mode()
def patch_dists(x: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    best = torch.full((x.shape[0],), -1.0, device=x.device)
    for j in range(0, bank.shape[0], 65536):
        s = (x @ bank[j:j + 65536].T).float().amax(dim=1)
        best = torch.maximum(best, s)
    return 1.0 - best


def frame_score(d: np.ndarray, q: float) -> float:
    if d.size == 0:
        return float("nan")
    k = max(1, int(round(q * d.size)))
    return float(np.sort(d)[-k:].mean())


def frame_hist(a: torch.Tensor) -> np.ndarray:
    h = np.bincount(a.cpu().numpy(), minlength=KMEANS_K).astype(np.float64)
    return h / max(1, h.sum())


def video_frames_tensors(npz_path: Path, device):
    """逐检出帧返回 (x_norm, fg) 张量。"""
    feats, fgs = video_patches(npz_path)
    out = []
    for f, m in zip(feats, fgs):
        x = torch.from_numpy(f).to(device, torch.float16)
        x = torch.nn.functional.normalize(x.float(), dim=-1).half()
        out.append((x, torch.from_numpy(m).to(device)))
    return out


def agg(vals: list[float], prefix: str, out: dict):
    v = np.array(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return
    out[f"{prefix}_max"] = float(v.max())
    out[f"{prefix}_p75"] = float(np.percentile(v, 75))
    out[f"{prefix}_mean"] = float(v.mean())


def score_video(npz_path: Path, bank, cents, bank_hists, device) -> dict:
    frames = video_frames_tensors(npz_path, device)
    if not frames:
        return {}
    pf = {f"{n}_{s}": [] for n in TOPQ for s in ("fg", "all")}
    hist_d = []
    for x, m in frames:
        d = patch_dists(x, bank).cpu().numpy()
        mn = m.cpu().numpy()
        for n, q in TOPQ.items():
            pf[f"{n}_fg"].append(frame_score(d[mn], q))
            pf[f"{n}_all"].append(frame_score(d, q))
        if mn.any() and bank_hists is not None:
            h = frame_hist(assign(x[m], cents))
            hist_d.append(float(np.abs(bank_hists - h).sum(1).min()))
    out = {}
    for key, vals in pf.items():
        agg(vals, f"s_{key}", out)
    if hist_d:
        agg(hist_d, "s_hist", out)
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/prod500/mech_subset.tsv"))
    ap.add_argument("--feat-dir", default=str(ROOT / "data/corpus_patch_feat"))
    ap.add_argument("--out", default=str(ROOT / "data/prod500/patch_bank_scores.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = Path(args.feat_dir)
    man = load_manifest(Path(args.manifest))
    goods = man[man.label == "good"]
    print("== 款分层划分 ==")
    bank_rels, eval_rels = split_goods(goods)
    bads = man[man.label != "good"].rel.tolist()
    print(f"total: bank {len(bank_rels)} / eval-good {len(eval_rels)} / bad {len(bads)}")

    bank = build_bank(feat_dir, bank_rels, device)
    print("kmeans...", flush=True)
    cents = spherical_kmeans(bank)
    bank_hists = []
    for rel in bank_rels:
        p = feat_dir / rel.replace(".mp4", ".npz")
        if not p.exists():
            continue
        for x, m in video_frames_tensors(p, device):
            if m.any():
                bank_hists.append(frame_hist(assign(x[m], cents)))
    bank_hists = np.stack(bank_hists)
    print(f"bank hists: {bank_hists.shape}", flush=True)

    rows = []
    for group, rels in (("eval_good", eval_rels), ("bad", bads), ("bank_good", bank_rels)):
        for i, rel in enumerate(rels):
            p = feat_dir / rel.replace(".mp4", ".npz")
            if not p.exists():
                continue
            rec = {"rel": rel, "style": rel.split("/")[0], "group": group}
            rec.update(score_video(p, bank, cents, bank_hists, device))
            rows.append(rec)
            if (i + 1) % 50 == 0:
                print(f"  {group} {i+1}/{len(rels)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"写出 {args.out} ({len(df)} rows)")

    print("\n== AUC: bad vs eval_good ==")
    sc = [c for c in df.columns if c.startswith("s_")]
    pos = df[df.group == "bad"]
    neg = df[df.group == "eval_good"]
    for a, c in sorted(((auc(pos[c].values, neg[c].values), c) for c in sc),
                       reverse=True):
        print(f"  {c:22s} AUC={a:.3f}")


if __name__ == "__main__":
    main()
