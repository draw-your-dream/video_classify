#!/usr/bin/env python
"""E21 码流专家:H.264 压缩域信号(2026-08-04 预注册)。

ffprobe 逐帧:pict_type(I/P/B)、pkt_size;ffmpeg export_mvs 逐帧运动向量。
特征 14 列:
  isz_cv 帧大小变异 | i_frac 非首帧I帧占比(跳变) | psz_max_z P帧大小最大z分
  mv_mag_mean/max MV幅值 | mv_div MV散度(空间不一致) | mv_zero_frac 零MV占比(冻结)
  mv_mag_cv 时间变异(卡顿) | mv_ent MV方向熵 | mv_jump 相邻帧MV场突变
  sz_slope 码率趋势 | n_frames | dur | fps"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

COLS = ("isz_cv i_frac psz_max_z mv_mag_mean mv_mag_max mv_div mv_zero_frac "
        "mv_mag_cv mv_ent mv_jump sz_slope n_frames dur fps").split()


def probe_frames(vp):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_frames", "-show_entries",
                        "frame=pict_type,pkt_size", "-of", "json", str(vp)],
                       capture_output=True, text=True, timeout=60)
    d = json.loads(r.stdout or "{}")
    return d.get("frames", [])


def extract_mvs(vp, max_frames=64):
    """ffmpeg codecview 太重;用 extract_mvs 风格:mpegflow 不在,改用
    ffmpeg -flags2 +export_mvs + vf codecview 需解码。轻量替代:
    用 ffprobe motion vectors 不可得时返回 None,由上层退化为仅码流统计。"""
    try:
        import av
    except ImportError:
        return None
    out = []
    try:
        with av.open(str(vp)) as c:
            stream = c.streams.video[0]
            stream.codec_context.options = {"flags2": "+export_mvs"}
            for i, frame in enumerate(c.decode(stream)):
                if i >= max_frames:
                    break
                sd = frame.side_data.get("MOTION_VECTORS") if hasattr(frame.side_data, "get") else None
                if sd is None:
                    try:
                        sd = frame.side_data[av.sidedata.sidedata.Type.MOTION_VECTORS]
                    except Exception:
                        sd = None
                if sd is None:
                    out.append(None)
                    continue
                arr = sd.to_ndarray()
                if arr.size == 0:
                    out.append(np.zeros((0, 2)))
                    continue
                mx = (arr["dst_x"].astype(float) - arr["src_x"].astype(float))
                my = (arr["dst_y"].astype(float) - arr["src_y"].astype(float))
                out.append(np.stack([mx, my], 1))
    except Exception:
        return None
    return out


def video_feats(vp):
    frames = probe_frames(vp)
    if len(frames) < 8:
        return None
    sizes = np.array([float(f.get("pkt_size", 0)) for f in frames])
    types = [f.get("pict_type", "?") for f in frames]
    n = len(frames)
    ft = {c: np.nan for c in COLS}
    ft["n_frames"] = float(n)
    ft["isz_cv"] = float(sizes.std() / (sizes.mean() + 1e-6))
    ft["i_frac"] = float(sum(1 for t in types[1:] if t == "I") / max(1, n - 1))
    ps = sizes[[i for i, t in enumerate(types) if t != "I"]]
    if len(ps) > 4:
        ft["psz_max_z"] = float((ps.max() - ps.mean()) / (ps.std() + 1e-6))
    t = np.arange(n)
    ft["sz_slope"] = float(np.polyfit(t, sizes / (sizes.mean() + 1e-6), 1)[0])
    mvs = extract_mvs(vp)
    if mvs:
        mags, divs, zeros, ents = [], [], [], []
        prev_mean = None
        jumps = []
        for m in mvs:
            if m is None or len(m) < 8:
                continue
            mag = np.sqrt((m ** 2).sum(1))
            mags.append(mag.mean())
            zeros.append(float((mag < 0.5).mean()))
            divs.append(float(mag.std()))
            ang = np.arctan2(m[:, 1], m[:, 0])
            hist, _ = np.histogram(ang, bins=8, range=(-np.pi, np.pi))
            p = hist / max(1, hist.sum())
            ents.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
            mean_v = m.mean(0)
            if prev_mean is not None:
                jumps.append(float(np.linalg.norm(mean_v - prev_mean)))
            prev_mean = mean_v
        if mags:
            mags = np.array(mags)
            ft["mv_mag_mean"] = float(mags.mean())
            ft["mv_mag_max"] = float(mags.max())
            ft["mv_div"] = float(np.mean(divs))
            ft["mv_zero_frac"] = float(np.mean(zeros))
            ft["mv_mag_cv"] = float(mags.std() / (mags.mean() + 1e-6))
            ft["mv_ent"] = float(np.mean(ents))
            if jumps:
                ft["mv_jump"] = float(np.max(jumps))
    return ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/root/mech/data/corpus_videos")
    ap.add_argument("--manifest", default="/root/mech/manifest_all.tsv")
    ap.add_argument("--out", default="/root/mech/data/e21_bitstream.csv")
    ap.add_argument("--workers", type=int, default=16)
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

    def one(rel):
        try:
            ft = video_feats(Path(args.videos_dir) / rel)
        except Exception:
            ft = None
        if ft is None:
            return rel, None
        return rel, [f"{ft[c]:.5g}" for c in COLS]

    import time
    from multiprocessing import Pool
    f = open(out_p, "a", newline=""); w = csv.writer(f)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (rel, row) in enumerate(pool.imap_unordered(one, todo, chunksize=8)):
            w.writerow([rel] + (row if row else ["nan"] * len(COLS)))
            if (i + 1) % 200 == 0:
                f.flush()
                print(f"[{i+1}/{len(todo)}] {(time.time()-t0)/(i+1):.2f}s/vid", flush=True)
    f.close()
    print("E21_DONE", flush=True)


if __name__ == "__main__":
    main()
