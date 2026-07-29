#!/usr/bin/env python
"""F1-F5 因子提取(FACTOR_PREREG.md,2026-07-28 预注册)。

单遍流水,每视频:
  16 帧均匀采样 -> GroundingDINO 角色框(与 c_first_last 家族同 prompt/阈值)
  -> 整体/菌盖(框上40%)/脸部(框中带30-70%)三种裁剪 -> SigLIP2 嵌入
  F1 cap_drift   : 菌盖首末帧 SigLIP2 余弦 + HSV 直方图 Bhattacharyya 距离
  F2 anchor_min  : min over t of cos(whole_F0, whole_Ft)
  F3 face_drift  : 脸部首末帧 SigLIP2 余弦
  F4 size_trend  : 框高时间序列,背景 ORB 相似变换补偿镜头推拉后取对数极差/斜率
  F5 nonchar_motion : 镜头补偿后角色框外残余光流能量(P95/mean)

输出 JSONL(逐视频落盘,可断点续跑)。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

PROMPT = "a mushroom character."
CONF_THRESH = 0.3
N_FRAMES = 16
FLOW_W, FLOW_H = 480, 270


def read_frames(path: Path, n: int = N_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    want = sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))
    frames, idx = [], 0
    wi = 0
    while wi < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if idx == want[wi]:
            frames.append(fr)  # BGR
            wi += 1
        idx += 1
    cap.release()
    return frames


class Models:
    def __init__(self, cache_dir: Path):
        from transformers import (AutoImageProcessor, AutoModel,
                                  AutoModelForZeroShotObjectDetection,
                                  AutoProcessor)
        cd = str(cache_dir)
        self.dino_proc = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base", cache_dir=cd)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base", cache_dir=cd,
            torch_dtype=torch.float16).to("cuda").eval()
        self.sig_proc = AutoImageProcessor.from_pretrained(
            "google/siglip2-base-patch16-224", cache_dir=cd)
        self.sig = AutoModel.from_pretrained(
            "google/siglip2-base-patch16-224", cache_dir=cd,
            torch_dtype=torch.float16).to("cuda").eval()

    @torch.inference_mode()
    def detect(self, img: Image.Image) -> tuple[np.ndarray, float] | None:
        inputs = self.dino_proc(images=img, text=PROMPT, return_tensors="pt").to("cuda")
        out = self.dino(**inputs)
        res = self.dino_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=CONF_THRESH, text_threshold=0.25,
            target_sizes=[img.size[::-1]])
        if not res or len(res[0]["boxes"]) == 0:
            return None
        top = int(torch.argmax(res[0]["scores"]).item())
        box = res[0]["boxes"][top].float().cpu().numpy()
        return box, float(res[0]["scores"][top])

    @torch.inference_mode()
    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        feats = []
        for i in range(0, len(crops), 32):
            batch = self.sig_proc(images=crops[i:i + 32], return_tensors="pt")
            pv = batch["pixel_values"].to("cuda", torch.float16)
            f = self.sig.get_image_features(pixel_values=pv)
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, 0)


def crop_regions(frame_bgr: np.ndarray, box: np.ndarray) -> dict[str, np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - 10); y1 = max(0, int(y1) - 10)
    x2 = min(w, int(x2) + 10); y2 = min(h, int(y2) + 10)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return {}
    whole = frame_bgr[y1:y2, x1:x2]
    bh = y2 - y1
    cap_ = frame_bgr[y1:y1 + max(8, int(bh * 0.40)), x1:x2]
    fy1 = y1 + int(bh * 0.30); fy2 = y1 + int(bh * 0.70)
    face = frame_bgr[fy1:max(fy1 + 8, fy2), x1:x2]
    return {"whole": whole, "cap": cap_, "face": face}


def hsv_hist(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def cam_scale_and_residual(prev_bgr, cur_bgr, prev_box, cur_box):
    """背景 ORB 相似变换(补偿镜头) + 补偿后角色框外残余光流。

    返回 (scale, n_matches, res_mean, res_p95);失败处为 NaN。
    """
    sx, sy = FLOW_W / prev_bgr.shape[1], FLOW_H / prev_bgr.shape[0]
    g0 = cv2.cvtColor(cv2.resize(prev_bgr, (FLOW_W, FLOW_H)), cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cv2.resize(cur_bgr, (FLOW_W, FLOW_H)), cv2.COLOR_BGR2GRAY)

    def char_mask(box):  # True = 背景
        m = np.ones((FLOW_H, FLOW_W), bool)
        if box is not None:
            x1, y1, x2, y2 = box
            cxa, cya = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
            hw, hh = (x2 - x1) / 2 * sx * 1.15, (y2 - y1) / 2 * sy * 1.15
            xa, xb = int(max(0, cxa - hw)), int(min(FLOW_W, cxa + hw))
            ya, yb = int(max(0, cya - hh)), int(min(FLOW_H, cya + hh))
            m[ya:yb, xa:xb] = False
        return m

    bg = char_mask(prev_box) & char_mask(cur_box)

    orb = cv2.ORB_create(1500)
    mask0 = (char_mask(prev_box) * 255).astype(np.uint8)
    mask1 = (char_mask(cur_box) * 255).astype(np.uint8)
    k0, d0 = orb.detectAndCompute(g0, mask0)
    k1, d1 = orb.detectAndCompute(g1, mask1)
    scale, n_match, M = float("nan"), 0, None
    if d0 is not None and d1 is not None and len(k0) >= 8 and len(k1) >= 8:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d0, d1)
        if len(matches) >= 8:
            src = np.float32([k0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([k1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, inl = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=3.0)
            if M is not None:
                n_match = int(inl.sum()) if inl is not None else len(matches)
                scale = float(np.sqrt(max(1e-12, np.linalg.det(M[:, :2]))))

    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    xx, yy = np.meshgrid(np.arange(FLOW_W, dtype=np.float32),
                         np.arange(FLOW_H, dtype=np.float32))
    if M is not None:
        px = M[0, 0] * xx + M[0, 1] * yy + M[0, 2] - xx
        py = M[1, 0] * xx + M[1, 1] * yy + M[1, 2] - yy
    else:
        px = np.full_like(xx, np.median(flow[..., 0]))
        py = np.full_like(yy, np.median(flow[..., 1]))
    res = np.sqrt((flow[..., 0] - px) ** 2 + (flow[..., 1] - py) ** 2)
    diag = math.hypot(FLOW_W, FLOW_H)
    rb = res[bg] / diag
    if rb.size < 100:
        return scale, n_match, float("nan"), float("nan")
    return scale, n_match, float(rb.mean()), float(np.percentile(rb, 95))


def process_video(models: Models, path: Path) -> dict:
    rec: dict = {"stem": path.stem}
    frames = read_frames(path)
    rec["n_frames_read"] = len(frames)
    if len(frames) < 4:
        rec["error"] = "too_few_frames"
        return rec

    boxes, scores = [], []
    for fr in frames:
        det = models.detect(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        boxes.append(None if det is None else det[0])
        scores.append(float("nan") if det is None else det[1])
    det_idx = [i for i, b in enumerate(boxes) if b is not None]
    rec["det_rate"] = len(det_idx) / len(frames)
    rec["det_score_mean"] = float(np.nanmean(scores)) if det_idx else float("nan")

    if len(det_idx) >= 2:
        crops = {k: {} for k in ("whole", "cap", "face")}
        hists = {}
        for i in det_idx:
            regs = crop_regions(frames[i], boxes[i])
            if not regs:
                continue
            for k, img in regs.items():
                crops[k][i] = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            hists[i] = hsv_hist(regs["cap"])
        ids = sorted(crops["whole"].keys())
        if len(ids) >= 2:
            flat, index = [], []
            for k in ("whole", "cap", "face"):
                for i in ids:
                    flat.append(crops[k][i]); index.append((k, i))
            emb = models.embed(flat)
            E = {k: {} for k in ("whole", "cap", "face")}
            for (k, i), e in zip(index, emb):
                E[k][i] = e
            a, z = ids[0], ids[-1]

            def cos(u, v):
                return float(np.dot(u, v))

            # F1 菌盖
            rec["f1_cap_cos_fl"] = cos(E["cap"][a], E["cap"][z])
            rec["f1_cap_hist_bhat"] = float(cv2.compareHist(
                hists[a], hists[z], cv2.HISTCMP_BHATTACHARYYA))
            rec["f1_cap_cos_min"] = min(cos(E["cap"][a], E["cap"][i]) for i in ids[1:])
            # F2 锚定整体
            anchor = [cos(E["whole"][a], E["whole"][i]) for i in ids[1:]]
            rec["f2_anchor_min"] = min(anchor)
            rec["f2_anchor_argmin"] = ids[1:][int(np.argmin(anchor))]
            rec["c_first_last_check"] = cos(E["whole"][a], E["whole"][z])
            # F3 脸部
            rec["f3_face_cos_fl"] = cos(E["face"][a], E["face"][z])
            rec["f3_face_cos_min"] = min(cos(E["face"][a], E["face"][i]) for i in ids[1:])

    # F4/F5:逐相邻采样帧对
    heights = [float(b[3] - b[1]) if b is not None else float("nan") for b in boxes]
    pair_scale, pair_match, pair_rm, pair_rp = [], [], [], []
    for i in range(len(frames) - 1):
        s, nm, rm, rp = cam_scale_and_residual(
            frames[i], frames[i + 1], boxes[i], boxes[i + 1])
        pair_scale.append(s); pair_match.append(nm)
        pair_rm.append(rm); pair_rp.append(rp)
    rec["f4_cam_match_min"] = int(min(pair_match)) if pair_match else 0

    cum, cs = [1.0], 1.0
    for s in pair_scale:
        cs *= s if (s == s and 0.5 < s < 2.0) else 1.0
        cum.append(cs)
    hc = np.array([h / c for h, c in zip(heights, cum)])
    valid = ~np.isnan(hc)
    if valid.sum() >= 4:
        lh = np.log(hc[valid])
        t = np.arange(len(hc))[valid].astype(float)
        t = (t - t.min()) / max(1.0, t.max() - t.min())
        rec["f4_logh_range"] = float(np.percentile(lh, 90) - np.percentile(lh, 10))
        rec["f4_logh_slope"] = float(np.polyfit(t, lh, 1)[0])
    rm = np.array(pair_rm); rp = np.array(pair_rp)
    if np.isfinite(rp).sum() >= 4:
        rec["f5_res_p95_mean"] = float(np.nanmean(rp))
        rec["f5_res_p95_max"] = float(np.nanmax(rp))
        rec["f5_res_mean"] = float(np.nanmean(rm))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "data/prod500/videos"))
    ap.add_argument("--out", default=str(ROOT / "data/prod500/factors_f1f5.jsonl"))
    ap.add_argument("--stems-file", default=None, help="只处理这些 stem(每行一个)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-dir", default=str(ROOT / ".hf_cache"))
    args = ap.parse_args()

    vids = sorted(Path(args.videos_dir).glob("*.mp4"))
    if args.stems_file:
        keep = set(Path(args.stems_file).read_text().split())
        vids = [v for v in vids if v.stem in keep or v.stem[:8] in keep]
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["stem"])
            except Exception:
                pass
    vids = [v for v in vids if v.stem not in done]
    if args.limit:
        vids = vids[: args.limit]
    print(f"todo {len(vids)} videos (skip {len(done)} done)", flush=True)

    models = Models(Path(args.cache_dir))
    import time
    t0 = time.time()
    with out.open("a") as f:
        for i, v in enumerate(vids):
            try:
                rec = process_video(models, v)
            except Exception as e:  # noqa: BLE001
                rec = {"stem": v.stem, "error": repr(e)[:200]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(vids)}] {el/ (i+1):.1f}s/vid elapsed {el/60:.1f}m",
                      flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
