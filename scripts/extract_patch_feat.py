#!/usr/bin/env python
"""单类 patch 异常检测——特征提取(机制验证,语料 919 子集 / prod500 通用)。

设计(2026-07-30 预注册前置实验,见 FACTOR_PREREG.md 追加条目):
  16 帧均匀采样 -> GroundingDINO 角色框(crop_feat_original 同款:三连 prompt,
  thr 0.25,方形裁剪 1.25 倍,无检测回退中央方框)
  -> 裁剪 resize 518x518 -> DINOv2-base patch tokens(37x37x768)
  -> 前景掩码 = patch 中心落在原检测框内
每视频存一个 npz:feat fp16 (T,1369,768) / fg bool / det bool / boxes。
库构建与打分在 patch_bank_eval.py,本脚本只产特征。
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

PROMPT = "a mushroom character. a plush toy. a mushroom."   # crop_feat_original 同款
DET_THRESH = 0.25
N_FRAMES = 16
PATCH_RES = 518          # 37x37 patches @ patch14
GRID = PATCH_RES // 14
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_frames(path: Path, n: int = N_FRAMES) -> list[np.ndarray]:
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


class Models:
    def __init__(self, cache_dir: Path):
        from transformers import (AutoModel, AutoModelForZeroShotObjectDetection,
                                  AutoProcessor)
        cd = str(cache_dir)
        self.dproc = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base", cache_dir=cd)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base", cache_dir=cd).to("cuda").eval()
        local = cache_dir / "dinov2-base-ms"       # ModelScope 镜像落地目录
        src = str(local) if local.exists() else "facebook/dinov2-base"
        self.vit = AutoModel.from_pretrained(
            src, cache_dir=cd, torch_dtype=torch.float16).to("cuda").eval()

    @torch.inference_mode()
    def detect(self, imgs: list[Image.Image]) -> list[np.ndarray | None]:
        boxes = []
        for j in range(0, len(imgs), 8):
            chunk = imgs[j:j + 8]
            inp = self.dproc(images=chunk, text=[PROMPT] * len(chunk),
                             return_tensors="pt").to("cuda")
            out = self.dino(**inp)
            res = self.dproc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=DET_THRESH, text_threshold=0.25,
                target_sizes=[im.size[::-1] for im in chunk])
            for r in res:
                if len(r["boxes"]) == 0:
                    boxes.append(None)
                else:
                    boxes.append(r["boxes"][int(r["scores"].argmax())]
                                 .float().cpu().numpy())
        return boxes

    @torch.inference_mode()
    def patch_feat(self, crops: list[Image.Image]) -> np.ndarray:
        """(B, GRID*GRID, 768) fp16;手工预处理保证分辨率恒为 518。"""
        arrs = []
        for im in crops:
            a = np.asarray(im.convert("RGB").resize(
                (PATCH_RES, PATCH_RES), Image.BICUBIC), dtype=np.float32) / 255.0
            arrs.append((a - IMNET_MEAN) / IMNET_STD)
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to("cuda", torch.float16)
        feats = []
        for j in range(0, len(x), 8):
            h = self.vit(pixel_values=x[j:j + 8]).last_hidden_state  # (b,1+N,768)
            feats.append(h[:, 1:, :].cpu().numpy().astype(np.float16))
        return np.concatenate(feats, 0)


def sq_crop(im: Image.Image, b: np.ndarray | None, marg: float = 0.25):
    """crop_feat_original.crop 同款;返回 (crop_img, crop_rect)。"""
    W, H = im.size
    if b is None:
        s = min(W, H)
        rect = ((W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2)
        return im.crop(rect), rect
    x0, y0, x1, y1 = b
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * (1 + marg)
    s = max(s, 32)
    rect = (max(0, cx - s / 2), max(0, cy - s / 2), min(W, cx + s / 2), min(H, cy + s / 2))
    return im.crop(rect), rect


def fg_mask(box: np.ndarray | None, rect: tuple) -> np.ndarray:
    """patch 中心是否落在检测框内(裁剪坐标系)。无检测帧全 True。"""
    m = np.ones((GRID, GRID), dtype=bool)
    if box is None:
        return m.reshape(-1)
    rx0, ry0, rx1, ry1 = rect
    cw, ch = rx1 - rx0, ry1 - ry0
    if cw <= 0 or ch <= 0:
        return m.reshape(-1)
    bx0 = (box[0] - rx0) / cw * GRID
    by0 = (box[1] - ry0) / ch * GRID
    bx1 = (box[2] - rx0) / cw * GRID
    by1 = (box[3] - ry0) / ch * GRID
    jj, ii = np.meshgrid(np.arange(GRID) + 0.5, np.arange(GRID) + 0.5)
    m = (jj >= bx0) & (jj <= bx1) & (ii >= by0) & (ii <= by1)
    return m.reshape(-1)


def process_video(models: Models, path: Path, out: Path) -> dict:
    frames = read_frames(path)
    if len(frames) < 4:
        return {"stem": path.stem, "error": "too_few_frames"}
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    boxes = models.detect(imgs)
    crops, rects = zip(*[sq_crop(im, b) for im, b in zip(imgs, boxes)])
    feat = models.patch_feat(list(crops))                     # (T,1369,768)
    fg = np.stack([fg_mask(b, r) for b, r in zip(boxes, rects)])
    det = np.array([b is not None for b in boxes])
    bx = np.stack([b if b is not None else np.full(4, np.nan, np.float32)
                   for b in boxes]).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feat=feat, fg=fg, det=det, boxes=bx)
    return {"stem": path.stem, "n_frames": len(frames),
            "det_rate": float(det.mean()), "fg_rate": float(fg[det].mean()) if det.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/prod500/mech_subset.tsv"),
                    help="tsv: 相对路径\\t标签;或用 --videos-dir 扫目录")
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/corpus_patch_feat"))
    ap.add_argument("--log", default=str(ROOT / "data/corpus_patch_feat/extract_log.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-dir", default=str(ROOT / ".hf_cache"))
    args = ap.parse_args()

    vdir, odir = Path(args.videos_dir), Path(args.out_dir)
    todo = []
    for line in Path(args.manifest).read_text().splitlines():
        rel = line.split("\t")[0]
        src = vdir / rel
        dst = odir / rel.replace(".mp4", ".npz")
        if src.exists() and not dst.exists():
            todo.append((src, dst))
    if args.limit:
        todo = todo[: args.limit]
    print(f"todo {len(todo)} videos", flush=True)
    if not todo:
        return

    models = Models(Path(args.cache_dir))
    logp = Path(args.log)
    logp.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with logp.open("a") as lf:
        for i, (src, dst) in enumerate(todo):
            try:
                rec = process_video(models, src, dst)
            except Exception as e:  # noqa: BLE001
                rec = {"stem": src.stem, "error": repr(e)[:200]}
            rec["rel"] = str(src.relative_to(vdir))
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lf.flush()
            if (i + 1) % 25 == 0 or i == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(todo)}] {el/(i+1):.1f}s/vid elapsed {el/60:.1f}m",
                      flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
