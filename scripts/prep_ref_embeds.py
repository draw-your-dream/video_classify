#!/usr/bin/env python
"""参照池嵌入准备(P3-R 轴,2026-08-01)。

renders/<款>/<id>.png + <id>_mask.png -> 白底 448 画布(与查询抠像同一几何)
-> so400m-512 图像塔嵌入 + DreamSim 嵌入 -> ref_embeds.npz(按款索引)。
参照池:饰品款 = 本款 9 张 + 基础款 98 张;基础款 = 98 张(评估时按款取并集)。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CANVAS = 448


def white_canvas(img: Image.Image, mask: Image.Image | None) -> Image.Image:
    im = np.array(img.convert("RGB")).astype(np.float32)
    if mask is not None:
        m = np.array(mask.convert("L")).astype(np.float32) / 255.0
        im = im * m[..., None] + 255.0 * (1 - m[..., None])
        ys, xs = np.where(m > 0.5)
        if len(ys):
            im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = im.astype(np.uint8)
    p = Image.fromarray(im)
    p.thumbnail((CANVAS, CANVAS), Image.LANCZOS)
    out = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    out.paste(p, ((CANVAS - p.width) // 2, (CANVAS - p.height) // 2))
    return out


class Embedders:
    def __init__(self, device="cuda"):
        from transformers import AutoModel, AutoProcessor
        mid = "google/siglip2-so400m-patch16-512"
        self.sp = AutoProcessor.from_pretrained(mid)
        self.sm = AutoModel.from_pretrained(mid, torch_dtype=torch.float16).to(device).eval()
        from dreamsim import dreamsim as ds_load
        self.dm, self.dp = ds_load(pretrained=True, device=device)
        self.device = device

    @torch.inference_mode()
    def so400m(self, ims: list[Image.Image]) -> np.ndarray:
        inp = self.sp(images=ims, return_tensors="pt").to(self.device)
        f = self.sm.get_image_features(pixel_values=inp["pixel_values"].half())
        if not torch.is_tensor(f):
            f = f.pooler_output
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy()

    @torch.inference_mode()
    def dreamsim(self, ims: list[Image.Image]) -> np.ndarray:
        xs = torch.cat([self.dp(im) for im in ims]).to(self.device)
        f = self.dm.embed(xs)
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default="/root/mech/renders")
    ap.add_argument("--out", default="/root/mech/ref_embeds.npz")
    args = ap.parse_args()

    emb = Embedders()
    styles, so_l, ds_l = [], [], []
    for sd in sorted(Path(args.renders).iterdir()):
        if not sd.is_dir():
            continue
        ims = []
        for f in sorted(sd.glob("*.png")):
            if f.name.endswith("_mask.png"):
                continue
            mp = f.with_name(f.stem + "_mask.png")
            ims.append(white_canvas(Image.open(f), Image.open(mp) if mp.exists() else None))
        if not ims:
            continue
        so = np.concatenate([emb.so400m(ims[i:i + 16]) for i in range(0, len(ims), 16)])
        dsv = np.concatenate([emb.dreamsim(ims[i:i + 16]) for i in range(0, len(ims), 16)])
        styles += [sd.name] * len(ims)
        so_l.append(so)
        ds_l.append(dsv)
        print(sd.name, len(ims), flush=True)
    np.savez_compressed(args.out, style=np.array(styles),
                        so400m=np.concatenate(so_l), dreamsim=np.concatenate(ds_l))
    print("REF_EMBEDS_DONE")


if __name__ == "__main__":
    main()
