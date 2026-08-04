#!/usr/bin/env python
"""重建参考池嵌入(so400m-only,v3 实例选择用)。
输入 /root/mech/renders_2026_04/export/<款>/*.png(带 *_mask.png 则白底化),
输出 /root/mech/ref_embeds.npz {style, so400m}。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")


def white_canvas(img: Image.Image, mask: Image.Image | None, size=448) -> Image.Image:
    img = img.convert("RGB")
    if mask is not None:
        m = np.array(mask.convert("L")) > 127
        a = np.array(img)
        a[~m] = 255
        ys, xs = np.where(m)
        if len(ys) > 50:
            a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        img = Image.fromarray(a)
    w, h = img.size
    s = size / max(w, h)
    img = img.resize((int(w * s), int(h * s)))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return canvas


def main():
    from transformers import AutoModel, AutoProcessor
    mid = "google/siglip2-so400m-patch16-512"
    sp = AutoProcessor.from_pretrained(mid)
    sm = AutoModel.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()

    @torch.inference_mode()
    def embed(ims):
        inp = sp(images=ims, return_tensors="pt").to("cuda")
        f = sm.get_image_features(pixel_values=inp["pixel_values"].half())
        if not torch.is_tensor(f):
            f = f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    styles, embs = [], []
    base = ROOT / "renders_2026_04/export"
    for sd in sorted(base.iterdir()):
        if not sd.is_dir():
            continue
        ims = []
        for p in sorted(sd.glob("*.png")):
            if p.name.endswith("_mask.png"):
                continue
            mp = p.with_name(p.stem + "_mask.png")
            ims.append(white_canvas(Image.open(p), Image.open(mp) if mp.exists() else None))
        for i in range(0, len(ims), 16):
            embs.append(embed(ims[i:i + 16]))
        styles += [sd.name] * len(ims)
        print(sd.name, len(ims), flush=True)
    np.savez_compressed(ROOT / "ref_embeds.npz", style=np.array(styles),
                        so400m=np.concatenate(embs))
    print("REF_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
