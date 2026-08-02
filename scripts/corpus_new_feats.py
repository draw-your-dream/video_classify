#!/usr/bin/env python
"""E3 全语料新特征(盒侧,2026-08-02 预注册族清单,冻结 21 列):

A 几何/计数(sam3_feat npz):sc_mean sc_min multi_frac zero_frac err_frac
  hc_med hc_drift area_slope area_range cxy_drift n_geo
B 参照相似(sam3_cutouts + ref_embeds 152 全池):rso_mean rso_p75 rso_max
  rds_mean rds_p75 rds_max   (d = 1 - max cos)
C 身体合成头 synth_head_v0:bh_p75 bh_max
D 脸部合成头 face_head_v1:fh_p75 fh_max fh_cover

输出 data/corpus_new_feats.tsv(rel + 21 列,缺失=nan,支持续跑)。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
COLS = ("sc_mean sc_min multi_frac zero_frac err_frac hc_med hc_drift area_slope "
        "area_range cxy_drift n_geo rso_mean rso_p75 rso_max rds_mean rds_p75 rds_max "
        "bh_p75 bh_max fh_p75 fh_max fh_cover").split()


def to_tensor(im_bgr):
    im = cv2.resize(im_bgr, (518, 518), interpolation=cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1))


class Head(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
    def so400m(self, ims):
        inp = self.sp(images=ims, return_tensors="pt").to(self.device)
        f = self.sm.get_image_features(pixel_values=inp["pixel_values"].half())
        if not torch.is_tensor(f):
            f = f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    @torch.inference_mode()
    def dreamsim(self, ims):
        xs = torch.cat([self.dp(im) for im in ims]).to(self.device)
        f = self.dm.embed(xs)
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()


def geo_feats(npz_p):
    out = {c: np.nan for c in COLS[:11]}
    if not npz_p.exists():
        return out
    z = np.load(npz_p, allow_pickle=True)
    ni = z["n_inst"].astype(int)
    sc = z["top_score"].astype(float)
    ok = ni >= 0
    if ok.sum():
        out["err_frac"] = float((~ok).mean())
        out["multi_frac"] = float((ni[ok] > 1).mean())
        out["zero_frac"] = float((ni[ok] == 0).mean())
        v = sc[ok & (sc > 0)]
        if len(v):
            out["sc_mean"], out["sc_min"] = float(v.mean()), float(v.min())
    geo = json.loads(str(z["geo"]))
    out["n_geo"] = float(len(geo))
    if len(geo) >= 4:
        hc = np.array([g["height"] / max(1.0, g["cap_width"]) for g in geo])
        ar = np.array([g["area"] for g in geo], float)
        cx = np.array([(g["bbox"][0] + g["bbox"][2]) / 2 for g in geo], float)
        cy = np.array([(g["bbox"][1] + g["bbox"][3]) / 2 for g in geo], float)
        w = np.array([g["bbox"][2] - g["bbox"][0] for g in geo], float)
        out["hc_med"] = float(np.median(hc))
        k = min(4, len(hc) // 2)
        out["hc_drift"] = float(abs(np.log(max(1e-3, hc[-k:].mean()) / max(1e-3, hc[:k].mean()))))
        t = np.arange(len(ar))
        out["area_slope"] = float(np.polyfit(t, ar / max(1.0, ar.mean()), 1)[0])
        out["area_range"] = float((ar.max() - ar.min()) / max(1.0, np.median(ar)))
        wm = max(1.0, np.median(w))
        out["cxy_drift"] = float((cx.std() + cy.std()) / wm)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data/corpus_new_feats.tsv"))
    args = ap.parse_args()
    device = "cuda"
    rows = []
    for mf in (ROOT / "corpus_full.tsv", ROOT / "manifest_rlhf.tsv"):
        rows += [l.split("\t")[0] for l in mf.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = [r for r in rows if (ROOT / "data/sam3_cutouts" / r.replace(".mp4", "")).exists()][:args.limit]
    out_p = Path(args.out)
    done = set()
    if out_p.exists():
        done = {l.split("\t")[0] for l in out_p.read_text().splitlines()[1:] if l.strip()}
    else:
        out_p.write_text("rel\t" + "\t".join(COLS) + "\n")
    todo = [r for r in rows if r not in done]
    print(f"total {len(rows)} done {len(done)} todo {len(todo)}", flush=True)

    ref = np.load(ROOT / "ref_embeds.npz", allow_pickle=True)
    R_so, R_ds = ref["so400m"].astype(np.float32), ref["dreamsim"].astype(np.float32)
    emb = Embedders(device)
    from transformers import AutoModel
    local = ROOT / ".hf_cache/dinov2-base-ms"
    src = str(local) if local.exists() else "facebook/dinov2-base"
    backbone = AutoModel.from_pretrained(src, torch_dtype=torch.float16).to(device).eval()
    bh = Head().to(device); bh.load_state_dict(torch.load(ROOT / "checkpoints/synth_head_v0.pt", map_location=device)); bh.eval()
    fh = Head().to(device); fh.load_state_dict(torch.load(ROOT / "checkpoints/face_head_v1.pt", map_location=device)); fh.eval()

    @torch.no_grad()
    def head_scores(head, ims_bgr):
        xs = torch.stack([to_tensor(im) for im in ims_bgr]).to(device)
        feats = backbone(pixel_values=xs.half()).last_hidden_state[:, 1:, :].float()
        p = torch.sigmoid(head(feats))
        return p.topk(16, dim=-1).values.mean(-1).cpu().numpy()

    import time
    t0 = time.time()
    f_out = open(out_p, "a", encoding="utf-8")
    for k, rel in enumerate(todo):
        stem = rel.replace(".mp4", "")
        feats = geo_feats(ROOT / "data/sam3_feat" / (stem + ".npz"))
        for c in COLS[11:]:
            feats[c] = np.nan
        cut_jpgs = sorted((ROOT / "data/sam3_cutouts" / stem).glob("f*.jpg"))
        if cut_jpgs:
            ims = [cv2.imread(str(p)) for p in cut_jpgs]
            ims = [im for im in ims if im is not None]
            if ims:
                pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in ims]
                for tag, E, Rp in (("rso", emb.so400m, R_so), ("rds", emb.dreamsim, R_ds)):
                    Q = E(pil)
                    d = 1.0 - (Q @ Rp.T).max(1)
                    feats[f"{tag}_mean"] = float(d.mean())
                    feats[f"{tag}_p75"] = float(np.percentile(d, 75))
                    feats[f"{tag}_max"] = float(d.max())
                s = head_scores(bh, ims)
                feats["bh_p75"], feats["bh_max"] = float(np.percentile(s, 75)), float(s.max())
        face_jpgs = sorted((ROOT / "data/face_crops" / stem).glob("f*.jpg"))
        feats["fh_cover"] = float(len(face_jpgs))
        if len(face_jpgs) >= 3:
            ims = [cv2.imread(str(p)) for p in face_jpgs]
            ims = [im for im in ims if im is not None]
            if ims:
                s = head_scores(fh, ims)
                feats["fh_p75"], feats["fh_max"] = float(np.percentile(s, 75)), float(s.max())
        f_out.write(rel + "\t" + "\t".join(f"{feats[c]:.6g}" for c in COLS) + "\n")
        if (k + 1) % 100 == 0:
            f_out.flush()
            el = time.time() - t0
            print(f"[{k+1}/{len(todo)}] {el/(k+1):.2f}s/vid elapsed {el/60:.1f}m", flush=True)
    f_out.close()
    print("NEW_FEATS_DONE", flush=True)


if __name__ == "__main__":
    main()
