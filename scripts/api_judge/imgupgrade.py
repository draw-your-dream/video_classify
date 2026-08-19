#!/usr/bin/env python3
"""图片探针升级实验:更大骨干 + MIL 补丁探针,4553 张标注图分组 5 折 CV AUC。
对照基线 so400m 池化探针 0.610。产出各配置 AUC 表 + refprobe 同款 OOF 存档。
marker=IMGUPGRADE_DONE。"""
import csv
import gc
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

R2 = Path("/workspace/r2")

rows = []
for f in ["tutu_image_annotations_2962.csv", "tutu_image_annotations_0813.csv"]:
    for r in csv.DictReader(open(R2 / "data" / f, encoding="utf-8-sig")):
        rows.append((r["dataset"], r["sample_id"], r["label"]))
items = []
for d, s, l in rows:
    fp = R2 / "qcimgs" / f"{d}__SLASH__{s}.png"
    if fp.exists() and fp.stat().st_size > 0:
        items.append((fp, s.split("__")[0], l))
print(f"images: {len(items)}", flush=True)
y = np.array([1 if l == "bad" else 0 for _, _, l in items])
groups = np.array([g for _, g, _ in items])


def load_pils(batch):
    out = []
    for fp, _, _ in batch:
        im = Image.open(fp).convert("RGB")
        im.thumbnail((512, 512))
        out.append(im)
    return out


import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


def cv_auc(X, tag):
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups):
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr])
        gb = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                min_child_samples=20, random_state=42, verbose=-1).fit(X[tr], y[tr])
        oof[te] = (rankdata(lr.predict_proba(X[te])[:, 1]) +
                   rankdata(gb.predict_proba(X[te])[:, 1])) / 2 / len(te)
    r = rankdata(oof)
    pos = r[y == 1]
    auc = (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (y == 0).sum())
    print(f"[imgupgrade] {tag}: CV AUC = {auc:.4f}", flush=True)
    return auc, oof


from transformers import AutoImageProcessor, AutoModel

EMB = {}
TOKENS = None  # so400m patch tokens 给 MIL 用

for tag, mn, kind in [("so400m", "google/siglip2-so400m-patch14-384", "sig"),
                      ("sg_giant", "google/siglip2-giant-opt-patch16-384", "sig"),
                      ("dino_g", "facebook/dinov2-giant", "dino")]:
    try:
        proc = AutoImageProcessor.from_pretrained(mn)
        model = AutoModel.from_pretrained(mn, torch_dtype=torch.float16).to("cuda").eval()
    except Exception as ex:
        print(f"[imgupgrade] {tag} load FAIL {repr(ex)[:100]}", flush=True)
        continue
    E_pool, tok_list = [], []
    t0 = time.time()
    for i in range(0, len(items), 24):
        pils = load_pils(items[i:i + 24])
        inputs = proc(images=pils, return_tensors="pt").to("cuda")
        with torch.no_grad():
            if kind == "sig":
                o = model.vision_model(pixel_values=inputs["pixel_values"].half())
                pooled = o.pooler_output
                if tag == "so400m":
                    tok_list.append(o.last_hidden_state.half().cpu())
            else:
                o = model(pixel_values=inputs["pixel_values"].half())
                pooled = o.last_hidden_state[:, 0]
        E_pool.append(pooled.float().cpu().numpy())
        if i % (24 * 40) == 0:
            print(f"  {tag} {i}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    E = np.concatenate(E_pool)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    EMB[tag] = E
    if tok_list:
        TOKENS = torch.cat(tok_list)  # (N, P, D) fp16 cpu
    del model
    gc.collect()
    torch.cuda.empty_cache()

results = {}
for tag, E in EMB.items():
    results[tag], _ = cv_auc(E, tag)
if len(EMB) > 1:
    results["concat"], oof_c = cv_auc(np.hstack(list(EMB.values())), "concat")
    np.savez_compressed(R2 / "imgup_oof.npz", oof=oof_c, y=y, groups=groups)

# ---- MIL 补丁探针(so400m tokens, top-k 池化线性头) ----
if TOKENS is not None:
    T = TOKENS.cuda()  # (N,P,D)
    T = T / T.norm(dim=-1, keepdim=True)
    N, P, Dd = T.shape
    yt = torch.tensor(y, dtype=torch.float16, device="cuda")
    oof = np.full(N, np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(N), y, groups):
        w = torch.zeros(Dd, device="cuda", dtype=torch.float32, requires_grad=True)
        b = torch.zeros(1, device="cuda", dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=3e-3, weight_decay=1e-4)
        tr_t = torch.tensor(tr, device="cuda")
        for step in range(300):
            idx = tr_t[torch.randint(0, len(tr_t), (256,), device="cuda")]
            tok = T[idx].float()
            sc = tok @ w + b  # (B,P)
            logit = sc.topk(32, dim=1).values.mean(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[idx].float())
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sc = (T[torch.tensor(te, device='cuda')].float() @ w + b)
            oof[te] = sc.topk(32, dim=1).values.mean(1).cpu().numpy()
    r = rankdata(oof); pos = r[y == 1]
    auc = (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (y == 0).sum())
    results["mil_so400m"] = auc
    print(f"[imgupgrade] mil_so400m: CV AUC = {auc:.4f}", flush=True)
    np.save(R2 / "imgup_mil_oof.npy", oof)

print("SUMMARY", {k: round(v, 4) for k, v in results.items()}, flush=True)
print("IMGUPGRADE_DONE", flush=True)
