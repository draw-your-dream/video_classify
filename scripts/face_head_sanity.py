#!/usr/bin/env python
"""线F 仪器自检:held-out 合成脸缺陷判别力。"""
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, torch, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axis_rself_local import split_goods, stem_seed, auc
from synth_face_defects import FACE_DEFECTS, synthesize_face
from train_synth_head import Head, load_backbone, patch_feats, to_tensor, frame_score

ROOT = Path(__file__).resolve().parents[1]
device = "cuda"
rows = [l.split("\t") for l in (ROOT/"data/prod500/mech_subset.tsv").read_text().splitlines() if l.strip()]
gb = defaultdict(list)
for rel, label in rows:
    if label == "good": gb[rel.split("/")[0]].append(rel)
_, eval_rels = split_goods(gb)
frames = []
for rel in eval_rels:
    frames += sorted((ROOT/"data/face_crops"/rel.replace(".mp4","")).glob("f*.jpg"))
rng = np.random.default_rng(stem_seed("face-sanity"))
picks = [frames[i] for i in rng.choice(len(frames), 240, replace=False)]
backbone = load_backbone(device)
head = Head().to(device)
head.load_state_dict(torch.load(ROOT/"data/prod500/face_head_v1.pt"))
head.eval()
kinds = list(FACE_DEFECTS)
per = {k: ([], []) for k in kinds}
with torch.no_grad():
    for p in picks:
        im = cv2.imread(str(p))
        if im is None: continue
        k = kinds[int(rng.integers(len(kinds)))]
        got = synthesize_face(im, k, rng)
        if got is None: continue
        xs = torch.stack([to_tensor(got[0]), to_tensor(im)]).to(device)
        fs = frame_score(head(patch_feats(backbone, xs))).cpu().numpy()
        per[k][0].append(fs[0]); per[k][1].append(fs[1])
allp = np.array(sum((v[0] for v in per.values()), []))
alln = np.array(sum((v[1] for v in per.values()), []))
print(f"held-out 合成脸缺陷 帧级 AUC = {auc(allp, alln):.3f} (n={len(allp)})")
for k,(p_,n_) in per.items():
    if len(p_) >= 10:
        print(f"  {k:12s} AUC={auc(np.array(p_), np.array(n_)):.3f} (n={len(p_)})")
print("FACE_SANITY_DONE")
