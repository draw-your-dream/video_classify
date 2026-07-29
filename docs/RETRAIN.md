# Train + run single-video inference (minimal, from scratch)

This repo is trimmed to exactly what's needed to **train the model `inference/predict.py`
uses and run single-video inference**. The model is:

```
video → per-video feature caches → 15 experts → LR meta + LGBM boost → P(bad)
```

The 14-pass **gate cascade is intentionally excluded**: it is rank-domain (defined only
over a batch) and was shown to be in-sample-overfit (it does not generalize to a new
video), so `inference/predict.py` skips it and returns the base score (meta + boost),
which is well-defined for a single sample.

## Train from scratch

```
bash scripts/train_for_inference.sh
```

This runs, from repo root (needs GPU + the HF models in `.hf_cache` / `~/.cache/huggingface`):

1. **Split** — `scripts/make_split_v3.py` (stratified train/eval over all videos).
2. **Per-video feature caches** consumed by the 15 experts → `data/cache/<group>/<label>/<stem>.{json,npy}`:
   - heuristic (`extract_features_v2`), dense motion/flow, bbox (GroundingDINO),
   - embeddings: clip / siglip2-base / siglip2-so400m / vjepa2 / vlm-embed,
   - VLM yes/no: fatalflaw / specific / judgment / refid,
   - fidelity: vs-rlhf (`fidelity_v2`), perceptual (LPIPS+DreamSim), cropped (GroundingDINO crop),
     sku-ref (`cache_sku_ref_clip` builds the reference pool → `cache_fidelity_sku_ref`),
     multipose (KMeans-20) / mp50 (KMeans-50), text-image-align,
   - asr (SenseVoice), ocr (EasyOCR),
   - LoRA SFT (`lora_sft_v3_qwen25vl`) → `lm_sft_v3_pred/` (symlinked to `lm_sft_v2_pred/`, identical schema).
3. **Train** — `training.build_experts` (15 experts + OOF) → `training.build_stack`
   (15-expert stack) → `training.train` (LR meta + LGBM boost + threshold) →
   `inference/artifacts_v3/`.

## Single-video inference

```
PYTHONPATH=src .venv/bin/python -m inference.predict path/to/video.mp4
# prints P(bad), P(good), label; --json for per-expert detail and feature-group status
# --fast : quick approximation (heavy groups default-filled) instead of the full model
```

The **only local input is the video**. By default `predict.py` pulls the trained params +
reference data from the HF Hub (`Picaa-AI/tutu-video-badcase-eval`, tag `v1`) via
`inference/hf_assets.py` and caches them under `~/.cache/tutu-video-eval`
(`artifacts_v1/` classifier, `lora_sft_v1_qwen25vl/` adapter, `references_v1.tar.gz` →
`TUTU_REF_DIR`). Base public models come from their own HF repos. Use `--local-artifacts DIR`
to run fully offline from a local artifacts dir instead (dev mode).

Extraction is **faithful**: `inference/extract_full.py` runs the real training extractors
(`scripts/*`) on the one video (each writes `data/cache/<group>/query/<stem>` under
`TUTU_QUERY_ABS`), passing the HF references via `TUTU_REF_DIR` and the HF adapter via
`--adapter` — so all 23 feature groups are the true heavy-model values (validated bit-faithful;
only `multipose`/`mp50` drift by sklearn KMeans float-nondeterminism, ~0.007 on `P(bad)`).
~2 min/video. `--fast` uses the `_extract.py` approximation instead (no reference data).

## The 15 experts (training.expert_definitions)

full, full_lean, hint, motion, bbox, vjepa, siglip, cropped_m10, per_src_{hint,motion,siglip,
cropped_m10}, sigb, tia, mp — each a 3-model rank-averaged ensemble (LR + LGBM-31 + LGBM-15)
over its feature subset; per-source variants train one ensemble per source (ti2i2v/skus/rlhf).

## Honest performance note

On a held-out fresh split the base model reaches **(good+normal)-recall ≈ 0.28 at
bad-recall = 0.95** (AUC ≈ 0.75). The historical "≈1.0" figures came from the cascade
fitting a fixed eval set (in-sample) and do not reflect new-video performance. See
`ITER_LOG.md` for the full experiment record.
