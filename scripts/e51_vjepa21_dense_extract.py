#!/usr/bin/env python3
"""Train-only persistent V-JEPA 2.1 Base-384 dense feature extractor.

The E51 extraction contract is frozen before G0:

* official ``vjepa2_1_vit_base_384`` EMA encoder weights;
* 64 uniformly sampled RGB frames, decoded once with Decord;
* deterministic GPU resize to short side 438, then two 384px crops at the
  endpoints of the long side (a square frame duplicates its centered crop);
* final encoder tokens ``[2, 18432, 768]`` reshaped to
  ``[2, 32, 24, 24, 768]`` (view, time, height, width, channel);
* 2x2 spatial average pooling to ``[2, 32, 12, 12, 768]`` float16;
* per-view full 24x24 scalar temporal-motion maps retained.

The script never accesses the manifest's label field and computes no metric.  It
accepts only the preregistered ``train_v3.jsonl`` bytes.  Encoder state is loaded
once, outside the video loop.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from e50_make_shadow import sample_token


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "e51_vjepa21_b384_f64_v2_g12_v1"
MODEL_NAME = "vjepa2_1_vit_base_384"
UPSTREAM_URL = "https://github.com/facebookresearch/vjepa2"
AUDITED_REPO_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
TRAIN_MANIFEST_NAME = "train_v3.jsonl"
TRAIN_MANIFEST_SHA256 = "3ae40b797113ab9d1195ef3566e1380fe2c53c83d1937f19161b082a4d1da40d"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/"
    "vjepa2_1_vitb_dist_vitG_384.pt"
)
CHECKPOINT_SIZE = 1_664_223_428
CHECKPOINT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
CHECKPOINT_KEY = "ema_encoder"

# Frozen E51 preregistration contract.  Do not expose casual tuning knobs.
NUM_FRAMES = 64
NUM_VIEWS = 2
CROP_SIZE = 384
SHORT_SIDE_SIZE = int(CROP_SIZE * 256 / 224)
PATCH_SIZE = 16
TUBELET_SIZE = 2
PATCH_GRID = 24
DENSE_GRID = 12
EMBED_DIM = 768
TOKEN_TIME = NUM_FRAMES // TUBELET_SIZE
EXPECTED_TOKEN_COUNT = TOKEN_TIME * PATCH_GRID * PATCH_GRID
AMP_DTYPE = "bfloat16"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
PREPROCESS_EQ_MAX_ATOL = 2.0e-2
PREPROCESS_EQ_MEAN_ATOL = 2.0e-4

MAP_NAMES = (
    "patch_step_cosine",
    "patch_step_l2",
    "patch_local_residual",
    "patch_curvature_l2",
)
STAT_NAMES = ("mean", "std", "p50", "p75", "p90", "p95", "max", "top10_mean")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "splits" / "train_v3.jsonl",
        help="JSONL; only video and abs_path are accessed.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "cache" / "e51_vjepa21_b384_f64_v2_g12",
    )
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path("/workspace/mech/external/vjepa2"),
        help="Pinned official facebookresearch/vjepa2 checkout.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/workspace/mech/checkpoints/vjepa2_1/"
            "vjepa2_1_vitb_dist_vitG_384.pt"
        ),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="0 means all selected rows.")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--only-token",
        action="append",
        default=[],
        help="Opaque S0 sample token to process; repeatable.",
    )
    ap.add_argument(
        "--discovery-ids",
        type=Path,
        help=(
            "S0 discovery_ids.txt membership. Required for an unbounded non-G0 run; "
            "shadow members are never selected."
        ),
    )
    ap.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Prefix replacement for stale abs_path values; repeatable.",
    )
    ap.add_argument(
        "--videos-root",
        type=Path,
        help="Fallback corpus root; also tries suffix after data/s3.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--g0",
        action="store_true",
        help=(
            "Blind G0: hash all selected train videos, take the first 3 by byte "
            "SHA-256, repeat each twice, and run reproducibility/collapse checks."
        ),
    )
    ap.add_argument("--max-failures", type=int, default=5)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve manifest/video paths but load no model and write nothing.",
    )
    return ap.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, Path):
        return str(value)
    return value


def load_manifest(path: Path) -> list[dict[str, str]]:
    manifest = path.expanduser().resolve()
    if manifest.name != TRAIN_MANIFEST_NAME:
        raise ValueError(
            f"E51 accepts only the preregistered {TRAIN_MANIFEST_NAME}; "
            f"got {manifest.name!r}"
        )
    manifest_sha = sha256_file(manifest)
    if manifest_sha != TRAIN_MANIFEST_SHA256:
        raise RuntimeError(
            f"train manifest sha256 {manifest_sha} != preregistered "
            f"{TRAIN_MANIFEST_SHA256}"
        )
    rows: list[dict[str, str]] = []
    seen_stems: set[str] = set()
    with manifest.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            # Deliberately do not read, copy, count, print, or persist item['label'].
            missing = {"video", "abs_path"} - set(item)
            if missing:
                raise ValueError(f"{manifest}:{line_no}: missing {sorted(missing)}")
            video = str(item["video"])
            abs_path = str(item["abs_path"])
            token = sample_token(video)
            if token in seen_stems:
                raise ValueError(
                    f"{manifest}:{line_no}: duplicate sample token; "
                    "a label-free cache would collide"
                )
            seen_stems.add(token)
            rows.append(
                {"video": video, "abs_path": abs_path, "sample_token": token}
            )
    return rows


def load_discovery_ids(path: Path) -> tuple[set[str], str]:
    resolved = path.expanduser().resolve()
    values: list[str] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
                raise ValueError("discovery_ids.txt contains an invalid sample token")
            values.append(token)
    if not values or len(values) != len(set(values)):
        raise ValueError("discovery_ids.txt must contain unique non-empty sample tokens")
    return set(values), sha256_file(resolved)


def select_rows(
    args: argparse.Namespace,
    rows: Sequence[dict[str, str]],
    discovery_ids: set[str] | None,
) -> list[dict[str, str]]:
    if args.offset < 0 or args.limit < 0:
        raise ValueError("--offset and --limit must be >= 0")
    selected = list(rows)
    if discovery_ids is not None:
        available = {row["sample_token"] for row in selected}
        if not discovery_ids <= available:
            raise ValueError(
                f"{len(discovery_ids - available)} discovery token(s) absent from train manifest"
            )
        selected = [row for row in selected if row["sample_token"] in discovery_ids]
    if args.only_token:
        wanted = set(args.only_token)
        selected = [row for row in selected if row["sample_token"] in wanted]
        absent = wanted - {row["sample_token"] for row in selected}
        if absent:
            raise ValueError(f"{len(absent)} --only-token value(s) absent from selection")
    selected = selected[args.offset :]
    if args.limit:
        selected = selected[: args.limit]
    return selected


def parse_path_maps(values: Sequence[str]) -> list[tuple[Path, Path]]:
    result: list[tuple[Path, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--path-map must be OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(f"--path-map must have two non-empty sides: {value!r}")
        result.append((Path(old).expanduser(), Path(new).expanduser()))
    return result


def suffix_after_data_s3(path: Path) -> Path | None:
    parts = path.parts
    for i in range(len(parts) - 1):
        if parts[i] == "data" and parts[i + 1] == "s3":
            suffix = parts[i + 2 :]
            return Path(*suffix) if suffix else None
    return None


def video_candidates(
    row: Mapping[str, str],
    path_maps: Sequence[tuple[Path, Path]],
    videos_root: Path | None,
) -> list[Path]:
    original = Path(row["abs_path"]).expanduser()
    candidates = [original]
    original_s = str(original)
    for old, new in path_maps:
        old_s = str(old)
        if original_s == old_s:
            candidates.append(new)
        elif original_s.startswith(old_s.rstrip("/") + "/"):
            candidates.append(new / original_s[len(old_s.rstrip("/")) + 1 :])
    if videos_root is not None:
        root = videos_root.expanduser()
        candidates.append(root / row["video"])
        suffix = suffix_after_data_s3(original)
        if suffix is not None:
            candidates.append(root / suffix)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_video(
    row: Mapping[str, str],
    path_maps: Sequence[tuple[Path, Path]],
    videos_root: Path | None,
) -> tuple[Path | None, list[Path]]:
    candidates = video_candidates(row, path_maps, videos_root)
    for path in candidates:
        if path.is_file():
            return path.resolve(), candidates
    return None, candidates


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_tracked_dirty(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return bool(result.stdout.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    required = [
        repo / "src" / "hub" / "backbones.py",
        repo / "evals" / "hub" / "preprocessor.py",
        repo / "app" / "vjepa_2_1" / "models" / "vision_transformer.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"official repo is incomplete: {missing}")
    commit = git_commit(repo)
    if commit != AUDITED_REPO_COMMIT:
        raise RuntimeError(
            f"repo commit {commit!r} differs from audited {AUDITED_REPO_COMMIT}; "
            "review and preregister a new source revision"
        )
    if git_tracked_dirty(repo):
        raise RuntimeError("official V-JEPA 2 repo has tracked modifications")
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(f"checkpoint must be a regular non-symlink file: {checkpoint}")
    size = checkpoint.stat().st_size
    if size != CHECKPOINT_SIZE:
        raise RuntimeError(f"checkpoint size {size} != expected {CHECKPOINT_SIZE}")
    checksum = sha256_file(checkpoint)
    if checksum != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint sha256 {checksum} != expected {CHECKPOINT_SHA256}"
        )
    args.repo = repo
    args.checkpoint = checkpoint
    return {
        "model_name": MODEL_NAME,
        "upstream_repo": UPSTREAM_URL,
        "repo_commit": commit,
        "repo_tracked_clean": True,
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_size": size,
        "checkpoint_sha256": checksum,
        "checkpoint_hash_verified": True,
        "checkpoint_key": CHECKPOINT_KEY,
    }


def decode_decord(path: Path) -> tuple[np.ndarray, np.ndarray, float, int]:
    import decord

    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=2)
    total = len(reader)
    if total <= 0:
        raise ValueError(f"video has no frames: {path}")
    indices = np.rint(np.linspace(0, total - 1, NUM_FRAMES)).astype(np.int64)
    frames = reader.get_batch(indices).asnumpy()
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        fps = 0.0
    return frames, indices, fps, total


def decode_video(path: Path) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    frames, indices, fps, total = decode_decord(path)
    return frames, indices, fps, total, "decord"


def clean_encoder_state(state: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        key = key.replace("module.", "").replace("backbone.", "")
        if key in cleaned:
            raise KeyError(f"checkpoint key collision after cleaning: {key}")
        cleaned[key] = value
    return cleaned


class PersistentEncoder:
    def __init__(self, args: argparse.Namespace, provenance: Mapping[str, Any]):
        import torch

        self.torch = torch
        self.device = torch.device(args.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E51 extraction requires one CUDA GPU")
        torch.cuda.set_device(self.device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("current extraction protocol requires CUDA bfloat16 support")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        repo_s = str(args.repo)
        if repo_s not in sys.path:
            sys.path.insert(0, repo_s)
        from app.vjepa_2_1.models import vision_transformer as vit_encoder
        from evals.hub.preprocessor import vjepa2_preprocessor

        # Used only by the synthetic G0 equivalence check.  Production clips
        # use the batched GPU implementation in preprocess().
        self.official_center_processor = vjepa2_preprocessor(crop_size=CROP_SIZE)
        encoder = vit_encoder.vit_base(
            patch_size=PATCH_SIZE,
            img_size=(CROP_SIZE, CROP_SIZE),
            num_frames=64,
            tubelet_size=TUBELET_SIZE,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if CHECKPOINT_KEY not in checkpoint:
            raise KeyError(f"checkpoint has no {CHECKPOINT_KEY!r}")
        state = clean_encoder_state(checkpoint[CHECKPOINT_KEY])
        encoder.load_state_dict(state, strict=True)
        self.parameter_count = int(sum(p.numel() for p in encoder.parameters()))
        del state, checkpoint
        gc.collect()

        self.encoder = encoder.eval().to(self.device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.image_mean = torch.tensor(
            IMAGE_MEAN, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            IMAGE_STD, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.provenance = dict(provenance)

    @staticmethod
    def resized_hw(source_h: int, source_w: int) -> tuple[int, int]:
        if source_w < source_h:
            return int(SHORT_SIDE_SIZE * source_h / source_w), SHORT_SIDE_SIZE
        return SHORT_SIDE_SIZE, int(SHORT_SIDE_SIZE * source_w / source_h)

    @staticmethod
    def crop_boxes(resized_h: int, resized_w: int, mode: str) -> list[list[int]]:
        center_y = int(round((resized_h - CROP_SIZE) / 2.0))
        center_x = int(round((resized_w - CROP_SIZE) / 2.0))
        if mode == "center":
            return [[center_y, center_x, center_y + CROP_SIZE, center_x + CROP_SIZE]]
        if mode != "endpoints":
            raise ValueError(f"unknown crop mode {mode!r}")
        if resized_h == resized_w:
            box = [center_y, center_x, center_y + CROP_SIZE, center_x + CROP_SIZE]
            return [box, list(box)]
        if resized_h > resized_w:
            return [
                [0, center_x, CROP_SIZE, center_x + CROP_SIZE],
                [resized_h - CROP_SIZE, center_x, resized_h, center_x + CROP_SIZE],
            ]
        return [
            [center_y, 0, center_y + CROP_SIZE, CROP_SIZE],
            [center_y, resized_w - CROP_SIZE, center_y + CROP_SIZE, resized_w],
        ]

    def preprocess(
        self, frames_thwc: np.ndarray, *, mode: str = "endpoints"
    ) -> tuple[Any, dict[str, Any]]:
        torch = self.torch
        import torch.nn.functional as functional

        if frames_thwc.ndim != 4 or frames_thwc.shape[-1] != 3:
            raise ValueError(f"decoded frames must be [T,H,W,3], got {frames_thwc.shape}")
        if mode == "endpoints" and frames_thwc.shape[0] != NUM_FRAMES:
            raise ValueError(
                f"production extraction requires {NUM_FRAMES} frames, got {frames_thwc.shape[0]}"
            )
        if frames_thwc.dtype != np.uint8:
            raise ValueError(f"decoded frames must be uint8, got {frames_thwc.dtype}")
        source_h, source_w = int(frames_thwc.shape[1]), int(frames_thwc.shape[2])
        target_h, target_w = self.resized_hw(source_h, source_w)
        frames_tchw = (
            torch.from_numpy(np.ascontiguousarray(frames_thwc))
            .permute(0, 3, 1, 2)
            .to(self.device, dtype=torch.float32, non_blocking=True)
        )
        resized = functional.interpolate(
            frames_tchw,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        # Official tensor Resize returns uint8 before ClipToTensor.  Reproduce
        # that quantization boundary, then scale to [0,1] before normalization.
        resized = resized.round_().clamp_(0.0, 255.0).div_(255.0)
        boxes = self.crop_boxes(target_h, target_w, mode)
        views = []
        for y0, x0, y1, x1 in boxes:
            crop = resized[:, :, y0:y1, x0:x1]
            if tuple(crop.shape[-2:]) != (CROP_SIZE, CROP_SIZE):
                raise RuntimeError(f"invalid crop shape {tuple(crop.shape)}")
            normalized = (crop - self.image_mean) / self.image_std
            views.append(normalized.permute(1, 0, 2, 3))
        clip = torch.stack(views, dim=0).contiguous()
        expected_views = NUM_VIEWS if mode == "endpoints" else 1
        expected = (
            expected_views,
            3,
            int(frames_thwc.shape[0]),
            CROP_SIZE,
            CROP_SIZE,
        )
        if tuple(clip.shape) != expected:
            raise RuntimeError(f"preprocessor output {tuple(clip.shape)} != {expected}")
        if not torch.isfinite(clip).all():
            raise RuntimeError("preprocessor output contains non-finite values")
        return clip, {"resized_hw": [target_h, target_w], "view_boxes_yxyx": boxes}

    def check_preprocess_equivalence(self) -> dict[str, Any]:
        """Compare optimized GPU center crop with the official CPU transform."""
        torch = self.torch
        synthetic_t, synthetic_h, synthetic_w = 3, 521, 913
        tt = np.arange(synthetic_t, dtype=np.int64)[:, None, None, None]
        yy = np.arange(synthetic_h, dtype=np.int64)[None, :, None, None]
        xx = np.arange(synthetic_w, dtype=np.int64)[None, None, :, None]
        cc = np.arange(3, dtype=np.int64)[None, None, None, :]
        synthetic = (
            (17 * xx + 29 * yy + 43 * cc + 61 * tt + (xx * yy) % 251) % 256
        ).astype(np.uint8)
        official_input = torch.from_numpy(synthetic.copy()).permute(0, 3, 1, 2)
        official_views = self.official_center_processor(official_input)
        if not isinstance(official_views, list) or len(official_views) != 1:
            raise RuntimeError("official center processor returned an unexpected view count")
        official = official_views[0].to(self.device)
        optimized, _ = self.preprocess(synthetic, mode="center")
        torch.cuda.synchronize(self.device)
        difference = (optimized[0] - official).abs().float()
        max_abs = float(difference.max())
        mean_abs = float(difference.mean())
        passed = max_abs <= PREPROCESS_EQ_MAX_ATOL and mean_abs <= PREPROCESS_EQ_MEAN_ATOL
        return {
            "synthetic_shape_thwc": list(synthetic.shape),
            "official_shape_cthw": list(official.shape),
            "optimized_shape_cthw": list(optimized[0].shape),
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "max_abs_tolerance": PREPROCESS_EQ_MAX_ATOL,
            "mean_abs_tolerance": PREPROCESS_EQ_MEAN_ATOL,
            "passed": passed,
        }

    def encode(self, clip):
        torch = self.torch
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            tokens = self.encoder(clip)
        expected = (NUM_VIEWS, EXPECTED_TOKEN_COUNT, EMBED_DIM)
        if tuple(tokens.shape) != expected:
            raise RuntimeError(f"encoder tokens {tuple(tokens.shape)} != fixed contract {expected}")
        if not torch.isfinite(tokens).all():
            raise RuntimeError("encoder output contains non-finite values")
        peak = int(torch.cuda.max_memory_allocated(self.device))
        return tokens, peak


def tensor_stats(tensor) -> list[float]:
    torch = __import__("torch")
    values = tensor.float().reshape(-1)
    quantiles = torch.quantile(
        values, torch.tensor([0.5, 0.75, 0.9, 0.95], device=values.device)
    )
    n_top = max(1, int(math.ceil(0.10 * values.numel())))
    top = torch.topk(values, k=n_top, sorted=False).values
    return [
        float(values.mean()),
        float(values.std(unbiased=False)),
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
        float(quantiles[3]),
        float(values.max()),
        float(top.mean()),
    ]


def aggregate_tokens(tokens) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    grid = tokens.float().reshape(
        NUM_VIEWS, TOKEN_TIME, PATCH_GRID, PATCH_GRID, EMBED_DIM
    )
    # Each cached spatial cell is exactly one non-overlapping 2x2 patch mean.
    pooled = functional.avg_pool2d(
        grid.permute(0, 1, 4, 2, 3).reshape(
            NUM_VIEWS * TOKEN_TIME, EMBED_DIM, PATCH_GRID, PATCH_GRID
        ),
        kernel_size=2,
        stride=2,
    )
    dense_grid = pooled.reshape(
        NUM_VIEWS, TOKEN_TIME, EMBED_DIM, DENSE_GRID, DENSE_GRID
    ).permute(0, 1, 3, 4, 2)
    expected_dense = (NUM_VIEWS, TOKEN_TIME, DENSE_GRID, DENSE_GRID, EMBED_DIM)
    if tuple(dense_grid.shape) != expected_dense:
        raise RuntimeError(f"dense grid {tuple(dense_grid.shape)} != {expected_dense}")

    temporal_mean = grid.mean(dim=(2, 3))
    temporal_std = grid.std(dim=(2, 3), unbiased=False)
    delta = grid[:, 1:] - grid[:, :-1]
    norm_grid = functional.normalize(grid, dim=-1, eps=1e-6)
    patch_step_cosine = 1.0 - (norm_grid[:, 1:] * norm_grid[:, :-1]).sum(dim=-1)
    patch_step_l2 = torch.linalg.vector_norm(delta, dim=-1) / math.sqrt(EMBED_DIM)
    global_delta = delta.mean(dim=(2, 3), keepdim=True)
    patch_local_residual = torch.linalg.vector_norm(
        delta - global_delta, dim=-1
    ) / math.sqrt(EMBED_DIM)
    curvature = grid[:, 2:] - 2.0 * grid[:, 1:-1] + grid[:, :-2]
    patch_curvature_l2 = torch.linalg.vector_norm(
        curvature, dim=-1
    ) / math.sqrt(EMBED_DIM)

    temporal_delta = temporal_mean[:, 1:] - temporal_mean[:, :-1]
    temporal_curvature = (
        temporal_mean[:, 2:]
        - 2.0 * temporal_mean[:, 1:-1]
        + temporal_mean[:, :-2]
    )
    global_blocks = [
        ("token_mean", grid.mean(dim=(1, 2, 3))),
        ("token_std", grid.std(dim=(1, 2, 3), unbiased=False)),
        ("temporal_step_abs_mean", temporal_delta.abs().mean(dim=1)),
        ("temporal_step_std", temporal_delta.std(dim=1, unbiased=False)),
        ("temporal_curvature_abs_mean", temporal_curvature.abs().mean(dim=1)),
    ]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for _, block in global_blocks:
        block_width = int(block.shape[-1])
        offsets.append((cursor, cursor + block_width))
        cursor += block_width
    global_vector = torch.cat([block for _, block in global_blocks], dim=-1)

    maps = {
        "patch_step_cosine": patch_step_cosine,
        "patch_step_l2": patch_step_l2,
        "patch_local_residual": patch_local_residual,
        "patch_curvature_l2": patch_curvature_l2,
    }
    summary_values: list[list[float]] = [[] for _ in range(NUM_VIEWS)]
    summary_names: list[str] = []
    for map_name in MAP_NAMES:
        for view_index in range(NUM_VIEWS):
            summary_values[view_index].extend(tensor_stats(maps[map_name][view_index]))
        summary_names.extend(f"{map_name}.{stat}" for stat in STAT_NAMES)

    arrays = {
        "dense_grid": dense_grid.cpu().numpy().astype(np.float16),
        "temporal_mean": temporal_mean.cpu().numpy().astype(np.float16),
        "temporal_std": temporal_std.cpu().numpy().astype(np.float16),
        "patch_step_cosine": patch_step_cosine.cpu().numpy().astype(np.float16),
        "patch_step_l2": patch_step_l2.cpu().numpy().astype(np.float16),
        "patch_local_residual": patch_local_residual.cpu().numpy().astype(np.float16),
        "patch_curvature_l2": patch_curvature_l2.cpu().numpy().astype(np.float16),
        "global_vector": global_vector.cpu().numpy().astype(np.float32),
        "global_block_names": np.asarray([name for name, _ in global_blocks], dtype=np.str_),
        "global_block_offsets": np.asarray(offsets, dtype=np.int32),
        "map_summary": np.asarray(summary_values, dtype=np.float32),
        "map_summary_names": np.asarray(summary_names, dtype=np.str_),
    }
    meta = {
        "raw_token_shape": [NUM_VIEWS, EXPECTED_TOKEN_COUNT, EMBED_DIM],
        "token_grid_shape": [
            NUM_VIEWS,
            TOKEN_TIME,
            PATCH_GRID,
            PATCH_GRID,
            EMBED_DIM,
        ],
        "dense_grid_shape": list(expected_dense),
        "dense_pool": {
            "kernel": [1, 2, 2],
            "stride": [1, 2, 2],
            "operation": "arithmetic mean over each non-overlapping 2x2 patch block",
        },
        "global_vector_shape": list(global_vector.shape),
        "map_shapes": {name: list(maps[name].shape) for name in MAP_NAMES},
        "map_summary_shape": [NUM_VIEWS, len(MAP_NAMES) * len(STAT_NAMES)],
        "finite": True,
    }
    return arrays, meta


def atomic_write(
    json_path: Path,
    npz_path: Path,
    payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.tmp")
    tmp_json = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    artifact_id = secrets.token_hex(16)
    committed_payload = dict(payload)
    committed_payload["artifact_id"] = artifact_id
    npz_payload = dict(arrays)
    npz_payload["schema_version"] = np.asarray(SCHEMA_VERSION)
    npz_payload["artifact_id"] = np.asarray(artifact_id)
    try:
        with tmp_npz.open("wb") as f:
            np.savez_compressed(f, **npz_payload)
            f.flush()
            os.fsync(f.fileno())
        with tmp_json.open("w", encoding="utf-8") as f:
            f.write(json.dumps(json_ready(committed_payload), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        # JSON is the final commit marker.  A crash between these replaces is
        # detected by the shared artifact_id during resume validation.
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_json, json_path)
    finally:
        for path in (tmp_npz, tmp_json):
            if path.exists():
                path.unlink()


def expected_array_specs() -> dict[str, tuple[tuple[int, ...], np.dtype[Any] | str]]:
    return {
        "dense_grid": (
            (NUM_VIEWS, TOKEN_TIME, DENSE_GRID, DENSE_GRID, EMBED_DIM),
            np.dtype(np.float16),
        ),
        "temporal_mean": ((NUM_VIEWS, TOKEN_TIME, EMBED_DIM), np.dtype(np.float16)),
        "temporal_std": ((NUM_VIEWS, TOKEN_TIME, EMBED_DIM), np.dtype(np.float16)),
        "patch_step_cosine": (
            (NUM_VIEWS, TOKEN_TIME - 1, PATCH_GRID, PATCH_GRID),
            np.dtype(np.float16),
        ),
        "patch_step_l2": (
            (NUM_VIEWS, TOKEN_TIME - 1, PATCH_GRID, PATCH_GRID),
            np.dtype(np.float16),
        ),
        "patch_local_residual": (
            (NUM_VIEWS, TOKEN_TIME - 1, PATCH_GRID, PATCH_GRID),
            np.dtype(np.float16),
        ),
        "patch_curvature_l2": (
            (NUM_VIEWS, TOKEN_TIME - 2, PATCH_GRID, PATCH_GRID),
            np.dtype(np.float16),
        ),
        "global_vector": ((NUM_VIEWS, 5 * EMBED_DIM), np.dtype(np.float32)),
        "global_block_names": ((5,), "U"),
        "global_block_offsets": ((5, 2), np.dtype(np.int32)),
        "map_summary": ((NUM_VIEWS, len(MAP_NAMES) * len(STAT_NAMES)), np.dtype(np.float32)),
        "map_summary_names": ((len(MAP_NAMES) * len(STAT_NAMES),), "U"),
        "frame_indices": ((NUM_FRAMES,), np.dtype(np.int32)),
        "frame_times_seconds": ((NUM_FRAMES,), np.dtype(np.float32)),
        "tubelet_frame_indices": ((TOKEN_TIME, TUBELET_SIZE), np.dtype(np.int32)),
        "resized_hw": ((2,), np.dtype(np.int32)),
        "view_boxes_yxyx": ((NUM_VIEWS, 4), np.dtype(np.int32)),
        "schema_version": ((), "U"),
        "artifact_id": ((), "U"),
    }


def output_pair_complete(json_path: Path, npz_path: Path) -> bool:
    """Validate a committed JSON/NPZ pair before resume skips it."""
    if not json_path.is_file() or not npz_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        artifact_id = payload.get("artifact_id")
        provenance = payload.get("provenance", {})
        if (
            payload.get("status") != "ok"
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("label_accessed") is not False
            or payload.get("metric_computed") is not False
            or not isinstance(artifact_id, str)
            or len(artifact_id) != 32
            or provenance.get("repo_commit") != AUDITED_REPO_COMMIT
            or provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or provenance.get("checkpoint_hash_verified") is not True
        ):
            return False
        specs = expected_array_specs()
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(specs):
                return False
            for key, (shape, dtype) in specs.items():
                value = archive[key]
                if tuple(value.shape) != shape:
                    return False
                if isinstance(dtype, str):
                    if value.dtype.kind != dtype:
                        return False
                elif value.dtype != dtype:
                    return False
            if str(archive["schema_version"].item()) != SCHEMA_VERSION:
                return False
            if str(archive["artifact_id"].item()) != artifact_id:
                return False
            for key in (
                "dense_grid",
                "temporal_mean",
                "temporal_std",
                "patch_step_cosine",
                "patch_step_l2",
                "patch_local_residual",
                "patch_curvature_l2",
                "global_vector",
                "map_summary",
            ):
                if not np.isfinite(archive[key]).all():
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def write_error(path: Path, token: str, exc: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "sample_token": token,
        "error_type": type(exc).__name__,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def extract_once(
    args: argparse.Namespace,
    runner: PersistentEncoder,
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Decode and encode once without writing; shared by normal and blind G0 paths."""
    import torch

    t0 = time.perf_counter()
    frames, indices, fps, total_frames, backend = decode_video(path)
    decode_seconds = time.perf_counter() - t0
    source_h, source_w = int(frames.shape[1]), int(frames.shape[2])

    t0 = time.perf_counter()
    clip, preprocess_meta = runner.preprocess(frames)
    torch.cuda.synchronize(runner.device)
    preprocess_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    tokens, peak_memory = runner.encode(clip)
    torch.cuda.synchronize(runner.device)
    inference_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    arrays, tensor_meta = aggregate_tokens(tokens)
    torch.cuda.synchronize(runner.device)
    aggregate_seconds = time.perf_counter() - t0

    arrays["frame_indices"] = indices.astype(np.int32)
    arrays["frame_times_seconds"] = (
        indices.astype(np.float32) / fps
        if fps > 0
        else np.full(NUM_FRAMES, np.nan, np.float32)
    )
    arrays["tubelet_frame_indices"] = indices.reshape(
        TOKEN_TIME, TUBELET_SIZE
    ).astype(np.int32)
    arrays["resized_hw"] = np.asarray(preprocess_meta["resized_hw"], dtype=np.int32)
    arrays["view_boxes_yxyx"] = np.asarray(
        preprocess_meta["view_boxes_yxyx"], dtype=np.int32
    )
    del frames, clip, tokens
    run_meta = {
        "frame_indices": indices,
        "fps": fps,
        "total_frames": total_frames,
        "backend": backend,
        "source_hw": [source_h, source_w],
        "resized_hw": preprocess_meta["resized_hw"],
        "view_boxes_yxyx": preprocess_meta["view_boxes_yxyx"],
        "decode_seconds": decode_seconds,
        "preprocess_seconds": preprocess_seconds,
        "inference_seconds": inference_seconds,
        "aggregate_seconds": aggregate_seconds,
        "peak_memory": peak_memory,
    }
    return arrays, tensor_meta, run_meta


def process_video(
    args: argparse.Namespace,
    runner: PersistentEncoder,
    provenance: Mapping[str, Any],
    row: Mapping[str, str],
    path: Path,
) -> tuple[str, float, dict[str, Any]]:
    token = row["sample_token"]
    json_path = args.output_root / f"{token}.json"
    npz_path = args.output_root / f"{token}.npz"
    error_path = args.output_root / f"{token}.error.json"
    if not args.overwrite and output_pair_complete(json_path, npz_path):
        return "skipped", 0.0, {}

    t_start = time.perf_counter()
    arrays, tensor_meta, run_meta = extract_once(args, runner, path)
    total_seconds_before_write = time.perf_counter() - t_start
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "sample_token": token,
        "selection": {
            "scope": "s0_discovery" if args.discovery_ids_sha256 else "technical_subset",
            "discovery_ids_sha256": args.discovery_ids_sha256,
            "sample_token_formula": "sha256(b'e50-sample-v1\\0' + video.encode('utf-8'))",
        },
        "provenance": dict(provenance),
        "model": {
            "official_name": MODEL_NAME,
            "parameter_count": runner.parameter_count,
            "checkpoint_key": CHECKPOINT_KEY,
            "encoder_forward": "encoder.eval(); encoder(x); training argument omitted",
            "amp_dtype": AMP_DTYPE,
        },
        "sampling": {
            "method": "round(linspace(0, total_frames-1, 64))",
            "num_frames": NUM_FRAMES,
            "frame_indices": run_meta["frame_indices"],
            "source_total_frames": run_meta["total_frames"],
            "source_fps": run_meta["fps"],
            "decode_backend": run_meta["backend"],
        },
        "preprocessing": {
            "source_hw": run_meta["source_hw"],
            "resized_hw": run_meta["resized_hw"],
            "view_boxes_yxyx": run_meta["view_boxes_yxyx"],
            "short_side_size": SHORT_SIDE_SIZE,
            "crop_size": CROP_SIZE,
            "views": NUM_VIEWS,
            "operation": (
                "GPU float32 bilinear resize with antialias to short side 438; "
                "round/clamp to reproduce official uint8 boundary; long-side "
                "endpoint crops; ImageNet mean/std normalization"
            ),
            "synthetic_official_center_equivalence": runner.preprocess_equivalence,
        },
        "tensors": tensor_meta,
        "timing_seconds": {
            "decode": run_meta["decode_seconds"],
            "preprocess": run_meta["preprocess_seconds"],
            "encoder": run_meta["inference_seconds"],
            "aggregate": run_meta["aggregate_seconds"],
            "before_write": total_seconds_before_write,
        },
        "cuda_peak_allocated_bytes": run_meta["peak_memory"],
        "label_accessed": False,
        "metric_computed": False,
    }
    t0 = time.perf_counter()
    atomic_write(json_path, npz_path, payload, arrays)
    write_seconds = time.perf_counter() - t0
    if error_path.exists():
        error_path.unlink()
    total_seconds = time.perf_counter() - t_start
    return "done", total_seconds, {
        "encoder_seconds": run_meta["inference_seconds"],
        "write_seconds": write_seconds,
        "peak_memory": run_meta["peak_memory"],
        "npz_bytes": npz_path.stat().st_size,
        "tensor_meta": tensor_meta,
    }


def select_g0_by_video_sha(
    resolved: Sequence[tuple[dict[str, str], Path | None, list[Path]]]
) -> list[tuple[dict[str, str], Path]]:
    """Hash all train video bytes and return the first three, without logging IDs."""
    ranked: list[tuple[str, int, dict[str, str], Path]] = []
    total = len(resolved)
    for index, (row, path, _) in enumerate(resolved, 1):
        if path is None:
            raise FileNotFoundError("G0 requires every selected train video to resolve")
        digest = sha256_file(path)
        ranked.append((digest, index, row, path))
        if index % 500 == 0 or index == total:
            print(f"G0_HASH_PROGRESS {index}/{total}", flush=True)
    ranked.sort(key=lambda item: (item[0], item[1]))
    if len(ranked) < 3:
        raise ValueError("G0 requires at least three train videos")
    return [(row, path) for _, _, row, path in ranked[:3]]


def compare_repeats(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> tuple[bool, bool, list[str]]:
    failures: list[str] = []
    if set(first) != set(second):
        return False, False, ["array_key_set"]
    exact = True
    close = True
    for key in sorted(first):
        left, right = np.asarray(first[key]), np.asarray(second[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            exact = close = False
            failures.append(f"{key}:shape_or_dtype")
            continue
        if np.issubdtype(left.dtype, np.number):
            key_exact = np.array_equal(left, right, equal_nan=True)
            key_close = np.allclose(left, right, rtol=1e-5, atol=1e-6, equal_nan=True)
        else:
            key_exact = np.array_equal(left, right)
            key_close = key_exact
        exact &= bool(key_exact)
        close &= bool(key_close)
        if not key_close:
            failures.append(f"{key}:not_allclose")
        elif not key_exact:
            failures.append(f"{key}:not_exact")
    return exact, close, failures


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(json_ready(payload), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_g0(
    args: argparse.Namespace,
    runner: PersistentEncoder,
    provenance: Mapping[str, Any],
    selected: Sequence[tuple[dict[str, str], Path]],
) -> int:
    """Run the preregistered blind reproducibility/non-collapse gate."""
    start = time.perf_counter()
    g0_root = args.output_root / "g0"
    exact_checks: list[bool] = []
    close_checks: list[bool] = []
    finite_checks: list[bool] = []
    signatures: list[np.ndarray] = []
    repeat_failures: list[list[str]] = []
    peaks: list[int] = []

    for anonymous_index, (_, path) in enumerate(selected):
        anonymous = f"anon_{anonymous_index:02d}"
        try:
            first, tensor_meta, first_meta = extract_once(args, runner, path)
            second, second_tensor_meta, second_meta = extract_once(args, runner, path)
        except Exception as exc:
            atomic_json(
                g0_root / f"{anonymous}.error.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "anonymous_id": anonymous,
                    "error_type": type(exc).__name__,
                    "label_accessed": False,
                    "metric_computed": False,
                },
            )
            print(
                f"G0_ITEM_FAIL anonymous_index={anonymous_index} "
                f"error_type={type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        exact, close, failures = compare_repeats(first, second)
        exact_checks.append(exact)
        close_checks.append(close)
        repeat_failures.append(failures)
        peaks.extend([int(first_meta["peak_memory"]), int(second_meta["peak_memory"])])
        finite = True
        for key in (
            "dense_grid",
            "temporal_mean",
            "temporal_std",
            "patch_step_cosine",
            "patch_step_l2",
            "patch_local_residual",
            "patch_curvature_l2",
            "global_vector",
            "map_summary",
        ):
            finite &= bool(np.isfinite(first[key]).all())
        finite_checks.append(finite)
        signatures.append(
            np.concatenate(
                [
                    first["global_vector"].astype(np.float64).reshape(-1),
                    first["map_summary"].astype(np.float64).reshape(-1),
                ]
            )
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "g0_anonymous": True,
            "anonymous_id": anonymous,
            "provenance": dict(provenance),
            "fixed_contract": {
                "frames": NUM_FRAMES,
                "raw_tokens": [NUM_VIEWS, EXPECTED_TOKEN_COUNT, EMBED_DIM],
                "token_grid": [
                    NUM_VIEWS, TOKEN_TIME, PATCH_GRID, PATCH_GRID, EMBED_DIM
                ],
                "dense_grid": [
                    NUM_VIEWS, TOKEN_TIME, DENSE_GRID, DENSE_GRID, EMBED_DIM
                ],
            },
            "first_tensor_meta": tensor_meta,
            "second_tensor_meta": second_tensor_meta,
            "repeat_exact": exact,
            "repeat_allclose_rtol1e5_atol1e6": close,
            "repeat_failures": failures,
            "finite": finite,
            "label_accessed": False,
            "metric_computed": False,
        }
        atomic_write(
            g0_root / f"{anonymous}.json",
            g0_root / f"{anonymous}.npz",
            payload,
            first,
        )
        del first, second

    pairwise_nonconstant: list[bool] = []
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            pairwise_nonconstant.append(
                not np.allclose(signatures[i], signatures[j], rtol=1e-5, atol=1e-6)
            )
    exact_all = bool(all(exact_checks))
    close_all = bool(all(close_checks))
    finite_all = bool(all(finite_checks))
    nonconstant_all_pairs = bool(all(pairwise_nonconstant))
    preprocess_equivalence_passed = bool(runner.preprocess_equivalence["passed"])
    passed = (
        exact_all
        and close_all
        and finite_all
        and nonconstant_all_pairs
        and preprocess_equivalence_passed
    )
    elapsed = time.perf_counter() - start
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "selection": "first 3 train videos by byte SHA-256; tie-break by manifest order",
        "selected_count": 3,
        "repeat_count_per_video": 2,
        "anonymous_ids": [f"anon_{index:02d}" for index in range(3)],
        "exact_all_arrays": exact_all,
        "allclose_all_arrays": close_all,
        "allclose_rtol": 1e-5,
        "allclose_atol": 1e-6,
        "finite_all_arrays": finite_all,
        "nonconstant_all_pairs": nonconstant_all_pairs,
        "repeat_failures": repeat_failures,
        "synthetic_official_center_equivalence": runner.preprocess_equivalence,
        "fixed_shapes": {
            "raw_tokens": [NUM_VIEWS, EXPECTED_TOKEN_COUNT, EMBED_DIM],
            "token_grid": [
                NUM_VIEWS, TOKEN_TIME, PATCH_GRID, PATCH_GRID, EMBED_DIM
            ],
            "dense_grid": [
                NUM_VIEWS, TOKEN_TIME, DENSE_GRID, DENSE_GRID, EMBED_DIM
            ],
        },
        "aggregate": {
            "elapsed_seconds": elapsed,
            "max_cuda_peak_allocated_bytes": max(peaks),
        },
        "label_accessed": False,
        "metric_computed": False,
    }
    atomic_json(g0_root / "g0_summary.json", summary)
    print(
        "G0_RESULT "
        f"selected=3 repeats=2 exact={str(exact_all).lower()} "
        f"allclose={str(close_all).lower()} finite={str(finite_all).lower()} "
        f"nonconstant={str(nonconstant_all_pairs).lower()} "
        f"preprocess_equivalent={str(preprocess_equivalence_passed).lower()} "
        f"preprocess_max_abs={runner.preprocess_equivalence['max_abs']:.8f} "
        f"preprocess_mean_abs={runner.preprocess_equivalence['mean_abs']:.8f} "
        f"raw_shape={NUM_VIEWS}x{EXPECTED_TOKEN_COUNT}x{EMBED_DIM} "
        f"dense_shape={NUM_VIEWS}x{TOKEN_TIME}x{DENSE_GRID}x{DENSE_GRID}x{EMBED_DIM} "
        f"elapsed={elapsed:.1f}s max_peak={max(peaks)/2**30:.2f}GiB "
        f"status={'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return 0 if passed else 1


def main() -> int:
    args = parse_args()
    if args.max_failures <= 0:
        raise ValueError("--max-failures must be > 0")
    all_rows = load_manifest(args.manifest)
    if args.g0:
        if args.limit or args.offset or args.only_token or args.discovery_ids:
            raise ValueError(
                "--g0 fixes its own SHA-ranked 3-video subset; no discovery/limit/offset/only-token"
            )
        args.discovery_ids_sha256 = None
        rows = all_rows
    else:
        if args.discovery_ids is None and args.limit == 0:
            raise ValueError(
                "an unbounded E51 extraction requires --discovery-ids; "
                "use a positive --limit only for technical smoke"
            )
        discovery_ids = None
        args.discovery_ids_sha256 = None
        if args.discovery_ids is not None:
            discovery_ids, args.discovery_ids_sha256 = load_discovery_ids(
                args.discovery_ids
            )
        rows = select_rows(args, all_rows, discovery_ids)
    path_maps = parse_path_maps(args.path_map)
    resolved = [
        (row, *resolve_video(row, path_maps, args.videos_root)) for row in rows
    ]
    missing = [item for item in resolved if item[1] is None]
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "selected": len(rows),
                "resolved": len(rows) - len(missing),
                "missing": len(missing),
                "discovery_membership": args.discovery_ids_sha256 is not None,
                "dry_run": args.dry_run,
                "g0": args.g0,
                "label_accessed": False,
                "fixed_contract": {
                    "frames": NUM_FRAMES,
                    "raw_tokens": [NUM_VIEWS, EXPECTED_TOKEN_COUNT, EMBED_DIM],
                    "token_grid": [
                        NUM_VIEWS, TOKEN_TIME, PATCH_GRID, PATCH_GRID, EMBED_DIM
                    ],
                    "dense_grid": [
                        NUM_VIEWS, TOKEN_TIME, DENSE_GRID, DENSE_GRID, EMBED_DIM
                    ],
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected train video(s) unresolved; fix --path-map before loading model"
        )

    if args.g0:
        try:
            g0_selected = select_g0_by_video_sha(resolved)
        except Exception as exc:
            print(
                f"G0_SELECTION_FAIL error_type={type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            return 1
    else:
        g0_selected = []
    if args.dry_run:
        if args.g0:
            print("G0_DRY_RUN selected=3 hashes_complete=true label_accessed=false")
        return 0

    args.output_root = args.output_root.expanduser().resolve()
    provenance = validate_sources(args)
    if not args.g0:
        todo = []
        for row, path, _ in resolved:
            stem = row["sample_token"]
            if not args.overwrite and output_pair_complete(
                args.output_root / f"{stem}.json",
                args.output_root / f"{stem}.npz",
            ):
                continue
            todo.append((row, path))
        if not todo:
            print(f"E51_VJEPA21_DONE done=0 skipped={len(rows)} failed=0")
            return 0

    print(
        f"loading {MODEL_NAME} once on {args.device}",
        flush=True,
    )
    load_start = time.perf_counter()
    runner = PersistentEncoder(args, provenance)
    load_seconds = time.perf_counter() - load_start
    print(
        f"model_ready params={runner.parameter_count} load_seconds={load_seconds:.1f}",
        flush=True,
    )
    preprocess_equivalence = runner.check_preprocess_equivalence()
    runner.preprocess_equivalence = preprocess_equivalence
    print(
        "PREPROCESS_EQUIVALENCE "
        f"max_abs={preprocess_equivalence['max_abs']:.8f} "
        f"mean_abs={preprocess_equivalence['mean_abs']:.8f} "
        f"max_tol={preprocess_equivalence['max_abs_tolerance']:.8f} "
        f"mean_tol={preprocess_equivalence['mean_abs_tolerance']:.8f} "
        f"status={'PASS' if preprocess_equivalence['passed'] else 'FAIL'}",
        flush=True,
    )
    if not preprocess_equivalence["passed"]:
        raise RuntimeError("optimized GPU preprocessing failed official equivalence gate")
    if args.g0:
        try:
            return run_g0(args, runner, provenance, g0_selected)
        except Exception as exc:
            print(
                f"G0_RUNTIME_FAIL error_type={type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            return 1

    done = skipped = failed = 0
    elapsed_done = 0.0
    encoder_done = 0.0
    npz_bytes_done = 0
    max_peak_memory = 0
    for index, (row, path, _) in enumerate(resolved, 1):
        stem = row["sample_token"]
        error_path = args.output_root / f"{stem}.error.json"
        try:
            status, seconds, detail = process_video(
                args, runner, provenance, row, path
            )
            if status == "skipped":
                skipped += 1
            else:
                done += 1
                elapsed_done += seconds
            if detail:
                encoder_done += float(detail["encoder_seconds"])
                npz_bytes_done += int(detail["npz_bytes"])
                max_peak_memory = max(max_peak_memory, int(detail["peak_memory"]))
            if index % 100 == 0 or index == len(resolved):
                avg = elapsed_done / max(done, 1)
                eta_min = avg * max(len(resolved) - index, 0) / 60.0
                print(
                    "E51_PROGRESS "
                    f"processed={index}/{len(resolved)} done={done} skipped={skipped} "
                    f"failed={failed} mean_seconds={avg:.3f} eta_minutes={eta_min:.1f} "
                    f"max_peak_gib={max_peak_memory/2**30:.3f} "
                    f"total_npz_mib={npz_bytes_done/2**20:.2f}",
                    flush=True,
                )
        except Exception as exc:
            failed += 1
            write_error(error_path, row["sample_token"], exc)
            print(
                f"E51_ITEM_FAIL processed={index}/{len(resolved)} "
                f"error_type={type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            if failed >= args.max_failures:
                print(f"stopping after {failed} failures", file=sys.stderr)
                break
    print(
        f"E51_VJEPA21_DONE done={done} skipped={skipped} failed={failed} "
        f"model_load_seconds={load_seconds:.1f}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
