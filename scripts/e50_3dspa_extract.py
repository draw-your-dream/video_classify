#!/usr/bin/env python3
"""Prepare compact, per-video E50 features from the official 3DSPA CLI.

This script is deliberately an extractor, not an evaluator.  It defaults to
``splits/train_v3.jsonl`` and refuses a manifest whose name contains ``eval``
unless ``--allow-eval`` is explicitly supplied.  It never reads E18 scores or
computes a release metric.

The upstream repository currently has an important interface limitation (at
commit 3c73353): ``inference.py`` instantiates the 2-D ``TrackAutoEncoder``;
DINO and depth are extracted and saved but are not passed to that model.  We
therefore label the three outputs honestly:

* ``trajectory2d``: official 2-D reconstruction plus track kinematics;
* ``semantic2d_posthoc``: DINO features sampled along observed/predicted tracks;
* ``depth3d_posthoc``: VDA depth sampled along tracks and lifted-track dynamics.

They are useful fixed ablations, but the latter two are not claimed to be a
full TrackAutoEncoder3D forward pass.  See docs/e50_3dspa_integration.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "e50_trajan_features_v2"
AUDITED_UPSTREAM_COMMIT = "3c73353bfa26bd83e856bfe05c72efecee4ed284"
OFFICIAL_TRAJAN_CHECKPOINT_SHA256 = (
    "05743748c50e3b8456e4f36f7d55bc652012bd14dc58c7eb66e9ad63cf9f0cae"
)
COORDINATE_CONTRACT = "trajan_unit_to_raster_v1"
TAPVID_METRIC_SIZE_WH = (256.0, 256.0)
TAPVID_THRESHOLDS = (1, 2, 4, 8, 16)
GROUP_ORDER = ("trajectory2d", "semantic2d_posthoc", "depth3d_posthoc")
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "splits" / "train_v3.jsonl",
        help="JSONL with label/video/abs_path. Defaults to train only.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "cache" / "e50_3dspa",
    )
    ap.add_argument(
        "--official-repo",
        type=Path,
        default=Path(os.environ["THREEDSPA_REPO"])
        if os.environ.get("THREEDSPA_REPO")
        else None,
        help="Checkout of https://github.com/TheProParadox/3dspa_code.",
    )
    ap.add_argument("--checkpoint-path", type=Path)
    ap.add_argument("--vda-model-path", type=Path)
    ap.add_argument(
        "--official-python",
        default=sys.executable,
        help="Python interpreter containing JAX/PyTorch/3DSPA dependencies.",
    )
    ap.add_argument(
        "--ablation-mode",
        choices=("trajectory", "semantic2d", "depth3d", "all"),
        default="trajectory",
        help="Which raw upstream signals to request and compact.",
    )
    ap.add_argument(
        "--runner-mode",
        choices=("persistent", "legacy-subprocess"),
        default="persistent",
        help=(
            "Persistent keeps CoTracker and TRAJAN/JAX weights resident. "
            "legacy-subprocess is retained only to hard-fail old outputs."
        ),
    )
    ap.add_argument("--num-output-frames", type=int, default=150)
    ap.add_argument("--num-query-points", type=int, default=512)
    ap.add_argument("--num-support-tracks", type=int, default=2048)
    ap.add_argument("--tracking-grid-size", type=int, default=64)
    ap.add_argument("--dino-model", default="facebook/dinov2-base")
    ap.add_argument("--limit", type=int, default=0, help="0 means all selected rows.")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--only-video",
        action="append",
        default=[],
        help="Process only these manifest video names; repeatable.",
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
        help="Fallback corpus root. Tries video, label/video, and suffix after data/s3.",
    )
    ap.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Do not invoke upstream; read predictions.npz from --predictions-root.",
    )
    ap.add_argument(
        "--predictions-root",
        type=Path,
        help="Existing upstream outputs for --postprocess-only.",
    )
    ap.add_argument("--keep-official-npz", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--max-failures", type=int, default=10)
    ap.add_argument("--allow-eval", action="store_true")
    ap.add_argument("--allow-unpinned-upstream", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve rows and paths but load no model and write no cache.",
    )
    return ap.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_label(label: str) -> str:
    if not label or label in {".", ".."} or Path(label).name != label:
        raise ValueError(f"unsafe label for cache path: {label!r}")
    return label


def load_manifest(path: Path, allow_eval: bool) -> list[dict[str, str]]:
    resolved = path.expanduser().resolve()
    if "eval" in resolved.name.lower() and not allow_eval:
        raise ValueError(
            f"refusing eval-like manifest {resolved}; use --allow-eval only after promotion"
        )
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with resolved.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = {"label", "video", "abs_path"} - set(item)
            if missing:
                raise ValueError(f"{resolved}:{lineno}: missing keys {sorted(missing)}")
            row = {k: str(item[k]) for k in ("label", "video", "abs_path")}
            _safe_label(row["label"])
            key = (row["label"], Path(row["video"]).stem)
            if key in seen:
                raise ValueError(f"duplicate output key in manifest: {key}")
            seen.add(key)
            rows.append(row)
    return rows


def parse_path_maps(values: Sequence[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--path-map must be OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(f"--path-map must have non-empty sides, got {value!r}")
        pairs.append((Path(old).expanduser(), Path(new).expanduser()))
    return pairs


def _suffix_after_data_s3(path: Path) -> Path | None:
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
        candidates.extend(
            [root / row["video"], root / row["label"] / row["video"]]
        )
        suffix = _suffix_after_data_s3(original)
        if suffix is not None:
            candidates.append(root / suffix)
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def resolve_video_path(
    row: Mapping[str, str],
    path_maps: Sequence[tuple[Path, Path]],
    videos_root: Path | None,
) -> tuple[Path | None, list[Path]]:
    candidates = video_candidates(row, path_maps, videos_root)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), candidates
    return None, candidates


def _git_commit(repo: Path) -> str | None:
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


def audit_official_repo(repo: Path, allow_unpinned: bool) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    required = ["inference.py", "track_autoencoder.py", "track_autoencoder_3d.py"]
    missing = [name for name in required if not (repo / name).is_file()]
    if missing:
        raise FileNotFoundError(f"official repo missing files: {missing}")
    commit = _git_commit(repo)
    if commit != AUDITED_UPSTREAM_COMMIT and not allow_unpinned:
        raise RuntimeError(
            "upstream commit differs from the audited interface: "
            f"got {commit!r}, expected {AUDITED_UPSTREAM_COMMIT}; "
            "inspect the diff or pass --allow-unpinned-upstream"
        )
    source = (repo / "inference.py").read_text(encoding="utf-8")
    uses_2d_model = "track_autoencoder.TrackAutoEncoder(" in source
    batch_start = source.find("batch = {")
    batch_end = source.find("}", batch_start)
    batch_source = source[batch_start:batch_end] if batch_start >= 0 else ""
    semantic_in_model_batch = "dino_features" in batch_source
    depth_in_model_batch = "depth_features" in batch_source
    return {
        "repo": str(repo),
        "commit": commit,
        "audited_commit": AUDITED_UPSTREAM_COMMIT,
        "uses_2d_track_autoencoder": uses_2d_model,
        "dino_passed_to_model": semantic_in_model_batch,
        "depth_passed_to_model": depth_in_model_batch,
        "full_3dspa_forward": bool(
            not uses_2d_model and semantic_in_model_batch and depth_in_model_batch
        ),
    }


def preflight_python(python: str, need_dino: bool, need_depth: bool) -> None:
    modules = ["numpy", "jax", "flax", "torch", "cv2", "absl"]
    if need_dino:
        modules.extend(["transformers", "torchvision"])
    code = "\n".join(f"import {name}" for name in modules)
    result = subprocess.run(
        [python, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"official Python dependency preflight failed ({python}):\n{result.stdout[-4000:]}"
        )
    if need_depth:
        # The VDA module is imported by upstream only after its checkout is put
        # on sys.path, so its existence is checked separately by the caller.
        return


def _finite(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), arr.shape)
    return arr[valid]


def add_stats(
    out: dict[str, float],
    prefix: str,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    vals = _finite(values, mask)
    names = ("mean", "std", "p50", "p75", "p90", "p95", "max", "top5_mean")
    if vals.size == 0:
        out.update({f"{prefix}_{name}": float("nan") for name in names})
        return
    q = np.percentile(vals, [50, 75, 90, 95])
    n_top = max(1, int(math.ceil(0.05 * vals.size)))
    top = np.partition(vals, vals.size - n_top)[-n_top:]
    values_out = (
        float(vals.mean()),
        float(vals.std()),
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
        float(vals.max()),
        float(top.mean()),
    )
    out.update({f"{prefix}_{name}": value for name, value in zip(names, values_out)})


def masked_axis_mean(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(vals)
    total = np.where(valid, vals, 0.0).sum(axis=axis)
    count = valid.sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=count > 0,
    )


def sigmoid(logits: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _npz_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape != ():
        raise ValueError(f"expected scalar string, got shape {arr.shape}")
    return str(arr.item())


def _require_finite(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        bad = int(arr.size - np.isfinite(arr).sum())
        raise ValueError(f"{name} contains {bad}/{arr.size} NaN/Inf values")
    return arr


def frame_hw(data: Mapping[str, np.ndarray]) -> tuple[int, int]:
    if "frame_size_hw" in data:
        shape = np.asarray(data["frame_size_hw"], dtype=np.int64).reshape(-1)
        if shape.size != 2 or np.any(shape <= 0):
            raise ValueError(f"invalid frame_size_hw: {shape.tolist()}")
        return int(shape[0]), int(shape[1])
    if "video" in data:
        return video_hw(np.asarray(data["video"]))
    raise KeyError("predictions must contain frame_size_hw or video")


def _check_unit_tracks(name: str, tracks: np.ndarray, visible: np.ndarray) -> None:
    arr = _require_finite(name, tracks).astype(np.float64, copy=False)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"{name} must be [N,T,2], got {arr.shape}")
    if np.max(np.abs(arr)) > 4.0:
        raise ValueError(
            f"{name} is not in TRAJAN unit coordinates: max_abs={np.max(np.abs(arr)):.6g}"
        )
    mask = np.broadcast_to(np.asarray(visible, dtype=bool)[..., None], arr.shape)
    vals = arr[mask]
    if vals.size:
        in_frame = (vals >= -0.25) & (vals <= 1.25)
        if float(in_frame.mean()) < 0.90:
            raise ValueError(
                f"{name} unit-domain coverage is only {float(in_frame.mean()):.4f}"
            )


def _assert_coordinate_contract(
    data: Mapping[str, np.ndarray],
    pred_px: np.ndarray,
    gt_px: np.ndarray,
    pred_unit: np.ndarray,
    gt_unit: np.ndarray,
    visible: np.ndarray,
    image_hw: tuple[int, int],
) -> None:
    if "coordinate_contract" not in data:
        raise ValueError(
            "missing coordinate_contract; refusing legacy pixel-input TRAJAN output"
        )
    got = _npz_string(data["coordinate_contract"])
    if got != COORDINATE_CONTRACT:
        raise ValueError(
            f"coordinate contract mismatch: got {got!r}, expected {COORDINATE_CONTRACT!r}"
        )
    _check_unit_tracks("pred_tracks_2d_normalized", pred_unit, visible)
    _check_unit_tracks("gt_query_2d_normalized", gt_unit, visible)
    h, w = image_hw
    scale = np.asarray([w, h], dtype=np.float64)
    if not np.allclose(pred_px, pred_unit * scale, rtol=0.0, atol=2e-3):
        delta = float(np.max(np.abs(pred_px - pred_unit * scale)))
        raise ValueError(f"pred normalized->pixel contract failed: max_delta={delta:.6g}")
    if not np.allclose(gt_px, gt_unit * scale, rtol=0.0, atol=2e-3):
        delta = float(np.max(np.abs(gt_px - gt_unit * scale)))
        raise ValueError(f"gt normalized->pixel contract failed: max_delta={delta:.6g}")
    if float(np.ptp(pred_unit)) <= 1e-7:
        raise ValueError("TRAJAN predictions collapsed to a within-video constant")


def tapvid_average_jaccard(
    query_points_txy: np.ndarray,
    gt_visible: np.ndarray,
    pred_visible: np.ndarray,
    gt_tracks_unit: np.ndarray,
    pred_tracks_unit: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Official TAP-Vid Jaccard on a fixed 256x256 metric grid.

    The TRAJAN model consumes/returns unit ``(x, y)`` coordinates.  TAP-Vid's
    thresholds are raster-pixel thresholds at the standard 256x256 evaluation
    resolution, so both tracks are mapped to that grid before applying the
    official TP / (GT positives + false positives) definition.  ``strided``
    query semantics exclude each track's actual query frame.
    """
    query = _require_finite("query_points", query_points_txy).astype(
        np.float64, copy=False
    )
    gt_vis = np.asarray(gt_visible, dtype=bool)
    pred_vis = np.asarray(pred_visible, dtype=bool)
    gt = _require_finite("gt_tracks_unit", gt_tracks_unit).astype(
        np.float64, copy=False
    )
    pred = _require_finite("pred_tracks_unit", pred_tracks_unit).astype(
        np.float64, copy=False
    )
    if gt.shape != pred.shape or gt.ndim != 3 or gt.shape[-1] != 2:
        raise ValueError(f"track shape mismatch: gt={gt.shape}, pred={pred.shape}")
    n, t_len, _ = gt.shape
    if gt_vis.shape != (n, t_len) or pred_vis.shape != (n, t_len):
        raise ValueError(
            f"visibility shape mismatch: gt={gt_vis.shape}, pred={pred_vis.shape}, "
            f"tracks={(n, t_len)}"
        )
    if query.shape != (n, 3):
        raise ValueError(f"query_points must be [N,3] (t,x,y), got {query.shape}")
    query_frame = np.rint(query[:, 0]).astype(np.int64)
    if np.any(query_frame < 0) or np.any(query_frame >= t_len):
        raise ValueError(
            f"query frame outside [0,{t_len}): "
            f"min={int(query_frame.min())}, max={int(query_frame.max())}"
        )
    if np.max(np.abs(query[:, 1:])) > 4.0:
        raise ValueError("query_points x/y are not in TRAJAN unit coordinates")

    evaluation_points = np.ones((n, t_len), dtype=bool)
    evaluation_points[np.arange(n), query_frame] = False
    scale = np.asarray(TAPVID_METRIC_SIZE_WH, dtype=np.float64)
    squared_error = np.sum(np.square(pred * scale - gt * scale), axis=-1)
    values: list[float] = []
    for threshold in TAPVID_THRESHOLDS:
        within = squared_error < float(threshold * threshold)
        is_correct = within & gt_vis
        true_positives = int(
            np.sum(is_correct & pred_vis & evaluation_points, dtype=np.int64)
        )
        gt_positives = int(np.sum(gt_vis & evaluation_points, dtype=np.int64))
        false_positives = ((~gt_vis) & pred_vis) | ((~within) & pred_vis)
        false_positives_n = int(
            np.sum(false_positives & evaluation_points, dtype=np.int64)
        )
        denominator = gt_positives + false_positives_n
        if denominator <= 0:
            raise ValueError(f"TAP-Vid Jaccard denominator is zero at {threshold}px")
        values.append(true_positives / denominator)
    by_threshold = np.asarray(values, dtype=np.float64)
    average = float(by_threshold.mean())
    if not math.isfinite(average) or not 0.0 <= average <= 1.0:
        raise ValueError(f"invalid TAP-Vid average Jaccard: {average}")
    return average, by_threshold


def video_hw(video: np.ndarray) -> tuple[int, int]:
    if video.ndim != 4:
        raise ValueError(f"video must be rank 4, got {video.shape}")
    if video.shape[1] in (1, 3, 4):  # upstream stores [T,C,H,W]
        return int(video.shape[2]), int(video.shape[3])
    if video.shape[-1] in (1, 3, 4):
        return int(video.shape[1]), int(video.shape[2])
    raise ValueError(f"cannot infer video layout from {video.shape}")


def _ensure_nt2(name: str, arr: np.ndarray) -> np.ndarray:
    value = np.asarray(arr, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 2:
        raise ValueError(f"{name} must be [N,T,2], got {value.shape}")
    return value


def similarity_residual_series(
    points_ntd: np.ndarray, visible_nt: np.ndarray
) -> np.ndarray:
    """Consecutive-frame Umeyama similarity residual, normalized by target RMS."""
    points = np.asarray(points_ntd, dtype=np.float64)
    visible = np.asarray(visible_nt, dtype=bool)
    n, t_len, dim = points.shape
    out = np.full(max(t_len - 1, 0), np.nan, dtype=np.float64)
    for t in range(t_len - 1):
        ok = visible[:, t] & visible[:, t + 1]
        if ok.sum() < dim + 2:
            continue
        src = points[ok, t]
        dst = points[ok, t + 1]
        src0 = src - src.mean(axis=0, keepdims=True)
        dst0 = dst - dst.mean(axis=0, keepdims=True)
        var_src = float(np.mean(np.sum(src0 * src0, axis=1)))
        if var_src <= EPS:
            continue
        cov = dst0.T @ src0 / src0.shape[0]
        u, singular, vt = np.linalg.svd(cov, full_matrices=False)
        sign = np.ones(dim)
        if np.linalg.det(u @ vt) < 0:
            sign[-1] = -1.0
        rot = u @ np.diag(sign) @ vt
        scale = float(np.sum(singular * sign) / var_src)
        pred0 = scale * (src0 @ rot.T)
        residual = np.sqrt(np.mean(np.sum((pred0 - dst0) ** 2, axis=1)))
        target_rms = np.sqrt(np.mean(np.sum(dst0 * dst0, axis=1)))
        out[t] = residual / (target_rms + EPS)
    return out


def pair_stretch_series(
    points_ntd: np.ndarray, visible_nt: np.ndarray, max_points: int = 64
) -> np.ndarray:
    """Median consecutive-frame |log pair-distance ratio| on a fixed subset."""
    points = np.asarray(points_ntd, dtype=np.float64)
    visible = np.asarray(visible_nt, dtype=bool)
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=int)
        points = points[idx]
        visible = visible[idx]
    n, t_len, _ = points.shape
    iu = np.triu_indices(n, 1)
    out = np.full(max(t_len - 1, 0), np.nan, dtype=np.float64)
    for t in range(t_len - 1):
        valid_points = visible[:, t] & visible[:, t + 1]
        pair_ok = valid_points[iu[0]] & valid_points[iu[1]]
        if pair_ok.sum() < 8:
            continue
        d0 = np.linalg.norm(points[iu[0], t] - points[iu[1], t], axis=-1)
        d1 = np.linalg.norm(
            points[iu[0], t + 1] - points[iu[1], t + 1], axis=-1
        )
        good = pair_ok & (d0 > EPS) & (d1 > EPS)
        if good.sum() >= 8:
            out[t] = float(np.median(np.abs(np.log(d1[good] / d0[good]))))
    return out


def kinematic_features(
    points_ntd: np.ndarray,
    visible_nt: np.ndarray,
    prefix: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    points = np.asarray(points_ntd, dtype=np.float64)
    visible = np.asarray(visible_nt, dtype=bool)
    visible &= np.all(np.isfinite(points), axis=-1)
    feats: dict[str, float] = {}
    if points.shape[1] < 2:
        add_stats(feats, f"{prefix}_speed", np.array([]))
        add_stats(feats, f"{prefix}_accel", np.array([]))
        add_stats(feats, f"{prefix}_similarity_resid", np.array([]))
        add_stats(feats, f"{prefix}_pair_stretch", np.array([]))
        return feats, {}
    delta = np.diff(points, axis=1)
    speed = np.linalg.norm(delta, axis=-1)
    speed_valid = visible[:, :-1] & visible[:, 1:]
    accel = np.linalg.norm(np.diff(delta, axis=1), axis=-1)
    accel_valid = visible[:, :-2] & visible[:, 1:-1] & visible[:, 2:]
    rigid = similarity_residual_series(points, visible)
    stretch = pair_stretch_series(points, visible)
    add_stats(feats, f"{prefix}_speed", speed, speed_valid)
    add_stats(feats, f"{prefix}_accel", accel, accel_valid)
    add_stats(feats, f"{prefix}_similarity_resid", rigid)
    add_stats(feats, f"{prefix}_pair_stretch", stretch)
    valid_speed = _finite(speed, speed_valid)
    feats[f"{prefix}_frozen_frac_1e3"] = (
        float(np.mean(valid_speed < 1e-3)) if valid_speed.size else float("nan")
    )
    feats[f"{prefix}_speed_cv"] = (
        float(valid_speed.std() / (valid_speed.mean() + EPS))
        if valid_speed.size
        else float("nan")
    )
    return feats, {
        "speed": speed.astype(np.float32),
        "speed_valid": speed_valid.astype(np.uint8),
        "similarity_resid": rigid.astype(np.float32),
        "pair_stretch": stretch.astype(np.float32),
    }


def trajectory_features(data: Mapping[str, np.ndarray]) -> tuple[
    dict[str, float], dict[str, np.ndarray], dict[str, Any]
]:
    pred_px = _ensure_nt2("pred_tracks_2d", data["pred_tracks_2d"])
    gt_px = _ensure_nt2("gt_query_2d", data["gt_query_2d"])
    pred_unit = _ensure_nt2(
        "pred_tracks_2d_normalized", data["pred_tracks_2d_normalized"]
    )
    gt_unit = _ensure_nt2(
        "gt_query_2d_normalized", data["gt_query_2d_normalized"]
    )
    if not (pred_px.shape == gt_px.shape == pred_unit.shape == gt_unit.shape):
        raise ValueError(
            "trajectory arrays must have identical [N,T,2] shapes: "
            f"pred_px={pred_px.shape}, gt_px={gt_px.shape}, "
            f"pred_unit={pred_unit.shape}, gt_unit={gt_unit.shape}"
        )
    n, t_len, _ = pred_px.shape
    vis_tn = np.asarray(data["visibs"], dtype=np.float32)
    if vis_tn.shape != (t_len, n):
        raise ValueError(f"visibs must be [T,N]={t_len,n}, got {vis_tn.shape}")
    visible = vis_tn.T > 0.5
    h, w = frame_hw(data)
    _assert_coordinate_contract(
        data, pred_px, gt_px, pred_unit, gt_unit, visible, (h, w)
    )

    query_points = np.asarray(data["query_points"], dtype=np.float32)
    if query_points.shape != (n, 3):
        raise ValueError(f"query_points must be [N,3], got {query_points.shape}")

    visible_logits = np.asarray(
        data.get("pred_visible_logits", data.get("pred_visible", np.empty((0,))))
    )
    certain_logits = np.asarray(
        data.get("pred_certain_logits", data.get("pred_certain", np.empty((0,))))
    )
    if visible_logits.ndim == 3 and visible_logits.shape[-1] == 1:
        visible_logits = visible_logits[..., 0]
    if certain_logits.ndim == 3 and certain_logits.shape[-1] == 1:
        certain_logits = certain_logits[..., 0]
    if visible_logits.shape != (n, t_len) or certain_logits.shape != (n, t_len):
        raise ValueError(
            "visibility/certainty logits must be [N,T]: "
            f"visible={visible_logits.shape}, certain={certain_logits.shape}"
        )
    _require_finite("pred_visible_logits", visible_logits)
    _require_finite("pred_certain_logits", certain_logits)
    visible_and_certain_prob = sigmoid(visible_logits) * sigmoid(certain_logits)
    predicted_visible = visible_and_certain_prob > 0.5
    if "pred_visible_and_certain" in data:
        stored = np.asarray(data["pred_visible_and_certain"])
        if stored.shape != (n, t_len):
            raise ValueError(
                f"pred_visible_and_certain must be [N,T], got {stored.shape}"
            )
        if not np.array_equal(stored > 0.5, predicted_visible):
            raise ValueError("stored visibility disagrees with TRAJAN logits")

    average_jaccard, jaccard_by_threshold = tapvid_average_jaccard(
        query_points_txy=query_points,
        gt_visible=visible,
        pred_visible=predicted_visible,
        gt_tracks_unit=gt_unit,
        pred_tracks_unit=pred_unit,
    )
    # E50 has exactly one preregistered candidate axis.  Diagnostics and
    # post-hoc controls must not silently enter this feature group.
    feats = {"one_minus_aj_trajan": 1.0 - average_jaccard}
    arrays: dict[str, np.ndarray] = {
        "tapvid_jaccard_by_threshold": jaccard_by_threshold.astype(np.float32),
        "tapvid_thresholds_px": np.asarray(TAPVID_THRESHOLDS, dtype=np.int16),
    }
    meta = {
        "n_query_points": n,
        "n_frames": t_len,
        "height": h,
        "width": w,
        "coordinate_contract": COORDINATE_CONTRACT,
        "tapvid_metric_size_wh": list(TAPVID_METRIC_SIZE_WH),
        "tapvid_query_mode": "strided_actual_query_frame",
    }
    return feats, arrays, meta


def sample_spatial_grid(
    grid_thwd: np.ndarray,
    tracks_nt2: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    grid = np.asarray(grid_thwd)
    tracks = np.asarray(tracks_nt2)
    if grid.ndim != 4:
        raise ValueError(f"spatial feature grid must be [T,H,W,D], got {grid.shape}")
    t_len = min(grid.shape[0], tracks.shape[1])
    n = tracks.shape[0]
    hp, wp, dim = grid.shape[1:]
    h, w = image_hw
    result = np.empty((n, t_len, dim), dtype=np.float32)
    for t in range(t_len):
        x = np.rint((tracks[:, t, 0] + 0.5) * wp / max(w, 1) - 0.5)
        y = np.rint((tracks[:, t, 1] + 0.5) * hp / max(h, 1) - 0.5)
        xi = np.clip(x.astype(np.int64), 0, wp - 1)
        yi = np.clip(y.astype(np.int64), 0, hp - 1)
        result[:, t] = np.asarray(grid[t, yi, xi], dtype=np.float32)
    return result


def cosine_loss(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    dot = np.sum(aa * bb, axis=-1)
    norm = np.linalg.norm(aa, axis=-1) * np.linalg.norm(bb, axis=-1)
    return 1.0 - np.divide(dot, norm, out=np.zeros_like(dot), where=norm > EPS)


def semantic_features(
    data: Mapping[str, np.ndarray], trajectory_meta: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if "dino_features" not in data:
        raise KeyError("upstream predictions.npz has no dino_features")
    n = int(trajectory_meta["n_query_points"])
    t_len = int(trajectory_meta["n_frames"])
    h, w = int(trajectory_meta["height"]), int(trajectory_meta["width"])
    gt = _ensure_nt2("gt_query_2d", data["gt_query_2d"])[:n, :t_len]
    pred = _ensure_nt2("pred_tracks_2d", data["pred_tracks_2d"])[:n, :t_len]
    visible = np.asarray(data["visibs"], dtype=np.float32)[:t_len, :n].T > 0.5
    grid = np.asarray(data["dino_features"])
    gt_sem = sample_spatial_grid(grid, gt, (h, w))
    pred_sem = sample_spatial_grid(grid, pred, (h, w))
    t_eff = min(gt_sem.shape[1], pred_sem.shape[1], visible.shape[1])
    gt_sem, pred_sem, visible = (
        gt_sem[:, :t_eff],
        pred_sem[:, :t_eff],
        visible[:, :t_eff],
    )
    recon_loss = cosine_loss(gt_sem, pred_sem)
    temporal_loss = cosine_loss(gt_sem[:, 1:], gt_sem[:, :-1])
    temporal_valid = visible[:, 1:] & visible[:, :-1]
    anchor_loss = np.full((n, t_eff), np.nan, dtype=np.float32)
    for i in range(n):
        valid_idx = np.flatnonzero(visible[i])
        if valid_idx.size:
            anchor = gt_sem[i, valid_idx[0]][None, :]
            anchor_loss[i] = cosine_loss(gt_sem[i], np.broadcast_to(anchor, gt_sem[i].shape))
    feats: dict[str, float] = {}
    add_stats(feats, "semantic_recon_cosloss", recon_loss, visible)
    add_stats(feats, "semantic_temporal_cosloss", temporal_loss, temporal_valid)
    add_stats(feats, "semantic_anchor_cosloss", anchor_loss, visible)

    motion = np.linalg.norm(np.diff(gt[:, :t_eff], axis=1), axis=-1)
    motion /= max(float(math.hypot(h, w)), EPS)
    static = temporal_valid & (motion < 1e-3)
    moving = temporal_valid & (motion >= 1e-3)
    vals_static = _finite(temporal_loss, static)
    vals_moving = _finite(temporal_loss, moving)
    feats["semantic_static_drift_mean"] = (
        float(vals_static.mean()) if vals_static.size else float("nan")
    )
    feats["semantic_moving_drift_mean"] = (
        float(vals_moving.mean()) if vals_moving.size else float("nan")
    )
    joint_valid = temporal_valid & np.isfinite(temporal_loss) & np.isfinite(motion)
    if joint_valid.sum() >= 8:
        x, y = motion[joint_valid], temporal_loss[joint_valid]
        feats["motion_semantic_drift_corr"] = (
            float(np.corrcoef(x, y)[0, 1])
            if x.std() > EPS and y.std() > EPS
            else 0.0
        )
    else:
        feats["motion_semantic_drift_corr"] = float("nan")
    feats["dino_grid_h"] = float(grid.shape[1])
    feats["dino_grid_w"] = float(grid.shape[2])
    feats["dino_dim"] = float(grid.shape[3])
    arrays = {
        "semantic_recon_cosloss": recon_loss.astype(np.float16),
        "semantic_temporal_cosloss": temporal_loss.astype(np.float16),
        "semantic_anchor_cosloss": anchor_loss.astype(np.float16),
    }
    return feats, arrays


def sample_scalar_map(
    maps_thw: np.ndarray,
    tracks_nt2: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    maps = np.asarray(maps_thw)
    if maps.ndim == 4 and maps.shape[-1] == 1:
        maps = maps[..., 0]
    if maps.ndim != 3:
        raise ValueError(f"depth must be [T,H,W], got {maps.shape}")
    tracks = np.asarray(tracks_nt2)
    t_len = min(maps.shape[0], tracks.shape[1])
    n = tracks.shape[0]
    hd, wd = maps.shape[1:]
    h, w = image_hw
    result = np.empty((n, t_len), dtype=np.float32)
    for t in range(t_len):
        x = np.rint((tracks[:, t, 0] + 0.5) * wd / max(w, 1) - 0.5)
        y = np.rint((tracks[:, t, 1] + 0.5) * hd / max(h, 1) - 0.5)
        xi = np.clip(x.astype(np.int64), 0, wd - 1)
        yi = np.clip(y.astype(np.int64), 0, hd - 1)
        result[:, t] = maps[t, yi, xi]
    return result


def robust_normalize_depth(depth_nt: np.ndarray, visible_nt: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_nt, dtype=np.float64)
    vals = _finite(depth, visible_nt)
    if vals.size == 0:
        return np.full_like(depth, np.nan, dtype=np.float32)
    median = float(np.median(vals))
    q25, q75 = np.percentile(vals, [25, 75])
    scale = float(q75 - q25)
    if scale <= EPS:
        scale = float(np.std(vals))
    if scale <= EPS:
        scale = 1.0
    return ((depth - median) / scale).astype(np.float32)


def depth_features(
    data: Mapping[str, np.ndarray], trajectory_meta: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if "depth" not in data:
        raise KeyError("upstream predictions.npz has no depth")
    n = int(trajectory_meta["n_query_points"])
    t_len = int(trajectory_meta["n_frames"])
    h, w = int(trajectory_meta["height"]), int(trajectory_meta["width"])
    gt = _ensure_nt2("gt_query_2d", data["gt_query_2d"])[:n, :t_len]
    pred = _ensure_nt2("pred_tracks_2d", data["pred_tracks_2d"])[:n, :t_len]
    visible = np.asarray(data["visibs"], dtype=np.float32)[:t_len, :n].T > 0.5
    gt_depth = sample_scalar_map(np.asarray(data["depth"]), gt, (h, w))
    pred_depth = sample_scalar_map(np.asarray(data["depth"]), pred, (h, w))
    t_eff = min(gt_depth.shape[1], pred_depth.shape[1], visible.shape[1])
    gt_depth = gt_depth[:, :t_eff]
    pred_depth = pred_depth[:, :t_eff]
    visible = visible[:, :t_eff]
    gt_z = robust_normalize_depth(gt_depth, visible)
    pred_z = robust_normalize_depth(pred_depth, visible)
    depth_recon = np.abs(pred_z - gt_z)
    depth_delta = np.abs(np.diff(gt_z, axis=1))
    delta_valid = visible[:, 1:] & visible[:, :-1]
    depth_accel = np.abs(np.diff(gt_z, n=2, axis=1))
    accel_valid = visible[:, 2:] & visible[:, 1:-1] & visible[:, :-2]
    feats: dict[str, float] = {}
    add_stats(feats, "depth_recon_robust", depth_recon, visible)
    add_stats(feats, "depth_temporal_delta", depth_delta, delta_valid)
    add_stats(feats, "depth_temporal_accel", depth_accel, accel_valid)

    coords = np.empty((n, t_eff, 3), dtype=np.float32)
    coords[..., 0] = gt[:, :t_eff, 0] / max(float(w), 1.0)
    coords[..., 1] = gt[:, :t_eff, 1] / max(float(h), 1.0)
    coords[..., 2] = gt_z
    kin, kin_arrays = kinematic_features(coords, visible, "track3d")
    feats.update(kin)

    extents = np.full(t_eff, np.nan, dtype=np.float64)
    for t in range(t_eff):
        ok = visible[:, t] & np.all(np.isfinite(coords[:, t]), axis=-1)
        if ok.sum() >= 8:
            centered = coords[ok, t] - np.median(coords[ok, t], axis=0)
            extents[t] = float(np.median(np.linalg.norm(centered, axis=-1)))
    finite_extents = _finite(extents)
    feats["track3d_extent_cv"] = (
        float(finite_extents.std() / (finite_extents.mean() + EPS))
        if finite_extents.size
        else float("nan")
    )
    add_stats(feats, "track3d_extent_delta", np.abs(np.diff(extents)))
    arrays: dict[str, np.ndarray] = {
        "sampled_depth": gt_depth.astype(np.float16),
        "sampled_depth_robust": gt_z.astype(np.float16),
        "depth_recon_robust": depth_recon.astype(np.float16),
        "track3d_extent": extents.astype(np.float32),
    }
    arrays.update({f"track3d_{k}": v for k, v in kin_arrays.items()})
    return feats, arrays


def compact_predictions(
    predictions_path: Path,
    mode: str,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, Any], list[str]]:
    warnings: list[str] = []
    with np.load(predictions_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {
        "pred_tracks_2d",
        "gt_query_2d",
        "pred_tracks_2d_normalized",
        "gt_query_2d_normalized",
        "pred_visible_logits",
        "pred_certain_logits",
        "query_points",
        "visibs",
        "coordinate_contract",
    }
    missing = sorted(required - set(data))
    if missing:
        raise KeyError(f"{predictions_path}: missing upstream arrays {missing}")
    if "frame_size_hw" not in data and "video" not in data:
        raise KeyError(f"{predictions_path}: missing frame_size_hw/video")
    groups: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {}
    trajectory, trajectory_arrays, trajectory_meta = trajectory_features(data)
    groups["trajectory2d"] = trajectory
    arrays.update(trajectory_arrays)

    if mode in {"semantic2d", "all"}:
        try:
            semantic, semantic_arrays = semantic_features(data, trajectory_meta)
            groups["semantic2d_posthoc"] = semantic
            arrays.update(semantic_arrays)
        except (KeyError, ValueError) as exc:
            warnings.append(f"semantic2d unavailable: {exc}")
    if mode in {"depth3d", "all"}:
        try:
            depth, depth_arrays = depth_features(data, trajectory_meta)
            groups["depth3d_posthoc"] = depth
            arrays.update(depth_arrays)
        except (KeyError, ValueError) as exc:
            warnings.append(f"depth3d unavailable: {exc}")

    return groups, arrays, trajectory_meta, warnings


def flatten_groups(
    groups: Mapping[str, Mapping[str, float]]
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    names: list[str] = []
    values: list[float] = []
    offsets: dict[str, tuple[int, int]] = {}
    for group in GROUP_ORDER:
        if group not in groups:
            continue
        start = len(values)
        for key in sorted(groups[group]):
            names.append(f"{group}.{key}")
            values.append(float(groups[group][key]))
        offsets[group] = (start, len(values))
    return (
        np.asarray(names, dtype=np.str_),
        np.asarray(values, dtype=np.float32),
        offsets,
    )


def atomic_write_outputs(
    json_path: Path,
    npz_path: Path,
    payload: Mapping[str, Any],
    groups: Mapping[str, Mapping[str, float]],
    arrays: Mapping[str, np.ndarray],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    names, features, offsets = flatten_groups(groups)
    npz_payload: dict[str, np.ndarray] = {
        "feature_names": names,
        "features": features,
        "group_names": np.asarray(list(offsets), dtype=np.str_),
        "group_offsets": np.asarray(list(offsets.values()), dtype=np.int32),
        "schema_version": np.asarray(SCHEMA_VERSION),
    }
    npz_payload.update(arrays)
    tmp_npz = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.tmp")
    tmp_json = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    try:
        with tmp_npz.open("wb") as f:
            np.savez_compressed(f, **npz_payload)
        tmp_json.write_text(
            json.dumps(_json_ready({**payload, "feature_groups": groups}), ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_json, json_path)
    finally:
        for path in (tmp_npz, tmp_json):
            if path.exists():
                path.unlink()


def write_error(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(_json_ready(payload), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _flatten_param_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            out.update(_flatten_param_tree(value, path))
    else:
        out[prefix] = tree
    return out


def _unflatten_param_tree(flat: Mapping[str, Any]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for key, value in flat.items():
        node = tree
        parts = key.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


class PersistentTrajanRunner:
    """Keep CoTracker and strict-loaded TRAJAN/JAX parameters resident."""

    def __init__(self, args: argparse.Namespace):
        if args.ablation_mode != "trajectory":
            raise RuntimeError(
                "persistent E50 currently supports only the preregistered TRAJAN "
                "trajectory axis; post-hoc controls remain unrun"
            )
        # JAX must not reserve nearly the whole GPU while the persistent Torch
        # tracker is resident.  These are set before importing track_autoencoder.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

        import cv2  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        self.args = args
        self.cv2 = cv2
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        t0 = time.perf_counter()
        self.tracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline"
        )
        self.tracker = self.tracker.to(self.device).eval()
        self.load_tracker_seconds = time.perf_counter() - t0

        repo_s = str(args.official_repo)
        if repo_s not in sys.path:
            sys.path.insert(0, repo_s)
        import importlib

        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        track_autoencoder = importlib.import_module("track_autoencoder")
        self.jax = jax
        self.jnp = jnp
        self.model = track_autoencoder.TrackAutoEncoder(
            num_output_frames=args.num_output_frames
        )

        t0 = time.perf_counter()
        with np.load(args.checkpoint_path, allow_pickle=False) as archive:
            checkpoint_flat_np = {
                key: np.asarray(archive[key]) for key in archive.files
            }
        self.load_checkpoint_seconds = time.perf_counter() - t0

        dummy = {
            "support_tracks": jnp.zeros(
                (1, args.num_support_tracks, args.num_output_frames, 2), jnp.float32
            ),
            "support_tracks_visible": jnp.ones(
                (1, args.num_support_tracks, args.num_output_frames, 1), jnp.float32
            ),
            "query_points": jnp.zeros(
                (1, args.num_query_points, 3), jnp.float32
            ),
            "boundary_frame": jnp.asarray([args.num_output_frames], jnp.int32),
        }
        t0 = time.perf_counter()
        expected_vars = jax.eval_shape(
            lambda key, batch: self.model.init(key, batch),
            jax.random.PRNGKey(0),
            dummy,
        )
        expected = _flatten_param_tree(expected_vars["params"])
        actual = checkpoint_flat_np
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            key
            for key in set(expected) & set(actual)
            if tuple(expected[key].shape) != tuple(actual[key].shape)
            or np.dtype(expected[key].dtype) != np.dtype(actual[key].dtype)
        )
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "strict TRAJAN checkpoint mismatch: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"shape_or_dtype={len(mismatched)}; "
                f"samples={(missing + unexpected + mismatched)[:8]}"
            )
        self.strict_load_seconds = time.perf_counter() - t0
        self.strict_report = {
            "expected_keys": len(expected),
            "actual_keys": len(actual),
            "missing_keys": 0,
            "unexpected_keys": 0,
            "shape_or_dtype_mismatches": 0,
        }

        params_np = _unflatten_param_tree(checkpoint_flat_np)
        self.params = jax.device_put(
            jax.tree_util.tree_map(lambda value: jnp.asarray(value), params_np)
        )

        def apply_model(params: Mapping[str, Any], batch: Mapping[str, Any]) -> Any:
            return self.model.apply({"params": params}, batch)

        self._forward = jax.jit(apply_model)

    def _synchronize_torch(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _load_video(self, path: Path) -> tuple[np.ndarray, float]:
        cap = self.cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("cannot decode input video")
        fps = float(cap.get(self.cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        frames: list[np.ndarray] = []
        while len(frames) < self.args.num_output_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError("input video has no decodable frames")
        return np.stack(frames, axis=0), fps

    def synthetic_smoke(self) -> dict[str, Any]:
        """Determinism/non-collapse check on two fixed unit-track tensors."""
        n_support = self.args.num_support_tracks
        n_query = self.args.num_query_points
        t_len = self.args.num_output_frames
        point = np.arange(n_support, dtype=np.float32)
        base_x = ((point % 64.0) + 0.5) / 64.0
        base_y = ((np.floor(point / 64.0) % 32.0) + 0.5) / 32.0
        phase = point[:, None] * np.float32(0.013)
        frame = np.arange(t_len, dtype=np.float32)[None, :]
        motion_x = np.float32(0.02) * np.sin(frame / 9.0 + phase)
        motion_y = np.float32(0.015) * np.cos(frame / 11.0 + phase)
        support_a = np.stack(
            [base_x[:, None] + motion_x, base_y[:, None] + motion_y], axis=-1
        ).astype(np.float32)
        support_b = support_a.copy()
        support_b[..., 0] += np.float32(0.01) * np.sin(frame / 5.0)
        visibility = np.ones((1, n_support, t_len, 1), dtype=np.float32)
        query_index = np.arange(n_query, dtype=np.int64)
        query_frame = query_index % t_len
        query = np.stack(
            [
                query_frame.astype(np.float32),
                support_a[query_index, query_frame, 0],
                support_a[query_index, query_frame, 1],
            ],
            axis=-1,
        ).astype(np.float32)

        def batch(support: np.ndarray) -> dict[str, Any]:
            return {
                "support_tracks": self.jnp.asarray(support[None]),
                "support_tracks_visible": self.jnp.asarray(visibility),
                "query_points": self.jnp.asarray(query[None]),
                "boundary_frame": self.jnp.asarray([t_len], self.jnp.int32),
            }

        first = self._forward(self.params, batch(support_a))
        first.tracks.block_until_ready()
        repeated = self._forward(self.params, batch(support_a))
        repeated.tracks.block_until_ready()
        changed = self._forward(self.params, batch(support_b))
        changed.tracks.block_until_ready()
        first_tree = [
            np.asarray(first.tracks),
            np.asarray(first.visible_logits),
            np.asarray(first.certain_logits),
        ]
        repeated_tree = [
            np.asarray(repeated.tracks),
            np.asarray(repeated.visible_logits),
            np.asarray(repeated.certain_logits),
        ]
        changed_tree = [
            np.asarray(changed.tracks),
            np.asarray(changed.visible_logits),
            np.asarray(changed.certain_logits),
        ]
        if not all(np.all(np.isfinite(value)) for value in first_tree + changed_tree):
            raise RuntimeError("synthetic TRAJAN output contains NaN/Inf")
        if not all(
            np.array_equal(left, right)
            for left, right in zip(first_tree, repeated_tree)
        ):
            raise RuntimeError("synthetic deterministic repeat differs")
        max_change = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(first_tree, changed_tree)
        )
        if max_change <= 1e-7:
            raise RuntimeError("synthetic distinct inputs collapsed to the same output")
        return {
            "deterministic_repeat": True,
            "distinct_input_nonconstant": True,
            "tracks_shape": list(first_tree[0].shape),
            "visible_logits_shape": list(first_tree[1].shape),
            "certain_logits_shape": list(first_tree[2].shape),
        }

    def run(
        self, video_path: Path, output_dir: Path
    ) -> tuple[Path, float, str, dict[str, float]]:
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        t0 = time.perf_counter()
        video, _fps = self._load_video(video_path)
        timings["decode"] = time.perf_counter() - t0
        original_t, height, width = video.shape[:3]

        t0 = time.perf_counter()
        # CoTracker's official predictor normalizes internally as
        # ``2 * (video / 255) - 1``; its public API therefore takes 0..255
        # float tensors.  Do not pre-divide here.
        tensor = (
            self.torch.from_numpy(video)
            .permute(0, 3, 1, 2)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        self._synchronize_torch()
        with self.torch.inference_mode():
            pred_tracks, pred_vis = self.tracker(
                tensor, grid_size=self.args.tracking_grid_size
            )
        self._synchronize_torch()
        timings["cotracker"] = time.perf_counter() - t0
        tracks_px = pred_tracks[0].permute(1, 0, 2).cpu().numpy().astype(np.float32)
        vis_tensor = pred_vis[0]
        if vis_tensor.ndim == 2:
            vis_tensor = vis_tensor.permute(1, 0).unsqueeze(-1)
        else:
            vis_tensor = vis_tensor.permute(1, 0, 2)
        tracker_visible = vis_tensor.cpu().numpy().astype(np.float32)
        del tensor, pred_tracks, pred_vis, vis_tensor

        total_needed = self.args.num_support_tracks + self.args.num_query_points
        if tracks_px.shape[0] < total_needed:
            raise ValueError(
                f"CoTracker returned {tracks_px.shape[0]} tracks, need {total_needed}"
            )
        if tracks_px.shape[1] != original_t or tracker_visible.shape != (
            tracks_px.shape[0],
            original_t,
            1,
        ):
            raise ValueError(
                f"unexpected CoTracker shapes tracks={tracks_px.shape}, "
                f"visible={tracker_visible.shape}, frames={original_t}"
            )
        _require_finite("CoTracker tracks", tracks_px)
        _require_finite("CoTracker visibility", tracker_visible)

        t0 = time.perf_counter()
        if original_t < self.args.num_output_frames:
            pad = self.args.num_output_frames - original_t
            tracks_px_model = np.concatenate(
                [tracks_px, np.repeat(tracks_px[:, -1:], pad, axis=1)], axis=1
            )
            visible_model = np.concatenate(
                [
                    tracker_visible,
                    np.zeros((tracks_px.shape[0], pad, 1), dtype=np.float32),
                ],
                axis=1,
            )
        else:
            tracks_px_model = tracks_px[:, : self.args.num_output_frames]
            visible_model = tracker_visible[:, : self.args.num_output_frames]

        scale = np.asarray([width, height], dtype=np.float32)
        tracks_unit = (tracks_px_model + 0.5) / scale
        rng = np.random.RandomState(0)
        permutation = rng.permutation(tracks_px.shape[0])
        support_idx = permutation[: self.args.num_support_tracks]
        query_idx = permutation[
            self.args.num_support_tracks : total_needed
        ]
        support_unit = tracks_unit[support_idx]
        support_visible = visible_model[support_idx]
        query_unit = tracks_unit[query_idx]
        query_points = np.zeros((self.args.num_query_points, 3), dtype=np.float32)
        for index, track_index in enumerate(query_idx):
            frames = np.flatnonzero(
                visible_model[track_index, :original_t, 0] > 0.5
            )
            query_frame = int(rng.choice(frames)) if frames.size else 0
            query_points[index] = (
                query_frame,
                query_unit[index, query_frame, 0],
                query_unit[index, query_frame, 1],
            )
        query_visible = visible_model[query_idx, :original_t]
        _check_unit_tracks("support_tracks_normalized", support_unit, support_visible[..., 0] > 0.5)
        _check_unit_tracks("query_tracks_normalized", query_unit, visible_model[query_idx, ..., 0] > 0.5)
        timings["prepare_trajan"] = time.perf_counter() - t0

        batch = {
            "support_tracks": self.jnp.asarray(support_unit[None]),
            "support_tracks_visible": self.jnp.asarray(support_visible[None]),
            "query_points": self.jnp.asarray(query_points[None]),
            "boundary_frame": self.jnp.asarray([original_t], self.jnp.int32),
        }
        t0 = time.perf_counter()
        outputs = self._forward(self.params, batch)
        outputs.tracks.block_until_ready()
        timings["trajan_forward"] = time.perf_counter() - t0

        pred_unit = np.asarray(outputs.tracks[0])[:, :original_t]
        visible_logits = np.asarray(outputs.visible_logits[0])[:, :original_t]
        certain_logits = np.asarray(outputs.certain_logits[0])[:, :original_t]
        gt_unit = query_unit[:, :original_t]
        pred_px_center = pred_unit * scale
        gt_px_center = gt_unit * scale
        pred_visible_and_certain = (
            sigmoid(visible_logits[..., 0]) * sigmoid(certain_logits[..., 0]) > 0.5
        )
        _require_finite("TRAJAN tracks", pred_unit)
        _require_finite("TRAJAN visible logits", visible_logits)
        _require_finite("TRAJAN certain logits", certain_logits)

        t0 = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = output_dir / "predictions.npz"
        with predictions.open("wb") as handle:
            np.savez_compressed(
                handle,
                coordinate_contract=np.asarray(COORDINATE_CONTRACT),
                frame_size_hw=np.asarray([height, width], dtype=np.int32),
                pred_tracks_2d=pred_px_center.astype(np.float32),
                gt_query_2d=gt_px_center.astype(np.float32),
                pred_tracks_2d_normalized=pred_unit.astype(np.float32),
                gt_query_2d_normalized=gt_unit.astype(np.float32),
                pred_visible_logits=visible_logits.astype(np.float32),
                pred_certain_logits=certain_logits.astype(np.float32),
                pred_visible_and_certain=pred_visible_and_certain.astype(np.uint8),
                visibs=query_visible[..., 0].T.astype(np.float32),
                query_points=query_points.astype(np.float32),
            )
        timings["write_intermediate"] = time.perf_counter() - t0
        total = time.perf_counter() - total_start
        timings["persistent_total"] = total
        return predictions, total, "", timings


def official_command(
    args: argparse.Namespace, video_path: Path, output_dir: Path
) -> list[str]:
    need_dino = args.ablation_mode in {"semantic2d", "all"}
    need_depth = args.ablation_mode in {"depth3d", "all"}
    command = [
        args.official_python,
        str(args.official_repo / "inference.py"),
        f"--checkpoint_path={args.checkpoint_path}",
        f"--video_path={video_path}",
        f"--output_dir={output_dir}",
        f"--num_output_frames={args.num_output_frames}",
        f"--num_query_points={args.num_query_points}",
        f"--num_support_tracks={args.num_support_tracks}",
        f"--tracking_grid_size={args.tracking_grid_size}",
        f"--dino_model={args.dino_model}",
        f"--use_dino={'true' if need_dino else 'false'}",
        f"--use_depth={'true' if need_depth else 'false'}",
    ]
    if need_depth:
        command.append(f"--vda_model_path={args.vda_model_path}")
    return command


def run_official(
    args: argparse.Namespace, video_path: Path, output_dir: Path
) -> tuple[Path, float, str]:
    command = official_command(args, video_path, output_dir)
    env = os.environ.copy()
    repo_s = str(args.official_repo)
    env["PYTHONPATH"] = (
        repo_s
        if not env.get("PYTHONPATH")
        else repo_s + os.pathsep + env["PYTHONPATH"]
    )
    t0 = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=args.official_repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.timeout_s,
    )
    elapsed = time.perf_counter() - t0
    log_tail = result.stdout[-12000:]
    if result.returncode:
        raise RuntimeError(
            f"official inference exited {result.returncode}; log tail:\n{log_tail}"
        )
    predictions = output_dir / "predictions.npz"
    if not predictions.is_file():
        raise FileNotFoundError(
            f"official inference succeeded but did not write {predictions}; log tail:\n{log_tail}"
        )
    return predictions, elapsed, log_tail


def find_existing_predictions(root: Path, label: str, stem: str) -> Path | None:
    candidates = [
        root / label / stem / "predictions.npz",
        root / label / f"{stem}.npz",
        root / stem / "predictions.npz",
        root / f"{stem}.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def process_one(
    args: argparse.Namespace,
    row: Mapping[str, str],
    video_path: Path | None,
    upstream_audit: Mapping[str, Any],
    persistent_runner: PersistentTrajanRunner | None = None,
) -> tuple[str, float]:
    label = _safe_label(row["label"])
    stem = Path(row["video"]).stem
    json_path = args.output_root / label / f"{stem}.json"
    npz_path = args.output_root / label / f"{stem}.npz"
    error_path = args.output_root / label / f"{stem}.error.json"
    if not args.overwrite and json_path.is_file() and npz_path.is_file():
        return "skipped", 0.0
    if video_path is None and not args.postprocess_only:
        raise FileNotFoundError(f"no resolved video path for {row['video']}")

    t0 = time.perf_counter()
    official_seconds = 0.0
    official_log_tail = ""
    runner_timing: dict[str, float] = {}
    if args.postprocess_only:
        predictions = find_existing_predictions(args.predictions_root, label, stem)
        if predictions is None:
            raise FileNotFoundError(
                f"no predictions.npz under {args.predictions_root} for {label}/{stem}"
            )
        groups, arrays, shape_meta, warnings = compact_predictions(
            predictions, args.ablation_mode
        )
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        tmp_parent = args.output_root / ".official_tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{label}-{stem}-", dir=tmp_parent) as td:
            if persistent_runner is not None:
                (
                    predictions,
                    official_seconds,
                    official_log_tail,
                    runner_timing,
                ) = persistent_runner.run(video_path, Path(td))
            else:
                predictions, official_seconds, official_log_tail = run_official(
                    args, video_path, Path(td)
                )
            groups, arrays, shape_meta, warnings = compact_predictions(
                predictions, args.ablation_mode
            )
            if args.keep_official_npz:
                keep_dir = args.output_root / "official_raw" / label / stem
                keep_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(predictions, keep_dir / "predictions.npz")

    total_seconds = time.perf_counter() - t0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "entry": {
            "label": label,
            "video": row["video"],
            "resolved_path": str(video_path) if video_path is not None else None,
        },
        "ablation_mode": args.ablation_mode,
        "upstream": dict(upstream_audit),
        "semantics": {
            "trajectory2d": (
                "single preregistered axis: 1 - official TAP-Vid average Jaccard "
                "for strict-loaded 2-D TRAJAN reconstruction"
            ),
            "semantic2d_posthoc": "DINO patch features sampled along tracks; not fed to upstream model",
            "depth3d_posthoc": "VDA scalar depth sampled along tracks; not a TrackAutoEncoder3D forward",
        },
        "shape": shape_meta,
        "timing_seconds": {
            "official_inference": official_seconds,
            "total": total_seconds,
            **runner_timing,
        },
        "warnings": warnings,
    }
    atomic_write_outputs(json_path, npz_path, payload, groups, arrays)
    if error_path.exists():
        error_path.unlink()
    # Keep normal logs out of the cache.  A tail is attached only on failure by
    # the outer loop; this variable is intentionally retained for debugging.
    del official_log_tail
    return "done", total_seconds


def selected_rows(args: argparse.Namespace, rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    selected = list(rows)
    if args.only_video:
        wanted = set(args.only_video)
        selected = [row for row in selected if row["video"] in wanted]
        missing = sorted(wanted - {row["video"] for row in selected})
        if missing:
            raise ValueError(f"--only-video values absent from manifest: {missing}")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    selected = selected[args.offset :]
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.limit:
        selected = selected[: args.limit]
    return selected


def validate_run_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_failures <= 0:
        raise ValueError("--max-failures must be > 0")
    if args.postprocess_only:
        if args.predictions_root is None:
            raise ValueError("--postprocess-only requires --predictions-root")
        return {
            "repo": None,
            "commit": None,
            "full_3dspa_forward": False,
            "postprocess_only": True,
        }
    if args.runner_mode == "legacy-subprocess":
        raise RuntimeError(
            "legacy upstream subprocess feeds pixel tracks to a unit-coordinate "
            "TRAJAN model and is hard-blocked; use --runner-mode=persistent"
        )
    if args.official_repo is None:
        raise ValueError("--official-repo or THREEDSPA_REPO is required")
    args.official_repo = args.official_repo.expanduser().resolve()
    if args.checkpoint_path is None or not args.checkpoint_path.expanduser().is_file():
        raise FileNotFoundError("--checkpoint-path must point to the downloaded .npz")
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    checkpoint_sha256 = sha256_file(args.checkpoint_path)
    if (
        checkpoint_sha256 != OFFICIAL_TRAJAN_CHECKPOINT_SHA256
        and not args.allow_unpinned_upstream
    ):
        raise RuntimeError(
            "checkpoint is not the audited official TRAJAN checkpoint: "
            f"got {checkpoint_sha256}, expected {OFFICIAL_TRAJAN_CHECKPOINT_SHA256}"
        )
    need_depth = args.ablation_mode in {"depth3d", "all"}
    if need_depth and (
        args.vda_model_path is None or not args.vda_model_path.expanduser().is_file()
    ):
        raise FileNotFoundError("depth ablation requires --vda-model-path")
    if args.vda_model_path is not None:
        args.vda_model_path = args.vda_model_path.expanduser().resolve()
    if args.num_support_tracks + args.num_query_points > args.tracking_grid_size ** 2:
        raise ValueError(
            "support + query tracks exceed grid_size**2; upstream would silently reduce queries"
        )
    audit = audit_official_repo(args.official_repo, args.allow_unpinned_upstream)
    audit.update(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "official_trajan_checkpoint_sha256": OFFICIAL_TRAJAN_CHECKPOINT_SHA256,
            "coordinate_contract": COORDINATE_CONTRACT,
            "tapvid_metric_size_wh": list(TAPVID_METRIC_SIZE_WH),
            "full_3dspa_status": "BLOCKED_BY_UPSTREAM",
        }
    )
    if need_depth:
        vda_candidates = [
            args.official_repo / "Video-Depth-Anything" / "video_depth_anything" / "video_depth.py",
            args.official_repo.parent
            / "Video-Depth-Anything"
            / "video_depth_anything"
            / "video_depth.py",
        ]
        if not any(path.is_file() for path in vda_candidates):
            raise FileNotFoundError(
                "Video-Depth-Anything checkout not found beside/inside --official-repo"
            )
    preflight_python(
        args.official_python,
        need_dino=args.ablation_mode in {"semantic2d", "all"},
        need_depth=need_depth,
    )
    return audit


def main() -> int:
    args = parse_args()
    rows = selected_rows(args, load_manifest(args.manifest, args.allow_eval))
    path_maps = parse_path_maps(args.path_map)
    resolved: list[tuple[dict[str, str], Path | None, list[Path]]] = []
    for row in rows:
        path, candidates = resolve_video_path(row, path_maps, args.videos_root)
        resolved.append((row, path, candidates))

    missing = [item for item in resolved if item[1] is None]
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "selected": len(rows),
                "resolved_videos": len(rows) - len(missing),
                "missing_videos": len(missing),
                "mode": args.ablation_mode,
                "dry_run": args.dry_run,
                "postprocess_only": args.postprocess_only,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if missing:
        for row, _, candidates in missing[:5]:
            print(
                f"MISSING {row['label']}/{row['video']} candidates="
                + ",".join(str(p) for p in candidates),
                file=sys.stderr,
            )
    if args.dry_run:
        return 0 if (not missing or args.postprocess_only) else 2

    upstream_audit = validate_run_args(args)
    persistent_runner: PersistentTrajanRunner | None = None
    if not args.postprocess_only and args.runner_mode == "persistent":
        persistent_runner = PersistentTrajanRunner(args)
        upstream_audit = {
            **upstream_audit,
            "runner": "persistent_cotracker_trajan",
            "strict_load": persistent_runner.strict_report,
            "one_time_load_seconds": {
                "cotracker": persistent_runner.load_tracker_seconds,
                "checkpoint_npz": persistent_runner.load_checkpoint_seconds,
                "strict_shape_audit": persistent_runner.strict_load_seconds,
            },
        }
    if not upstream_audit.get("full_3dspa_forward", False):
        print(
            "NOTICE full semantic/depth 3DSPA is BLOCKED_BY_UPSTREAM; "
            "this run is strict official-checkpoint TRAJAN-2D only.",
            file=sys.stderr,
            flush=True,
        )

    done = skipped = failed = 0
    elapsed_done = 0.0
    for i, (row, video_path, _) in enumerate(resolved, 1):
        label = _safe_label(row["label"])
        stem = Path(row["video"]).stem
        error_path = args.output_root / label / f"{stem}.error.json"
        try:
            status, seconds = process_one(
                args,
                row,
                video_path,
                upstream_audit=upstream_audit,
                persistent_runner=persistent_runner,
            )
            if status == "skipped":
                skipped += 1
            else:
                done += 1
                elapsed_done += seconds
            avg = elapsed_done / max(done, 1)
            eta_min = avg * max(len(resolved) - i, 0) / 60.0
            print(
                f"[{i}/{len(resolved)}] {status} {label}/{row['video']} "
                f"{seconds:.1f}s eta={eta_min:.1f}m",
                flush=True,
            )
        except Exception as exc:  # Per-video failures must be resumable.
            failed += 1
            write_error(
                error_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "entry": {
                        "label": label,
                        "video": row["video"],
                        "resolved_path": str(video_path) if video_path else None,
                    },
                    "error_type": type(exc).__name__,
                    "error": str(exc)[-12000:],
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            print(
                f"[{i}/{len(resolved)}] FAIL {label}/{row['video']}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if failed >= args.max_failures:
                print(f"stopping after {failed} failures", file=sys.stderr)
                break
    print(
        f"E50_TRAJAN_DONE done={done} skipped={skipped} failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
