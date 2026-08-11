#!/usr/bin/env python3
"""Label-blind G0 and timing smoke for the persistent E50 TRAJAN runner.

This program never opens a manifest, split, prediction, reason, or label file.
It selects the first three decodable videos after sorting all candidate videos
by SHA-256 of their bytes.  Only aggregate pass/fail, shapes, and timing are
written; video names and per-video AJ values are never printed or persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import e50_3dspa_extract as core


SCHEMA = "e50_trajan_g0_v1"
NUM_OUTPUT_FRAMES = 150
NUM_SUPPORT_TRACKS = 2048
NUM_QUERY_POINTS = 512
TRACKING_GRID_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _video_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_label_blind_videos(root: Path) -> tuple[list[Path], int]:
    import cv2  # Imported only for decode probing; no model is loaded here.

    candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if len(candidates) < 3:
        raise RuntimeError(f"need at least 3 candidate videos, found {len(candidates)}")
    ranked = sorted((_video_sha256(path), path) for path in candidates)
    selected: list[Path] = []
    seen_content: set[str] = set()
    for content_hash, path in ranked:
        if content_hash in seen_content:
            continue
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if ok and frame is not None and frame.size:
            selected.append(path)
            seen_content.add(content_hash)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError(
            f"found only {len(selected)} distinct decodable videos among {len(candidates)}"
        )
    return selected, len(candidates)


def deterministic_arrays_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            value = np.asarray(archive[key])
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(value.dtype.str.encode("ascii") + b"\0")
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def timing_summary(rows: list[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        result[key] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"G0 output is immutable and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(core._json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def main() -> int:
    cli = parse_args()
    started = time.perf_counter()
    videos_root = cli.videos_root.expanduser().resolve()
    official_repo = cli.official_repo.expanduser().resolve()
    checkpoint = cli.checkpoint_path.expanduser().resolve()
    if not videos_root.is_dir():
        raise FileNotFoundError(f"videos root is not a directory: {videos_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {checkpoint}")
    checkpoint_sha = core.sha256_file(checkpoint)
    if checkpoint_sha != core.OFFICIAL_TRAJAN_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA mismatch: {checkpoint_sha} != "
            f"{core.OFFICIAL_TRAJAN_CHECKPOINT_SHA256}"
        )
    audit = core.audit_official_repo(official_repo, allow_unpinned=False)
    selection_start = time.perf_counter()
    videos, candidate_count = select_label_blind_videos(videos_root)
    selection_seconds = time.perf_counter() - selection_start

    runner_args = argparse.Namespace(
        ablation_mode="trajectory",
        official_repo=official_repo,
        checkpoint_path=checkpoint,
        num_output_frames=NUM_OUTPUT_FRAMES,
        num_support_tracks=NUM_SUPPORT_TRACKS,
        num_query_points=NUM_QUERY_POINTS,
        tracking_grid_size=TRACKING_GRID_SIZE,
    )
    runner = core.PersistentTrajanRunner(runner_args)
    synthetic_start = time.perf_counter()
    synthetic = runner.synthetic_smoke()
    synthetic_seconds = time.perf_counter() - synthetic_start

    axes: list[float] = []
    array_digests: list[str] = []
    timings: list[dict[str, float]] = []
    output_parent = cli.output_json.expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    # Four forwards: video 0 twice for determinism, then videos 1 and 2.
    schedule = [videos[0], videos[0], videos[1], videos[2]]
    for path in schedule:
        with tempfile.TemporaryDirectory(prefix="e50-g0-", dir=output_parent) as temp:
            prediction, _elapsed, _log, timing = runner.run(path, Path(temp))
            groups, _arrays, _shape, warnings = core.compact_predictions(
                prediction, "trajectory"
            )
            if warnings:
                raise RuntimeError(f"unexpected compact warnings: {len(warnings)}")
            axis = float(groups["trajectory2d"]["one_minus_aj_trajan"])
            if not math.isfinite(axis) or not 0.0 <= axis <= 1.0:
                raise RuntimeError("non-finite/out-of-range TRAJAN axis")
            axes.append(axis)
            array_digests.append(deterministic_arrays_digest(prediction))
            timings.append(timing)

    if array_digests[0] != array_digests[1] or axes[0] != axes[1]:
        raise RuntimeError("same-video deterministic repeat differs")
    unique_video_axes = np.asarray([axes[0], axes[2], axes[3]], dtype=np.float64)
    if float(np.ptp(unique_video_axes)) <= 1e-6:
        raise RuntimeError("three distinct videos collapsed to a constant TRAJAN axis")

    payload = {
        "schema_version": SCHEMA,
        "status": "G0_PASS",
        "label_access": False,
        "sample_identifiers_persisted": False,
        "selection": {
            "rule": "ascending SHA-256(video bytes), first 3 distinct decodable files",
            "candidate_count": candidate_count,
            "selected_count": 3,
            "selection_seconds": selection_seconds,
        },
        "configuration": {
            "num_output_frames": NUM_OUTPUT_FRAMES,
            "num_support_tracks": NUM_SUPPORT_TRACKS,
            "num_query_points": NUM_QUERY_POINTS,
            "tracking_grid_size": TRACKING_GRID_SIZE,
            "coordinate_contract": core.COORDINATE_CONTRACT,
            "tapvid_metric_size_wh": list(core.TAPVID_METRIC_SIZE_WH),
            "tapvid_thresholds": list(core.TAPVID_THRESHOLDS),
            "query_mode": "strided_actual_query_frame",
        },
        "source": audit,
        "checkpoint_sha256": checkpoint_sha,
        "strict_load": runner.strict_report,
        "synthetic": {**synthetic, "seconds": synthetic_seconds},
        "video_smoke": {
            "unique_videos": 3,
            "forward_runs": 4,
            "all_finite_and_in_range": True,
            "deterministic_repeat": True,
            "distinct_video_axis_nonconstant": True,
        },
        "one_time_load_seconds": {
            "cotracker": runner.load_tracker_seconds,
            "checkpoint_npz": runner.load_checkpoint_seconds,
            "strict_shape_audit": runner.strict_load_seconds,
        },
        "per_stage_timing_seconds": timing_summary(timings),
        "total_seconds": time.perf_counter() - started,
        "full_3dspa_status": "BLOCKED_BY_UPSTREAM",
    }
    write_once(cli.output_json, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_count": 3,
                "forward_runs": 4,
                "strict_keys": runner.strict_report["actual_keys"],
                "deterministic": True,
                "nonconstant": True,
                "total_seconds": payload["total_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
