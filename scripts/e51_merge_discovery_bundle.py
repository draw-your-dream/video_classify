#!/usr/bin/env python3
"""Merge paired E18/E51 discovery nested-OOF bundles for the strict harness.

This is deliberately a thin, metric-free join.  It requires exact SHA-256
values for the frozen S0 split and both model bundles.  Labels, strata, outer
folds, and the inner-fold matrix come from S0; both model bundles must align
exactly to that source of truth.  It then delegates baseline and candidate
nested-contract validation to ``e50_frontier_harness.py`` and writes one
atomic, train-only NPZ.

``selftest`` uses only synthetic arrays and also invokes the real harness CLI
in ``inspect`` and explicit ``protocol-e51`` modes for targets 0.35 and 0.50.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from e50_frontier_harness import validate_nested_axis, validate_nested_bundle


SCHEMA_VERSION = "e51_discovery_harness_bundle_v1"
REQUIRED_BASE = {
    "sample_token",
    "strata",
    "y",
    "fold_id",
    "inner_fold_id",
    "base_oof",
    "inner_base",
}
REQUIRED_S0 = {
    "sample_token",
    "strata",
    "y",
    "fold_id",
    "inner_fold_id",
}
REQUIRED_CANDIDATE = {
    "sample_token",
    "strata",
    "y",
    "fold_id",
    "expert_oof",
    "inner_expert",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_sha(value: str, role: str) -> str:
    normalized = value.strip().casefold()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{role} SHA-256 must be 64 lowercase hex characters")
    return normalized


def safe_discovery_npz(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix.casefold() != ".npz" or not resolved.is_file():
        raise ValueError(f"{role} must be an existing .npz")
    if "eval" in resolved.name.casefold():
        raise ValueError(f"{role} refuses an eval-named path")
    return resolved


def load_required(path: Path, keys: set[str], role: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = keys - set(archive.files)
        if missing:
            raise KeyError(f"{role} missing required arrays: {sorted(missing)}")
        return {key: np.asarray(archive[key]) for key in sorted(keys)}


def string_vector(name: str, values: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a 1-D string array")
    if n is not None and array.size != n:
        raise ValueError(f"{name} length mismatch")
    return array.astype(str, copy=False)


def validate_tokens(values: np.ndarray) -> np.ndarray:
    tokens = string_vector("sample_token", values)
    if tokens.size < 2 or not np.all(tokens[:-1] < tokens[1:]):
        raise ValueError("sample_token must be unique and strictly sorted")
    if any(
        len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token)
        for token in tokens.tolist()
    ):
        raise ValueError("sample_token must contain lowercase SHA-256 hex")
    return tokens


def merge_arrays(
    base: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    s0: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    tokens = validate_tokens(s0["sample_token"])
    n = tokens.size
    base_tokens = validate_tokens(base["sample_token"])
    candidate_tokens = validate_tokens(candidate["sample_token"])
    if not np.array_equal(tokens, base_tokens) or not np.array_equal(tokens, candidate_tokens):
        raise ValueError("S0/E18/E51 sample_token order differs; refusing positional join")

    y = np.asarray(s0["y"])
    fold_id = np.asarray(s0["fold_id"])
    strata = string_vector("S0 strata", s0["strata"], n)
    inner_fold_id = np.asarray(s0["inner_fold_id"])
    if y.shape != (n,) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("S0 y contract failed")
    if fold_id.shape != (n,) or set(np.unique(fold_id).tolist()) != set(range(5)):
        raise ValueError("S0 outer-fold contract failed")
    if inner_fold_id.shape != (5, n):
        raise ValueError("S0 inner-fold shape failed")
    for outer_id in range(5):
        outer_valid = fold_id == outer_id
        if not np.all(inner_fold_id[outer_id, outer_valid] == -1):
            raise ValueError("S0 inner folds leak onto outer-valid")
        if not np.all(np.isin(inner_fold_id[outer_id, ~outer_valid], range(4))):
            raise ValueError("S0 inner fold id outside 0..3")
    for role, bundle, include_inner in (
        ("E18", base, True),
        ("E51", candidate, False),
    ):
        comparisons = [
            ("y", y, np.asarray(bundle["y"])),
            ("fold_id", fold_id, np.asarray(bundle["fold_id"])),
            ("strata", strata, string_vector(f"{role} strata", bundle["strata"], n)),
        ]
        if include_inner:
            comparisons.append(
                ("inner_fold_id", inner_fold_id, np.asarray(bundle["inner_fold_id"]))
            )
        for key, left, right in comparisons:
            if not np.array_equal(left, right):
                raise ValueError(f"S0/{role} {key} differs")

    yy, folds, base_oof, inner_base = validate_nested_bundle(
        y,
        fold_id,
        np.asarray(base["base_oof"]),
        np.asarray(base["inner_base"]),
    )
    expert_oof, inner_expert = validate_nested_axis(
        "expert_oof",
        np.asarray(candidate["expert_oof"]),
        np.asarray(candidate["inner_expert"]),
        folds,
    )
    if not np.isfinite(expert_oof).all():
        raise ValueError("E51 nested head must cover every discovery outer-valid sample")
    if not np.isfinite(inner_expert[~np.isnan(inner_expert)]).all():
        raise ValueError("inner_expert contains a non-finite non-NaN value")
    return {
        "sample_token": tokens.astype("<U64", copy=False),
        "strata": strata,
        "y": yy.astype(np.int8, copy=False),
        "fold_id": folds.astype(np.int8, copy=False),
        "inner_fold_id": inner_fold_id.astype(np.int8, copy=False),
        "base_oof": base_oof.astype(np.float64, copy=False),
        "inner_base": inner_base.astype(np.float64, copy=False),
        "expert_oof": expert_oof.astype(np.float32, copy=False),
        "inner_expert": inner_expert.astype(np.float32, copy=False),
    }


def atomic_write_bundle(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    base_sha: str,
    candidate_sha: str,
    s0_sha: str,
) -> str:
    target = output.expanduser().resolve()
    if target.suffix.casefold() != ".npz":
        raise ValueError("--output must end in .npz")
    if "eval" in target.name.casefold():
        raise ValueError("output refuses an eval-named path")
    if target.exists():
        raise FileExistsError("refusing to overwrite merged harness bundle")
    target.parent.mkdir(parents=True, exist_ok=True)
    artifact_id = secrets.token_hex(16)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = dict(arrays)
    payload.update(
        {
            "schema_version": np.asarray(SCHEMA_VERSION),
            "artifact_id": np.asarray(artifact_id),
            "e18_bundle_sha256": np.asarray(base_sha),
            "e51_bundle_sha256": np.asarray(candidate_sha),
            "s0_discovery_split_sha256": np.asarray(s0_sha),
            "protocol_mode": np.asarray("protocol-e51"),
        }
    )
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(target)


def merge_command(args: argparse.Namespace) -> int:
    if not args.confirm_discovery_only:
        raise RuntimeError("merge requires explicit --confirm-discovery-only")
    base_path = safe_discovery_npz(args.e18_bundle, "E18 bundle")
    candidate_path = safe_discovery_npz(args.e51_bundle, "E51 bundle")
    s0_path = safe_discovery_npz(args.s0_split, "S0 discovery split")
    base_sha = sha256_file(base_path)
    candidate_sha = sha256_file(candidate_path)
    s0_sha = sha256_file(s0_path)
    if base_sha != checked_sha(args.expected_e18_sha256, "E18"):
        raise RuntimeError("E18 bundle SHA-256 mismatch")
    if candidate_sha != checked_sha(args.expected_e51_sha256, "E51"):
        raise RuntimeError("E51 bundle SHA-256 mismatch")
    if s0_sha != checked_sha(args.expected_s0_sha256, "S0"):
        raise RuntimeError("S0 discovery split SHA-256 mismatch")
    base = load_required(base_path, REQUIRED_BASE, "E18 bundle")
    candidate = load_required(candidate_path, REQUIRED_CANDIDATE, "E51 bundle")
    s0 = load_required(s0_path, REQUIRED_S0, "S0 discovery split")
    arrays = merge_arrays(base, candidate, s0)
    output_sha = atomic_write_bundle(
        args.output,
        arrays,
        base_sha,
        candidate_sha,
        s0_sha,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "schema_version": SCHEMA_VERSION,
                "samples": int(arrays["y"].size),
                "folds": int(arrays["inner_base"].shape[0]),
                "token_alignment_exact": True,
                "label_fold_strata_alignment_exact": True,
                "s0_alignment_exact": True,
                "baseline_nested_contract": True,
                "candidate_nested_contract": True,
                "metric_computed": False,
                "output_sha256": output_sha,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def synthetic_arrays(n: int = 200) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if n % 10:
        raise ValueError("synthetic n must be divisible by 10")
    tokens = np.asarray([f"{index:064x}" for index in range(n)], dtype="<U64")
    y = np.asarray([index % 2 for index in range(n)], dtype=np.int8)
    folds = np.asarray([(index // 2) % 5 for index in range(n)], dtype=np.int8)
    strata = np.asarray([f"source-{(index // 2) % 3}" for index in range(n)])
    inner_fold_id = np.full((5, n), -1, dtype=np.int8)
    for outer_id in range(5):
        train = folds != outer_id
        for label in (0, 1):
            indices = np.flatnonzero(train & (y == label))
            inner_fold_id[outer_id, indices] = np.arange(indices.size) % 4

    pair_rank = (np.arange(n) // 2) / (n // 2 - 1)
    base_oof = np.where(y == 1, 0.05 + 0.90 * pair_rank, 0.90 * pair_rank)
    expert_oof = np.clip(0.20 + 0.60 * y + 0.05 * np.sin(np.arange(n)), 0.0, 1.0)
    inner_base = np.full((5, n), np.nan, dtype=np.float64)
    inner_expert = np.full((5, n), np.nan, dtype=np.float32)
    for outer_id in range(5):
        train = folds != outer_id
        inner_base[outer_id, train] = base_oof[train] + 0.001 * outer_id
        inner_expert[outer_id, train] = expert_oof[train] + 0.001 * outer_id
    common = {
        "sample_token": tokens,
        "strata": strata,
        "y": y,
        "fold_id": folds,
        "inner_fold_id": inner_fold_id,
    }
    base = {**common, "base_oof": base_oof, "inner_base": inner_base}
    s0 = dict(common)
    candidate_common = {key: value for key, value in common.items() if key != "inner_fold_id"}
    candidate = {
        **candidate_common,
        "expert_oof": expert_oof.astype(np.float32),
        "inner_expert": inner_expert,
    }
    return base, candidate, s0


def run_harness(harness: Path, bundle: Path, arguments: Sequence[str]) -> int:
    result = subprocess.run(
        [sys.executable, str(harness), *arguments, "--bundle", str(bundle)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    for line in lines:
        json.loads(line)
    if not lines:
        raise AssertionError("harness emitted no JSON records")
    return len(lines)


def selftest() -> int:
    base, candidate, s0 = synthetic_arrays()
    harness = Path(__file__).resolve().with_name("e50_frontier_harness.py")
    with tempfile.TemporaryDirectory(prefix="e51-merge-selftest-") as temporary:
        root = Path(temporary)
        base_path = root / "synthetic_e18.npz"
        candidate_path = root / "synthetic_e51.npz"
        s0_path = root / "synthetic_s0_split.npz"
        bundle_path = Path(temporary) / "synthetic_discovery.npz"
        np.savez_compressed(base_path, **base)
        np.savez_compressed(candidate_path, **candidate)
        np.savez_compressed(s0_path, **s0)
        merge_command(
            argparse.Namespace(
                confirm_discovery_only=True,
                e18_bundle=base_path,
                e51_bundle=candidate_path,
                expected_e18_sha256=sha256_file(base_path),
                expected_e51_sha256=sha256_file(candidate_path),
                s0_split=s0_path,
                expected_s0_sha256=sha256_file(s0_path),
                output=bundle_path,
            )
        )
        output_sha = sha256_file(bundle_path)
        inspect_records = run_harness(
            harness,
            bundle_path,
            ["inspect", "--targets", "0.35", "0.50"],
        )
        evaluate35_records = run_harness(
            harness,
            bundle_path,
            ["evaluate", "--mode", "protocol-e51", "--target", "0.35"],
        )
        evaluate50_records = run_harness(
            harness,
            bundle_path,
            ["evaluate", "--mode", "protocol-e51", "--target", "0.50"],
        )
        with np.load(bundle_path, allow_pickle=False) as archive:
            if str(archive["protocol_mode"].item()) != "protocol-e51":
                raise AssertionError("protocol mode provenance failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "schema_version": SCHEMA_VERSION,
                "synthetic_samples": int(base["y"].size),
                "token_alignment_exact": True,
                "s0_alignment_exact": True,
                "baseline_nested_contract": True,
                "candidate_nested_contract": True,
                "harness_inspect_records": inspect_records,
                "harness_protocol_e51_target35_records": evaluate35_records,
                "harness_protocol_e51_target50_records": evaluate50_records,
                "temporary_bundle_sha256_valid": len(output_sha) == 64,
                "real_bundle_accessed": False,
                "real_label_accessed": False,
                "real_metric_computed": False,
                "eval_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("selftest", help="Synthetic merge plus real harness CLI checks.")
    merge = subparsers.add_parser("merge", help="Build a strict train-only harness bundle.")
    merge.add_argument("--e18-bundle", type=Path, required=True)
    merge.add_argument("--e51-bundle", type=Path, required=True)
    merge.add_argument("--s0-split", type=Path, required=True)
    merge.add_argument("--expected-e18-sha256", required=True)
    merge.add_argument("--expected-e51-sha256", required=True)
    merge.add_argument("--expected-s0-sha256", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--confirm-discovery-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "selftest":
        return selftest()
    if args.command == "merge":
        return merge_command(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
