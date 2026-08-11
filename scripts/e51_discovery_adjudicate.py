#!/usr/bin/env python3
"""Discovery-only statistical adjudicator for the frozen E51 protocol.

Production use has no project-data defaults.  It requires an explicit merged
discovery NPZ, its expected SHA-256, the frozen train manifest and its expected
SHA-256, an immutable output path, and ``--confirm-discovery-only``.  The NPZ
key allowlist is checked before any array is read, so a mixed train/eval archive
is rejected without opening an unexpected member.

The computation reuses the SHA-pinned ``protocol-e51`` foldwise inner-OOF
mid-CDF and strict ``<T`` GN@95 implementation from
``e50_frontier_harness.py``.  It adds the preregistered control plane:

* fixed Frontier-35/50 bad shells and GN bands, recomputed per outer fold;
* paired point estimates and fold signs;
* 5,000 paired group-bootstrap replicates, with every GN threshold and
  frontier shell/band recomputed inside the replicate;
* 200 outer-fold-by-source permutations for the single completed E51 family;
* an independent Gaussian-axis control with the exact candidate missing masks.

It never reads eval or shadow data and never trains or selects a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "e51_discovery_adjudication_v1"
SEED = 20260810
RECALL = 0.95
BOOTSTRAP_REPEATS = 5000
NULL_REPEATS = 200
FRONTIER_HIGH = {"35": 0.067, "50": 0.145}
HARNESS_NAME = "e50_frontier_harness.py"
HARNESS_SHA256 = "66dc519841074c9f578cb186d4732788a8e2d0fb5dfafd700e8b74ea1a1f7119"
EXPECTED_PROTOCOL = "protocol-e51"
EXPECTED_BUNDLE_SCHEMA = "e51_discovery_harness_bundle_v1"
REQUIRED_ARRAYS = {
    "sample_token",
    "strata",
    "y",
    "fold_id",
    "inner_fold_id",
    "base_oof",
    "inner_base",
    "expert_oof",
    "inner_expert",
}
REQUIRED_METADATA = {"schema_version", "protocol_mode"}
ALLOWED_MEMBERS = REQUIRED_ARRAYS | {
    "schema_version",
    "artifact_id",
    "e18_bundle_sha256",
    "e51_bundle_sha256",
    "s0_discovery_split_sha256",
    "protocol_mode",
}
VALID_LABELS = frozenset({"bad", "good", "normal"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checked_sha(value: str, role: str) -> str:
    normalized = value.strip().casefold()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{role} SHA-256 must be 64 lowercase hexadecimal characters")
    return normalized


def sample_token(video: str) -> str:
    return sha256_bytes(b"e50-sample-v1\0" + video.encode("utf-8"))


def derive_source(row: Mapping[str, Any]) -> str:
    raw = str(row.get("abs_path", "")).replace("\\", "/")
    parts = PurePosixPath(raw).parts
    for marker in ("s3", "corpus_videos"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def opaque_group_key(row: Mapping[str, Any], token: str) -> tuple[str, bool, str]:
    for field in ("asset_id", "request_id", "batch_id"):
        raw_value = row.get(field, "")
        value = "" if raw_value is None else str(raw_value).strip()
        if value and value.casefold() not in {"nan", "none", "null"}:
            return f"{field}:{sha256_bytes(value.encode('utf-8'))}", True, field
    raw_prompt = row.get("prompt", "")
    prompt = "" if raw_prompt is None else str(raw_prompt).strip()
    if prompt and prompt.casefold() not in {"nan", "none", "null"}:
        normalized = " ".join(prompt.casefold().split())
        return f"prompt:{sha256_bytes(normalized.encode('utf-8'))}", True, "prompt"
    return f"sample:{token}", False, "none"


def _scalar_string(name: str, value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a scalar string")
    return str(array.item())


def _string_vector(name: str, value: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a one-dimensional string array")
    if n is not None and array.size != n:
        raise ValueError(f"{name} length mismatch")
    result = array.astype(str, copy=False)
    if any(not item or item.casefold() == "nan" for item in result.tolist()):
        raise ValueError(f"{name} contains an empty/NaN-like value")
    return result


def _validate_tokens(value: np.ndarray) -> np.ndarray:
    tokens = _string_vector("sample_token", value)
    if tokens.size < 2 or not np.all(tokens[:-1] < tokens[1:]):
        raise ValueError("sample_token must be unique and strictly sorted")
    if any(
        len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token)
        for token in tokens.tolist()
    ):
        raise ValueError("sample_token must contain lowercase SHA-256 hexadecimal values")
    return tokens


def _safe_input(path: Path, suffix: str, role: str, require_discovery: bool) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.suffix.casefold() != suffix:
        raise ValueError(f"{role} must be an existing {suffix} file")
    lowered = resolved.name.casefold()
    local_scope = f"{resolved.parent.name}/{resolved.name}".casefold()
    if "eval" in local_scope or "shadow" in local_scope:
        raise ValueError(f"{role} refuses an eval/shadow-named file")
    if require_discovery and "discovery" not in local_scope:
        raise ValueError(f"{role} filename or immediate parent must explicitly contain 'discovery'")
    return resolved


def load_frozen_harness(script_dir: Path | None = None) -> ModuleType:
    root = script_dir if script_dir is not None else Path(__file__).resolve().parent
    path = root / HARNESS_NAME
    if not path.is_file() or sha256_file(path) != HARNESS_SHA256:
        raise RuntimeError(f"frozen harness missing or SHA mismatch: {path}")
    spec = importlib.util.spec_from_file_location("_e51_frozen_frontier_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import frozen harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_discovery_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        members = set(archive.files)
        missing = (REQUIRED_ARRAYS | REQUIRED_METADATA) - members
        unexpected = members - ALLOWED_MEMBERS
        if missing:
            raise KeyError(f"discovery bundle missing required arrays: {sorted(missing)}")
        if unexpected:
            raise RuntimeError(
                "discovery bundle has unexpected members; refusing mixed archive before "
                f"opening them: {sorted(unexpected)}"
            )
        schema = _scalar_string("schema_version", archive["schema_version"])
        protocol = _scalar_string("protocol_mode", archive["protocol_mode"])
        if schema != EXPECTED_BUNDLE_SCHEMA or protocol != EXPECTED_PROTOCOL:
            raise RuntimeError(
                f"bundle provenance mismatch: schema={schema!r}, protocol={protocol!r}"
            )
        result = {name: np.asarray(archive[name]) for name in sorted(REQUIRED_ARRAYS)}
        result["schema_version"] = np.asarray(schema)
        result["protocol_mode"] = np.asarray(protocol)
    return result


def validate_bundle(
    bundle: Mapping[str, np.ndarray], harness: ModuleType
) -> dict[str, np.ndarray]:
    tokens = _validate_tokens(bundle["sample_token"])
    n = tokens.size
    source = _string_vector("strata/source", bundle["strata"], n)
    yy, folds, base, inner_base = harness.validate_nested_bundle(
        bundle["y"], bundle["fold_id"], bundle["base_oof"], bundle["inner_base"]
    )
    if inner_base.shape != (5, n) or not np.array_equal(np.unique(folds), np.arange(5)):
        raise ValueError("protocol requires exactly five outer folds")
    inner_fold = np.asarray(bundle["inner_fold_id"])
    if inner_fold.shape != (5, n):
        raise ValueError("inner_fold_id must have shape (5,N)")
    for fold in range(5):
        valid = folds == fold
        if not np.all(inner_fold[fold, valid] == -1):
            raise ValueError("inner_fold_id leaks onto outer validation")
        if not np.all(np.isin(inner_fold[fold, ~valid], np.arange(4))):
            raise ValueError("protocol requires four inner folds 0..3")
    expert, inner_expert = harness.validate_nested_axis(
        "expert_oof", bundle["expert_oof"], bundle["inner_expert"], folds
    )
    return {
        "sample_token": tokens,
        "source": source,
        "y": yy,
        "fold_id": folds,
        "base_oof": base,
        "inner_base": inner_base,
        "expert_oof": expert,
        "inner_expert": inner_expert,
    }


def align_manifest_groups(
    manifest: Path, tokens: np.ndarray, expected_y: np.ndarray, expected_source: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    wanted = {str(token): index for index, token in enumerate(tokens.tolist())}
    groups = np.empty(tokens.size, dtype=object)
    seen = np.zeros(tokens.size, dtype=bool)
    true_group = np.zeros(tokens.size, dtype=bool)
    kinds: dict[str, int] = {"asset_id": 0, "request_id": 0, "batch_id": 0, "prompt": 0, "none": 0}
    with manifest.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            token = sample_token(str(raw.get("video", "")))
            index = wanted.get(token)
            if index is None:
                # Membership is resolved before touching label/source/group fields.
                continue
            if seen[index]:
                raise RuntimeError("duplicate discovery sample_token in train manifest")
            label = str(raw.get("label", ""))
            if label not in VALID_LABELS:
                raise RuntimeError(f"line {lineno}: invalid discovery label")
            binary = 1 if label == "bad" else 0
            source = derive_source(raw)
            if binary != int(expected_y[index]) or source != str(expected_source[index]):
                raise RuntimeError("manifest/bundle discovery label or source alignment differs")
            key, is_true, kind = opaque_group_key(raw, token)
            groups[index] = key
            true_group[index] = is_true
            kinds[kind] += 1
            seen[index] = True
    if not seen.all():
        raise RuntimeError("not every discovery sample_token aligned to the train manifest")
    use_group_bootstrap = bool(true_group.any())
    return groups.astype(str), {
        "mode": "asset_request_batch_prompt_group" if use_group_bootstrap else "bad_gn_stratified",
        "true_group_rows": int(true_group.sum()),
        "singleton_fallback_rows": int((~true_group).sum()),
        "unique_bootstrap_units": int(np.unique(groups).size) if use_group_bootstrap else 2,
        "group_field_rows": kinds,
    }


def _sign(value: float, center: float = 0.0) -> str:
    if not math.isfinite(value):
        return "NA"
    if value > center:
        return "positive"
    if value < center:
        return "negative"
    return "zero"


def frontier_auc(
    base_rank: np.ndarray,
    axis_rank: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    high_quantile: float,
    harness: ModuleType,
) -> dict[str, Any]:
    """Fixed q-bad shell with strict support-order thresholds, fold by fold."""
    pooled_bad: list[np.ndarray] = []
    pooled_gn: list[np.ndarray] = []
    fold_records: list[dict[str, Any]] = []
    for fold in range(5):
        valid = np.flatnonzero(fold_id == fold)
        bad = valid[y[valid] == 1]
        gn = valid[y[valid] == 0]
        if bad.size == 0 or gn.size == 0:
            fold_records.append({"fold": fold, "auc": float("nan"), "reason": "empty_class"})
            continue
        order = bad[np.argsort(base_rank[bad], kind="stable")]
        ordered_scores = base_rank[order]
        start = bad.size - int(math.ceil(RECALL * bad.size))
        end = int(math.ceil(high_quantile * bad.size - 1e-12))
        if end <= start or end >= bad.size:
            fold_records.append({"fold": fold, "auc": float("nan"), "reason": "empty_shell"})
            continue
        low_threshold = float(ordered_scores[start])
        high_threshold = float(ordered_scores[end])
        shell = order[start:end]
        band = gn[(base_rank[gn] >= low_threshold) & (base_rank[gn] < high_threshold)]
        shell_covered = shell[np.isfinite(axis_rank[shell])]
        band_covered = band[np.isfinite(axis_rank[band])]
        auc = harness.midrank_auc(axis_rank[shell_covered], axis_rank[band_covered])
        fold_records.append(
            {
                "fold": fold,
                "auc": auc,
                "sign_vs_0_5": _sign(auc, 0.5),
                "n_bad": int(shell.size),
                "n_gn": int(band.size),
                "covered_bad": int(shell_covered.size),
                "covered_gn": int(band_covered.size),
            }
        )
        if shell_covered.size:
            pooled_bad.append(axis_rank[shell_covered])
        if band_covered.size:
            pooled_gn.append(axis_rank[band_covered])
    bad_values = np.concatenate(pooled_bad) if pooled_bad else np.empty(0)
    gn_values = np.concatenate(pooled_gn) if pooled_gn else np.empty(0)
    auc = harness.midrank_auc(bad_values, gn_values)
    return {
        "auc": auc,
        "n_bad": int(sum(int(item.get("n_bad", 0)) for item in fold_records)),
        "n_gn": int(sum(int(item.get("n_gn", 0)) for item in fold_records)),
        "covered_bad": int(bad_values.size),
        "covered_gn": int(gn_values.size),
        "folds": fold_records,
        "rank_convention": (
            "start=n_bad-ceil(0.95*n_bad); end=ceil(q_high*n_bad); "
            "thresholds=bad_order[start],bad_order[end]; left-inclusive/right-exclusive"
        ),
    }


def point_statistics(arrays: Mapping[str, np.ndarray], harness: ModuleType) -> dict[str, Any]:
    y = arrays["y"]
    folds = arrays["fold_id"]
    r0 = harness.crossfit_mid_cdf(
        arrays["base_oof"], arrays["inner_base"], folds,
        name="base_oof", allow_missing=False,
    )
    rj = harness.crossfit_mid_cdf(
        arrays["expert_oof"], arrays["inner_expert"], folds,
        name="expert_oof", allow_missing=True,
    )
    final = harness.protocol_e51_scores(r0, rj)
    baseline = harness.gn_at_recall(r0, y, RECALL)
    candidate = harness.gn_at_recall(final, y, RECALL)
    baseline_auc = harness.midrank_auc(r0[y == 1], r0[y == 0])
    candidate_auc = harness.midrank_auc(final[y == 1], final[y == 0])
    fold_records: list[dict[str, Any]] = []
    for fold in range(5):
        idx = folds == fold
        base_fold = harness.gn_at_recall(r0[idx], y[idx], RECALL)
        candidate_fold = harness.gn_at_recall(final[idx], y[idx], RECALL)
        delta = candidate_fold.value - base_fold.value
        fold_records.append(
            {
                "fold": fold,
                "baseline_gn_at_95": base_fold.value,
                "e51_gn_at_95": candidate_fold.value,
                "paired_delta": delta,
                "delta_sign": _sign(delta),
            }
        )
    frontier35 = frontier_auc(r0, rj, y, folds, FRONTIER_HIGH["35"], harness)
    frontier50 = frontier_auc(r0, rj, y, folds, FRONTIER_HIGH["50"], harness)
    return {
        "r0": r0,
        "rj": rj,
        "final": final,
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "paired_delta": candidate.value - baseline.value,
        "baseline_midrank_auc": baseline_auc,
        "candidate_midrank_auc": candidate_auc,
        "paired_midrank_auc_delta": candidate_auc - baseline_auc,
        "folds": fold_records,
        "frontier35": frontier35,
        "frontier50": frontier50,
    }


def bootstrap_indices(
    rng: np.random.Generator,
    y: np.ndarray,
    groups: np.ndarray,
    use_groups: bool,
) -> np.ndarray:
    if use_groups:
        units = np.unique(groups)
        members = {unit: np.flatnonzero(groups == unit) for unit in units}
        drawn = units[rng.integers(0, units.size, size=units.size)]
        return np.concatenate([members[unit] for unit in drawn])
    return np.concatenate(
        [
            indices[rng.integers(0, indices.size, size=indices.size)]
            for label in (0, 1)
            for indices in [np.flatnonzero(y == label)]
        ]
    )


def _percentile_summary(values: np.ndarray, point: float) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"point": point, "lower_95": float("nan"), "upper_95": float("nan"), "finite": 0}
    return {
        "point": point,
        "lower_95": float(np.quantile(finite, 0.025, method="linear")),
        "upper_95": float(np.quantile(finite, 0.975, method="linear")),
        "finite": int(finite.size),
        "nonfinite": int(values.size - finite.size),
    }


def paired_bootstrap(
    point: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    groups: np.ndarray,
    use_groups: bool,
    harness: ModuleType,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    y = arrays["y"]
    folds = arrays["fold_id"]
    r0 = point["r0"]
    rj = point["rj"]
    final = point["final"]
    candidate_values = np.full(repeats, np.nan)
    delta_values = np.full(repeats, np.nan)
    baseline_auc_values = np.full(repeats, np.nan)
    candidate_auc_values = np.full(repeats, np.nan)
    auc_delta_values = np.full(repeats, np.nan)
    auc35_values = np.full(repeats, np.nan)
    auc50_values = np.full(repeats, np.nan)
    group_members: list[np.ndarray] | None = None
    stratified_members: list[np.ndarray] | None = None
    if use_groups:
        units, inverse = np.unique(groups, return_inverse=True)
        group_members = [np.flatnonzero(inverse == index) for index in range(units.size)]
    else:
        stratified_members = [np.flatnonzero(y == label) for label in (0, 1)]
    for repeat in range(repeats):
        if group_members is not None:
            drawn = rng.integers(0, len(group_members), size=len(group_members))
            idx = np.concatenate([group_members[index] for index in drawn])
        else:
            assert stratified_members is not None
            idx = np.concatenate(
                [
                    members[rng.integers(0, members.size, size=members.size)]
                    for members in stratified_members
                ]
            )
        yy = y[idx]
        try:
            baseline = harness.gn_at_recall(r0[idx], yy, RECALL).value
            candidate = harness.gn_at_recall(final[idx], yy, RECALL).value
        except ValueError:
            continue
        candidate_values[repeat] = candidate
        delta_values[repeat] = candidate - baseline
        baseline_auc_values[repeat] = harness.midrank_auc(r0[idx][yy == 1], r0[idx][yy == 0])
        candidate_auc_values[repeat] = harness.midrank_auc(
            final[idx][yy == 1], final[idx][yy == 0]
        )
        auc_delta_values[repeat] = candidate_auc_values[repeat] - baseline_auc_values[repeat]
        auc35_values[repeat] = frontier_auc(
            r0[idx], rj[idx], yy, folds[idx], FRONTIER_HIGH["35"], harness
        )["auc"]
        auc50_values[repeat] = frontier_auc(
            r0[idx], rj[idx], yy, folds[idx], FRONTIER_HIGH["50"], harness
        )["auc"]
    return {
        "repeats": repeats,
        "seed": SEED,
        "percentile_method": "linear",
        "thresholds_and_frontiers_recomputed_each_replicate": True,
        "e51_gn_at_95": _percentile_summary(candidate_values, point["candidate"]["value"]),
        "paired_delta": _percentile_summary(delta_values, point["paired_delta"]),
        "baseline_midrank_auc": _percentile_summary(
            baseline_auc_values, point["baseline_midrank_auc"]
        ),
        "candidate_midrank_auc": _percentile_summary(
            candidate_auc_values, point["candidate_midrank_auc"]
        ),
        "paired_midrank_auc_delta": _percentile_summary(
            auc_delta_values, point["paired_midrank_auc_delta"]
        ),
        "frontier_auc35": _percentile_summary(auc35_values, point["frontier35"]["auc"]),
        "frontier_auc50": _percentile_summary(auc50_values, point["frontier50"]["auc"]),
    }


def permute_within_fold_source(
    axis: np.ndarray,
    fold_id: np.ndarray,
    source: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Permute finite outer ranks only inside outer-fold x source blocks.

    The E51 outer rank is calibrated against a different inner-OOF reference
    row for each outer fold.  Keeping donors inside the same outer fold avoids
    turning fold calibration differences into null signal.  Source blocking is
    the preregistered matched-null stratum.  NaNs never move.
    """
    out = axis.copy()
    movable = 0
    for fold in np.unique(fold_id):
        for value in np.unique(source[fold_id == fold]):
            indices = np.flatnonzero(
                (fold_id == fold) & (source == value) & np.isfinite(axis)
            )
            if indices.size >= 2:
                out[indices] = axis[rng.permutation(indices)]
                movable += int(indices.size)
            if not np.array_equal(np.sort(out[indices]), np.sort(axis[indices])):
                raise AssertionError("permutation crossed an outer-fold/source block")
    if not np.array_equal(np.isnan(out), np.isnan(axis)):
        raise AssertionError("fold/source permutation changed the missing mask")
    return out, movable


def _null_summary(values: np.ndarray, real: float) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"real": real, "max_null_99": float("nan"), "finite": 0, "real_gt_max_null_99": False}
    q99 = float(np.quantile(finite, 0.99, method="linear"))
    return {
        "real": real,
        "mean": float(finite.mean()),
        "max_null_99": q99,
        "maximum": float(finite.max()),
        "finite": int(finite.size),
        "nonfinite": int(values.size - finite.size),
        "real_gt_max_null_99": bool(real > q99),
        "empirical_p_ge_real": (1 + int(np.count_nonzero(finite >= real))) / (finite.size + 1),
    }


def null_controls(
    point: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    harness: ModuleType,
    repeats: int = NULL_REPEATS,
) -> dict[str, Any]:
    y = arrays["y"]
    folds = arrays["fold_id"]
    source = arrays["source"]
    r0 = point["r0"]
    rj = point["rj"]
    baseline = point["baseline"]["value"]
    perm_rng = np.random.default_rng(SEED + 1_000_003)
    gaussian_rng = np.random.default_rng(SEED + 2_000_003)
    perm = {key: np.full(repeats, np.nan) for key in ("delta", "auc35", "auc50")}
    gaussian = {key: np.full(repeats, np.nan) for key in ("delta", "auc35", "auc50")}
    movable_counts: list[int] = []
    outer_missing = np.isnan(arrays["expert_oof"])
    inner_missing = np.isnan(arrays["inner_expert"])
    for repeat in range(repeats):
        fake, movable = permute_within_fold_source(rj, folds, source, perm_rng)
        movable_counts.append(movable)
        fake_final = harness.protocol_e51_scores(r0, fake)
        perm["delta"][repeat] = harness.gn_at_recall(fake_final, y, RECALL).value - baseline
        perm["auc35"][repeat] = frontier_auc(
            r0, fake, y, folds, FRONTIER_HIGH["35"], harness
        )["auc"] - 0.5
        perm["auc50"][repeat] = frontier_auc(
            r0, fake, y, folds, FRONTIER_HIGH["50"], harness
        )["auc"] - 0.5

        gaussian_oof = gaussian_rng.standard_normal(y.size)
        gaussian_oof[outer_missing] = np.nan
        gaussian_inner = gaussian_rng.standard_normal(arrays["inner_expert"].shape)
        gaussian_inner[inner_missing] = np.nan
        gaussian_rank = harness.crossfit_mid_cdf(
            gaussian_oof, gaussian_inner, folds,
            name="independent_gaussian", allow_missing=True,
        )
        if not np.array_equal(np.isnan(gaussian_rank), outer_missing):
            raise AssertionError("Gaussian control changed the outer missing mask")
        gaussian_final = harness.protocol_e51_scores(r0, gaussian_rank)
        gaussian["delta"][repeat] = harness.gn_at_recall(gaussian_final, y, RECALL).value - baseline
        gaussian["auc35"][repeat] = frontier_auc(
            r0, gaussian_rank, y, folds, FRONTIER_HIGH["35"], harness
        )["auc"] - 0.5
        gaussian["auc50"][repeat] = frontier_auc(
            r0, gaussian_rank, y, folds, FRONTIER_HIGH["50"], harness
        )["auc"] - 0.5

    real = {
        "delta": point["paired_delta"],
        "auc35": point["frontier35"]["auc"] - 0.5,
        "auc50": point["frontier50"]["auc"] - 0.5,
    }
    return {
        "repeats": repeats,
        "seed": SEED,
        "family_count": 1,
        "scope": "single_family_e51_max_null",
        "permutation_partition": "outer_fold_x_source",
        "permutation_basis": (
            "outer ranks are fold-calibrated; donors stay within the same outer fold "
            "and preregistered source stratum"
        ),
        "permutation": {
            "missing_mask_preserved": True,
            "movable_finite_min": int(min(movable_counts)),
            "movable_finite_max": int(max(movable_counts)),
            "paired_delta": _null_summary(perm["delta"], real["delta"]),
            "frontier_auc35_minus_0_5": _null_summary(perm["auc35"], real["auc35"]),
            "frontier_auc50_minus_0_5": _null_summary(perm["auc50"], real["auc50"]),
        },
        "independent_gaussian": {
            "outer_and_inner_missing_masks_preserved": True,
            "paired_delta": _null_summary(gaussian["delta"], real["delta"]),
            "frontier_auc35_minus_0_5": _null_summary(gaussian["auc35"], real["auc35"]),
            "frontier_auc50_minus_0_5": _null_summary(gaussian["auc50"], real["auc50"]),
        },
    }


def _fold_gate(frontier: Mapping[str, Any]) -> dict[str, Any]:
    values = [float(item.get("auc", float("nan"))) for item in frontier["folds"]]
    finite = [value for value in values if math.isfinite(value)]
    positives = sum(value > 0.5 for value in finite)
    passed = positives >= 4 if len(finite) == 5 else bool(finite) and positives == len(finite)
    return {"defined_folds": len(finite), "positive_folds": positives, "passed": passed}


def adjudication(
    point: Mapping[str, Any], bootstrap: Mapping[str, Any], null: Mapping[str, Any]
) -> dict[str, Any]:
    delta_positive = sum(item["paired_delta"] > 0 for item in point["folds"])
    perm = null["permutation"]
    frontier35_fold = _fold_gate(point["frontier35"])
    frontier50_fold = _fold_gate(point["frontier50"])
    gates35 = {
        "gn_at_95_ge_0_38": point["candidate"]["value"] >= 0.38,
        "paired_delta_ge_0_060": point["paired_delta"] >= 0.060,
        "paired_delta_ci_lower_ge_0_030": bootstrap["paired_delta"]["lower_95"] >= 0.030,
        "delta_positive_at_least_4_of_5": delta_positive >= 4,
        "delta_gt_single_family_max_null_99": perm["paired_delta"]["real_gt_max_null_99"],
        "frontier_auc35_ci_lower_gt_0_5": bootstrap["frontier_auc35"]["lower_95"] > 0.5,
        "frontier_auc35_gt_single_family_max_null_99": perm["frontier_auc35_minus_0_5"]["real_gt_max_null_99"],
        "frontier_auc35_fold_sign_gate": frontier35_fold["passed"],
        "all_5000_required_bootstrap_values_finite": (
            bootstrap["paired_delta"]["finite"] == BOOTSTRAP_REPEATS
            and bootstrap["frontier_auc35"]["finite"] == BOOTSTRAP_REPEATS
        ),
        "all_200_required_null_values_finite": (
            perm["paired_delta"]["finite"] == NULL_REPEATS
            and perm["frontier_auc35_minus_0_5"]["finite"] == NULL_REPEATS
        ),
    }
    gates50 = {
        "gn_at_95_ge_0_54": point["candidate"]["value"] >= 0.54,
        "paired_delta_ge_0_20": point["paired_delta"] >= 0.20,
        "paired_delta_ci_lower_ge_0_14": bootstrap["paired_delta"]["lower_95"] >= 0.14,
        "frontier_auc50_ci_lower_gt_0_5": bootstrap["frontier_auc50"]["lower_95"] > 0.5,
        "frontier_auc50_gt_single_family_max_null_99": perm["frontier_auc50_minus_0_5"]["real_gt_max_null_99"],
        "target_35_discovery_statistical_gate_also_passed": all(gates35.values()),
        "all_5000_required_bootstrap_values_finite": (
            bootstrap["paired_delta"]["finite"] == BOOTSTRAP_REPEATS
            and bootstrap["frontier_auc50"]["finite"] == BOOTSTRAP_REPEATS
        ),
        "all_200_required_null_values_finite": (
            perm["paired_delta"]["finite"] == NULL_REPEATS
            and perm["frontier_auc50_minus_0_5"]["finite"] == NULL_REPEATS
        ),
    }
    return {
        "target_35": {
            "point_values": {
                "e51_gn_at_95": point["candidate"]["value"],
                "paired_delta": point["paired_delta"],
                "frontier_auc35": point["frontier35"]["auc"],
            },
            "fold_delta_signs": [item["delta_sign"] for item in point["folds"]],
            "frontier_fold_gate": frontier35_fold,
            "gates": gates35,
            "all_listed_discovery_statistical_gates_pass": all(gates35.values()),
        },
        "target_50": {
            "point_values": {
                "e51_gn_at_95": point["candidate"]["value"],
                "paired_delta": point["paired_delta"],
                "frontier_auc50": point["frontier50"]["auc"],
            },
            "frontier_fold_diagnostic": frontier50_fold,
            "gates": gates50,
            "all_listed_discovery_statistical_gates_pass": all(gates50.values()),
        },
        "limitations": [
            "discovery-only; prospective shadow gates are not evaluated",
            "single completed family max-null; combine replicate-synchronously if another family completes",
            "experiment-specific mechanism/data/source gates remain external",
            "this adjudicator does not select a winner or authorize eval",
        ],
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def build_report(
    arrays: Mapping[str, np.ndarray],
    groups: np.ndarray,
    group_audit: Mapping[str, Any],
    harness: ModuleType,
    *,
    bundle_sha: str,
    manifest_sha: str,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
    null_repeats: int = NULL_REPEATS,
) -> dict[str, Any]:
    point = point_statistics(arrays, harness)
    use_groups = group_audit["mode"] != "bad_gn_stratified"
    bootstrap = paired_bootstrap(
        point, arrays, groups, use_groups, harness, repeats=bootstrap_repeats
    )
    null = null_controls(point, arrays, harness, repeats=null_repeats)
    public_point = {
        key: value for key, value in point.items() if key not in {"r0", "rj", "final"}
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "discovery_only",
        "eval_accessed": False,
        "shadow_accessed": False,
        "protocol": {
            "mode": EXPECTED_PROTOCOL,
            "seed": SEED,
            "recall": RECALL,
            "strict_release": "score < T",
            "formula": "r0+0.25*sigmoid((0.45-r0)/0.08)*(rj-0.5)",
            "mid_cdf": "foldwise outer-train inner-OOF empirical mid-CDF",
            "harness_sha256": HARNESS_SHA256,
            "bootstrap_repeats": bootstrap_repeats,
            "null_repeats": null_repeats,
        },
        "input_provenance": {
            "bundle_sha256": bundle_sha,
            "train_manifest_sha256": manifest_sha,
            "sample_token_alignment_exact": True,
            "label_source_alignment_exact": True,
            "npz_member_allowlist_enforced_before_array_reads": True,
        },
        "coverage": {
            "samples": int(arrays["y"].size),
            "outer_folds": 5,
            "expert_oof": float(np.isfinite(arrays["expert_oof"]).mean()),
            "inner_expert_finite_train_cells": int(np.isfinite(arrays["inner_expert"]).sum()),
        },
        "bootstrap_units": dict(group_audit),
        "point_estimates": public_point,
        "paired_group_bootstrap": bootstrap,
        "matched_null": null,
        "adjudication": adjudication(point, bootstrap, null),
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    lowered = target.name.casefold()
    if target.suffix.casefold() != ".json" or "eval" in lowered or "shadow" in lowered:
        raise ValueError("output must be a non-eval/non-shadow .json")
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable adjudication output")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def adjudicate_command(args: argparse.Namespace) -> int:
    if not args.confirm_discovery_only:
        raise RuntimeError("adjudication requires explicit --confirm-discovery-only")
    harness = load_frozen_harness()
    bundle_path = _safe_input(args.discovery_bundle, ".npz", "discovery bundle", True)
    manifest_path = _safe_input(args.train_manifest, ".jsonl", "train manifest", False)
    if manifest_path.name != "train_v3.jsonl":
        raise ValueError("train manifest must be the explicit train_v3.jsonl artifact")
    bundle_sha = sha256_file(bundle_path)
    manifest_sha = sha256_file(manifest_path)
    if bundle_sha != checked_sha(args.expected_bundle_sha256, "bundle"):
        raise RuntimeError("discovery bundle SHA-256 mismatch")
    if manifest_sha != checked_sha(args.expected_manifest_sha256, "manifest"):
        raise RuntimeError("train manifest SHA-256 mismatch")
    bundle = load_discovery_bundle(bundle_path)
    arrays = validate_bundle(bundle, harness)
    groups, group_audit = align_manifest_groups(
        manifest_path, arrays["sample_token"], arrays["y"], arrays["source"]
    )
    report = build_report(
        arrays,
        groups,
        group_audit,
        harness,
        bundle_sha=bundle_sha,
        manifest_sha=manifest_sha,
    )
    atomic_write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "scope": "discovery_only",
                "samples": int(arrays["y"].size),
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "null_repeats": NULL_REPEATS,
                "output_sha256": sha256_file(args.output_json.expanduser().resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def synthetic_fixture(n: int = 500) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    if n % 10:
        raise ValueError("synthetic sample count must be divisible by ten")
    raw_tokens = np.asarray([sample_token(f"synthetic/{index:05d}.mp4") for index in range(n)])
    order = np.argsort(raw_tokens)
    original = np.arange(n)[order]
    tokens = raw_tokens[order]
    y0 = np.asarray([index % 2 for index in range(n)], dtype=np.int8)
    folds0 = np.asarray([(index // 2) % 5 for index in range(n)], dtype=np.int8)
    within0 = np.asarray([(index // 10) / (n / 10 - 1) for index in range(n)])
    base0 = np.clip(within0 + 0.005 * y0, 0.0, 1.0)
    expert0 = np.clip(0.15 + 0.70 * y0 + 0.03 * np.sin(np.arange(n)), 0.0, 1.0)
    missing0 = np.arange(n) % 97 == 0
    expert0[missing0] = np.nan
    source0 = np.asarray([f"source-{(index // 4) % 4}" for index in range(n)])
    groups0 = np.asarray([f"asset:{(index // 4):04d}" for index in range(n)])
    y = y0[order]
    folds = folds0[order]
    base = base0[order]
    expert = expert0[order]
    source = source0[order]
    groups = groups0[order]
    inner_base = np.full((5, n), np.nan)
    inner_expert = np.full((5, n), np.nan)
    inner_fold = np.full((5, n), -1, dtype=np.int8)
    for fold in range(5):
        train = folds != fold
        inner_base[fold, train] = base[train] + 0.0001 * fold
        inner_expert[fold, train] = expert[train]
        positions = np.flatnonzero(train)
        inner_fold[fold, positions] = np.arange(positions.size) % 4
    arrays = {
        "sample_token": tokens,
        "source": source,
        "y": y,
        "fold_id": folds,
        "base_oof": base,
        "inner_base": inner_base,
        "expert_oof": expert,
        "inner_expert": inner_expert,
        "inner_fold_id": inner_fold,
        "_original_index": original,
    }
    audit = {
        "mode": "asset_request_batch_prompt_group",
        "true_group_rows": n,
        "singleton_fallback_rows": 0,
        "unique_bootstrap_units": int(np.unique(groups).size),
        "group_field_rows": {"asset_id": n, "request_id": 0, "batch_id": 0, "prompt": 0, "none": 0},
    }
    return arrays, groups, audit


def selftest() -> int:
    harness = load_frozen_harness()
    arrays, groups, audit = synthetic_fixture()
    with tempfile.TemporaryDirectory(prefix="e51-adjudicate-selftest-") as temporary:
        root = Path(temporary)
        bundle_path = root / "synthetic_discovery_bundle.npz"
        manifest_path = root / "train_v3.jsonl"
        payload = {
            "sample_token": arrays["sample_token"],
            "strata": arrays["source"],
            "y": arrays["y"],
            "fold_id": arrays["fold_id"],
            "inner_fold_id": arrays["inner_fold_id"],
            "base_oof": arrays["base_oof"],
            "inner_base": arrays["inner_base"],
            "expert_oof": arrays["expert_oof"],
            "inner_expert": arrays["inner_expert"],
            "schema_version": np.asarray(EXPECTED_BUNDLE_SCHEMA),
            "protocol_mode": np.asarray(EXPECTED_PROTOCOL),
        }
        np.savez_compressed(bundle_path, **payload)
        rows = []
        for index in range(arrays["y"].size):
            rows.append(
                json.dumps(
                    {
                        "video": f"synthetic/{index:05d}.mp4",
                        "label": "bad" if index % 2 else "good",
                        "abs_path": (
                            f"/synthetic/corpus_videos/source-{(index // 4) % 4}/"
                            f"{index:05d}.mp4"
                        ),
                        "asset_id": f"asset-{index // 4:04d}",
                    },
                    sort_keys=True,
                )
            )
        manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        loaded = load_discovery_bundle(bundle_path)
        validated = validate_bundle(loaded, harness)
        aligned_groups, aligned_audit = align_manifest_groups(
            manifest_path,
            validated["sample_token"],
            validated["y"],
            validated["source"],
        )
        if aligned_audit["mode"] != "asset_request_batch_prompt_group":
            raise AssertionError("synthetic manifest group alignment failed")

        # Unique donor ids make a cross-fold or cross-source move observable,
        # independent of whether score values themselves contain ties.
        donor_probe = np.arange(validated["y"].size, dtype=float)
        donor_probe[np.isnan(validated["expert_oof"])] = np.nan
        permuted_probe, _ = permute_within_fold_source(
            donor_probe,
            validated["fold_id"],
            validated["source"],
            np.random.default_rng(SEED + 77),
        )
        if not np.array_equal(np.isnan(permuted_probe), np.isnan(donor_probe)):
            raise AssertionError("synthetic permutation changed missing positions")
        for recipient in np.flatnonzero(np.isfinite(permuted_probe)):
            donor = int(permuted_probe[recipient])
            if (
                validated["fold_id"][recipient] != validated["fold_id"][donor]
                or validated["source"][recipient] != validated["source"][donor]
            ):
                raise AssertionError("synthetic permutation crossed fold/source")

        mixed_path = root / "synthetic_discovery_mixed_bundle.npz"
        np.savez_compressed(mixed_path, **payload, forbidden_eval_member=np.zeros(1))
        try:
            load_discovery_bundle(mixed_path)
        except RuntimeError as exc:
            if "unexpected members" not in str(exc):
                raise
        else:
            raise AssertionError("NPZ allowlist failed to reject a mixed member")

        report = build_report(
            validated,
            aligned_groups,
            aligned_audit,
            harness,
            bundle_sha=sha256_file(bundle_path),
            manifest_sha=sha256_file(manifest_path),
        )
    assert report["protocol"]["bootstrap_repeats"] == 5000
    assert report["protocol"]["null_repeats"] == 200
    assert report["paired_group_bootstrap"]["thresholds_and_frontiers_recomputed_each_replicate"]
    assert report["matched_null"]["permutation"]["missing_mask_preserved"]
    assert report["matched_null"]["independent_gaussian"]["outer_and_inner_missing_masks_preserved"]
    assert report["paired_group_bootstrap"]["baseline_midrank_auc"]["finite"] == 5000
    assert report["paired_group_bootstrap"]["candidate_midrank_auc"]["finite"] == 5000
    assert report["point_estimates"]["frontier35"]["n_bad"] > 0
    assert report["point_estimates"]["frontier50"]["n_gn"] > 0
    print(
        json.dumps(
            {
                "status": "PASS",
                "schema_version": SCHEMA_VERSION,
                "synthetic_samples": int(arrays["y"].size),
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "null_repeats": NULL_REPEATS,
                "frozen_harness_sha256": HARNESS_SHA256,
                "bundle_allowlist_guard": True,
                "manifest_group_alignment": True,
                "permutation_never_crosses_fold_or_source": True,
                "real_bundle_accessed": False,
                "real_manifest_accessed": False,
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
    subparsers.add_parser("selftest", help="run the fixed 5000/200 protocol on synthetic data")
    adjudicate = subparsers.add_parser("adjudicate", help="adjudicate one explicit discovery-only bundle")
    adjudicate.add_argument("--discovery-bundle", type=Path, required=True)
    adjudicate.add_argument("--expected-bundle-sha256", required=True)
    adjudicate.add_argument("--train-manifest", type=Path, required=True)
    adjudicate.add_argument("--expected-manifest-sha256", required=True)
    adjudicate.add_argument("--output-json", type=Path, required=True)
    adjudicate.add_argument("--confirm-discovery-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "selftest":
        return selftest()
    if args.command == "adjudicate":
        return adjudicate_command(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
