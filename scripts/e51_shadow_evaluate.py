#!/usr/bin/env python3
"""One-shot prospective-shadow evaluator for locked E51 and paired E18.

The evaluator first verifies both expected prediction SHA-256 values, their
lock receipts, exact S0 shadow membership, frozen discovery reference, and the
frozen strict-rank harness. It then atomically claims the one permitted shadow
evaluation. Only after that claim may it parse shadow labels. It reports strict
gn@95, paired 5000-draw grouped percentile bootstrap (seed 20260810), and all
S0 sources. No config field was frozen, so config is NOT_ASSESSABLE and this
script can never claim a complete shadow PASS or final-eval eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

HARNESS_SHA256 = "66dc519841074c9f578cb186d4732788a8e2d0fb5dfafd700e8b74ea1a1f7119"
E51_SCHEMA = "e51_shadow_prediction_v1"
E18_SCHEMA = "e18_shadow_prediction_v1"
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_REPEATS = 5000
RECALL = 0.95
CONFIG_FIELD = None
VIDEO_FIELD = re.compile(r'"video"\s*:\s*("(?:\\.|[^"\\])*")')
VALID_LABELS = {"bad", "good", "normal"}
EPS = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_sha(value: str, role: str) -> str:
    value = value.strip().casefold()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{role} SHA-256 must be 64 lowercase hex")
    return value


def verify_sha(path: Path, expected: str, role: str) -> str:
    actual = sha256_file(path)
    if actual != checked_sha(expected, role):
        raise RuntimeError(f"{role} SHA mismatch: expected {expected}, got {actual}")
    return actual


def sample_token(video: str) -> str:
    return hashlib.sha256(b"e50-sample-v1\0" + video.encode("utf-8")).hexdigest()


def read_tokens(path: Path) -> list[str]:
    values = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("shadow_ids must contain unique non-empty tokens")
    if any(len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in values):
        raise ValueError("shadow_ids contains invalid token")
    return sorted(values)


def string_vector(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a 1-D string vector")
    return array.astype(str, copy=False)


def import_harness(path: Path):
    verify_sha(path, HARNESS_SHA256, "frozen frontier harness")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_frozen_frontier_harness", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_locked_prediction(
    path: Path, expected_sha: str, lock_path: Path, expected_schema: str, role: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    path, lock_path = path.expanduser().resolve(), lock_path.expanduser().resolve()
    prediction_sha = verify_sha(path, expected_sha, f"{role} prediction")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("status") != "PREDICTIONS_LOCKED"
        or lock.get("prediction_sha256") != prediction_sha
        or lock.get("label_accessed") is not False
        or lock.get("metric_computed") is not False
    ):
        raise RuntimeError(f"{role} prediction lock contract failed")
    with np.load(path, allow_pickle=False) as archive:
        forbidden = {"y", "label", "labels", "source", "strata", "group", "metric"}
        if forbidden.intersection(archive.files):
            raise RuntimeError(f"{role} prediction artifact contains sealed fields")
        required = {"schema_version", "artifact_id", "sample_token", "prediction"}
        if required - set(archive.files):
            raise KeyError(f"{role} prediction missing {sorted(required - set(archive.files))}")
        schema = str(np.asarray(archive["schema_version"]).item())
        artifact_id = str(np.asarray(archive["artifact_id"]).item())
        tokens = string_vector(f"{role} tokens", archive["sample_token"])
        predictions = np.asarray(archive["prediction"], dtype=np.float64)
    if schema != expected_schema or lock.get("artifact_id") != artifact_id:
        raise RuntimeError(f"{role} schema/artifact_id lock mismatch")
    if tokens.size < 2 or not np.all(tokens[:-1] < tokens[1:]):
        raise ValueError(f"{role} tokens must be strictly sorted")
    if predictions.shape != tokens.shape or not np.isfinite(predictions).all():
        raise ValueError(f"{role} prediction vector contract failed")
    if np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError(f"{role} predictions must lie in [0,1]")
    if int(lock.get("sample_count", -1)) != tokens.size:
        raise RuntimeError(f"{role} lock sample count mismatch")
    return tokens, predictions, lock, sha256_file(lock_path)


def load_discovery_reference(path: Path, expected_sha: str):
    artifact_sha = verify_sha(path, expected_sha, "discovery reference")
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_token", "strata", "base_oof", "expert_oof"}
        if required - set(archive.files):
            raise KeyError(f"discovery reference missing {sorted(required - set(archive.files))}")
        tokens = string_vector("discovery tokens", archive["sample_token"])
        strata = string_vector("discovery strata", archive["strata"])
        base = np.asarray(archive["base_oof"], dtype=np.float64)
        expert = np.asarray(archive["expert_oof"], dtype=np.float64)
    if tokens.size < 2 or not np.all(tokens[:-1] < tokens[1:]):
        raise ValueError("discovery tokens must be strictly sorted")
    if strata.shape != tokens.shape or base.shape != tokens.shape or expert.shape != tokens.shape:
        raise ValueError("discovery reference shape mismatch")
    if not np.isfinite(base).all() or not np.isfinite(expert).all():
        raise ValueError("discovery reference must be complete and finite")
    expected_sources = sorted(np.unique(strata).tolist())
    if len(expected_sources) != 3:
        raise RuntimeError(
            f"frozen S0 discovery reference must contain exactly 3 sources, got {len(expected_sources)}"
        )
    return base, expert, expected_sources, artifact_sha


def derive_source(row: Mapping[str, Any]) -> str:
    raw = str(row.get("abs_path", "")).replace("\\", "/")
    parts = PurePosixPath(raw).parts
    for marker in ("s3", "corpus_videos"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return parts[-2] if len(parts) >= 2 else "unknown"


def opaque_group(row: Mapping[str, Any], token: str) -> str:
    for field in ("asset_id", "request_id", "batch_id"):
        value = str(row.get(field, "")).strip()
        if value:
            return f"{field}:{hashlib.sha256(value.encode()).hexdigest()}"
    prompt = " ".join(str(row.get("prompt", "")).casefold().split())
    if prompt:
        return f"prompt:{hashlib.sha256(prompt.encode()).hexdigest()}"
    return f"sample:{token}"


def load_shadow_labels(manifest: Path, tokens: Sequence[str]):
    """Parse full JSON only for shadow members; discovery labels stay untouched."""
    wanted = set(tokens)
    rows: dict[str, tuple[int, str, str]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            matches = VIDEO_FIELD.findall(line)
            if len(matches) != 1:
                raise ValueError(f"manifest line {lineno}: expected one video field")
            token = sample_token(json.loads(matches[0]))
            if token not in wanted:
                continue
            raw = json.loads(line)
            label = str(raw.get("label", ""))
            if label not in VALID_LABELS or token in rows:
                raise ValueError(f"shadow manifest row contract failed at line {lineno}")
            rows[token] = (1 if label == "bad" else 0, derive_source(raw), opaque_group(raw, token))
    if set(rows) != wanted:
        raise RuntimeError("not every S0 shadow token has one manifest row")
    y = np.asarray([rows[token][0] for token in tokens], dtype=np.int8)
    source = np.asarray([rows[token][1] for token in tokens], dtype=np.str_)
    group = np.asarray([rows[token][2] for token in tokens], dtype=np.str_)
    if np.unique(y).size != 2:
        raise RuntimeError("shadow must contain both binary classes")
    return y, source, group


def paired_bootstrap(
    base: np.ndarray, candidate: np.ndarray, y: np.ndarray, groups: np.ndarray,
    harness: Any, repeats: int = BOOTSTRAP_REPEATS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    grouped = unique_groups.size < y.size
    group_indices = [np.flatnonzero(inverse == index) for index in range(unique_groups.size)]
    bad_indices, rel_indices = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = np.empty(repeats, dtype=np.float64)
    for draw in range(repeats):
        if grouped:
            sampled = rng.integers(0, unique_groups.size, size=unique_groups.size)
            index = np.concatenate([group_indices[value] for value in sampled])
            if np.unique(y[index]).size != 2:
                raise RuntimeError("group bootstrap draw lost a class; one-shot evaluation stops")
        else:
            index = np.concatenate([
                rng.choice(bad_indices, size=bad_indices.size, replace=True),
                rng.choice(rel_indices, size=rel_indices.size, replace=True),
            ])
        base_value = harness.gn_at_recall(base[index], y[index], RECALL).value
        candidate_value = harness.gn_at_recall(candidate[index], y[index], RECALL).value
        deltas[draw] = candidate_value - base_value
    lower, upper = np.quantile(deltas, [0.025, 0.975], method="linear")
    return {
        "method": "paired grouped percentile bootstrap" if grouped else "paired label-stratified percentile bootstrap",
        "group_priority": "asset/request/batch/prompt; sample fallback",
        "repeats": repeats, "seed": seed,
        "ci95_lower": float(lower), "ci95_upper": float(upper),
        "delta_mean": float(deltas.mean()), "group_count": int(unique_groups.size),
    }


def source_gate(
    sources: np.ndarray, expected_sources: Sequence[str], y: np.ndarray,
    base: np.ndarray, candidate: np.ndarray, base_threshold: float, candidate_threshold: float,
) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    passed = True
    for source in expected_sources:
        index = (sources == source) & (y == 1)
        count = int(index.sum())
        if count:
            base_recall = float(np.mean(base[index] >= base_threshold))
            candidate_recall = float(np.mean(candidate[index] >= candidate_threshold))
            delta = candidate_recall - base_recall
            source_pass = delta >= -0.05 - EPS
        else:
            base_recall = candidate_recall = delta = None
            source_pass = False
        passed = passed and source_pass
        records.append({
            "source": source, "n_bad": count, "e18_bad_recall": base_recall,
            "candidate_bad_recall": candidate_recall, "delta": delta,
            "max_allowed_drop_pt": 5.0, "pass": source_pass,
        })
    unexpected = sorted(set(np.unique(sources).tolist()) - set(expected_sources))
    if unexpected:
        passed = False
        records.append({"unexpected_shadow_sources": unexpected, "pass": False})
    return records, passed


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def claim_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("prospective shadow has already been claimed/evaluated") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def evaluate_once(
    args: argparse.Namespace,
    label_loader: Callable[[Path, Sequence[str]], tuple[np.ndarray, np.ndarray, np.ndarray]] = load_shadow_labels,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    # This confirmation and both prediction locks are deliberately the first gate.
    if not args.confirm_predictions_locked:
        raise RuntimeError("evaluation requires --confirm-predictions-locked")
    e51_tokens, e51_prediction, _, e51_lock_sha = load_locked_prediction(
        args.e51_predictions, args.expected_e51_prediction_sha256,
        args.e51_lock, E51_SCHEMA, "E51",
    )
    e18_tokens, e18_prediction, _, e18_lock_sha = load_locked_prediction(
        args.e18_predictions, args.expected_e18_prediction_sha256,
        args.e18_lock, E18_SCHEMA, "E18",
    )
    if not np.array_equal(e51_tokens, e18_tokens):
        raise RuntimeError("E51/E18 shadow token order mismatch")
    shadow_ids_path = args.shadow_ids.expanduser().resolve()
    shadow_ids_sha = verify_sha(shadow_ids_path, args.expected_shadow_ids_sha256, "shadow ids")
    shadow_tokens = read_tokens(shadow_ids_path)
    if shadow_tokens != e51_tokens.tolist():
        raise RuntimeError("locked predictions do not exactly cover S0 shadow")
    base_ref, expert_ref, expected_sources, discovery_sha = load_discovery_reference(
        args.discovery_reference.expanduser().resolve(), args.expected_discovery_reference_sha256
    )
    harness = import_harness(args.harness_script.expanduser().resolve())
    output = args.output_result.expanduser().resolve()
    claim = output.with_suffix(".claim.json")
    for path in (output, claim):
        folded = path.name.casefold()
        if "eval" in folded or "shadow" not in folded:
            raise ValueError("result/claim must be shadow-named and not eval-named")
        if path.exists():
            raise RuntimeError("shadow result or one-shot claim already exists")
    manifest = args.train_manifest.expanduser().resolve()
    if "eval" in manifest.name.casefold():
        raise ValueError("refusing eval-like manifest")
    manifest_expected = checked_sha(args.expected_train_manifest_sha256, "train manifest")
    claim_once(claim, {
        "status": "SHADOW_EVALUATION_CLAIMED",
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "e51_prediction_sha256": checked_sha(args.expected_e51_prediction_sha256, "E51 prediction"),
        "e18_prediction_sha256": checked_sha(args.expected_e18_prediction_sha256, "E18 prediction"),
        "shadow_ids_sha256": shadow_ids_sha, "discovery_reference_sha256": discovery_sha,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_repeats": bootstrap_repeats,
        "config_gate": "NOT_ASSESSABLE_CONFIG_FIELD_NOT_FROZEN",
    })
    # First permitted label-bearing input access begins only after the durable claim.
    manifest_sha = verify_sha(manifest, manifest_expected, "train manifest")
    y, sources, groups = label_loader(manifest, shadow_tokens)
    r0 = harness.empirical_mid_cdf(base_ref, e18_prediction)
    rj = harness.empirical_mid_cdf(expert_ref, e51_prediction)
    candidate = harness.protocol_e51_scores(r0, rj)
    baseline_metric = harness.gn_at_recall(r0, y, RECALL)
    candidate_metric = harness.gn_at_recall(candidate, y, RECALL)
    delta = float(candidate_metric.value - baseline_metric.value)
    bootstrap = paired_bootstrap(r0, candidate, y, groups, harness, bootstrap_repeats, BOOTSTRAP_SEED)
    source_records, sources_pass = source_gate(
        sources, expected_sources, y, r0, candidate,
        baseline_metric.threshold, candidate_metric.threshold,
    )
    empirical_recall = candidate_metric.caught_bad / candidate_metric.n_bad
    gates = {
        "gn95_at_least_0_35": candidate_metric.value >= 0.35 - EPS,
        "delta_at_least_0_05": delta >= 0.05 - EPS,
        "paired_bootstrap_ci_lower_gt_0": bootstrap["ci95_lower"] > 0.0,
        "empirical_bad_recall_at_least_0_95": empirical_recall >= 0.95 - EPS,
        "all_s0_sources_bad_recall_drop_within_5pt": sources_pass,
        "config_bad_recall_drop_within_5pt": "NOT_ASSESSABLE_CONFIG_FIELD_NOT_FROZEN",
    }
    assessable_pass = all(value is True for key, value in gates.items() if not key.startswith("config_"))
    status = "NOT_ASSESSABLE_CONFIG_FIELD_NOT_FROZEN" if assessable_pass else "FAIL_ASSESSABLE_SHADOW_GATE"
    result = {
        "schema_version": "e51_prospective_shadow_result_v1",
        "status": status, "complete_shadow_gate_pass": False,
        "final_eval_eligible": False,
        "prospective_shadow_notice": "prospective for the new hypothesis, not historically untouched",
        "protocol": {
            "recall": RECALL, "release_rule": "strict score < threshold",
            "rank_reference": "full discovery cross-fitted base_oof/expert_oof empirical mid-CDF",
            "formula": "r0+0.25*sigmoid((0.45-r0)/0.08)*(rj-0.5)",
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_repeats": bootstrap_repeats,
            "config_field": CONFIG_FIELD,
        },
        "baseline_e18": baseline_metric.__dict__,
        "candidate_e51": candidate_metric.__dict__,
        "delta_gn95": delta, "empirical_candidate_bad_recall": empirical_recall,
        "paired_bootstrap": bootstrap, "source_diagnostics": source_records,
        "gates": gates,
        "stretch_50_shadow_only": {
            "gn95_at_least_0_50": candidate_metric.value >= 0.50 - EPS,
            "note": "does not independently establish discovery 50-stretch gates",
        },
        "locked_inputs": {
            "e51_prediction_sha256": checked_sha(args.expected_e51_prediction_sha256, "E51 prediction"),
            "e51_lock_sha256": e51_lock_sha,
            "e18_prediction_sha256": checked_sha(args.expected_e18_prediction_sha256, "E18 prediction"),
            "e18_lock_sha256": e18_lock_sha, "shadow_ids_sha256": shadow_ids_sha,
            "discovery_reference_sha256": discovery_sha, "train_manifest_sha256": manifest_sha,
            "harness_sha256": HARNESS_SHA256,
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "sample_count": len(shadow_tokens), "single_sample_output": False,
    }
    atomic_json(output, result)
    return result


def write_synthetic_prediction(path: Path, schema: str, tokens: np.ndarray, prediction: np.ndarray) -> tuple[str, Path]:
    artifact_id = "a" * 32
    np.savez_compressed(
        path, schema_version=np.asarray(schema), artifact_id=np.asarray(artifact_id),
        sample_token=tokens, prediction=prediction,
    )
    prediction_sha = sha256_file(path)
    lock_path = path.with_suffix(".lock.json")
    lock_path.write_text(json.dumps({
        "status": "PREDICTIONS_LOCKED", "artifact_id": artifact_id,
        "prediction_sha256": prediction_sha, "sample_count": int(tokens.size),
        "label_accessed": False, "metric_computed": False,
    }), encoding="utf-8")
    return prediction_sha, lock_path


def selftest() -> int:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    with tempfile.TemporaryDirectory(prefix="e51-shadow-eval-selftest-") as temp:
        root, n, d = Path(temp), 120, 300
        shadow_tokens = np.asarray([f"{i:064x}" for i in range(n)], dtype="<U64")
        discovery_tokens = np.asarray([f"{1000+i:064x}" for i in range(d)], dtype="<U64")
        shadow_ids = root / "shadow_ids.txt"
        shadow_ids.write_text("\n".join(shadow_tokens.tolist()) + "\n", encoding="utf-8")
        e18_pred = np.linspace(0.05, 0.95, n)
        e51_pred = np.clip(e18_pred + 0.15 * np.sin(np.arange(n)), 0, 1)
        e51_path, e18_path = root / "shadow_e51.npz", root / "shadow_e18.npz"
        e51_sha, e51_lock = write_synthetic_prediction(e51_path, E51_SCHEMA, shadow_tokens, e51_pred)
        e18_sha, e18_lock = write_synthetic_prediction(e18_path, E18_SCHEMA, shadow_tokens, e18_pred)
        reference = root / "discovery_reference.npz"
        np.savez_compressed(
            reference, sample_token=discovery_tokens,
            strata=np.asarray(["s0", "s1", "s2"] * (d // 3)),
            base_oof=np.linspace(0, 1, d), expert_oof=np.clip(rng.normal(0.5, 0.2, d), 0, 1),
        )
        manifest = root / "train_v3.jsonl"
        manifest.write_text("{}\n", encoding="utf-8")
        harness_path = Path(__file__).resolve().with_name("e50_frontier_harness.py")
        output = root / "shadow_result.json"
        args = argparse.Namespace(
            confirm_predictions_locked=False,
            e51_predictions=e51_path, expected_e51_prediction_sha256=e51_sha, e51_lock=e51_lock,
            e18_predictions=e18_path, expected_e18_prediction_sha256=e18_sha, e18_lock=e18_lock,
            shadow_ids=shadow_ids, expected_shadow_ids_sha256=sha256_file(shadow_ids),
            discovery_reference=reference, expected_discovery_reference_sha256=sha256_file(reference),
            harness_script=harness_path, output_result=output,
            train_manifest=manifest, expected_train_manifest_sha256=sha256_file(manifest),
        )
        calls = {"count": 0}
        y = np.asarray(([1, 0] * (n // 2)), dtype=np.int8)
        sources = np.asarray(["s0", "s1", "s2"] * (n // 3))
        groups = np.asarray([f"sample:{i}" for i in range(n)])
        def loader(_path: Path, _tokens: Sequence[str]):
            calls["count"] += 1
            return y, sources, groups
        try:
            evaluate_once(args, loader, 100)
        except RuntimeError as exc:
            assert "confirm" in str(exc)
        else:
            raise AssertionError("missing confirmation accepted")
        assert calls["count"] == 0 and not output.with_suffix(".claim.json").exists()
        args.confirm_predictions_locked = True
        original_sha = args.expected_e51_prediction_sha256
        args.expected_e51_prediction_sha256 = "0" * 64
        try:
            evaluate_once(args, loader, 100)
        except RuntimeError as exc:
            assert "SHA mismatch" in str(exc)
        else:
            raise AssertionError("wrong prediction SHA accepted")
        assert calls["count"] == 0 and not output.with_suffix(".claim.json").exists()
        args.expected_e51_prediction_sha256 = original_sha
        result = evaluate_once(args, loader, 100)
        assert calls["count"] == 1 and output.exists()
        assert result["complete_shadow_gate_pass"] is False
        assert result["status"] in {"NOT_ASSESSABLE_CONFIG_FIELD_NOT_FROZEN", "FAIL_ASSESSABLE_SHADOW_GATE"}
        try:
            evaluate_once(args, loader, 100)
        except RuntimeError:
            pass
        else:
            raise AssertionError("one-shot claim did not block repeat")
        assert calls["count"] == 1
    print("E51 prospective-shadow one-shot evaluator self-test: OK")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    ap = sub.add_parser("evaluate")
    ap.add_argument("--e51-predictions", type=Path, required=True)
    ap.add_argument("--expected-e51-prediction-sha256", required=True)
    ap.add_argument("--e51-lock", type=Path, required=True)
    ap.add_argument("--e18-predictions", type=Path, required=True)
    ap.add_argument("--expected-e18-prediction-sha256", required=True)
    ap.add_argument("--e18-lock", type=Path, required=True)
    ap.add_argument("--shadow-ids", type=Path, required=True)
    ap.add_argument("--expected-shadow-ids-sha256", required=True)
    ap.add_argument("--discovery-reference", type=Path, required=True)
    ap.add_argument("--expected-discovery-reference-sha256", required=True)
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--expected-train-manifest-sha256", required=True)
    ap.add_argument("--harness-script", type=Path, required=True)
    ap.add_argument("--output-result", type=Path, required=True)
    ap.add_argument("--confirm-predictions-locked", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "selftest":
        return selftest()
    result = evaluate_once(args)
    print(json.dumps({
        "status": result["status"], "complete_shadow_gate_pass": False,
        "final_eval_eligible": False, "output": str(args.output_result),
        "output_sha256": sha256_file(args.output_result.expanduser().resolve()),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
