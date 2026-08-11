#!/usr/bin/env python3
"""Full-discovery frozen E51 refit and label-sealed shadow prediction.

Five outer epochs come only from an immutable candidate bundle; their upper
median is used with seed 20260900. Frozen b639 receipts must cover exactly the
shadow. Output has no labels; a final lock commits model and prediction SHA.
The selftest is synthetic-only and never imports torch or reads project data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

HEAD_SHA256 = "b10a0dcf0bc15ffb15e568f81d2587998284c9497424e3a21af40b3b7dd423ee"
EXTRACTOR_SHA256 = "b639dcc04dc60fae26c3a5725f3eebc46121b62e270c89f7e260ebb4b96bc37e"
CHECKPOINT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
REFIT_SEED, OUTER_FOLDS, MAX_EPOCHS = 20260900, 5, 30
SCHEMA = "e51_shadow_prediction_v1"
LOCK_SCHEMA = "e51_shadow_prediction_lock_v1"


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


def read_tokens(path: Path, role: str) -> list[str]:
    values = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{role} must contain unique non-empty tokens")
    if any(len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in values):
        raise ValueError(f"{role} contains an invalid token")
    return sorted(values)


def upper_median(values: Sequence[int]) -> int:
    ordered = sorted(int(value) for value in values)
    if len(ordered) != OUTER_FOLDS:
        raise ValueError("outer_best_epoch must contain exactly five values")
    if any(value < 1 or value > MAX_EPOCHS for value in ordered):
        raise ValueError("outer_best_epoch outside frozen 1..30 range")
    return ordered[len(ordered) // 2]


def string_vector(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a 1-D string vector")
    return array.astype(str, copy=False)


def validate_candidate(candidate_path: Path, metadata_path: Path, split_path: Path):
    with np.load(candidate_path, allow_pickle=False) as archive:
        keys = {"sample_token", "y", "fold_id", "outer_best_epoch", "artifact_id"}
        if keys - set(archive.files):
            raise KeyError(f"candidate missing {sorted(keys - set(archive.files))}")
        candidate = {key: np.asarray(archive[key]) for key in keys}
    with np.load(split_path, allow_pickle=False) as archive:
        keys = {"sample_token", "y", "fold_id"}
        if keys - set(archive.files):
            raise KeyError(f"S0 split missing {sorted(keys - set(archive.files))}")
        split = {key: np.asarray(archive[key]) for key in keys}
    tokens = string_vector("candidate sample_token", candidate["sample_token"])
    if tokens.size < 2 or not np.all(tokens[:-1] < tokens[1:]):
        raise ValueError("candidate tokens must be unique and strictly sorted")
    for key in ("sample_token", "y", "fold_id"):
        if not np.array_equal(candidate[key], split[key]):
            raise RuntimeError(f"candidate/S0 {key} mismatch")
    y, folds = np.asarray(candidate["y"]), np.asarray(candidate["fold_id"])
    if y.shape != (tokens.size,) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("candidate y contract failed")
    if folds.shape != (tokens.size,) or set(np.unique(folds).tolist()) != set(range(5)):
        raise ValueError("candidate fold contract failed")
    epoch = upper_median(np.asarray(candidate["outer_best_epoch"]).tolist())
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_id") != str(np.asarray(candidate["artifact_id"]).item()):
        raise RuntimeError("candidate bundle/metadata artifact_id mismatch")
    if metadata.get("code_sha256") != HEAD_SHA256:
        raise RuntimeError("candidate was not produced by frozen b10 head")
    if metadata.get("feature_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("candidate checkpoint provenance mismatch")
    if metadata.get("training", {}).get("shadow_refit_seed_reserved") != REFIT_SEED:
        raise RuntimeError("candidate does not reserve frozen shadow seed")
    return tokens, y.astype(np.int8, copy=False), epoch, metadata


def validate_receipts(
    paths: Sequence[Path], expected: Sequence[str], feature_root: Path,
    shadow_tokens: Sequence[str], shadow_sha: str, discovery_sha: str,
) -> list[str]:
    if not paths or len(paths) != len(expected):
        raise ValueError("receipt paths/SHA lists must be non-empty and equal")
    seen: set[str] = set()
    receipt_hashes: list[str] = []
    for index, (path, want) in enumerate(zip(paths, expected)):
        path = path.expanduser().resolve()
        receipt_hashes.append(verify_sha(path, want, f"feature receipt {index}"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = [str(value) for value in payload.get("selected_tokens", [])]
        if (
            payload.get("status") != "complete"
            or payload.get("scope") != "s0_prospective_shadow_label_blind"
            or payload.get("extractor_sha256") != EXTRACTOR_SHA256
            or payload.get("shadow_ids_sha256") != shadow_sha
            or payload.get("discovery_ids_sha256") != discovery_sha
            or payload.get("label_accessed") is not False
            or payload.get("metric_computed") is not False
            or not tokens
        ):
            raise RuntimeError(f"feature receipt {index} contract failed")
        artifacts = payload.get("artifacts", {})
        for token in tokens:
            if token in seen:
                raise RuntimeError("feature receipts overlap")
            seen.add(token)
            hashes = artifacts.get(token, {})
            meta_path, array_path = feature_root / f"{token}.json", feature_root / f"{token}.npz"
            if sha256_file(meta_path) != hashes.get("json_sha256"):
                raise RuntimeError(f"feature JSON changed after receipt: {token}")
            if sha256_file(array_path) != hashes.get("npz_sha256"):
                raise RuntimeError(f"feature NPZ changed after receipt: {token}")
    if seen != set(shadow_tokens):
        raise RuntimeError("feature receipts do not exactly cover S0 shadow")
    return receipt_hashes


def validate_feature_scope(root: Path, tokens: Sequence[str], scope: str, discovery_sha: str | None) -> None:
    for token in tokens:
        path = root / f"{token}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        selection = payload.get("selection", {})
        if (
            payload.get("sample_token") != token
            or payload.get("status") != "ok"
            or payload.get("label_accessed") is not False
            or payload.get("metric_computed") is not False
            or selection.get("scope") != scope
            or selection.get("discovery_ids_sha256") != discovery_sha
        ):
            raise RuntimeError(f"feature scope mismatch for {token}")


def import_frozen_head(path: Path):
    verify_sha(path, HEAD_SHA256, "frozen head")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_frozen_e51_head", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, expected in (("SHADOW_REFIT_SEED", REFIT_SEED), ("MAX_EPOCHS", 30), ("OUTER_FOLDS", 5)):
        if getattr(module, name) != expected:
            raise RuntimeError(f"frozen head constant changed: {name}")
    return module


def train_and_predict(head: Any, discovery_root: Path, tokens: np.ndarray, y: np.ndarray,
                      shadow_root: Path, shadow_tokens: np.ndarray, epochs: int, device_name: str):
    import torch
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production shadow refit requires one CUDA GPU")
    train_store = head.DenseFeatureStore(discovery_root, tokens)
    model = head.new_model(REFIT_SEED, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=head.LEARNING_RATE, weight_decay=head.WEIGHT_DECAY)
    indices = np.arange(tokens.size, dtype=np.int64)
    for epoch in range(1, epochs + 1):
        head.train_one_epoch(
            model, optimizer, train_store, y, indices, device,
            order_seed=REFIT_SEED * 100 + epoch,
        )
    del optimizer, train_store
    gc.collect()
    shadow_store = head.DenseFeatureStore(shadow_root, shadow_tokens)
    predictions = head.predict(model, shadow_store, np.arange(shadow_tokens.size), device)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    del shadow_store, model
    torch.cuda.empty_cache()
    return predictions, state


def commit_predictions(
    prediction_path: Path, model_path: Path, tokens: np.ndarray, predictions: np.ndarray,
    provenance: Mapping[str, Any], write_model: Callable[[Path], None],
) -> tuple[str, str, str]:
    prediction_path, model_path = prediction_path.expanduser().resolve(), model_path.expanduser().resolve()
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != np.asarray(tokens).shape or not np.isfinite(predictions).all():
        raise ValueError("prediction vector shape/finite contract failed")
    if np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("E51 predictions must lie in [0,1]")
    lock_path = prediction_path.with_suffix(".lock.json")
    for path in (prediction_path, model_path, lock_path):
        folded = path.name.casefold()
        if "eval" in folded or "shadow" not in folded:
            raise ValueError("outputs must be shadow-named and not eval-named")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_id = secrets.token_hex(16)
    tmp_p = prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.tmp")
    tmp_m = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    tmp_l = lock_path.with_name(f".{lock_path.name}.{os.getpid()}.tmp")
    try:
        with tmp_p.open("wb") as handle:
            np.savez_compressed(
                handle, schema_version=np.asarray(SCHEMA), artifact_id=np.asarray(artifact_id),
                sample_token=np.asarray(tokens, dtype="<U64"),
                prediction=predictions.astype(np.float32),
                model_kind=np.asarray("e51_vjepa21_b384_dense_local"),
                fixed_epoch=np.asarray(int(provenance["fixed_epoch"]), dtype=np.int16),
                refit_seed=np.asarray(REFIT_SEED, dtype=np.int64),
                provenance_json=np.asarray(json.dumps(dict(provenance), sort_keys=True)),
            )
            handle.flush(); os.fsync(handle.fileno())
        write_model(tmp_m)
        prediction_sha, model_sha = sha256_file(tmp_p), sha256_file(tmp_m)
        lock = {
            "schema_version": LOCK_SCHEMA, "status": "PREDICTIONS_LOCKED",
            "artifact_id": artifact_id, "prediction_sha256": prediction_sha,
            "model_sha256": model_sha, "sample_count": int(tokens.size),
            "sample_tokens_sha256": hashlib.sha256("\0".join(tokens.tolist()).encode()).hexdigest(),
            "label_accessed": False, "metric_computed": False, "provenance": dict(provenance),
        }
        with tmp_l.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(lock, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp_m, model_path); os.replace(tmp_p, prediction_path); os.replace(tmp_l, lock_path)
    finally:
        for path in (tmp_p, tmp_m, tmp_l):
            if path.exists():
                path.unlink()
    return sha256_file(prediction_path), sha256_file(model_path), sha256_file(lock_path)


def apply(args: argparse.Namespace) -> int:
    if not args.confirm_shadow_labels_sealed:
        raise RuntimeError("apply requires --confirm-shadow-labels-sealed")
    paths = {
        "candidate": args.candidate_bundle.expanduser().resolve(),
        "metadata": args.candidate_metadata.expanduser().resolve(),
        "split": args.s0_split.expanduser().resolve(),
        "discovery_ids": args.discovery_ids.expanduser().resolve(),
        "shadow_ids": args.shadow_ids.expanduser().resolve(),
        "head": args.head_script.expanduser().resolve(),
    }
    hashes = {
        "candidate_bundle_sha256": verify_sha(paths["candidate"], args.expected_candidate_sha256, "candidate"),
        "candidate_metadata_sha256": verify_sha(paths["metadata"], args.expected_candidate_metadata_sha256, "candidate metadata"),
        "s0_split_sha256": verify_sha(paths["split"], args.expected_s0_split_sha256, "S0 split"),
        "discovery_ids_sha256": verify_sha(paths["discovery_ids"], args.expected_discovery_ids_sha256, "discovery ids"),
        "shadow_ids_sha256": verify_sha(paths["shadow_ids"], args.expected_shadow_ids_sha256, "shadow ids"),
    }
    tokens, y, epochs, _ = validate_candidate(paths["candidate"], paths["metadata"], paths["split"])
    discovery_ids, shadow_ids = read_tokens(paths["discovery_ids"], "discovery ids"), read_tokens(paths["shadow_ids"], "shadow ids")
    if discovery_ids != tokens.tolist() or set(discovery_ids).intersection(shadow_ids):
        raise RuntimeError("S0 candidate/discovery/shadow membership contract failed")
    discovery_root = args.discovery_features_root.expanduser().resolve()
    shadow_root = args.shadow_features_root.expanduser().resolve()
    receipt_hashes = validate_receipts(
        args.shadow_feature_receipt, args.expected_shadow_feature_receipt_sha256,
        shadow_root, shadow_ids, hashes["shadow_ids_sha256"], hashes["discovery_ids_sha256"],
    )
    validate_feature_scope(discovery_root, discovery_ids, "s0_discovery", hashes["discovery_ids_sha256"])
    validate_feature_scope(shadow_root, shadow_ids, "technical_subset", None)
    head = import_frozen_head(paths["head"])
    predictions, state = train_and_predict(
        head, discovery_root, tokens, y, shadow_root,
        np.asarray(shadow_ids, dtype="<U64"), epochs, args.device,
    )
    provenance = {
        **hashes, "head_sha256": HEAD_SHA256, "extractor_sha256": EXTRACTOR_SHA256,
        "shadow_apply_code_sha256": sha256_file(Path(__file__).resolve()),
        "feature_checkpoint_sha256": CHECKPOINT_SHA256,
        "shadow_feature_receipt_sha256": receipt_hashes, "fixed_epoch": epochs,
        "outer_best_epoch_rule": "upper median (third sorted) of 5 frozen outer_best_epoch",
        "refit_seed": REFIT_SEED, "optimizer": "AdamW(lr=3e-4,weight_decay=1e-4)",
        "batch_size": 32,
        "discovery_cache_aggregate_sha256": "NOT_AVAILABLE_IN_FROZEN_DISCOVERY_BUNDLE",
        "shadow_labels_accessed": False, "metric_computed": False,
    }
    def write_model(path: Path) -> None:
        import torch
        torch.save({"schema_version": "e51_shadow_model_v1", "state_dict": state, "provenance": provenance}, path)
    prediction_sha, model_sha, lock_sha = commit_predictions(
        args.output_predictions, args.output_model, np.asarray(shadow_ids, dtype="<U64"),
        predictions, provenance, write_model,
    )
    print(json.dumps({
        "status": "PREDICTIONS_LOCKED", "samples": len(shadow_ids),
        "fixed_epoch": epochs, "refit_seed": REFIT_SEED,
        "prediction_sha256": prediction_sha, "model_sha256": model_sha,
        "lock_sha256": lock_sha, "shadow_labels_accessed": False, "metric_computed": False,
    }, sort_keys=True), flush=True)
    return 0


def selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="e51-shadow-apply-selftest-") as temp:
        root, n = Path(temp), 20
        tokens = np.asarray([f"{i:064x}" for i in range(n)], dtype="<U64")
        y, folds, artifact_id = np.arange(n) % 2, np.arange(n) % 5, "1" * 32
        candidate, split, metadata = root / "candidate.npz", root / "split.npz", root / "candidate.json"
        np.savez_compressed(candidate, sample_token=tokens, y=y, fold_id=folds,
                            outer_best_epoch=np.asarray([7, 1, 6, 4, 5]), artifact_id=np.asarray(artifact_id))
        np.savez_compressed(split, sample_token=tokens, y=y, fold_id=folds)
        metadata.write_text(json.dumps({
            "artifact_id": artifact_id, "code_sha256": HEAD_SHA256,
            "feature_checkpoint_sha256": CHECKPOINT_SHA256,
            "training": {"shadow_refit_seed_reserved": REFIT_SEED},
        }), encoding="utf-8")
        checked_tokens, checked_y, epoch, _ = validate_candidate(candidate, metadata, split)
        assert np.array_equal(checked_tokens, tokens) and np.array_equal(checked_y, y) and epoch == 5
        output, model = root / "shadow_predictions.npz", root / "shadow_model.pt"
        prediction_sha, model_sha, _ = commit_predictions(
            output, model, tokens[:6], np.linspace(0.1, 0.9, 6),
            {"fixed_epoch": epoch, "synthetic_only": True},
            lambda path: path.write_bytes(b"synthetic-model"),
        )
        assert sha256_file(output) == prediction_sha and sha256_file(model) == model_sha
        with np.load(output, allow_pickle=False) as archive:
            assert "y" not in archive.files and "label" not in archive.files
        lock = json.loads(output.with_suffix(".lock.json").read_text(encoding="utf-8"))
        assert lock["status"] == "PREDICTIONS_LOCKED" and lock["prediction_sha256"] == prediction_sha
    print("E51 shadow full-discovery refit/apply self-test: OK")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    ap = sub.add_parser("apply")
    ap.add_argument("--candidate-bundle", type=Path, required=True)
    ap.add_argument("--expected-candidate-sha256", required=True)
    ap.add_argument("--candidate-metadata", type=Path, required=True)
    ap.add_argument("--expected-candidate-metadata-sha256", required=True)
    ap.add_argument("--s0-split", type=Path, required=True)
    ap.add_argument("--expected-s0-split-sha256", required=True)
    ap.add_argument("--discovery-ids", type=Path, required=True)
    ap.add_argument("--expected-discovery-ids-sha256", required=True)
    ap.add_argument("--shadow-ids", type=Path, required=True)
    ap.add_argument("--expected-shadow-ids-sha256", required=True)
    ap.add_argument("--discovery-features-root", type=Path, required=True)
    ap.add_argument("--shadow-features-root", type=Path, required=True)
    ap.add_argument("--shadow-feature-receipt", type=Path, action="append", required=True)
    ap.add_argument("--expected-shadow-feature-receipt-sha256", action="append", required=True)
    ap.add_argument("--head-script", type=Path, required=True)
    ap.add_argument("--output-predictions", type=Path, required=True)
    ap.add_argument("--output-model", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--confirm-shadow-labels-sealed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return selftest() if args.command == "selftest" else apply(args)


if __name__ == "__main__":
    raise SystemExit(main())
