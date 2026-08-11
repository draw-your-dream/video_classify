#!/usr/bin/env python3
"""Frozen E51 local dense head and strict nested-OOF trainer.

The default-safe entry point is ``selftest`` and consumes only synthetic
tensors.  The ``train`` entry point has no project-data defaults: it requires
an immutable S0 directory, the preregistered train manifest, and a complete E51
discovery feature cache.  It never opens shadow_ids.txt or any eval artifact.

Candidate bundle contract:

* ``expert_oof [N]``: one prediction from the full outer-train refit for each
  outer-valid sample;
* ``inner_expert [5,N]``: row k is four-way inner OOF on outer-train k and is
  NaN on outer-valid k;
* ``y [N]``, ``fold_id [N]``, ``sample_token [N]``, and ``strata [N]`` allow a
  later merge with the paired E18 bundle for e50_frontier_harness.py.

No metric, frontier, threshold, rank-CDF, model selection, or eval prediction
is computed here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import secrets
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from e50_make_shadow import derive_source, sample_token, seeded_rank, sha256_file
from e51_vjepa21_dense_extract import (
    CHECKPOINT_SHA256,
    SCHEMA_VERSION as FEATURE_SCHEMA,
    TRAIN_MANIFEST_NAME,
    TRAIN_MANIFEST_SHA256,
    output_pair_complete,
)


HEAD_SCHEMA = "e51_dense_head_nested_v1"
VIEWS = 2
TOKEN_TIME = 32
DIFF_TIME = TOKEN_TIME - 1
GRID = 12
INPUT_DIM = 768
HIDDEN_DIM = 256
ATTENTION_HEADS = 4
DROPOUT = 0.2
LME_TEMPERATURE = 0.10

OUTER_FOLDS = 5
INNER_FOLDS = 4
INNER_SPLIT_SEED = 20260811
INNER_MODEL_SEED_BASE = 20260820
OUTER_REFIT_SEED_BASE = 20260880
SHADOW_REFIT_SEED = 20260900

LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
BATCH_SIZE = 32
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5

DENSE_SHAPE = (VIEWS, TOKEN_TIME, GRID, GRID, INPUT_DIM)
VALID_LABELS = frozenset({"bad", "good", "normal"})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def set_deterministic(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Training determinism is more important than Flash-SDPA throughput for
        # the preregistered small head.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div)
    return encoding.unsqueeze(0)


class E51DenseLocalHead(nn.Module):
    """The single preregistered E51 local temporal architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(
            INPUT_DIM, HIDDEN_DIM, kernel_size=3, stride=1, padding=1
        )
        self.temporal_attention = nn.MultiheadAttention(
            HIDDEN_DIM,
            ATTENTION_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.bad_head = nn.Linear(HIDDEN_DIM, 1)
        self.register_buffer(
            "temporal_position",
            sinusoidal_encoding(DIFF_TIME, HIDDEN_DIM),
            persistent=True,
        )

    @staticmethod
    def normalized_log_mean_exp(
        positions: torch.Tensor, temperature: float = LME_TEMPERATURE
    ) -> torch.Tensor:
        if positions.ndim != 3 or positions.shape[-1] != HIDDEN_DIM:
            raise ValueError(
                f"positions must be [B,P,{HIDDEN_DIM}], got {tuple(positions.shape)}"
            )
        count = positions.shape[1]
        if count <= 0:
            raise ValueError("position count must be positive")
        values = positions.float() / temperature
        return temperature * (
            torch.logsumexp(values, dim=1) - math.log(float(count))
        )

    def forward(
        self, dense_grid: torch.Tensor, *, return_local: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        expected_tail = DENSE_SHAPE
        if dense_grid.ndim != 6 or tuple(dense_grid.shape[1:]) != expected_tail:
            raise ValueError(
                f"dense_grid must be [B,{','.join(map(str, expected_tail))}], "
                f"got {tuple(dense_grid.shape)}"
            )
        batch = int(dense_grid.shape[0])
        delta = dense_grid[:, :, 1:] - dense_grid[:, :, :-1]
        spatial_input = delta.permute(0, 1, 2, 5, 3, 4).reshape(
            batch * VIEWS * DIFF_TIME, INPUT_DIM, GRID, GRID
        )
        spatial = self.spatial(spatial_input)
        spatial = spatial.reshape(
            batch, VIEWS, DIFF_TIME, HIDDEN_DIM, GRID, GRID
        )
        sequences = spatial.permute(0, 1, 4, 5, 2, 3).reshape(
            batch * VIEWS * GRID * GRID, DIFF_TIME, HIDDEN_DIM
        )
        sequences = sequences + self.temporal_position.to(sequences.dtype)
        attended, _ = self.temporal_attention(
            sequences, sequences, sequences, need_weights=False
        )
        local = attended.mean(dim=1).reshape(
            batch, VIEWS * GRID * GRID, HIDDEN_DIM
        )
        pooled = self.normalized_log_mean_exp(local)
        logits = self.bad_head(pooled).squeeze(-1)
        if return_local:
            return logits, local
        return logits


def upper_median(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("upper median requires at least one value")
    ordered = sorted(int(value) for value in values)
    return ordered[len(ordered) // 2]


def opaque_group_key(raw: Mapping[str, Any], token: str) -> str:
    for field in ("asset_id", "request_id", "batch_id"):
        value = str(raw.get(field, "")).strip()
        if value:
            return f"{field}:{sha256_bytes(value.encode('utf-8'))}"
    prompt = str(raw.get("prompt", "")).strip()
    if prompt:
        normalized = " ".join(prompt.casefold().split())
        return f"prompt:{sha256_bytes(normalized.encode('utf-8'))}"
    return f"sample:{token}"


def verify_s0(
    s0_dir: Path,
) -> tuple[list[str], dict[str, int], dict[str, Any], dict[str, np.ndarray]]:
    root = s0_dir.expanduser().resolve()
    lock_path = root / "LOCK.json"
    summary_path = root / "summary.json"
    discovery_path = root / "discovery_ids.txt"
    outer_path = root / "discovery_outer_folds.jsonl"
    split_path = root / "discovery_split.npz"
    for path in (lock_path, summary_path, discovery_path, outer_path, split_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("S0 directory is incomplete or contains a symlink")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if lock.get("status") != "S0_FROZEN" or summary.get("status") != "S0_FROZEN":
        raise RuntimeError("S0 is not frozen")
    if sha256_file(summary_path) != lock.get("summary_sha256"):
        raise RuntimeError("S0 summary hash mismatch")
    artifact_hashes = lock.get("artifact_sha256", {})
    if sha256_file(discovery_path) != artifact_hashes.get("discovery_ids.txt"):
        raise RuntimeError("S0 discovery_ids hash mismatch")
    if sha256_file(outer_path) != artifact_hashes.get(
        "discovery_outer_folds.jsonl"
    ):
        raise RuntimeError("S0 outer-fold hash mismatch")
    if sha256_file(split_path) != artifact_hashes.get("discovery_split.npz"):
        raise RuntimeError("S0 discovery split hash mismatch")

    tokens = [line.strip() for line in discovery_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens or len(tokens) != len(set(tokens)):
        raise RuntimeError("S0 discovery membership is empty or duplicated")
    if any(
        len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token)
        for token in tokens
    ):
        raise RuntimeError("S0 discovery membership contains an invalid token")
    outer: dict[str, int] = {}
    with outer_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            token = str(item["sample_token"])
            fold = int(item["outer_fold"])
            if token in outer or fold not in range(OUTER_FOLDS):
                raise RuntimeError("S0 outer-fold artifact is invalid")
            outer[token] = fold
    if set(tokens) != set(outer):
        raise RuntimeError("S0 discovery and outer-fold memberships differ")
    with np.load(split_path, allow_pickle=False) as archive:
        required = {"sample_token", "y", "strata", "fold_id", "inner_fold_id"}
        if required - set(archive.files):
            raise RuntimeError("S0 discovery split is missing required arrays")
        split = {key: np.asarray(archive[key]) for key in required}
    sorted_tokens = np.asarray(sorted(tokens), dtype=np.str_)
    if not np.array_equal(split["sample_token"], sorted_tokens):
        raise RuntimeError("S0 discovery split token order mismatch")
    n = len(sorted_tokens)
    if split["y"].shape != (n,) or not np.all(np.isin(split["y"], [0, 1])):
        raise RuntimeError("S0 discovery split y contract failed")
    if split["strata"].shape != (n,) or split["fold_id"].shape != (n,):
        raise RuntimeError("S0 discovery split 1-D contract failed")
    if split["inner_fold_id"].shape != (OUTER_FOLDS, n):
        raise RuntimeError("S0 discovery split inner-fold shape failed")
    expected_outer = np.asarray([outer[str(token)] for token in sorted_tokens], dtype=np.int8)
    if not np.array_equal(split["fold_id"], expected_outer):
        raise RuntimeError("S0 discovery split outer folds disagree with audit JSONL")
    for outer_id in range(OUTER_FOLDS):
        valid = expected_outer == outer_id
        if not np.all(split["inner_fold_id"][outer_id, valid] == -1):
            raise RuntimeError("S0 inner folds leak onto outer-valid")
        if not np.all(np.isin(split["inner_fold_id"][outer_id, ~valid], range(INNER_FOLDS))):
            raise RuntimeError("S0 inner fold id outside 0..3")
    rows = {
        "sample_token": sorted_tokens,
        "y": split["y"].astype(np.int8, copy=False),
        "strata": split["strata"].astype(np.str_, copy=False),
        "fold_id": expected_outer,
        "inner_fold_id": split["inner_fold_id"].astype(np.int8, copy=False),
    }
    return sorted(tokens), outer, {
        "summary_sha256": lock["summary_sha256"],
        "discovery_ids_sha256": artifact_hashes["discovery_ids.txt"],
        "outer_folds_sha256": artifact_hashes["discovery_outer_folds.jsonl"],
        "discovery_split_sha256": artifact_hashes["discovery_split.npz"],
    }, rows


def read_discovery_training_rows(
    manifest: Path,
    discovery_tokens: Sequence[str],
    outer_map: Mapping[str, int],
) -> dict[str, np.ndarray]:
    path = manifest.expanduser().resolve()
    if path.name != TRAIN_MANIFEST_NAME or sha256_file(path) != TRAIN_MANIFEST_SHA256:
        raise RuntimeError("training requires the preregistered train_v3 manifest bytes")
    wanted = set(discovery_tokens)
    rows: dict[str, tuple[int, str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            # Derive membership before touching label/source/group fields.  A
            # non-discovery row (including shadow) is skipped without label access.
            token = sample_token(str(raw["video"]))
            if token not in wanted:
                continue
            label = str(raw.get("label", ""))
            if label not in VALID_LABELS:
                raise RuntimeError("discovery row has an invalid label")
            source = derive_source(raw)
            group = opaque_group_key(raw, token)
            if token in rows:
                raise RuntimeError("duplicate discovery token in train manifest")
            rows[token] = (1 if label == "bad" else 0, source, group)
    if set(rows) != wanted:
        raise RuntimeError("not every S0 discovery token has a training row")
    tokens = np.asarray(sorted(wanted), dtype=np.str_)
    y = np.asarray([rows[str(token)][0] for token in tokens], dtype=np.int8)
    source = np.asarray([rows[str(token)][1] for token in tokens], dtype=np.str_)
    group = np.asarray([rows[str(token)][2] for token in tokens], dtype=np.str_)
    fold = np.asarray([outer_map[str(token)] for token in tokens], dtype=np.int8)
    if set(np.unique(fold).tolist()) != set(range(OUTER_FOLDS)):
        raise RuntimeError("outer folds must be contiguous 0..4")
    for outer_id in range(OUTER_FOLDS):
        valid_y = y[fold == outer_id]
        if valid_y.size == 0 or np.unique(valid_y).size != 2:
            raise RuntimeError("every outer-valid fold must contain both binary classes")
    return {
        "sample_token": tokens,
        "y": y,
        "strata": source,
        "group": group,
        "fold_id": fold,
    }


def make_inner_folds(
    tokens: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_id: int,
) -> np.ndarray:
    """Deterministic grouped greedy balance for one outer-train partition."""
    local_groups: dict[str, list[int]] = defaultdict(list)
    for index in outer_train_indices.tolist():
        local_groups[str(groups[index])].append(index)
    stratum_keys = sorted(
        {(int(y[index]), str(strata[index])) for index in outer_train_indices.tolist()}
    )
    stratum_index = {key: i for i, key in enumerate(stratum_keys)}
    total = np.zeros(len(stratum_keys), dtype=np.float64)
    group_counts: dict[str, np.ndarray] = {}
    for group_key, indices in local_groups.items():
        counts = np.zeros(len(stratum_keys), dtype=np.float64)
        for index in indices:
            counts[stratum_index[(int(y[index]), str(strata[index]))]] += 1.0
        group_counts[group_key] = counts
        total += counts
    target = total / INNER_FOLDS
    fold_counts = np.zeros((INNER_FOLDS, len(stratum_keys)), dtype=np.float64)
    fold_sizes = np.zeros(INNER_FOLDS, dtype=np.float64)
    ordered_groups = sorted(
        local_groups,
        key=lambda key: (
            -len(local_groups[key]),
            seeded_rank(INNER_SPLIT_SEED, f"e51-inner-order-{outer_id}", key),
        ),
    )
    assignment: dict[str, int] = {}
    for group_key in ordered_groups:
        counts = group_counts[group_key]
        group_size = float(len(local_groups[group_key]))
        tie_offset = int(
            seeded_rank(
                INNER_SPLIT_SEED, f"e51-inner-tie-{outer_id}", group_key
            )[:8],
            16,
        ) % INNER_FOLDS
        candidates: list[tuple[float, float, int, int]] = []
        current_stratum_cost = float(np.square(fold_counts - target[None, :]).sum())
        target_size = len(outer_train_indices) / INNER_FOLDS
        current_size_cost = float(np.square(fold_sizes - target_size).sum())
        for fold_id in range(INNER_FOLDS):
            proposed = fold_counts[fold_id] + counts
            stratum_cost = (
                current_stratum_cost
                - float(np.square(fold_counts[fold_id] - target).sum())
                + float(np.square(proposed - target).sum())
            )
            size_cost = (
                current_size_cost
                - float((fold_sizes[fold_id] - target_size) ** 2)
                + float((fold_sizes[fold_id] + group_size - target_size) ** 2)
            )
            tie_rank = (fold_id - tie_offset) % INNER_FOLDS
            candidates.append((stratum_cost, size_cost, tie_rank, fold_id))
        chosen = min(candidates)[-1]
        assignment[group_key] = chosen
        fold_counts[chosen] += counts
        fold_sizes[chosen] += group_size
    result = np.full(y.shape[0], -1, dtype=np.int8)
    for group_key, indices in local_groups.items():
        result[np.asarray(indices, dtype=np.int64)] = assignment[group_key]
    if np.any(result[outer_train_indices] < 0):
        raise AssertionError("inner fold assignment is incomplete")
    for group_key, indices in local_groups.items():
        if np.unique(result[np.asarray(indices)]).size != 1:
            raise AssertionError("a group crossed inner folds")
    for inner_id in range(INNER_FOLDS):
        inner_y = y[outer_train_indices[result[outer_train_indices] == inner_id]]
        if inner_y.size == 0 or np.unique(inner_y).size != 2:
            raise RuntimeError("every inner-valid fold must contain both binary classes")
    return result


class DenseFeatureStore:
    """Preload the fixed discovery dense grids once into host float16 memory."""

    def __init__(self, root: Path, tokens: Sequence[str]) -> None:
        self.root = root.expanduser().resolve()
        self.values = np.empty((len(tokens), *DENSE_SHAPE), dtype=np.float16)
        for index, token_value in enumerate(tokens):
            token = str(token_value)
            json_path = self.root / f"{token}.json"
            npz_path = self.root / f"{token}.npz"
            if not output_pair_complete(json_path, npz_path):
                raise RuntimeError("E51 feature cache has an invalid artifact pair")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("sample_token") != token:
                raise RuntimeError("E51 cache token/file mismatch")
            with np.load(npz_path, allow_pickle=False) as archive:
                dense = archive["dense_grid"]
                if dense.shape != DENSE_SHAPE or dense.dtype != np.float16:
                    raise RuntimeError("E51 dense_grid contract mismatch")
                self.values[index] = dense
            if (index + 1) % 100 == 0 or index + 1 == len(tokens):
                print(
                    f"E51_FEATURE_LOAD loaded={index + 1}/{len(tokens)}",
                    flush=True,
                )

    def batch(self, indices: np.ndarray, device: torch.device) -> torch.Tensor:
        values = np.ascontiguousarray(self.values[indices])
        return torch.from_numpy(values).to(device, non_blocking=True)


def batches(
    indices: np.ndarray, *, shuffle: bool, seed: int
) -> Iterable[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for start in range(0, len(order), BATCH_SIZE):
        yield order[start : start + BATCH_SIZE]


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def train_one_epoch(
    model: E51DenseLocalHead,
    optimizer: torch.optim.Optimizer,
    store: DenseFeatureStore,
    y: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    order_seed: int,
) -> None:
    model.train()
    for batch_indices in batches(indices, shuffle=True, seed=order_seed):
        features = store.batch(batch_indices, device)
        targets = torch.from_numpy(y[batch_indices].astype(np.float32)).to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = model(features)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def validation_loss(
    model: E51DenseLocalHead,
    store: DenseFeatureStore,
    y: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch_indices in batches(indices, shuffle=False, seed=0):
        features = store.batch(batch_indices, device)
        targets = torch.from_numpy(y[batch_indices].astype(np.float32)).to(device)
        with autocast_context(device):
            logits = model(features)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
        total += float(losses.float().sum())
        count += int(losses.numel())
    if count == 0:
        raise ValueError("validation partition is empty")
    return total / count


@torch.inference_mode()
def predict(
    model: E51DenseLocalHead,
    store: DenseFeatureStore,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    cursor = 0
    for batch_indices in batches(indices, shuffle=False, seed=0):
        features = store.batch(batch_indices, device)
        with autocast_context(device):
            probabilities = torch.sigmoid(model(features))
        size = len(batch_indices)
        output[cursor : cursor + size] = probabilities.float().cpu().numpy()
        cursor += size
    if cursor != len(indices) or not np.isfinite(output).all():
        raise RuntimeError("prediction output is incomplete or non-finite")
    return output


def new_model(seed: int, device: torch.device) -> E51DenseLocalHead:
    set_deterministic(seed, device)
    return E51DenseLocalHead().to(device)


def train_inner_head(
    store: DenseFeatureStore,
    y: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    model = new_model(seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(
            model,
            optimizer,
            store,
            y,
            train_indices,
            device,
            order_seed=seed * 100 + epoch,
        )
        loss = validation_loss(model, store, y, valid_indices, device)
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("inner early stopping did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    predictions = predict(model, store, valid_indices, device)
    del optimizer, model, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, best_epoch


def train_fixed_epochs(
    store: DenseFeatureStore,
    y: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    if epochs <= 0 or epochs > MAX_EPOCHS:
        raise ValueError("fixed refit epochs outside preregistered range")
    model = new_model(seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model,
            optimizer,
            store,
            y,
            train_indices,
            device,
            order_seed=seed * 100 + epoch,
        )
    predictions = predict(model, store, predict_indices, device)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def run_nested(
    store: DenseFeatureStore,
    rows: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    tokens = rows["sample_token"]
    y = rows["y"]
    strata = rows["strata"]
    outer_folds = rows["fold_id"]
    frozen_inner_folds = rows["inner_fold_id"]
    n = len(tokens)
    expert_oof = np.full(n, np.nan, dtype=np.float32)
    inner_expert = np.full((OUTER_FOLDS, n), np.nan, dtype=np.float32)
    inner_best_epoch = np.zeros((OUTER_FOLDS, INNER_FOLDS), dtype=np.int16)
    outer_best_epoch = np.zeros(OUTER_FOLDS, dtype=np.int16)

    all_indices = np.arange(n, dtype=np.int64)
    for outer_id in range(OUTER_FOLDS):
        outer_valid = all_indices[outer_folds == outer_id]
        outer_train = all_indices[outer_folds != outer_id]
        inner_fold = frozen_inner_folds[outer_id]
        fold_epochs: list[int] = []
        for inner_id in range(INNER_FOLDS):
            inner_valid = outer_train[inner_fold[outer_train] == inner_id]
            inner_train = outer_train[inner_fold[outer_train] != inner_id]
            seed = INNER_MODEL_SEED_BASE + 10 * outer_id + inner_id
            predictions, best_epoch = train_inner_head(
                store,
                y,
                inner_train,
                inner_valid,
                seed,
                device,
            )
            inner_expert[outer_id, inner_valid] = predictions
            inner_best_epoch[outer_id, inner_id] = best_epoch
            fold_epochs.append(best_epoch)
            print(
                f"E51_INNER_COMPLETE outer={outer_id} inner={inner_id} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if not np.isfinite(inner_expert[outer_id, outer_train]).all():
            raise RuntimeError("inner_expert is incomplete on outer-train")
        if not np.isnan(inner_expert[outer_id, outer_valid]).all():
            raise RuntimeError("inner_expert leaks onto outer-valid")
        chosen_epoch = upper_median(fold_epochs)
        outer_best_epoch[outer_id] = chosen_epoch
        outer_predictions = train_fixed_epochs(
            store,
            y,
            outer_train,
            outer_valid,
            chosen_epoch,
            OUTER_REFIT_SEED_BASE + outer_id,
            device,
        )
        expert_oof[outer_valid] = outer_predictions
        print(
            f"E51_OUTER_COMPLETE outer={outer_id} refit_epoch={chosen_epoch} "
            f"valid_count={len(outer_valid)}",
            flush=True,
        )
    if not np.isfinite(expert_oof).all():
        raise RuntimeError("expert_oof is incomplete")
    for outer_id in range(OUTER_FOLDS):
        if not np.isnan(inner_expert[outer_id, outer_folds == outer_id]).all():
            raise RuntimeError("final inner_expert outer-valid NaN contract failed")
        if not np.isfinite(inner_expert[outer_id, outer_folds != outer_id]).all():
            raise RuntimeError("final inner_expert outer-train coverage failed")
    return {
        "sample_token": tokens,
        "y": y,
        "strata": strata,
        "fold_id": outer_folds,
        "expert_oof": expert_oof,
        "inner_expert": inner_expert,
        "inner_best_epoch": inner_best_epoch,
        "outer_best_epoch": outer_best_epoch,
    }


def atomic_bundle(
    output_path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> tuple[str, str]:
    target = output_path.expanduser().resolve()
    if target.suffix != ".npz":
        raise ValueError("--output-bundle must end in .npz")
    meta_path = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (target.exists() or meta_path.exists()):
        raise FileExistsError("output bundle already exists")
    artifact_id = secrets.token_hex(16)
    temp_npz = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temp_json = meta_path.with_name(f".{meta_path.name}.{os.getpid()}.tmp")
    payload = dict(arrays)
    payload["schema_version"] = np.asarray(HEAD_SCHEMA)
    payload["artifact_id"] = np.asarray(artifact_id)
    meta = dict(metadata)
    meta.update({"schema_version": HEAD_SCHEMA, "artifact_id": artifact_id})
    try:
        with temp_npz.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        with temp_json.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(json_ready(meta), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_npz, target)
        os.replace(temp_json, meta_path)
    finally:
        for path in (temp_npz, temp_json):
            if path.exists():
                path.unlink()
    return sha256_file(target), sha256_file(meta_path)


def selftest(device_name: str) -> int:
    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    set_deterministic(20260820, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = E51DenseLocalHead().to(device).train()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    synthetic = torch.randn((1, *DENSE_SHAPE), dtype=dtype, device=device)
    target = torch.ones(1, dtype=torch.float32, device=device)
    with autocast_context(device):
        logits, local = model(synthetic, return_local=True)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if logits.shape != (1,) or local.shape != (1, VIEWS * GRID * GRID, HIDDEN_DIM):
        raise AssertionError("synthetic forward shape failed")
    if not gradients or any(gradient is None for gradient in gradients):
        raise AssertionError("synthetic backward missed a trainable parameter")
    if not all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None):
        raise AssertionError("synthetic backward produced a non-finite gradient")
    if not any(float(gradient.abs().sum()) > 0.0 for gradient in gradients if gradient is not None):
        raise AssertionError("synthetic backward gradients collapsed to zero")

    model.eval()
    with torch.inference_mode(), autocast_context(device):
        first = model(synthetic)
        second = model(synthetic)
        positions = torch.randn(1, GRID * GRID, HIDDEN_DIM, device=device)
        once = model.normalized_log_mean_exp(positions)
        twice = model.normalized_log_mean_exp(torch.cat([positions, positions], dim=1))
    if not torch.equal(first, second):
        raise AssertionError("eval forward is not exact-repeat deterministic")
    duplicate_max_abs = float((once - twice).abs().max())
    if duplicate_max_abs > 2.0e-6:
        raise AssertionError("normalized log-mean-exp is not duplicate invariant")
    if upper_median([2, 5, 3, 7]) != 5:
        raise AssertionError("upper-median epoch rule failed")

    # Synthetic grouped inner folds and strict outer-valid NaN contract.
    n = 80
    tokens = np.asarray([f"{index:064x}" for index in range(n)], dtype=np.str_)
    yy = np.asarray([index % 2 for index in range(n)], dtype=np.int8)
    strata = np.asarray([f"s{index % 4}" for index in range(n)], dtype=np.str_)
    groups = np.asarray([f"g{index // 2}" for index in range(n)], dtype=np.str_)
    outer = np.asarray([index % OUTER_FOLDS for index in range(n)], dtype=np.int8)
    train_indices = np.flatnonzero(outer != 0)
    inner = make_inner_folds(tokens, yy, strata, groups, train_indices, 0)
    for group_name in np.unique(groups[train_indices]):
        member = train_indices[groups[train_indices] == group_name]
        if np.unique(inner[member]).size != 1:
            raise AssertionError("synthetic group crossed inner folds")
    candidate = np.full((OUTER_FOLDS, n), np.nan, dtype=np.float32)
    for outer_id in range(OUTER_FOLDS):
        candidate[outer_id, outer != outer_id] = 0.5
        if not np.isnan(candidate[outer_id, outer == outer_id]).all():
            raise AssertionError("synthetic inner_expert leakage contract failed")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "schema_version": HEAD_SCHEMA,
                "input_shape": [1, *DENSE_SHAPE],
                "local_shape": [1, VIEWS * GRID * GRID, HIDDEN_DIM],
                "logit_shape": [1],
                "parameter_count": parameter_count,
                "all_trainable_gradients_finite": True,
                "eval_repeat_exact": True,
                "duplicate_view_lme_invariant": True,
                "duplicate_view_max_abs": duplicate_max_abs,
                "upper_median_rule": True,
                "grouped_inner_split": True,
                "inner_expert_nan_contract": True,
                "cuda_peak_allocated_bytes": peak,
                "real_labels_accessed": False,
                "real_features_accessed": False,
                "metric_computed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def train(args: argparse.Namespace) -> int:
    if not args.confirm_s0_frozen:
        raise RuntimeError("train requires explicit --confirm-s0-frozen")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("nested E51 training requires one CUDA GPU")
    discovery_tokens, _outer_map, s0_meta, rows = verify_s0(args.s0_dir)
    manifest = args.manifest.expanduser().resolve()
    if manifest.name != TRAIN_MANIFEST_NAME or sha256_file(manifest) != TRAIN_MANIFEST_SHA256:
        raise RuntimeError("training requires the preregistered train_v3 manifest bytes")
    print(
        json.dumps(
            {
                "schema_version": HEAD_SCHEMA,
                "s0_verified": True,
                "discovery_count": len(discovery_tokens),
                "outer_folds": OUTER_FOLDS,
                "inner_folds": INNER_FOLDS,
                "feature_shape": list(DENSE_SHAPE),
                "shadow_membership_opened": False,
                "eval_opened": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    store = DenseFeatureStore(args.features_root, rows["sample_token"])
    started = time.perf_counter()
    arrays = run_nested(store, rows, device)
    elapsed = time.perf_counter() - started
    code_sha = sha256_file(Path(__file__).resolve())
    metadata = {
        "status": "complete",
        "code_sha256": code_sha,
        "feature_schema": FEATURE_SCHEMA,
        "feature_checkpoint_sha256": CHECKPOINT_SHA256,
        "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "s0": s0_meta,
        "architecture": {
            "input": [VIEWS, TOKEN_TIME, GRID, GRID, INPUT_DIM],
            "temporal_difference": [VIEWS, DIFF_TIME, GRID, GRID, INPUT_DIM],
            "spatial_conv": "3x3, 768->256, shared over view/time",
            "temporal_attention": {
                "layers": 1,
                "heads": ATTENTION_HEADS,
                "dropout": DROPOUT,
                "sinusoidal_position": True,
            },
            "temporal_pool": "mean over 31",
            "spatial_view_pool": "position-normalized log-mean-exp",
            "temperature": LME_TEMPERATURE,
            "bad_head": "linear 256->1",
        },
        "training": {
            "optimizer": "AdamW",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "inner_split_seed": INNER_SPLIT_SEED,
            "inner_model_seed_formula": "20260820 + 10*outer_id + inner_id",
            "outer_refit_seed_formula": "20260880 + outer_id",
            "outer_refit_epoch": "upper median (third sorted) of four inner best epochs",
            "shadow_refit_seed_reserved": SHADOW_REFIT_SEED,
        },
        "bundle_contract": {
            "expert_oof": [len(discovery_tokens)],
            "inner_expert": [OUTER_FOLDS, len(discovery_tokens)],
            "inner_outer_valid": "NaN",
            "direction": "higher probability means bad",
        },
        "elapsed_seconds": elapsed,
        "shadow_membership_opened": False,
        "eval_opened": False,
        "metric_computed": False,
    }
    npz_sha, json_sha = atomic_bundle(
        args.output_bundle, arrays, metadata, args.overwrite
    )
    print(
        "E51_NESTED_DONE "
        f"samples={len(discovery_tokens)} elapsed_seconds={elapsed:.1f} "
        f"bundle_sha256={npz_sha} metadata_sha256={json_sha}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    test_parser = subparsers.add_parser("selftest", help="Synthetic-only forward/backward and protocol checks.")
    test_parser.add_argument("--device", default="auto")

    train_parser = subparsers.add_parser("train", help="Run strict discovery nested OOF after S0.")
    train_parser.add_argument("--s0-dir", type=Path, required=True)
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--features-root", type=Path, required=True)
    train_parser.add_argument("--output-bundle", type=Path, required=True)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--confirm-s0-frozen", action="store_true")
    train_parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "selftest":
        return selftest(args.device)
    if args.command == "train":
        return train(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
