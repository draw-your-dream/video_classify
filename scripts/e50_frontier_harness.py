#!/usr/bin/env python3
"""Training-only harness for E50 frontier-shell experiments.

This module deliberately has no project-data defaults and never opens eval_v3.
It consumes an explicit NPZ bundle of *outer-fold OOF* predictions. The bundle
contract is:

``y``
    Binary labels, shape ``(n,)``; bad=1 and good/normal=0.
``fold_id``
    Contiguous outer-fold ids ``0..K-1``, shape ``(n,)``.
``base_oof``
    Baseline scores for each sample from its outer-fold model, shape ``(n,)``.
``inner_base``
    Shape ``(K, n)``. Row k contains baseline scores produced by inner OOF on
    the samples whose ``fold_id != k`` and NaN on the outer validation fold
    ``fold_id == k``. The strict NaN contract prevents a global-OOF shortcut.
``expert_oof`` (needed by protocol ``evaluate``)
    Candidate scores from the same outer folds, shape ``(n,)``. Higher means
    more likely bad. NaN is allowed and means "candidate abstains".
``inner_expert`` (needed by protocol ``evaluate``)
    Shape ``(K, n)``. Row k contains candidate scores produced by inner OOF on
    ``fold_id != k`` and NaN on ``fold_id == k``. Missing outer-train candidate
    scores may also be NaN and are omitted only from that fold's empirical CDF.
``expert2_oof`` / ``inner_expert2`` (needed by E53 protocol mode)
    A second independently cross-fitted candidate axis with the same contracts.
``strata`` (optional)
    Source/group ids used to keep permutation/random controls within strata.

The frontier shell is the minimum-score-displacement band immediately above
the current 95%-recall bad boundary. For every outer fold it is derived only
from that fold's inner OOF training predictions. No eval labels or eval scores
are needed or accepted implicitly.

Examples:

    python scripts/e50_frontier_harness.py --self-test
    python scripts/e50_frontier_harness.py inspect --bundle work/e50_oof.npz
    python scripts/e50_frontier_harness.py evaluate --bundle work/e50_oof.npz \
        --target 0.35 \
        --controls permutation random --control-repeats 50
    python scripts/e50_frontier_harness.py evaluate --bundle work/e50_oof.npz \
        --mode protocol-e53 --target 0.50

The default ``evaluate`` mode is the frozen E51 rank-residual formula. E53 is
available only via explicit ``--mode protocol-e53``. The former raw-score soft
and veto gates remain available via explicit ``--mode legacy-raw --gate ...``;
they are not protocol defaults and their output is labelled legacy.

Multiple variants may be printed for bookkeeping, but this script never picks
a winner. Variant/seed selection must be preregistered outside this harness.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class GnResult:
    value: float
    threshold: float
    released_gn: int
    n_gn: int
    caught_bad: int
    n_bad: int
    bad_below_threshold: int
    requested_recall: float


@dataclass(frozen=True)
class FrontierDefinition:
    target_release: float
    current_threshold: float
    target_threshold: float
    current_released_gn: int
    target_released_gn: int
    required_released_gn: int
    n_gn: int
    n_bad: int
    allowed_bad_below: int
    bad_to_promote: int
    bad_shell_indices: np.ndarray
    gn_frontier_indices: np.ndarray
    extreme_tail_indices: np.ndarray
    support_bad_index: int

    def summary(self) -> dict[str, object]:
        return {
            "target_release": self.target_release,
            "current_threshold": self.current_threshold,
            "target_threshold": self.target_threshold,
            "current_released_gn": self.current_released_gn,
            "target_released_gn": self.target_released_gn,
            "required_released_gn": self.required_released_gn,
            "n_gn": self.n_gn,
            "n_bad": self.n_bad,
            "allowed_bad_below": self.allowed_bad_below,
            "bad_to_promote": self.bad_to_promote,
            "bad_shell_size": int(self.bad_shell_indices.size),
            "gn_frontier_size": int(self.gn_frontier_indices.size),
            "extreme_tail_size": int(self.extreme_tail_indices.size),
            "support_bad_index": self.support_bad_index,
        }


@dataclass(frozen=True)
class FoldFrontier:
    fold: int
    train_indices: np.ndarray
    valid_indices: np.ndarray
    definition: FrontierDefinition

    def summary(self) -> dict[str, object]:
        out = {
            "fold": self.fold,
            "n_train": int(self.train_indices.size),
            "n_valid": int(self.valid_indices.size),
        }
        out.update(self.definition.summary())
        return out


def _as_1d(name: str, values: np.ndarray | Sequence[float]) -> np.ndarray:
    out = np.asarray(values)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    return out


def _binary_labels(y: np.ndarray | Sequence[int]) -> np.ndarray:
    raw = _as_1d("y", y)
    if not np.all(np.isin(raw, [0, 1])):
        raise ValueError(f"y must contain only binary labels 0 and 1; got {np.unique(raw).tolist()}")
    out = raw.astype(np.int8, copy=False)
    unique = np.unique(out)
    if not np.array_equal(unique, np.array([0, 1], dtype=np.int8)):
        raise ValueError(f"y must contain both binary labels 0 and 1; got {unique.tolist()}")
    return out


def _finite_scores(name: str, scores: np.ndarray | Sequence[float]) -> np.ndarray:
    out = _as_1d(name, scores).astype(float, copy=False)
    if not np.all(np.isfinite(out)):
        bad = int((~np.isfinite(out)).sum())
        raise ValueError(f"{name} contains {bad} non-finite values")
    return out


def _finite_or_nan(name: str, scores: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a float vector that may contain NaN, but never +/-Inf."""
    out = _as_1d(name, scores).astype(float, copy=False)
    if np.isinf(out).any():
        count = int(np.isinf(out).sum())
        raise ValueError(f"{name} contains {count} infinite values")
    return out


def _rank_axis(
    name: str,
    scores: np.ndarray | Sequence[float],
    *,
    allow_nan: bool,
) -> np.ndarray:
    out = _finite_or_nan(name, scores)
    if not allow_nan and np.isnan(out).any():
        raise ValueError(f"{name} must be finite")
    finite = np.isfinite(out)
    if np.any((out[finite] < 0.0) | (out[finite] > 1.0)):
        raise ValueError(f"{name} finite values must lie in [0, 1]")
    return out


def gn_at_recall(
    scores: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[int],
    recall: float = 0.95,
) -> GnResult:
    """Return GN release at fixed bad recall using the project's strict ``< T`` rule."""
    p = _finite_scores("scores", scores)
    yy = _binary_labels(y)
    if p.size != yy.size:
        raise ValueError(f"scores/y length mismatch: {p.size} vs {yy.size}")
    if not 0.0 < recall <= 1.0:
        raise ValueError(f"recall must be in (0, 1], got {recall}")

    bad = np.sort(p[yy == 1], kind="stable")
    gn = p[yy == 0]
    n_catch = int(math.ceil(recall * bad.size))
    threshold = float(bad[bad.size - n_catch])
    released = int(np.count_nonzero(gn < threshold))  # Deliberately strict.
    caught = int(np.count_nonzero(bad >= threshold))
    below = int(np.count_nonzero(bad < threshold))
    return GnResult(
        value=released / gn.size,
        threshold=threshold,
        released_gn=released,
        n_gn=int(gn.size),
        caught_bad=caught,
        n_bad=int(bad.size),
        bad_below_threshold=below,
        requested_recall=float(recall),
    )


def define_frontier(
    inner_scores: np.ndarray | Sequence[float],
    y_train: np.ndarray | Sequence[int],
    target_release: float,
    recall: float = 0.95,
) -> FrontierDefinition:
    """Define the minimum-displacement frontier shell on inner-OOF training scores.

    The bad shell contains the bad ranks from the current threshold-support bad
    up to (but excluding) the target threshold-support bad. With unique scores
    its size is exactly the minimum number of bads that must move upward.
    Stable rank slicing, rather than threshold masks, makes ties deterministic.
    """
    p = _finite_scores("inner_scores", inner_scores)
    yy = _binary_labels(y_train)
    if p.size != yy.size:
        raise ValueError(f"inner_scores/y_train length mismatch: {p.size} vs {yy.size}")
    if not 0.0 < target_release <= 1.0:
        raise ValueError(f"target_release must be in (0, 1], got {target_release}")

    bad_idx = np.flatnonzero(yy == 1)
    gn_idx = np.flatnonzero(yy == 0)
    order = bad_idx[np.argsort(p[bad_idx], kind="stable")]
    bad_scores = p[order]
    allowed = bad_scores.size - int(math.ceil(recall * bad_scores.size))
    current_pos = allowed
    current_t = float(bad_scores[current_pos])
    current_release = int(np.count_nonzero(p[gn_idx] < current_t))
    required = int(math.ceil(target_release * gn_idx.size - EPS))

    target_pos = current_pos
    while target_pos < bad_scores.size:
        candidate_t = float(bad_scores[target_pos])
        if int(np.count_nonzero(p[gn_idx] < candidate_t)) >= required:
            break
        target_pos += 1
    if target_pos == bad_scores.size:
        max_release = int(np.count_nonzero(p[gn_idx] < bad_scores[-1]))
        raise ValueError(
            f"target_release={target_release:.6f} is unreachable: at most "
            f"{max_release}/{gn_idx.size} GN are strictly below the largest bad score"
        )

    target_t = float(bad_scores[target_pos])
    target_released = int(np.count_nonzero(p[gn_idx] < target_t))
    shell = order[current_pos:target_pos].copy()
    gn_frontier = gn_idx[(p[gn_idx] >= current_t) & (p[gn_idx] < target_t)].copy()
    extreme = order[:current_pos].copy()
    return FrontierDefinition(
        target_release=float(target_release),
        current_threshold=current_t,
        target_threshold=target_t,
        current_released_gn=current_release,
        target_released_gn=target_released,
        required_released_gn=required,
        n_gn=int(gn_idx.size),
        n_bad=int(bad_idx.size),
        allowed_bad_below=int(allowed),
        bad_to_promote=int(target_pos - current_pos),
        bad_shell_indices=shell,
        gn_frontier_indices=gn_frontier,
        extreme_tail_indices=extreme,
        support_bad_index=int(order[target_pos]),
    )


def validate_nested_bundle(
    y: np.ndarray,
    fold_id: np.ndarray,
    base_oof: np.ndarray,
    inner_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the strict outer-fold/inner-OOF bundle contract."""
    yy = _binary_labels(y)
    folds = _as_1d("fold_id", fold_id)
    base = _finite_scores("base_oof", base_oof)
    inner = np.asarray(inner_base, dtype=float)
    n = yy.size
    if folds.size != n or base.size != n:
        raise ValueError("y, fold_id, and base_oof must have the same length")
    if inner.ndim != 2 or inner.shape[1] != n:
        raise ValueError(f"inner_base must have shape (K, {n}), got {inner.shape}")
    try:
        integral = np.equal(folds, np.floor(folds))
    except TypeError as exc:
        raise ValueError("fold_id values must be numeric integers") from exc
    if not np.all(integral):
        raise ValueError("fold_id values must be integers")
    folds = folds.astype(int, copy=False)
    unique = np.unique(folds)
    expected = np.arange(inner.shape[0])
    if not np.array_equal(unique, expected):
        raise ValueError(
            f"fold_id must be contiguous 0..K-1 and match inner_base rows; "
            f"got {unique.tolist()}, expected {expected.tolist()}"
        )

    for k in expected:
        valid = folds == k
        train = ~valid
        if not valid.any() or not train.any():
            raise ValueError(f"fold {k} has an empty train or validation partition")
        if np.isfinite(inner[k, valid]).any():
            raise ValueError(
                f"inner_base row {k} has finite scores on its outer validation fold; "
                "this violates the strict nested-OOF contract"
            )
        if not np.isfinite(inner[k, train]).all():
            count = int((~np.isfinite(inner[k, train])).sum())
            raise ValueError(f"inner_base row {k} has {count} missing train-fold scores")
        if np.unique(yy[train]).size != 2 or np.unique(yy[valid]).size != 2:
            raise ValueError(f"fold {k} train and validation partitions must both contain 0/1")
    return yy, folds, base, inner


def validate_nested_axis(
    name: str,
    axis_oof: np.ndarray,
    inner_axis: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a candidate axis and its strict inner-OOF matrix.

    Candidate abstentions may be NaN on either outer OOF or outer-train inner
    OOF cells. Every outer-validation cell of row k must be NaN specifically;
    +/-Inf is never accepted. At least one finite outer-train reference value
    is required per fold so its empirical CDF is defined.
    """
    axis = _finite_or_nan(name, axis_oof)
    folds = _as_1d("fold_id", fold_id).astype(int, copy=False)
    inner = np.asarray(inner_axis, dtype=float)
    n = folds.size
    k_count = int(np.max(folds)) + 1
    if axis.size != n:
        raise ValueError(f"{name}/fold_id length mismatch: {axis.size} vs {n}")
    if inner.shape != (k_count, n):
        raise ValueError(
            f"inner_{name} must have shape ({k_count}, {n}), got {inner.shape}"
        )
    if np.isinf(inner).any():
        count = int(np.isinf(inner).sum())
        raise ValueError(f"inner_{name} contains {count} infinite values")
    for k in range(k_count):
        valid = folds == k
        train = ~valid
        if not np.isnan(inner[k, valid]).all():
            count = int((~np.isnan(inner[k, valid])).sum())
            raise ValueError(
                f"inner_{name} row {k} has {count} non-NaN values on its outer "
                "validation fold; this violates the strict nested-OOF contract"
            )
        if not np.isfinite(inner[k, train]).any():
            raise ValueError(f"inner_{name} row {k} has no finite train-fold scores")
    return axis, inner


def empirical_mid_cdf(
    reference: np.ndarray | Sequence[float],
    values: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Map values by an empirical mid-CDF fitted only on finite references.

    For value x the map is ``(# ref < x + 0.5 * # ref == x) / n``. NaN input
    values remain NaN so the fixed protocol formula can impute them neutrally.
    """
    ref = _finite_or_nan("reference", reference)
    ref = np.sort(ref[np.isfinite(ref)], kind="stable")
    if ref.size == 0:
        raise ValueError("empirical mid-CDF needs at least one finite reference")
    query = _finite_or_nan("values", values)
    out = np.full(query.shape, np.nan, dtype=float)
    finite = np.isfinite(query)
    left = np.searchsorted(ref, query[finite], side="left")
    right = np.searchsorted(ref, query[finite], side="right")
    out[finite] = 0.5 * (left + right) / ref.size
    return out


def crossfit_mid_cdf(
    axis_oof: np.ndarray,
    inner_axis: np.ndarray,
    fold_id: np.ndarray,
    *,
    name: str,
    allow_missing: bool,
) -> np.ndarray:
    """Foldwise rank-map outer-valid scores using only outer-train inner OOF."""
    folds = _as_1d("fold_id", fold_id).astype(int, copy=False)
    if allow_missing:
        axis, inner = validate_nested_axis(name, axis_oof, inner_axis, folds)
    else:
        axis = _finite_scores(name, axis_oof)
        inner = np.asarray(inner_axis, dtype=float)
        expected = (int(np.max(folds)) + 1, folds.size)
        if inner.shape != expected:
            raise ValueError(f"inner_{name} must have shape {expected}, got {inner.shape}")
        if np.isinf(inner).any():
            raise ValueError(f"inner_{name} contains infinite values")
        for k in range(expected[0]):
            valid = folds == k
            train = ~valid
            if not np.isnan(inner[k, valid]).all():
                raise ValueError(
                    f"inner_{name} row {k} has non-NaN scores on its outer validation fold"
                )
            if not np.isfinite(inner[k, train]).all():
                raise ValueError(f"inner_{name} row {k} has missing train-fold scores")

    out = np.full(axis.shape, np.nan, dtype=float)
    for k in range(inner.shape[0]):
        valid = folds == k
        train = ~valid
        out[valid] = empirical_mid_cdf(inner[k, train], axis[valid])
    if not allow_missing and not np.isfinite(out).all():
        raise AssertionError(f"internal error: {name} mid-CDF produced missing ranks")
    return out


def build_nested_frontiers(
    y: np.ndarray,
    fold_id: np.ndarray,
    base_oof: np.ndarray,
    inner_base: np.ndarray,
    targets: Iterable[float] = (0.35, 0.50),
    recall: float = 0.95,
) -> dict[float, list[FoldFrontier]]:
    """Build train-only frontier definitions for every outer fold and target."""
    yy, folds, _base, inner = validate_nested_bundle(y, fold_id, base_oof, inner_base)
    result: dict[float, list[FoldFrontier]] = {}
    for target in targets:
        target = float(target)
        per_fold: list[FoldFrontier] = []
        for k in range(inner.shape[0]):
            train = np.flatnonzero(folds != k)
            valid = np.flatnonzero(folds == k)
            local = define_frontier(inner[k, train], yy[train], target, recall)
            mapped = FrontierDefinition(
                target_release=local.target_release,
                current_threshold=local.current_threshold,
                target_threshold=local.target_threshold,
                current_released_gn=local.current_released_gn,
                target_released_gn=local.target_released_gn,
                required_released_gn=local.required_released_gn,
                n_gn=local.n_gn,
                n_bad=local.n_bad,
                allowed_bad_below=local.allowed_bad_below,
                bad_to_promote=local.bad_to_promote,
                bad_shell_indices=train[local.bad_shell_indices],
                gn_frontier_indices=train[local.gn_frontier_indices],
                extreme_tail_indices=train[local.extreme_tail_indices],
                support_bad_index=int(train[local.support_bad_index]),
            )
            per_fold.append(FoldFrontier(k, train, valid, mapped))
        result[target] = per_fold
    return result


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def protocol_e51_scores(
    r0: np.ndarray | Sequence[float],
    rj: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Frozen preregistered E51 single-axis rank-residual formula."""
    base_rank = _rank_axis("r0", r0, allow_nan=False)
    expert_rank = _rank_axis("rj", rj, allow_nan=True)
    if expert_rank.size != base_rank.size:
        raise ValueError("r0/rj length mismatch")
    neutral = np.where(np.isfinite(expert_rank), expert_rank, 0.5)
    gate = _sigmoid((0.45 - base_rank) / 0.08)
    return base_rank + 0.25 * gate * (neutral - 0.5)


def protocol_e53_scores(
    r0: np.ndarray | Sequence[float],
    rj: np.ndarray | Sequence[float],
    rj2: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Frozen preregistered E53 two-axis soft residual/veto formula."""
    base_rank = _rank_axis("r0", r0, allow_nan=False)
    first = _rank_axis("rj", rj, allow_nan=True)
    second = _rank_axis("rj2", rj2, allow_nan=True)
    if first.size != base_rank.size or second.size != base_rank.size:
        raise ValueError("r0/rj/rj2 length mismatch")
    first = np.where(np.isfinite(first), first, 0.5)
    second = np.where(np.isfinite(second), second, 0.5)
    gate = _sigmoid((0.45 - base_rank) / 0.08)
    residual = 0.20 * gate * ((first - 0.5) + (second - 0.5))
    veto_axis = np.maximum(first, second)
    veto = 0.15 * gate * np.maximum((veto_axis - 0.80) / 0.20, 0.0)
    return base_rank + residual + veto


def gate_weights(
    base_oof: np.ndarray,
    fold_id: np.ndarray,
    frontiers: Sequence[FoldFrontier],
    region: str = "frontier",
    temperature: float = 0.02,
    soft: bool = True,
) -> np.ndarray:
    """Create label-free validation gates from thresholds learned on inner OOF."""
    base = _finite_scores("base_oof", base_oof)
    folds = _as_1d("fold_id", fold_id).astype(int, copy=False)
    if region not in {"frontier", "below-target", "all"}:
        raise ValueError(f"unknown gate region: {region}")
    if soft and temperature <= 0:
        raise ValueError("temperature must be > 0 for a soft gate")
    out = np.zeros(base.size, dtype=float)
    for item in frontiers:
        idx = item.valid_indices
        if not np.all(folds[idx] == item.fold):
            raise ValueError(f"frontier/validation fold mismatch for fold {item.fold}")
        lo = item.definition.current_threshold
        hi = item.definition.target_threshold
        b = base[idx]
        if region == "all":
            weight = np.ones(idx.size)
        elif soft:
            upper = _sigmoid((hi - b) / temperature)
            if region == "frontier":
                lower = _sigmoid((b - lo) / temperature)
                weight = lower * upper
            else:
                weight = upper
        elif region == "frontier":
            weight = ((b >= lo) & (b < hi)).astype(float)
        else:
            weight = (b < hi).astype(float)
        out[idx] = weight
    return out


def soft_gate_scores(
    base_oof: np.ndarray,
    expert_oof: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    expert_center: float = 0.5,
) -> np.ndarray:
    """Add a bounded, centered expert residual inside a soft frontier gate."""
    base = _finite_scores("base_oof", base_oof)
    expert = _as_1d("expert_oof", expert_oof).astype(float, copy=False)
    w = _finite_scores("weights", weights)
    if expert.size != base.size or w.size != base.size:
        raise ValueError("base_oof, expert_oof, and weights must have equal lengths")
    delta = np.zeros(base.size, dtype=float)
    covered = np.isfinite(expert)
    delta[covered] = expert[covered] - expert_center
    return base + float(alpha) * w * delta


def veto_gate_scores(
    base_oof: np.ndarray,
    expert_oof: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    lift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Raise scores for covered expert hits inside a hard gate; never lower scores."""
    base = _finite_scores("base_oof", base_oof)
    expert = _as_1d("expert_oof", expert_oof).astype(float, copy=False)
    w = _finite_scores("weights", weights)
    if expert.size != base.size or w.size != base.size:
        raise ValueError("base_oof, expert_oof, and weights must have equal lengths")
    hit = np.isfinite(expert) & (expert >= threshold) & (w > 0.0)
    out = base.copy()
    out[hit] += float(lift) * w[hit]
    return out, hit


def midrank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-correct AUC without scipy; returns NaN when either side is empty."""
    a = np.asarray(pos, dtype=float)
    b = np.asarray(neg, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    values = np.concatenate([a, b])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    n = a.size
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * b.size))


def validation_frontier_auc(
    expert_oof: np.ndarray,
    base_oof: np.ndarray,
    y: np.ndarray,
    frontiers: Sequence[FoldFrontier],
) -> dict[str, object]:
    """Diagnostic AUC on outer validation samples selected by train-only thresholds."""
    expert = _as_1d("expert_oof", expert_oof).astype(float, copy=False)
    base = _finite_scores("base_oof", base_oof)
    yy = _binary_labels(y)
    pos: list[int] = []
    neg: list[int] = []
    for item in frontiers:
        idx = item.valid_indices
        lo = item.definition.current_threshold
        hi = item.definition.target_threshold
        band = (base[idx] >= lo) & (base[idx] < hi)
        pos.extend(idx[band & (yy[idx] == 1)].tolist())
        neg.extend(idx[band & (yy[idx] == 0)].tolist())
    pos_a = np.asarray(pos, dtype=int)
    neg_a = np.asarray(neg, dtype=int)
    pos_ok = pos_a[np.isfinite(expert[pos_a])]
    neg_ok = neg_a[np.isfinite(expert[neg_a])]
    return {
        "auc": midrank_auc(expert[pos_ok], expert[neg_ok]),
        "n_bad": int(pos_a.size),
        "n_gn": int(neg_a.size),
        "covered_bad": int(pos_ok.size),
        "covered_gn": int(neg_ok.size),
    }


def control_scores(
    expert_oof: np.ndarray,
    fold_id: np.ndarray,
    kind: str,
    seed: int,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Break expert/sample association while preserving fold, stratum, and missingness."""
    expert = _as_1d("expert_oof", expert_oof).astype(float, copy=True)
    folds = _as_1d("fold_id", fold_id)
    if expert.size != folds.size:
        raise ValueError("expert_oof/fold_id length mismatch")
    if kind not in {"permutation", "random"}:
        raise ValueError(f"control kind must be permutation or random, got {kind}")
    if strata is None:
        groups = np.zeros(expert.size, dtype=int)
    else:
        groups = _as_1d("strata", strata)
        if groups.size != expert.size:
            raise ValueError("strata length mismatch")
    rng = np.random.default_rng(seed)
    out = expert.copy()
    for fold in np.unique(folds):
        for group in np.unique(groups[folds == fold]):
            block = np.flatnonzero((folds == fold) & (groups == group) & np.isfinite(expert))
            if block.size < 2:
                continue
            if kind == "permutation":
                out[block] = expert[rng.permutation(block)]
            else:
                # Sampling with replacement preserves local scale and the exact missing mask.
                out[block] = rng.choice(expert[block], size=block.size, replace=True)
    return out


def _json_ready(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if not math.isfinite(v) else v
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _print_json(record: dict[str, object]) -> None:
    print(json.dumps(_json_ready(record), ensure_ascii=False, sort_keys=True))


def _load_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _require(bundle: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in bundle:
        raise KeyError(f"bundle is missing required array {key!r}; keys={sorted(bundle)}")
    return bundle[key]


def inspect_bundle(args: argparse.Namespace) -> None:
    bundle = _load_bundle(args.bundle)
    y = _require(bundle, args.y_key)
    fold_id = _require(bundle, args.fold_key)
    base = _require(bundle, args.base_key)
    inner = _require(bundle, args.inner_base_key)
    nested = build_nested_frontiers(y, fold_id, base, inner, args.targets, args.recall)
    _print_json({"kind": "baseline", **asdict(gn_at_recall(base, y, args.recall))})
    for target, items in nested.items():
        for item in items:
            _print_json({"kind": "frontier", **item.summary()})
        _print_json({
            "kind": "frontier_aggregate",
            "target_release": target,
            "folds": len(items),
            "bad_to_promote_total": sum(x.definition.bad_to_promote for x in items),
            "bad_to_promote_mean": np.mean([x.definition.bad_to_promote for x in items]),
            "gn_frontier_total": sum(x.definition.gn_frontier_indices.size for x in items),
        })


def _evaluate_legacy_one(
    args: argparse.Namespace,
    y: np.ndarray,
    fold_id: np.ndarray,
    base: np.ndarray,
    expert: np.ndarray,
    items: Sequence[FoldFrontier],
) -> tuple[dict[str, object], np.ndarray]:
    if args.gate == "soft":
        weights = gate_weights(base, fold_id, items, args.region, args.temperature, soft=True)
        final = soft_gate_scores(base, expert, weights, args.alpha, args.expert_center)
        detail = {
            "gate": "soft",
            "alpha": args.alpha,
            "expert_center": args.expert_center,
            "temperature": args.temperature,
            "region": args.region,
            "gate_weight_sum": float(weights.sum()),
        }
    else:
        weights = gate_weights(base, fold_id, items, args.region, args.temperature, soft=False)
        final, hit = veto_gate_scores(
            base, expert, weights, args.veto_threshold, args.veto_lift
        )
        detail = {
            "gate": "veto",
            "veto_threshold": args.veto_threshold,
            "veto_lift": args.veto_lift,
            "region": args.region,
            "veto_hits": int(hit.sum()),
            "veto_bad_hits": int(hit[np.asarray(y) == 1].sum()),
            "veto_gn_hits": int(hit[np.asarray(y) == 0].sum()),
        }
    metric = gn_at_recall(final, y, args.recall)
    baseline = gn_at_recall(base, y, args.recall)
    return {
        "protocol_mode": "legacy-raw",
        **detail,
        **asdict(metric),
        "delta_vs_base": metric.value - baseline.value,
    }, final


def _evaluate_protocol_one(
    mode: str,
    y: np.ndarray,
    r0: np.ndarray,
    rj: np.ndarray,
    rj2: np.ndarray | None = None,
    recall: float = 0.95,
) -> tuple[dict[str, object], np.ndarray]:
    if mode == "protocol-e51":
        final = protocol_e51_scores(r0, rj)
        detail: dict[str, object] = {
            "protocol_mode": mode,
            "formula": "r0+0.25*sigmoid((0.45-r0)/0.08)*(rj-0.5)",
            "missing_expert_rank": 0.5,
        }
    elif mode == "protocol-e53":
        if rj2 is None:
            raise ValueError("protocol-e53 requires a second expert rank axis")
        final = protocol_e53_scores(r0, rj, rj2)
        detail = {
            "protocol_mode": mode,
            "formula": (
                "r0+0.20*g*((rj-0.5)+(rj2-0.5))"
                "+0.15*g*relu((max(rj,rj2)-0.80)/0.20)"
            ),
            "gate": "sigmoid((0.45-r0)/0.08)",
            "missing_expert_rank": 0.5,
        }
    else:
        raise ValueError(f"unknown strict protocol mode: {mode}")
    metric = gn_at_recall(final, y, recall)
    baseline = gn_at_recall(r0, y, recall)
    return {
        **detail,
        **asdict(metric),
        "delta_vs_base": metric.value - baseline.value,
    }, final


def _resolve_evaluate_mode(args: argparse.Namespace) -> str:
    # Backward compatibility: an old invocation that explicitly passes
    # --gate soft/veto is legacy. With no mode or gate, strict E51 is default.
    if args.mode is None:
        return "legacy-raw" if args.gate is not None else "protocol-e51"
    if args.mode == "legacy-raw":
        if args.gate is None:
            raise ValueError("--mode legacy-raw requires explicit --gate soft or veto")
    elif args.gate is not None:
        raise ValueError("--gate is legacy-only; omit it in strict protocol modes")
    return args.mode


def _print_control_summary(
    *,
    control: str,
    mode: str,
    target: float,
    real_value: float,
    values: list[float],
    deltas: list[float],
    repeats: int,
    independent_axes: bool,
) -> None:
    arr = np.asarray(values)
    delta = np.asarray(deltas)
    _print_json({
        "kind": "control",
        "control": control,
        "protocol_mode": mode,
        "matched_null_partition": "outer_fold_x_strata",
        "independently_permuted_axes": independent_axes,
        "target_release": target,
        "repeats": repeats,
        "value_min": arr.min(),
        "value_mean": arr.mean(),
        "value_p95": np.quantile(arr, 0.95),
        "value_max": arr.max(),
        "delta_mean": delta.mean(),
        "delta_max": delta.max(),
        "real_gt_control_max": bool(real_value > arr.max()),
        "empirical_p_ge_real": (
            1 + int(np.count_nonzero(arr >= real_value))
        ) / (arr.size + 1),
    })


def evaluate_bundle(args: argparse.Namespace) -> None:
    bundle = _load_bundle(args.bundle)
    y = _require(bundle, args.y_key)
    fold_id = _require(bundle, args.fold_key)
    base = _require(bundle, args.base_key)
    inner = _require(bundle, args.inner_base_key)
    yy, folds, base, inner = validate_nested_bundle(y, fold_id, base, inner)
    strata = bundle.get(args.strata_key) if args.strata_key else None
    if strata is not None and _as_1d(args.strata_key, strata).size != yy.size:
        raise ValueError("strata length mismatch")
    nested = build_nested_frontiers(yy, folds, base, inner, [args.target], args.recall)
    items = nested[float(args.target)]
    mode = _resolve_evaluate_mode(args)

    expert = _require(bundle, args.expert_key)
    if mode == "legacy-raw":
        expert = _finite_or_nan(args.expert_key, expert)
        if expert.size != yy.size:
            raise ValueError("expert_oof length mismatch")
        baseline = gn_at_recall(base, yy, args.recall)
        diagnostic = validation_frontier_auc(expert, base, yy, items)
        real, _ = _evaluate_legacy_one(args, yy, folds, base, expert, items)
        _print_json({
            "kind": "real",
            "protocol_mode": mode,
            "target_release": args.target,
            "baseline_value": baseline.value,
            "baseline_score_space": "raw_outer_oof_legacy",
            "expert_coverage": float(np.isfinite(expert).mean()),
            "validation_frontier": diagnostic,
            **real,
        })
        for kind in args.controls:
            values: list[float] = []
            deltas: list[float] = []
            for repeat in range(args.control_repeats):
                fake = control_scores(
                    expert, folds, kind, args.control_seed + repeat, strata=strata
                )
                record, _ = _evaluate_legacy_one(
                    args, yy, folds, base, fake, items
                )
                values.append(float(record["value"]))
                deltas.append(float(record["delta_vs_base"]))
            _print_control_summary(
                control=kind,
                mode=mode,
                target=args.target,
                real_value=float(real["value"]),
                values=values,
                deltas=deltas,
                repeats=args.control_repeats,
                independent_axes=False,
            )
        return

    inner_expert = _require(bundle, args.inner_expert_key)
    r0 = crossfit_mid_cdf(
        base, inner, folds, name=args.base_key, allow_missing=False
    )
    rj = crossfit_mid_cdf(
        expert,
        inner_expert,
        folds,
        name=args.expert_key,
        allow_missing=True,
    )
    rj2: np.ndarray | None = None
    if mode == "protocol-e53":
        expert2 = _require(bundle, args.expert2_key)
        inner_expert2 = _require(bundle, args.inner_expert2_key)
        rj2 = crossfit_mid_cdf(
            expert2,
            inner_expert2,
            folds,
            name=args.expert2_key,
            allow_missing=True,
        )

    baseline = gn_at_recall(r0, yy, args.recall)
    real, final = _evaluate_protocol_one(
        mode, yy, r0, rj, rj2, recall=args.recall
    )
    diagnostics: dict[str, object] = {
        "axis1": validation_frontier_auc(rj, base, yy, items),
        "fused": validation_frontier_auc(final, base, yy, items),
    }
    if rj2 is not None:
        diagnostics["axis2"] = validation_frontier_auc(rj2, base, yy, items)
    record: dict[str, object] = {
        "kind": "real",
        "protocol_mode": mode,
        "target_release": args.target,
        "baseline_value": baseline.value,
        "baseline_score_space": "foldwise_inner_oof_mid_cdf_r0",
        "expert_coverage": float(np.isfinite(rj).mean()),
        "validation_frontier": diagnostics,
        **real,
    }
    if rj2 is not None:
        record["expert2_coverage"] = float(np.isfinite(rj2).mean())
    _print_json(record)

    for kind in args.controls:
        values = []
        deltas = []
        for repeat in range(args.control_repeats):
            first_seed = args.control_seed + repeat
            fake = control_scores(rj, folds, kind, first_seed, strata=strata)
            fake2 = None
            if rj2 is not None:
                # A fixed disjoint seed stream prevents coupled E53 permutations.
                second_seed = args.control_seed + 1_000_003 + repeat
                fake2 = control_scores(
                    rj2, folds, kind, second_seed, strata=strata
                )
            null_record, _ = _evaluate_protocol_one(
                mode, yy, r0, fake, fake2, recall=args.recall
            )
            values.append(float(null_record["value"]))
            deltas.append(float(null_record["delta_vs_base"]))
        _print_control_summary(
            control=kind,
            mode=mode,
            target=args.target,
            real_value=float(real["value"]),
            values=values,
            deltas=deltas,
            repeats=args.control_repeats,
            independent_axes=(rj2 is not None),
        )


def self_test() -> None:
    # Strict <T: one GN ties T and must not be released.
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.2, 0.19])
    metric = gn_at_recall(p, y, recall=0.75)
    assert metric.threshold == 0.2
    assert metric.released_gn == 1 and metric.value == 0.5

    # A deterministic frontier with a non-empty shell and strict GN band.
    y2 = np.array([1] * 20 + [0] * 20)
    p2 = np.concatenate([np.linspace(0.05, 1.0, 20), np.linspace(0.01, 0.96, 20)])
    frontier = define_frontier(p2, y2, target_release=0.35, recall=0.80)
    assert frontier.bad_to_promote > 0
    assert frontier.bad_shell_indices.size == frontier.bad_to_promote
    assert np.all(y2[frontier.bad_shell_indices] == 1)
    assert np.all(y2[frontier.gn_frontier_indices] == 0)

    # Strict nested matrix: outer validation cells are NaN, training cells finite.
    n = 60
    folds = np.arange(n) % 3
    y3 = np.tile([0, 1], n // 2)
    base = np.linspace(0.01, 0.99, n) + 0.03 * y3
    inner = np.tile(base, (3, 1))
    for k in range(3):
        inner[k, folds == k] = np.nan
    nested = build_nested_frontiers(
        y3, folds, base, inner, targets=(0.35, 0.50), recall=0.80
    )
    assert set(nested) == {0.35, 0.50}
    broken = inner.copy()
    broken[0, folds == 0] = 0.5
    try:
        validate_nested_bundle(y3, folds, base, broken)
    except ValueError as exc:
        assert "validation fold" in str(exc)
    else:
        raise AssertionError("nested leakage guard did not reject validation scores")

    # Foldwise empirical mid-CDF uses only row-k outer-train inner OOF.
    folds4 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    axis4 = np.array([0.0, 1.5, 3.0, 4.0, 10.0, 25.0, 40.0, 50.0])
    inner4 = np.full((2, 8), np.nan)
    inner4[0, folds4 != 0] = np.array([0.0, 1.0, 2.0, 3.0])
    inner4[1, folds4 != 1] = np.array([10.0, 20.0, 30.0, 40.0])
    ranks4 = crossfit_mid_cdf(
        axis4, inner4, folds4, name="synthetic_base", allow_missing=False
    )
    expected4 = np.array([0.125, 0.5, 0.875, 1.0, 0.125, 0.5, 0.875, 1.0])
    assert np.allclose(ranks4, expected4)

    # Changing only fold-1's reference row cannot alter fold-0 validation ranks.
    altered4 = inner4.copy()
    altered4[1, folds4 != 1] = np.array([-100.0, -10.0, 100.0, 1000.0])
    altered_ranks4 = crossfit_mid_cdf(
        axis4, altered4, folds4, name="synthetic_base", allow_missing=False
    )
    assert np.allclose(altered_ranks4[folds4 == 0], ranks4[folds4 == 0])

    # Candidate NaN means abstention, while any non-NaN outer-valid inner cell leaks.
    expert4 = axis4.copy()
    expert4[2] = np.nan
    inner_expert4 = inner4.copy()
    inner_expert4[0, 5] = np.nan  # Missing outer-train candidate is allowed.
    expert_ranks4 = crossfit_mid_cdf(
        expert4,
        inner_expert4,
        folds4,
        name="synthetic_expert",
        allow_missing=True,
    )
    assert np.isnan(expert_ranks4[2])
    leaked_expert4 = inner_expert4.copy()
    leaked_expert4[0, 0] = 0.5
    try:
        crossfit_mid_cdf(
            expert4,
            leaked_expert4,
            folds4,
            name="synthetic_expert",
            allow_missing=True,
        )
    except ValueError as exc:
        assert "validation fold" in str(exc)
    else:
        raise AssertionError("candidate NaN contract did not reject outer-valid score")

    # Exact frozen E51/E53 formulas and neutral 0.5 missing-axis behavior.
    r0_formula = np.array([0.2, 0.4, 0.6])
    r1_formula = np.array([np.nan, 0.9, 0.1])
    g_formula = _sigmoid((0.45 - r0_formula) / 0.08)
    e51 = protocol_e51_scores(r0_formula, r1_formula)
    r1_neutral = np.array([0.5, 0.9, 0.1])
    expected_e51 = r0_formula + 0.25 * g_formula * (r1_neutral - 0.5)
    assert np.allclose(e51, expected_e51)
    assert e51[0] == r0_formula[0]

    r2_formula = np.array([0.95, np.nan, 0.85])
    r2_neutral = np.array([0.95, 0.5, 0.85])
    expected_e53 = (
        r0_formula
        + 0.20 * g_formula * ((r1_neutral - 0.5) + (r2_neutral - 0.5))
        + 0.15
        * g_formula
        * np.maximum((np.maximum(r1_neutral, r2_neutral) - 0.80) / 0.20, 0.0)
    )
    assert np.allclose(
        protocol_e53_scores(r0_formula, r1_formula, r2_formula), expected_e53
    )

    # Gate mechanics and abstention behavior.
    items = nested[0.50]
    soft_w = gate_weights(base, folds, items, temperature=0.05, soft=True)
    hard_w = gate_weights(base, folds, items, soft=False)
    assert np.all((soft_w >= 0.0) & (soft_w <= 1.0))
    expert = np.linspace(0.0, 1.0, n)
    expert[0] = np.nan
    fused = soft_gate_scores(base, expert, soft_w, alpha=0.1, expert_center=0.5)
    assert fused[0] == base[0]
    vetoed, hit = veto_gate_scores(base, expert, hard_w, threshold=0.8, lift=1.0)
    assert np.all(vetoed >= base)
    assert not hit[0]

    # Midrank, control reproducibility, missing-mask preservation.
    assert midrank_auc(np.array([1.0, 1.0]), np.array([0.0, 1.0])) == 0.75
    c1 = control_scores(expert, folds, "permutation", 7)
    c2 = control_scores(expert, folds, "permutation", 7)
    assert np.allclose(c1, c2, equal_nan=True)
    assert np.array_equal(np.isnan(c1), np.isnan(expert))
    print("E50 frontier harness strict-protocol self-test: OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict nested-OOF frontier-shell and gate evaluation harness (train only)."
    )
    parser.add_argument("--self-test", action="store_true", help="run synthetic tests and exit")
    sub = parser.add_subparsers(dest="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--bundle", type=Path, required=True, help="explicit train-only NPZ bundle")
        p.add_argument("--y-key", default="y")
        p.add_argument("--fold-key", default="fold_id")
        p.add_argument("--base-key", default="base_oof")
        p.add_argument("--inner-base-key", default="inner_base")
        p.add_argument("--recall", type=float, default=0.95)

    inspect_p = sub.add_parser("inspect", help="validate bundle and print per-fold shell definitions")
    common(inspect_p)
    inspect_p.add_argument("--targets", type=float, nargs="+", default=[0.35, 0.50])

    eval_p = sub.add_parser(
        "evaluate",
        help="evaluate the fixed E51/E53 rank protocol (default E51)",
    )
    common(eval_p)
    eval_p.add_argument("--expert-key", default="expert_oof")
    eval_p.add_argument("--inner-expert-key", default="inner_expert")
    eval_p.add_argument("--expert2-key", default="expert2_oof")
    eval_p.add_argument("--inner-expert2-key", default="inner_expert2")
    eval_p.add_argument("--target", type=float, choices=[0.35, 0.50], required=True)
    eval_p.add_argument(
        "--mode",
        choices=["protocol-e51", "protocol-e53", "legacy-raw"],
        default=None,
        help="default: protocol-e51; legacy-raw must be explicitly selected",
    )
    eval_p.add_argument(
        "--gate",
        choices=["soft", "veto"],
        default=None,
        help="legacy raw-score gate only; old --gate-only calls remain legacy",
    )
    eval_p.add_argument(
        "--region", choices=["frontier", "below-target", "all"], default="frontier"
    )
    eval_p.add_argument("--temperature", type=float, default=0.02)
    eval_p.add_argument(
        "--alpha", type=float, default=0.04,
        help="legacy soft residual multiplier; ignored by strict protocol modes",
    )
    eval_p.add_argument(
        "--expert-center", type=float, default=0.5,
        help="legacy expert center; ignored by strict protocol modes",
    )
    eval_p.add_argument("--veto-threshold", type=float, default=0.90)
    eval_p.add_argument("--veto-lift", type=float, default=1.0)
    eval_p.add_argument(
        "--controls", nargs="*", choices=["permutation", "random"], default=[]
    )
    eval_p.add_argument("--control-repeats", type=int, default=50)
    eval_p.add_argument("--control-seed", type=int, default=20260810)
    eval_p.add_argument(
        "--strata-key", default="strata",
        help="optional bundle key; controls stay within fold x stratum",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.command == "inspect":
        inspect_bundle(args)
    elif args.command == "evaluate":
        if args.control_repeats <= 0:
            parser.error("--control-repeats must be positive")
        evaluate_bundle(args)
    else:
        parser.error("choose inspect/evaluate or pass --self-test")


if __name__ == "__main__":
    main()
