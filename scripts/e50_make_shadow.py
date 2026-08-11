#!/usr/bin/env python3
"""Build the one-shot E50 discovery/shadow split and sealed discovery contract.

Human-facing membership files remain token-only.  The same atomic S0 directory
also contains ``discovery_split.npz`` for train-only code: rows are sorted by
``sample_token(video)`` and contain token, binary label, source stratum, frozen
outer fold, and strict four-fold inner ids.  No video names, paths, predictions,
shadow labels, or eval content are written.  The output directory is immutable
by construction: this tool refuses to run if it already exists.

This is the S0 split builder described by ``E50_PREREG_DRAFT.md``.  It must not
be used on eval manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "e50_s0_membership_v2"
VALID_LABELS = frozenset({"bad", "good", "normal"})
INNER_FOLDS = 4
INNER_SEED_BASE = 20260811


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_token(video: str) -> str:
    """Return the irreversible stable join key used by all S0 artifacts."""
    return sha256_bytes(b"e50-sample-v1\0" + video.encode("utf-8"))


def seeded_rank(seed: int, namespace: str, value: str) -> str:
    return sha256_bytes(f"{seed}\0{namespace}\0{value}".encode("utf-8"))


def derive_source(row: Mapping[str, object]) -> str:
    """Derive the corpus source without retaining the original path."""
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


def read_rows(path: Path) -> list[dict[str, str]]:
    if "eval" in path.name.lower():
        raise ValueError(f"refusing eval-like manifest: {path}")
    rows: list[dict[str, str]] = []
    seen_video: set[str] = set()
    seen_token: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            label = str(raw.get("label", ""))
            video = str(raw.get("video", ""))
            if label not in VALID_LABELS:
                raise ValueError(f"line {lineno}: invalid label {label!r}")
            if not video:
                raise ValueError(f"line {lineno}: missing video")
            token = sample_token(video)
            if video in seen_video or token in seen_token:
                raise ValueError(f"line {lineno}: duplicate video/token")
            seen_video.add(video)
            seen_token.add(token)
            rows.append(
                {
                    "token": token,
                    "label": label,
                    "source": derive_source(raw),
                }
            )
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def allocate_shadow_quotas(
    strata: Mapping[tuple[str, str], list[dict[str, str]]], fraction: float
) -> dict[tuple[str, str], int]:
    """Largest-remainder allocation with an exact global shadow size."""
    total = sum(len(items) for items in strata.values())
    target = int(round(total * fraction))
    quotas = {key: int(math.floor(len(items) * fraction)) for key, items in strata.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        strata,
        key=lambda key: (
            -(len(strata[key]) * fraction - quotas[key]),
            key[0],
            key[1],
        ),
    )
    for key in order:
        if remaining <= 0:
            break
        if quotas[key] < len(strata[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise AssertionError(f"unable to allocate {remaining} shadow rows")
    if sum(quotas.values()) != target:
        raise AssertionError("shadow quota total mismatch")
    return quotas


def make_split(
    rows: Iterable[dict[str, str]], seed: int, fraction: float, outer_folds: int
) -> tuple[set[str], set[str], dict[str, int]]:
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[(row["label"], row["source"])].append(row)
    quotas = allocate_shadow_quotas(strata, fraction)
    shadow: set[str] = set()
    discovery_rows: list[dict[str, str]] = []
    for stratum, items in sorted(strata.items()):
        ranked = sorted(
            items,
            key=lambda row: seeded_rank(seed, "shadow", row["token"]),
        )
        quota = quotas[stratum]
        shadow.update(row["token"] for row in ranked[:quota])
        discovery_rows.extend(ranked[quota:])

    discovery = {row["token"] for row in discovery_rows}
    if discovery & shadow:
        raise AssertionError("discovery/shadow overlap")
    if len(discovery) + len(shadow) != len(discovery_rows) + len(shadow):
        raise AssertionError("membership cardinality mismatch")

    # Freeze outer folds at S0 as well.  Round-robin assignment is performed
    # independently within label x source strata after a seeded hash sort.
    outer: dict[str, int] = {}
    discovery_strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in discovery_rows:
        discovery_strata[(row["label"], row["source"])].append(row)
    for stratum, items in sorted(discovery_strata.items()):
        ranked = sorted(items, key=lambda row: seeded_rank(seed, "outer", row["token"]))
        offset = int(seeded_rank(seed, "fold-offset", "\0".join(stratum))[:8], 16) % outer_folds
        for index, row in enumerate(ranked):
            outer[row["token"]] = (offset + index) % outer_folds
    if set(outer) != discovery:
        raise AssertionError("outer fold membership mismatch")
    return discovery, shadow, outer


def assign_inner_folds(
    rows: list[dict[str, str]], seed: int, n_folds: int = INNER_FOLDS
) -> dict[str, int]:
    """Deterministically balance label x source strata over inner folds."""
    if n_folds != INNER_FOLDS:
        raise ValueError(f"protocol requires exactly {INNER_FOLDS} inner folds")
    binary_counts = Counter(1 if row["label"] == "bad" else 0 for row in rows)
    if any(binary_counts[label] < n_folds for label in (0, 1)):
        raise ValueError(
            f"inner split needs at least {n_folds} rows of each binary class; "
            f"got {dict(binary_counts)}"
        )

    by_label: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_label[row["label"]][row["source"]].append(row)

    assignment: dict[str, int] = {}
    for label in sorted(by_label):
        fold_counts = np.zeros(n_folds, dtype=np.int64)
        sources = sorted(
            by_label[label],
            key=lambda source: (
                -len(by_label[label][source]),
                seeded_rank(seed, "inner-stratum-order", f"{label}\0{source}"),
                source,
            ),
        )
        for source in sources:
            items = sorted(
                by_label[label][source],
                key=lambda row: seeded_rank(
                    seed, f"inner-row-{label}-{source}", row["token"]
                ),
            )
            choices: list[tuple[tuple[int, int, int, str], int, np.ndarray]] = []
            for offset in range(n_folds):
                proposed = (offset + np.arange(len(items))) % n_folds
                trial = fold_counts + np.bincount(proposed, minlength=n_folds)
                cost = (
                    int(np.count_nonzero(trial == 0)),
                    int(trial.max() - trial.min()),
                    int(np.dot(trial, trial)),
                    seeded_rank(
                        seed,
                        "inner-offset",
                        f"{label}\0{source}\0{offset}",
                    ),
                )
                choices.append((cost, offset, proposed))
            _cost, _offset, selected = min(choices, key=lambda item: item[0])
            for row, fold in zip(items, selected.tolist()):
                assignment[row["token"]] = int(fold)
            fold_counts += np.bincount(selected, minlength=n_folds)

    if set(assignment) != {row["token"] for row in rows}:
        raise AssertionError("inner fold assignment coverage mismatch")
    for fold in range(n_folds):
        labels = {
            1 if row["label"] == "bad" else 0
            for row in rows
            if assignment[row["token"]] == fold
        }
        if labels != {0, 1}:
            raise ValueError(
                f"inner fold {fold} lacks a binary class after deterministic "
                "label x source allocation"
            )
    return assignment


def build_discovery_split_arrays(
    rows: list[dict[str, str]],
    discovery: set[str],
    outer: Mapping[str, int],
    outer_folds: int,
) -> dict[str, np.ndarray]:
    """Create the sorted S0 discovery arrays consumed by nested builders."""
    selected = sorted(
        (row for row in rows if row["token"] in discovery),
        key=lambda row: row["token"],
    )
    if len(selected) != len(discovery) or set(outer) != discovery:
        raise AssertionError("discovery array membership mismatch")
    tokens = np.asarray([row["token"] for row in selected], dtype="<U64")
    if tokens.size and not np.all(tokens[:-1] < tokens[1:]):
        raise AssertionError("sample_token order must be strictly sorted")
    y = np.asarray([1 if row["label"] == "bad" else 0 for row in selected], dtype=np.int8)
    strata = np.asarray([row["source"] for row in selected], dtype=str)
    fold_id = np.asarray([outer[row["token"]] for row in selected], dtype=np.int16)
    if not np.array_equal(np.unique(fold_id), np.arange(outer_folds)):
        raise ValueError("outer folds must be contiguous and all represented")

    inner = np.full((outer_folds, len(selected)), -1, dtype=np.int16)
    for k in range(outer_folds):
        train_rows = [row for row in selected if outer[row["token"]] != k]
        assignment = assign_inner_folds(train_rows, INNER_SEED_BASE + k)
        train_mask = fold_id != k
        for index in np.flatnonzero(train_mask):
            inner[k, index] = assignment[str(tokens[index])]
        if not (inner[k, ~train_mask] == -1).all():
            raise AssertionError(f"outer {k} valid inner ids must be -1")
        if not np.array_equal(np.unique(inner[k, train_mask]), np.arange(INNER_FOLDS)):
            raise AssertionError(f"outer {k} train inner ids must be 0..3")
        for j in range(INNER_FOLDS):
            inner_valid = train_mask & (inner[k] == j)
            if set(y[inner_valid].tolist()) != {0, 1}:
                raise AssertionError(f"outer {k} inner {j} must contain both classes")

    return {
        "sample_token": tokens,
        "y": y,
        "strata": strata,
        "fold_id": fold_id,
        "inner_fold_id": inner,
    }


def aggregate(rows: Iterable[dict[str, str]], members: set[str]) -> dict[str, object]:
    selected = [row for row in rows if row["token"] in members]
    return {
        "n": len(selected),
        "label_counts": dict(sorted(Counter(row["label"] for row in selected).items())),
        "source_counts": dict(sorted(Counter(row["source"] for row in selected).items())),
        "stratum_counts": {
            f"{label}|{source}": count
            for (label, source), count in sorted(
                Counter((row["label"], row["source"]) for row in selected).items()
            )
        },
    }


def lines_digest(lines: Iterable[str]) -> tuple[str, bytes]:
    data = ("\n".join(lines) + "\n").encode("utf-8")
    return sha256_bytes(data), data


def write_s0(
    output_dir: Path,
    manifest: Path,
    rows: list[dict[str, str]],
    discovery: set[str],
    shadow: set[str],
    outer: Mapping[str, int],
    seed: int,
    fraction: float,
    outer_folds: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"S0 output already exists and is immutable: {output_dir}"
        )
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temp_dir.exists():
        raise FileExistsError(f"temporary output already exists: {temp_dir}")
    temp_dir.mkdir(parents=True)

    discovery_sha, discovery_data = lines_digest(sorted(discovery))
    shadow_sha, shadow_data = lines_digest(sorted(shadow))
    outer_lines = [
        json.dumps({"sample_token": token, "outer_fold": outer[token]}, sort_keys=True)
        for token in sorted(outer)
    ]
    outer_sha, outer_data = lines_digest(outer_lines)
    (temp_dir / "discovery_ids.txt").write_bytes(discovery_data)
    (temp_dir / "shadow_ids.txt").write_bytes(shadow_data)
    (temp_dir / "discovery_outer_folds.jsonl").write_bytes(outer_data)
    discovery_arrays = build_discovery_split_arrays(
        rows, discovery, outer, outer_folds
    )
    discovery_split_path = temp_dir / "discovery_split.npz"
    np.savez_compressed(discovery_split_path, **discovery_arrays)
    discovery_split_sha = sha256_file(discovery_split_path)

    summary = {
        "schema_version": SCHEMA,
        "status": "S0_FROZEN",
        "seed": seed,
        "shadow_fraction": fraction,
        "outer_folds": outer_folds,
        "inner_folds": INNER_FOLDS,
        "inner_seed": "20260811 + outer_fold_id",
        "discovery_split_order": "strictly sorted sample_token",
        "discovery_split_fields": [
            "sample_token",
            "y",
            "strata",
            "fold_id",
            "inner_fold_id",
        ],
        "selection": "label_x_source largest-remainder; seeded SHA-256 rank",
        "source_derivation": "first component after s3/corpus_videos, else parent",
        "sample_token": "sha256('e50-sample-v1\\0' + video)",
        "manifest_sha256": sha256_file(manifest),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "all": aggregate(rows, {row["token"] for row in rows}),
        "discovery": aggregate(rows, discovery),
        "shadow": aggregate(rows, shadow),
        "fold_counts": dict(sorted(Counter(outer.values()).items())),
        "artifacts": {
            "discovery_ids.txt": discovery_sha,
            "shadow_ids.txt": shadow_sha,
            "discovery_outer_folds.jsonl": outer_sha,
            "discovery_split.npz": discovery_split_sha,
        },
    }
    summary_data = (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    (temp_dir / "summary.json").write_bytes(summary_data)
    lock = {
        "schema_version": SCHEMA,
        "status": "S0_FROZEN",
        "summary_sha256": sha256_bytes(summary_data),
        "artifact_sha256": summary["artifacts"],
    }
    (temp_dir / "LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp_dir.replace(output_dir)


def self_test() -> None:
    rows: list[dict[str, str]] = []
    for label_index, label in enumerate(("bad", "good", "normal")):
        for source_index, source in enumerate(("a", "b", "c")):
            for index in range(11 + label_index + source_index):
                video = f"{label}-{source}-{index}.mp4"
                rows.append({"token": sample_token(video), "label": label, "source": source})
    first = make_split(rows, 20260810, 0.25, 5)
    second = make_split(list(reversed(rows)), 20260810, 0.25, 5)
    assert first == second
    discovery, shadow, outer = first
    assert not discovery & shadow
    assert discovery | shadow == {row["token"] for row in rows}
    assert set(outer) == discovery
    assert len(shadow) == round(len(rows) * 0.25)
    assert len(set(outer.values())) == 5
    arrays = build_discovery_split_arrays(rows, discovery, outer, 5)
    reversed_arrays = build_discovery_split_arrays(
        list(reversed(rows)), discovery, outer, 5
    )
    for key in arrays:
        assert np.array_equal(arrays[key], reversed_arrays[key])
    tokens = arrays["sample_token"]
    assert np.all(tokens[:-1] < tokens[1:])
    assert arrays["inner_fold_id"].shape == (5, len(discovery))
    for k in range(5):
        valid = arrays["fold_id"] == k
        assert (arrays["inner_fold_id"][k, valid] == -1).all()
        for j in range(INNER_FOLDS):
            inner_valid = (~valid) & (arrays["inner_fold_id"][k] == j)
            assert set(arrays["y"][inner_valid].tolist()) == {0, 1}

    # Exercise atomic S0 writing and the summary/LOCK hash chain on synthetic data.
    with tempfile.TemporaryDirectory(prefix="e50-s0-selftest-") as tmp:
        root = Path(tmp)
        manifest = root / "synthetic_train.jsonl"
        manifest.write_text("{}\n", encoding="utf-8")
        output = root / "s0"
        write_s0(
            output,
            manifest,
            rows,
            discovery,
            shadow,
            outer,
            20260810,
            0.25,
            5,
        )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        lock = json.loads((output / "LOCK.json").read_text(encoding="utf-8"))
        split_sha = sha256_file(output / "discovery_split.npz")
        assert summary["artifacts"]["discovery_split.npz"] == split_sha
        assert lock["artifact_sha256"]["discovery_split.npz"] == split_sha
        with np.load(output / "discovery_split.npz", allow_pickle=False) as saved:
            for key in arrays:
                assert np.array_equal(saved[key], arrays[key])
    print("E50 S0 split-builder v2 self-test: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits" / "train_v3.jsonl")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "prod500" / "e50_s0"
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--shadow-fraction", type=float, default=0.25)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not 0.0 < args.shadow_fraction < 1.0:
        raise ValueError("--shadow-fraction must be in (0, 1)")
    if args.outer_folds < 2:
        raise ValueError("--outer-folds must be at least 2")
    manifest = args.manifest.resolve()
    rows = read_rows(manifest)
    discovery, shadow, outer = make_split(
        rows, args.seed, args.shadow_fraction, args.outer_folds
    )
    discovery_arrays = build_discovery_split_arrays(
        rows, discovery, outer, args.outer_folds
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "DRY_RUN_NOT_FROZEN",
                    "n": len(rows),
                    "discovery_n": len(discovery),
                    "shadow_n": len(shadow),
                    "outer_fold_counts": dict(sorted(Counter(outer.values()).items())),
                    "inner_fold_shape": list(discovery_arrays["inner_fold_id"].shape),
                    "inner_seed": "20260811 + outer_fold_id",
                },
                sort_keys=True,
            )
        )
        return
    write_s0(
        args.output_dir.resolve(),
        manifest,
        rows,
        discovery,
        shadow,
        outer,
        args.seed,
        args.shadow_fraction,
        args.outer_folds,
    )
    print(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": "S0_FROZEN",
                "output_dir": str(args.output_dir.resolve()),
                "n": len(rows),
                "discovery_n": len(discovery),
                "shadow_n": len(shadow),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
