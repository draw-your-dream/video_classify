#!/usr/bin/env python3
"""Label-blind token-explicit wrapper for prospective E51 shadow extraction.

The frozen extractor is invoked with repeated --only-token plus a positive
--limit and never with --discovery-ids. Its own artifact therefore truthfully
remains a technical_subset; this wrapper's receipt records that the subset is
the frozen S0 prospective shadow. No label or metric field is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

EXTRACTOR_SHA256 = "b639dcc04dc60fae26c3a5725f3eebc46121b62e270c89f7e260ebb4b96bc37e"
MANIFEST_SHA256 = "3ae40b797113ab9d1195ef3566e1380fe2c53c83d1937f19161b082a4d1da40d"
SCHEMA = "e51_shadow_extract_receipt_v1"


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


def safe_shadow_path(path: Path, role: str, suffix: str | None = None) -> Path:
    resolved = path.expanduser().resolve()
    folded = resolved.name.casefold()
    if "eval" in folded:
        raise ValueError(f"{role} refuses an eval-named path")
    if "shadow" not in folded:
        raise ValueError(f"{role} must explicitly contain 'shadow'")
    if suffix and resolved.suffix.casefold() != suffix:
        raise ValueError(f"{role} must end in {suffix}")
    return resolved


def select_window(tokens: Sequence[str], offset: int, limit: int) -> list[str]:
    if offset < 0 or limit <= 0:
        raise ValueError("token offset must be >=0 and token limit must be >0")
    selected = list(tokens[offset : offset + limit])
    if not selected:
        raise ValueError("selected shadow shard is empty")
    return selected


def build_command(args: argparse.Namespace, selected: Sequence[str]) -> list[str]:
    command = [
        str(args.python), str(args.extractor),
        "--manifest", str(args.manifest),
        "--output-root", str(args.output_root),
        "--repo", str(args.repo),
        "--checkpoint", str(args.checkpoint),
        "--device", args.device,
        "--limit", str(len(selected)),
        "--max-failures", str(args.max_failures),
    ]
    for value in args.path_map:
        command += ["--path-map", value]
    if args.videos_root:
        command += ["--videos-root", str(args.videos_root)]
    for token in selected:
        command += ["--only-token", token]
    if args.extractor_dry_run:
        command.append("--dry-run")
    if "--discovery-ids" in command or int(command[command.index("--limit") + 1]) <= 0:
        raise AssertionError("unsafe shadow extractor command")
    return command


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_pair(root: Path, token: str) -> dict[str, str]:
    meta_path, array_path = root / f"{token}.json", root / f"{token}.npz"
    if not meta_path.is_file() or not array_path.is_file():
        raise RuntimeError(f"missing feature pair for {token}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    selection = payload.get("selection", {})
    if (
        payload.get("status") != "ok"
        or payload.get("sample_token") != token
        or payload.get("label_accessed") is not False
        or payload.get("metric_computed") is not False
        or selection.get("scope") != "technical_subset"
        or selection.get("discovery_ids_sha256") is not None
    ):
        raise RuntimeError(f"feature metadata contract failed for {token}")
    return {"json_sha256": sha256_file(meta_path), "npz_sha256": sha256_file(array_path)}


def run(args: argparse.Namespace) -> int:
    if not args.confirm_label_blind_shadow:
        raise RuntimeError("run requires --confirm-label-blind-shadow")
    args.extractor = args.extractor.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.shadow_ids = args.shadow_ids.expanduser().resolve()
    args.discovery_ids = args.discovery_ids.expanduser().resolve()
    args.output_root = safe_shadow_path(args.output_root, "output root")
    args.receipt = safe_shadow_path(args.receipt, "receipt", ".json")
    for path, role in (
        (args.extractor, "extractor"), (args.manifest, "manifest"),
        (args.shadow_ids, "shadow ids"), (args.discovery_ids, "discovery ids"),
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{role} must be a regular file: {path}")
    verify_sha(args.extractor, EXTRACTOR_SHA256, "extractor")
    manifest_sha = verify_sha(args.manifest, args.expected_manifest_sha256, "manifest")
    if manifest_sha != MANIFEST_SHA256:
        raise RuntimeError("manifest differs from frozen E51 train manifest")
    shadow_sha = verify_sha(args.shadow_ids, args.expected_shadow_ids_sha256, "shadow ids")
    discovery_sha = verify_sha(args.discovery_ids, args.expected_discovery_ids_sha256, "discovery ids")
    shadow_tokens = read_tokens(args.shadow_ids, "shadow ids")
    discovery_tokens = set(read_tokens(args.discovery_ids, "discovery ids"))
    if discovery_tokens.intersection(shadow_tokens):
        raise RuntimeError("S0 discovery and shadow memberships overlap")
    selected = select_window(shadow_tokens, args.token_offset, args.token_limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    command = build_command(args, selected)
    plan = {
        "schema_version": SCHEMA,
        "scope": "s0_prospective_shadow_label_blind",
        "selected_count": len(selected),
        "token_offset_in_sorted_shadow": args.token_offset,
        "extractor_sha256": EXTRACTOR_SHA256,
        "manifest_sha256": manifest_sha,
        "shadow_ids_sha256": shadow_sha,
        "discovery_ids_sha256": discovery_sha,
        "extractor_selection_scope": "technical_subset",
        "passes_discovery_ids_to_extractor": False,
        "positive_limit": len(selected),
        "label_accessed": False,
        "metric_computed": False,
    }
    print(json.dumps({**plan, "wrapper_dry_run": args.wrapper_dry_run}, sort_keys=True), flush=True)
    if args.wrapper_dry_run:
        return 0
    subprocess.run(command, check=True)
    if args.extractor_dry_run:
        print("E51_SHADOW_EXTRACT_DRY_RUN receipt_written=false", flush=True)
        return 0
    artifacts = {token: validate_pair(args.output_root, token) for token in selected}
    atomic_json(args.receipt, {
        **plan,
        "status": "complete",
        "selected_tokens": selected,
        "artifacts": artifacts,
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps({
        "status": "complete", "samples": len(selected),
        "receipt": str(args.receipt), "receipt_sha256": sha256_file(args.receipt),
        "label_accessed": False, "metric_computed": False,
    }, sort_keys=True), flush=True)
    return 0


def selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="e51-shadow-extract-selftest-") as temp:
        root = Path(temp)
        tokens = [f"{i:064x}" for i in range(8)]
        path = root / "shadow_ids.txt"
        path.write_text("\n".join(reversed(tokens)) + "\n", encoding="utf-8")
        assert read_tokens(path, "shadow") == tokens
        selected = select_window(tokens, 2, 3)
        args = argparse.Namespace(
            python=Path("python"), extractor=Path("extractor.py"),
            manifest=Path("train_v3.jsonl"), output_root=root / "shadow_features",
            repo=Path("repo"), checkpoint=Path("checkpoint"), device="cuda:0",
            max_failures=1, path_map=[], videos_root=None, extractor_dry_run=False,
        )
        command = build_command(args, selected)
        assert "--discovery-ids" not in command
        assert command.count("--only-token") == 3
        assert command[command.index("--limit") + 1] == "3"
        try:
            select_window(tokens, 0, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("zero limit accepted")
    print("E51 prospective-shadow extractor wrapper self-test: OK")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    ap = sub.add_parser("run")
    ap.add_argument("--extractor", type=Path, required=True)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--expected-manifest-sha256", required=True)
    ap.add_argument("--shadow-ids", type=Path, required=True)
    ap.add_argument("--expected-shadow-ids-sha256", required=True)
    ap.add_argument("--discovery-ids", type=Path, required=True)
    ap.add_argument("--expected-discovery-ids-sha256", required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--token-offset", type=int, default=0)
    ap.add_argument("--token-limit", type=int, required=True)
    ap.add_argument("--path-map", action="append", default=[])
    ap.add_argument("--videos-root", type=Path)
    ap.add_argument("--max-failures", type=int, default=5)
    ap.add_argument("--confirm-label-blind-shadow", action="store_true")
    ap.add_argument("--wrapper-dry-run", action="store_true")
    ap.add_argument("--extractor-dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return selftest() if args.command == "selftest" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
