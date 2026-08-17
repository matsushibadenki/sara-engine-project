#!/usr/bin/env python3
"""Collect the preregistered commit-pinned CPython source snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_cpython_snapshot import (  # noqa: E402
    COMMIT,
    FILE_ALLOWLIST,
    build_manifest,
    canonical_digest,
    fetch_source_entry,
    utc_now,
    validate_preregistration,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    raw_data_path,
    workspace_path,
)


DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation", "phase34_cpython_v3_14_6_snapshot_preregistration.json"
)
DEFAULT_RAW = raw_data_path("phase34_cpython_snapshot", "source_rows.jsonl")
DEFAULT_MANIFEST = processed_data_path(
    "autobot", "phase34_cpython_v3_14_6_snapshot_manifest.jsonl"
)
DEFAULT_REPORT = workspace_path(
    "evaluation", "phase34_cpython_v3_14_6_snapshot_collection.json"
)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _write_jsonl_exclusive_or_identical(
    path: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    resolved = ensure_parent_directory(path)
    candidate = [dict(row) for row in rows]
    if os.path.exists(resolved):
        if _read_jsonl(resolved) != candidate:
            raise ValueError(f"immutable snapshot output mismatch: {resolved}")
        return "already_present_identical"
    parent = os.path.dirname(resolved)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=parent, delete=False, prefix=".snapshot-", suffix=".jsonl"
    ) as handle:
        temporary = handle.name
        for row in candidate:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        os.link(temporary, resolved)
    except FileExistsError:
        if _read_jsonl(resolved) != candidate:
            raise ValueError(f"concurrent immutable snapshot output mismatch: {resolved}")
    finally:
        os.unlink(temporary)
    return "written_new"


def _write_report(path: str, report: Mapping[str, Any]) -> None:
    resolved = ensure_parent_directory(path)
    parent = os.path.dirname(resolved)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=parent, delete=False, prefix=".snapshot-report-", suffix=".json"
    ) as handle:
        temporary = handle.name
        json.dump(dict(report), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, resolved)


def collect(
    registration: Mapping[str, Any],
    *,
    timeout_seconds: float,
    existing_rows: Sequence[Mapping[str, Any]] = (),
    fetcher: Callable[..., Dict[str, Any]] = fetch_source_entry,
) -> List[Dict[str, Any]]:
    validation = validate_preregistration(registration)
    if not validation["valid"]:
        raise ValueError(
            "invalid CPython snapshot preregistration: "
            + "; ".join(validation["errors"])
        )
    existing_by_path = {
        str(row.get("repository_path", "")): row for row in existing_rows
    }
    if existing_by_path and set(existing_by_path) != set(FILE_ALLOWLIST):
        raise ValueError("existing snapshot rows do not match the frozen file allowlist")
    run_time = utc_now()
    rows: List[Dict[str, Any]] = []
    for entry in registration["sources"]:
        prior = existing_by_path.get(str(entry["repository_path"]))
        row = fetcher(
            entry,
            timeout_seconds=timeout_seconds,
            collection_time=(str(prior.get("collection_time")) if prior else run_time),
        )
        if prior is not None and dict(prior) != row:
            raise ValueError(
                f"commit-addressed source changed or stored row was altered: {entry['repository_path']}"
            )
        rows.append(row)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--raw-path", default=DEFAULT_RAW)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    try:
        registration = _read_json(args.preregistration_path)
        existing = _read_jsonl(args.raw_path)
        rows = collect(
            registration,
            timeout_seconds=args.timeout_seconds,
            existing_rows=existing,
        )
        manifest = build_manifest(rows, registration)
        raw_status = _write_jsonl_exclusive_or_identical(args.raw_path, rows)
        manifest_status = _write_jsonl_exclusive_or_identical(args.manifest_path, manifest)
        report = {
            "schema": "sara-phase34-cpython-source-snapshot-collection-v1",
            "observed_only": True,
            "collection_complete": True,
            "promotion_ready": False,
            "semantic_delayed_recall_allowed": False,
            "commit": COMMIT,
            "protocol_fingerprint": registration["protocol_fingerprint"],
            "source_count": len(rows),
            "unique_source_hash_count": len({row["source_hash"] for row in rows}),
            "allowlist_complete": [row["repository_path"] for row in rows]
            == list(FILE_ALLOWLIST),
            "all_commit_pinned": all(row["source_revision"] == COMMIT for row in rows),
            "all_untruncated": all(row["content_truncated"] is False for row in rows),
            "raw_snapshot_fingerprint": canonical_digest(rows),
            "manifest_fingerprint": canonical_digest(manifest),
            "raw_write_status": raw_status,
            "manifest_write_status": manifest_status,
            "raw_path": os.path.realpath(args.raw_path),
            "manifest_path": os.path.realpath(args.manifest_path),
            "next_actions": [
                "Complete human source-alignment review for the six historical transcribed excerpts.",
                "Do not preregister semantic delayed recall until that review is explicitly approved.",
            ],
        }
        _write_report(args.report_path, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
