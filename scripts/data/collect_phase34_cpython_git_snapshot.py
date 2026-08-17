#!/usr/bin/env python3
"""Collect the registered CPython snapshot through one shallow Git fetch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_cpython_git_snapshot import (  # noqa: E402
    REPOSITORY_URL,
    build_manifest,
    build_rows,
    validate_preregistration,
)
from sara_engine.evaluation.phase34_cpython_snapshot import COMMIT, canonical_digest, utc_now  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_output_directory,
    ensure_parent_directory,
    processed_data_path,
    raw_data_path,
    workspace_path,
)


DEFAULT_PREREGISTRATION = workspace_path("evaluation", "phase34_cpython_v3_14_6_git_snapshot_preregistration.json")
DEFAULT_RAW = raw_data_path("phase34_cpython_git_snapshot", "source_rows.jsonl")
DEFAULT_MANIFEST = processed_data_path("autobot", "phase34_cpython_v3_14_6_git_snapshot_manifest.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "phase34_cpython_v3_14_6_git_snapshot_collection.json")


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
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl_exclusive_or_identical(path: str, rows: Sequence[Mapping[str, Any]]) -> str:
    resolved = ensure_parent_directory(path)
    candidate = [dict(row) for row in rows]
    if os.path.exists(resolved):
        if _read_jsonl(resolved) != candidate:
            raise ValueError(f"immutable snapshot output mismatch: {resolved}")
        return "already_present_identical"
    with open(resolved, "x", encoding="utf-8") as handle:
        for row in candidate:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return "written_new"


def _run_git(arguments: Sequence[str], *, environment: Mapping[str, str]) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=dict(environment),
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(f"Git snapshot command failed: {message}")
    return result.stdout


def collect(registration: Mapping[str, Any], *, existing_rows: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    validation = validate_preregistration(registration)
    if not validation["valid"]:
        raise ValueError("invalid Git snapshot preregistration: " + "; ".join(validation["errors"]))
    existing_by_path = {str(row.get("repository_path", "")): row for row in existing_rows}
    collection_time = (
        str(next(iter(existing_by_path.values())).get("collection_time", ""))
        if existing_by_path
        else utc_now()
    )
    scratch = ensure_output_directory(workspace_path("phase34_cpython_git_snapshot"))
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    with tempfile.TemporaryDirectory(dir=scratch, prefix="fetch-") as repository_path:
        _run_git(["init", "--bare", repository_path], environment=environment)
        _run_git(["-C", repository_path, "remote", "add", "origin", REPOSITORY_URL], environment=environment)
        _run_git(
            ["-C", repository_path, "fetch", "--depth=1", "--no-tags", "origin", COMMIT],
            environment=environment,
        )
        observed = _run_git(["-C", repository_path, "rev-parse", "FETCH_HEAD"], environment=environment).decode("ascii").strip()
        if observed != COMMIT:
            raise ValueError("fetched Git head does not match the preregistered commit")

        def blob_loader(path: str) -> bytes:
            return _run_git(["-C", repository_path, "show", f"{COMMIT}:{path}"], environment=environment)

        rows = build_rows(registration, blob_loader, collection_time=collection_time)
    if existing_rows and [dict(row) for row in existing_rows] != rows:
        raise ValueError("commit snapshot changed or stored rows were altered")
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--raw-path", default=DEFAULT_RAW)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        registration = _read_json(args.preregistration_path)
        rows = collect(registration, existing_rows=_read_jsonl(args.raw_path))
        manifest = build_manifest(rows, registration)
        raw_status = _write_jsonl_exclusive_or_identical(args.raw_path, rows)
        manifest_status = _write_jsonl_exclusive_or_identical(args.manifest_path, manifest)
        report = {
            "schema": "sara-phase34-cpython-git-source-snapshot-collection-v1",
            "observed_only": True,
            "collection_complete": True,
            "promotion_ready": False,
            "semantic_delayed_recall_allowed": False,
            "commit": COMMIT,
            "protocol_fingerprint": registration["protocol_fingerprint"],
            "source_count": len(rows),
            "unique_source_hash_count": len({row["source_hash"] for row in rows}),
            "all_commit_pinned": all(row["source_revision"] == COMMIT for row in rows),
            "all_untruncated": all(row["content_truncated"] is False for row in rows),
            "raw_http_collection_preserved_as_failed": True,
            "acquisition_transport": "git_smart_http_shallow_fetch",
            "raw_snapshot_fingerprint": canonical_digest(rows),
            "manifest_fingerprint": canonical_digest(manifest),
            "raw_write_status": raw_status,
            "manifest_write_status": manifest_status,
            "next_actions": ["Complete human source-alignment review for the six historical transcribed excerpts."],
        }
        with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
