#!/usr/bin/env python3
"""Collect the preregistered Phase 34 review sources and build a comparison packet."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.request import Request, urlopen


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_cpython_snapshot import COMMIT, canonical_digest  # noqa: E402
from sara_engine.evaluation.phase34_review_support import (  # noqa: E402
    CPYTHON_REPOSITORY_URL,
    build_comparison_packet,
    build_source_rows,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_output_directory,
    ensure_parent_directory,
    raw_data_path,
    workspace_path,
)


DEFAULT_REQUEST = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_request.json"
)
DEFAULT_REGISTRATION = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_review_support_preregistration.json"
)
DEFAULT_RAW = raw_data_path("phase34_review_support", "source_rows.jsonl")
DEFAULT_PACKET = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_review_comparison_packet.json"
)
DEFAULT_REPORT = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_review_support_collection.json"
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
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: str, value: Mapping[str, Any]) -> None:
    with open(ensure_parent_directory(path), "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_exclusive_or_identical(
    path: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    resolved = ensure_parent_directory(path)
    candidate = [dict(row) for row in rows]
    if os.path.exists(resolved):
        if _read_jsonl(resolved) != candidate:
            raise ValueError("existing review-support snapshot differs")
        return "already_present_identical"
    with open(resolved, "x", encoding="utf-8") as handle:
        for row in candidate:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return "written_new"


def _run_git(arguments: Sequence[str], environment: Mapping[str, str]) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=dict(environment),
    )
    if result.returncode:
        raise OSError(
            "Git review-support command failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def collect(
    request: Mapping[str, Any],
    registration: Mapping[str, Any],
    *,
    collection_time: str,
) -> List[Dict[str, Any]]:
    scratch = ensure_output_directory(workspace_path("phase34_review_support"))
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    with tempfile.TemporaryDirectory(dir=scratch, prefix="fetch-") as repository_path:
        _run_git(["init", "--bare", repository_path], environment)
        _run_git(
            ["-C", repository_path, "remote", "add", "origin", CPYTHON_REPOSITORY_URL],
            environment,
        )
        _run_git(
            ["-C", repository_path, "fetch", "--depth=1", "--no-tags", "origin", COMMIT],
            environment,
        )
        observed = _run_git(
            ["-C", repository_path, "rev-parse", "FETCH_HEAD"], environment
        ).decode("ascii").strip()
        if observed != COMMIT:
            raise ValueError("review-support Git head does not match the frozen commit")

        def git_blob_loader(path: str) -> bytes:
            return _run_git(
                ["-C", repository_path, "show", f"{COMMIT}:{path}"], environment
            )

        def http_loader(url: str) -> bytes:
            request_value = Request(url, headers={"User-Agent": "SARA-Research/1.1"})
            with urlopen(request_value, timeout=30.0) as response:
                if response.geturl() != url:
                    raise ValueError("review-support RFC source redirected")
                return response.read()

        return build_source_rows(
            registration,
            request,
            git_blob_loader=git_blob_loader,
            http_loader=http_loader,
            collection_time=collection_time,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-path", default=DEFAULT_REQUEST)
    parser.add_argument("--preregistration-path", default=DEFAULT_REGISTRATION)
    parser.add_argument("--raw-path", default=DEFAULT_RAW)
    parser.add_argument("--packet-path", default=DEFAULT_PACKET)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        request = _read_json(args.request_path)
        registration = _read_json(args.preregistration_path)
        existing = _read_jsonl(args.raw_path)
        collection_time = (
            str(existing[0]["collection_time"])
            if existing
            else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        rows = collect(request, registration, collection_time=collection_time)
        raw_status = _write_jsonl_exclusive_or_identical(args.raw_path, rows)
        packet = build_comparison_packet(request, registration, rows)
        _write_json(args.packet_path, packet)
        report = {
            "schema": "sara-phase34-transcribed-excerpt-review-support-collection-v1",
            "observed_only": True,
            "protocol_fingerprint": registration["protocol_fingerprint"],
            "source_count": len(rows),
            "target_count": packet["target_count"],
            "raw_snapshot_fingerprint": canonical_digest(rows),
            "packet_fingerprint": packet["packet_fingerprint"],
            "raw_write_status": raw_status,
            "automated_alignment_decision_made": False,
            "review_complete": False,
            "review_gate_passed": False,
            "promotion_ready": False,
            "next_action": "perform_six_human_source_alignment_reviews",
        }
        _write_json(args.report_path, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
