#!/usr/bin/env python3
"""Audit train/evaluation isolation for managed Phase 7 learning materials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_TRAIN_PATH = workspace_path("autobot", "phase7_train_materials.jsonl")
DEFAULT_EVALUATION_PATH = workspace_path("autobot", "phase7_evaluation_materials.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase7_isolation_audit.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase7_isolation_audit_summary.txt")


def _read_jsonl(path: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except FileNotFoundError:
        pass
    return rows


def _text(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get(key, "") or "") for key in ("source_text", "content", "answer", "prompt"))


def _source_hash(row: Mapping[str, Any]) -> str:
    value = str(row.get("source_hash", "") or "").strip()
    if value:
        return value
    text = _text(row).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def _signature(row: Mapping[str, Any]) -> str:
    value = str(row.get("near_duplicate_signature", "") or "").strip().lower()
    if value:
        return value
    tokens = re.findall(r"[\w-]+", _text(row).lower())[:256]
    if not tokens:
        return ""
    weights = [0] * 64
    for token in tokens:
        token_hash = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if token_hash & (1 << bit) else -1
    value = sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)
    return f"{value:016x}"


def _values(rows: Iterable[Mapping[str, Any]], key: str) -> set[str]:
    return {str(row.get(key, "") or "").strip() for row in rows if str(row.get(key, "") or "").strip()}


def _hamming(left: str, right: str) -> Optional[int]:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return None


def build_report(
    train_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    max_signature_hamming_distance: int = 3,
) -> Dict[str, Any]:
    train = [dict(row) for row in train_rows if isinstance(row, Mapping)]
    evaluation = [dict(row) for row in evaluation_rows if isinstance(row, Mapping)]
    for row in train + evaluation:
        row["source_hash"] = _source_hash(row)
        row["near_duplicate_signature"] = _signature(row)
    train_hashes = _values(train, "source_hash")
    evaluation_hashes = _values(evaluation, "source_hash")
    train_revisions = _values(train, "source_revision")
    evaluation_revisions = _values(evaluation, "source_revision")
    train_domains = _values(train, "source_domain")
    evaluation_domains = _values(evaluation, "source_domain")
    train_times = _values(train, "collection_time")
    evaluation_times = _values(evaluation, "collection_time")
    train_signatures = _values(train, "near_duplicate_signature")
    evaluation_signatures = _values(evaluation, "near_duplicate_signature")
    near_duplicate_pairs = []
    for left in sorted(train_signatures):
        for right in sorted(evaluation_signatures):
            distance = _hamming(left, right)
            if distance is not None and distance <= int(max_signature_hamming_distance):
                near_duplicate_pairs.append({"train_signature": left, "evaluation_signature": right, "hamming_distance": distance})
    required_fields = ("source_hash", "source_revision", "source_domain", "collection_time", "near_duplicate_signature")
    metadata_complete = all(all(str(row.get(field, "") or "").strip() for field in required_fields) for row in train + evaluation)
    time_split_valid = bool(train_times and evaluation_times and max(train_times) < min(evaluation_times))
    checks = {
        "train_rows_present": bool(train),
        "evaluation_rows_present": bool(evaluation),
        "metadata_complete": metadata_complete,
        "source_hash_isolated": not bool(train_hashes & evaluation_hashes),
        "source_revision_isolated": not bool(train_revisions & evaluation_revisions),
        "source_domain_isolated": not bool(train_domains & evaluation_domains),
        "time_split_isolated": time_split_valid,
        "near_duplicate_signature_isolated": not bool(near_duplicate_pairs),
    }
    return {
        "schema": "sara-phase7-isolation-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "train_row_count": len(train),
            "evaluation_row_count": len(evaluation),
            "shared_source_hashes": sorted(train_hashes & evaluation_hashes),
            "shared_source_revisions": sorted(train_revisions & evaluation_revisions),
            "shared_source_domains": sorted(train_domains & evaluation_domains),
            "near_duplicate_pairs": near_duplicate_pairs,
            "max_signature_hamming_distance": int(max_signature_hamming_distance),
        },
        "policy_notes": [
            "Phase 7 isolation is a release guard for autonomous material generation, not a quality benchmark.",
            "Rows with missing provenance remain non-promotable even when no overlap is observed.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--evaluation-path", default=DEFAULT_EVALUATION_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--max-signature-hamming-distance", type=int, default=3)
    args = parser.parse_args(argv)
    report = build_report(_read_jsonl(args.train_path), _read_jsonl(args.evaluation_path), max_signature_hamming_distance=args.max_signature_hamming_distance)
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"Phase 7 isolation audit: {'PASS' if report['passed'] else 'FAIL'}\n")
        for name, passed in sorted(report["checks"].items()):
            handle.write(f"{name}: {passed}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
