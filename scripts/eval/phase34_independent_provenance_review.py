#!/usr/bin/env python3
"""Audit offline and online provenance for sampled Phase 34 identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.data.collect_continual_horizon_external import fetch_source  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    raw_data_path,
    workspace_path,
)


DEFAULT_RAW = raw_data_path("architecture_migration", "source_rows.jsonl")
DEFAULT_MANIFEST = processed_data_path("autobot", "architecture_migration_latent_manifest.jsonl")
DEFAULT_CASE_PLAN = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_case_plan.json"
)
DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_adapter_v2_preregistration.json"
)
DEFAULT_BENCHMARK = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_adapter_v2_benchmark.json"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_provenance_review.json"
)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def build_offline_review(
    raw_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    case_plan: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> Dict[str, Any]:
    planned_hashes = {
        str(value)
        for case in case_plan.get("cases", [])
        for value in case.get("stream_material_hashes", [])
    }
    planned_refs = {
        str(value)
        for case in case_plan.get("cases", [])
        for value in case.get("stream_source_refs", [])
    }
    raw_by_hash = {
        str(row.get("source_hash") or row.get("material_hash") or ""): row
        for row in raw_rows
    }
    manifest_by_hash = {
        str(row.get("material_hash", "")): row for row in manifest_rows
    }
    selected_raw = [raw_by_hash[value] for value in sorted(planned_hashes) if value in raw_by_hash]
    selected_manifest = [
        manifest_by_hash[value] for value in sorted(planned_hashes) if value in manifest_by_hash
    ]
    content_hash_matches = all(
        hashlib.sha256(str(row.get("content", "")).encode("utf-8")).hexdigest()
        == str(row.get("source_hash", ""))
        for row in selected_raw
    )
    checks = {
        "source_manifest_fingerprint_matches": _digest(list(manifest_rows))
        == preregistration.get("source_manifest_fingerprint"),
        "case_plan_fingerprint_matches": _digest(dict(case_plan))
        == preregistration.get("case_plan_fingerprint"),
        "benchmark_protocol_matches": benchmark.get("protocol_fingerprint")
        == preregistration.get("protocol_fingerprint"),
        "benchmark_execution_passed": benchmark.get("execution_passed") is True,
        "benchmark_identity_gate_passed": benchmark.get("identity_gate_passed") is True,
        "sampled_material_count_matches": len(planned_hashes) == 66,
        "sampled_source_ref_count_matches": len(planned_refs) == 66,
        "all_sampled_raw_rows_present": len(selected_raw) == len(planned_hashes),
        "all_sampled_manifest_rows_present": len(selected_manifest) == len(planned_hashes),
        "stored_content_hashes_recompute": content_hash_matches,
        "raw_manifest_hashes_match": all(
            str(raw_by_hash[value].get("source_hash", ""))
            == str(manifest_by_hash[value].get("material_hash", ""))
            for value in planned_hashes
            if value in raw_by_hash and value in manifest_by_hash
        ),
        "planned_refs_match_stored_sources": all(
            str(manifest_by_hash[value].get("source_ref", "")) in planned_refs
            and str(raw_by_hash[value].get("source_ref") or raw_by_hash[value].get("source_url"))
            == str(manifest_by_hash[value].get("source_ref", ""))
            for value in planned_hashes
            if value in raw_by_hash and value in manifest_by_hash
        ),
        "revision_and_collection_metadata_present": all(
            str(row.get("source_revision", ""))
            and str(row.get("collection_time", ""))
            and str(row.get("license_hint", ""))
            for row in selected_raw
        ),
        "observed_allow_only": all(
            row.get("observed_only") is True
            and str(row.get("compliance_level", "")) == "allow"
            for row in selected_raw
        ),
    }
    fetched = [
        dict(row)
        for row in selected_raw
        if str(row.get("content_origin", "")) == "fetched_authoritative_document"
    ]
    transcribed = [
        dict(row)
        for row in selected_raw
        if str(row.get("content_origin", "")) == "transcribed_source_excerpt"
    ]
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "planned_hashes": sorted(planned_hashes),
        "planned_refs": sorted(planned_refs),
        "fetched_rows": fetched,
        "transcribed_rows": transcribed,
        "metrics": {
            "sampled_material_count": len(planned_hashes),
            "fetched_authoritative_count": len(fetched),
            "transcribed_excerpt_count": len(transcribed),
        },
    }


def audit_online(
    fetched_rows: Sequence[Mapping[str, Any]], *, timeout_seconds: float
) -> List[Dict[str, Any]]:
    audit_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    observations: List[Dict[str, Any]] = []
    for row in fetched_rows:
        try:
            current = fetch_source(
                {
                    "source_url": str(row["source_url"]),
                    "source_revision_hint": str(row.get("source_revision", "")),
                    "license_hint": str(row.get("license_hint", "")),
                    "catalog_stage": "provenance_recheck",
                },
                audit_time,
                timeout_seconds,
            )
            observations.append(
                {
                    "source_url": str(row["source_url"]),
                    "source_ref": str(row.get("source_ref") or row["source_url"]),
                    "expected_source_hash": str(row["source_hash"]),
                    "observed_source_hash": str(current["source_hash"]),
                    "content_hash_matches": current["source_hash"] == row["source_hash"],
                    "expected_response_body_hash": str(row.get("response_body_hash", "")),
                    "observed_response_body_hash": str(current.get("response_body_hash", "")),
                    "response_body_hash_matches": current.get("response_body_hash")
                    == row.get("response_body_hash"),
                    "expected_revision": str(row.get("source_revision", "")),
                    "observed_revision": str(current.get("source_revision", "")),
                    "revision_matches": str(current.get("source_revision", ""))
                    == str(row.get("source_revision", "")),
                    "retrieval_succeeded": True,
                    "error": "",
                }
            )
        except (OSError, ValueError) as exc:
            observations.append(
                {
                    "source_url": str(row.get("source_url", "")),
                    "source_ref": str(row.get("source_ref") or row.get("source_url", "")),
                    "expected_source_hash": str(row.get("source_hash", "")),
                    "observed_source_hash": "",
                    "content_hash_matches": False,
                    "response_body_hash_matches": False,
                    "revision_matches": False,
                    "retrieval_succeeded": False,
                    "error": str(exc),
                }
            )
    return observations


def build_report(
    offline: Mapping[str, Any], online_observations: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    online_checks = {
        "all_fetched_sources_retrieved": len(online_observations)
        == int(offline["metrics"]["fetched_authoritative_count"])
        and all(row.get("retrieval_succeeded") is True for row in online_observations),
        "all_normalized_content_hashes_reproduced": bool(online_observations)
        and all(row.get("content_hash_matches") is True for row in online_observations),
    }
    automated_passed = bool(offline.get("passed")) and all(online_checks.values())
    transcribed = list(offline.get("transcribed_rows", []))
    complete = automated_passed and not transcribed
    domain_metrics: Dict[str, Dict[str, int]] = {}
    for row in online_observations:
        domain = str(urlparse(str(row.get("source_url", ""))).hostname or "")
        metrics = domain_metrics.setdefault(
            domain,
            {
                "retrieval_count": 0,
                "content_hash_match_count": 0,
                "content_hash_drift_count": 0,
                "response_body_hash_match_count": 0,
                "revision_match_count": 0,
                "revision_drift_count": 0,
            },
        )
        metrics["retrieval_count"] += int(row.get("retrieval_succeeded") is True)
        metrics["content_hash_match_count"] += int(row.get("content_hash_matches") is True)
        metrics["content_hash_drift_count"] += int(row.get("content_hash_matches") is not True)
        metrics["response_body_hash_match_count"] += int(
            row.get("response_body_hash_matches") is True
        )
        metrics["revision_match_count"] += int(row.get("revision_matches") is True)
        metrics["revision_drift_count"] += int(row.get("revision_matches") is not True)
    drifted_domains = sorted(
        domain for domain, metrics in domain_metrics.items() if metrics["content_hash_drift_count"]
    )
    stable_domains = sorted(
        domain
        for domain, metrics in domain_metrics.items()
        if metrics["retrieval_count"] and not metrics["content_hash_drift_count"]
    )
    return {
        "schema": "sara-phase34-independent-provenance-review-v1",
        "observed_only": True,
        "offline_integrity_passed": bool(offline.get("passed")),
        "automated_provenance_passed": automated_passed,
        "provenance_review_complete": complete,
        "promotion_ready": False,
        "offline_checks": dict(offline["checks"]),
        "online_checks": online_checks,
        "metrics": {
            **dict(offline["metrics"]),
            "online_retrieval_count": len(online_observations),
            "online_content_hash_match_count": sum(
                row.get("content_hash_matches") is True for row in online_observations
            ),
            "online_response_body_hash_match_count": sum(
                row.get("response_body_hash_matches") is True for row in online_observations
            ),
            "manual_review_required_count": len(transcribed),
            "online_content_hash_drift_count": sum(
                row.get("content_hash_matches") is not True for row in online_observations
            ),
            "online_revision_drift_count": sum(
                row.get("revision_matches") is not True for row in online_observations
            ),
        },
        "domain_metrics": dict(sorted(domain_metrics.items())),
        "stable_domains": stable_domains,
        "drifted_domains": drifted_domains,
        "online_observations": list(online_observations),
        "manual_review_targets": [
            {
                "record_id": str(row.get("record_id", "")),
                "source_url": str(row.get("source_url", "")),
                "source_ref": str(row.get("source_ref") or row.get("source_url", "")),
                "source_hash": str(row.get("source_hash", "")),
                "source_revision": str(row.get("source_revision", "")),
                "reason": "transcribed_source_excerpt_requires_human_source_alignment_review",
            }
            for row in transcribed
        ],
        "next_actions": (
            ([
                "Replace mutable documentation URLs only through a new preregistered source snapshot pinned to an immutable release artifact.",
                "Preserve the current drift result and do not rewrite the executed v2 source fingerprint.",
            ] if drifted_domains else [])
            + [
                "Review every transcribed excerpt against the cited authoritative section.",
                "Do not replace or silently reclassify a historical excerpt after observing benchmark results.",
                "Preregister a new source snapshot if reviewed excerpts are replaced by fetched section records.",
            ]
            if not complete
            else []
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default=DEFAULT_RAW)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--case-plan-path", default=DEFAULT_CASE_PLAN)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--benchmark-path", default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        offline = build_offline_review(
            _read_jsonl(args.raw_path),
            _read_jsonl(args.manifest_path),
            _read_json(args.case_plan_path),
            _read_json(args.preregistration_path),
            _read_json(args.benchmark_path),
        )
        observations = (
            []
            if args.offline_only
            else audit_online(offline["fetched_rows"], timeout_seconds=args.timeout_seconds)
        )
        report = build_report(offline, observations)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "offline_integrity_passed": report["offline_integrity_passed"],
                "automated_provenance_passed": report["automated_provenance_passed"],
                "provenance_review_complete": report["provenance_review_complete"],
                "manual_review_required_count": report["metrics"]["manual_review_required_count"],
                "output_path": os.path.realpath(args.output_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["offline_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
