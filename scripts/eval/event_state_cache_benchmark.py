#!/usr/bin/env python3
"""Run the observed-only verified hierarchical event-state cache benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.memory.event_state_cache import (  # noqa: E402
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    interim_data_path,
    processed_data_path,
    workspace_path,
)


DEFAULT_FIXTURE_PATH = processed_data_path(
    "benchmark_fixtures", "event_state_cache_cases.jsonl"
)
DEFAULT_CANDIDATE_PATH = interim_data_path(
    "event_state_cache", "candidates.jsonl"
)
DEFAULT_MANIFEST_PATH = processed_data_path(
    "event_state_cache", "manifest.jsonl"
)
DEFAULT_REPORT_PATH = workspace_path(
    "evaluation", "event_state_cache_benchmark.json"
)
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation", "event_state_cache_benchmark_summary.txt"
)
DEFAULT_TRACE_PATH = workspace_path(
    "evaluation", "event_state_cache_traces.jsonl"
)
DEFAULT_STATE_PATH = workspace_path(
    "evaluation", "event_state_cache_state.json"
)


def default_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "schema": "sara-event-state-cache-case-v1",
            "kind": "candidate",
            "entry_id": "early_verified_fact",
            "signature": [11, 13, 17],
            "source_ref": "fixture:early-fact",
            "time_segment": 0,
            "own_latent_id": "latent:prime-fact",
            "causal_predecessors": ["event:introduction"],
            "confidence": 0.98,
            "uncertainty": 0.02,
            "source_reliability": 0.98,
            "resonance_score": 0.98,
            "metabolic_headroom": 0.9,
            "observed": True,
            "source_backed": True,
            "verified": True,
        },
        {
            "schema": "sara-event-state-cache-case-v1",
            "kind": "candidate",
            "entry_id": "early_verified_procedure",
            "signature": [23, 29, 31],
            "source_ref": "fixture:early-procedure",
            "time_segment": 1,
            "own_latent_id": "latent:procedure",
            "causal_predecessors": ["event:instruction"],
            "confidence": 0.96,
            "uncertainty": 0.04,
            "source_reliability": 0.96,
            "resonance_score": 0.94,
            "metabolic_headroom": 0.9,
            "observed": True,
            "source_backed": True,
            "verified": True,
        },
        {
            "schema": "sara-event-state-cache-case-v1",
            "kind": "candidate",
            "entry_id": "blocked_contradiction",
            "signature": [41, 43],
            "source_ref": "fixture:contradiction",
            "time_segment": 2,
            "confidence": 0.9,
            "source_reliability": 0.9,
            "resonance_score": 0.92,
            "metabolic_headroom": 0.9,
            "observed": True,
            "source_backed": True,
            "verified": True,
            "contradicted": True,
            "expected_decision": "block_contradiction",
        },
        {
            "schema": "sara-event-state-cache-case-v1",
            "kind": "candidate",
            "entry_id": "blocked_predicted_only",
            "signature": [47, 53],
            "source_ref": "fixture:prediction",
            "time_segment": 3,
            "confidence": 0.9,
            "source_reliability": 0.9,
            "resonance_score": 0.92,
            "metabolic_headroom": 0.9,
            "observed": False,
            "source_backed": True,
            "verified": True,
            "expected_decision": "block_predicted_only",
        },
        {
            "schema": "sara-event-state-cache-case-v1",
            "kind": "candidate",
            "entry_id": "blocked_unverified_source",
            "signature": [59, 61],
            "source_ref": "",
            "time_segment": 4,
            "confidence": 0.9,
            "source_reliability": 0.4,
            "resonance_score": 0.9,
            "metabolic_headroom": 0.9,
            "observed": True,
            "source_backed": False,
            "verified": True,
            "expected_decision": "block_unverified_source",
        },
    ]
    for index in range(18):
        rows.append(
            {
                "schema": "sara-event-state-cache-case-v1",
                "kind": "candidate",
                "entry_id": f"distractor_{index:02d}",
                "signature": [100 + index, 200 + index],
                "source_ref": f"fixture:distractor:{index}",
                "time_segment": 10 + index,
                "own_latent_id": f"latent:distractor:{index}",
                "confidence": 0.66,
                "uncertainty": 0.3,
                "source_reliability": 0.7,
                "resonance_score": 0.68,
                "metabolic_headroom": 0.8,
                "observed": True,
                "source_backed": True,
                "verified": True,
            }
        )
    rows.extend(
        [
            {
                "schema": "sara-event-state-cache-case-v1",
                "kind": "query",
                "query_id": "delayed_fact_recall",
                "signature": [11, 13, 17],
                "own_latent_id": "latent:prime-fact",
                "expected_entry_id": "early_verified_fact",
            },
            {
                "schema": "sara-event-state-cache-case-v1",
                "kind": "query",
                "query_id": "delayed_procedure_recall",
                "signature": [23, 29, 31],
                "causal_context": ["event:instruction"],
                "expected_entry_id": "early_verified_procedure",
            },
            {
                "schema": "sara-event-state-cache-case-v1",
                "kind": "query",
                "query_id": "unknown_sparse_state",
                "signature": [701, 709, 719],
                "expect_abstention": True,
            },
        ]
    )
    return rows


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return resolved


def ensure_fixture(path: str) -> str:
    rows = read_jsonl(path)
    if rows and all(
        row.get("schema") == "sara-event-state-cache-case-v1" for row in rows
    ):
        return path
    return write_jsonl(path, default_cases())


def _candidate(row: Dict[str, Any]) -> EventStateCandidate:
    return EventStateCandidate.from_verified_evidence(
        verifier_id="event-state-cache-benchmark",
        evidence=row,
        entry_id=str(row.get("entry_id", "")),
        signature=tuple(int(value) for value in row.get("signature", [])),
        source_ref=str(row.get("source_ref", "")),
        time_segment=int(row.get("time_segment", 0)),
        own_latent_id=str(row.get("own_latent_id", "")),
        causal_predecessors=tuple(
            str(value) for value in row.get("causal_predecessors", [])
        ),
        confidence=float(row.get("confidence", 1.0)),
        uncertainty=float(row.get("uncertainty", 0.0)),
        source_reliability=float(row.get("source_reliability", 1.0)),
        resonance_score=float(row.get("resonance_score", 0.0)),
        sequence_support_score=float(row.get("sequence_support_score", 0.0)),
        sequence_support_count=int(row.get("sequence_support_count", 0)),
        metabolic_headroom=float(row.get("metabolic_headroom", 1.0)),
        observed=bool(row.get("observed", True)),
        source_backed=bool(row.get("source_backed", True)),
        verified=bool(row.get("verified", True)),
        contradicted=bool(row.get("contradicted", False)),
        abstained=bool(row.get("abstained", False)),
        event_cost=int(row.get("event_cost", 0)),
        expires_at=(
            None if row.get("expires_at") is None else int(row["expires_at"])
        ),
    )


def _run_profile(
    profile: str,
    candidates: Sequence[Dict[str, Any]],
    queries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if profile == "none":
        return {
            "profile": profile,
            "delayed_recall_success": 0,
            "delayed_recall_total": sum(
                1 for query in queries if query.get("expected_entry_id")
            ),
            "negative_abstention_success": sum(
                1 for query in queries if query.get("expect_abstention")
            ),
            "negative_abstention_total": sum(
                1 for query in queries if query.get("expect_abstention")
            ),
            "blocked_decision_integrity": 1.0,
            "entry_count": 0,
            "max_entry_count": 0,
            "eviction_count": 0,
            "merge_count": 0,
            "max_retrieval_event_cost": 0,
            "traces": [],
            "state": None,
        }

    cache = VerifiedHierarchicalEventStateCache(
        retention_profile=profile,
        max_entries=12,
        top_k=2,
    )
    traces: List[Dict[str, Any]] = []
    blocked_total = 0
    blocked_correct = 0
    max_entry_count = 0
    for row in candidates:
        admission = cache.admit(_candidate(row))
        expected = str(row.get("expected_decision", ""))
        if expected:
            blocked_total += 1
            blocked_correct += int(admission.decision == expected)
        traces.append(
            {
                "profile": profile,
                "kind": "admission",
                "case_id": row.get("entry_id"),
                **admission.to_dict(),
            }
        )
        max_entry_count = max(max_entry_count, len(cache.entries))

    recall_success = 0
    recall_total = 0
    abstention_success = 0
    abstention_total = 0
    max_retrieval_cost = 0
    for row in queries:
        result = cache.retrieve(
            row.get("signature", []),
            own_latent_id=str(row.get("own_latent_id", "")),
            causal_context=row.get("causal_context", []),
            source_ref=str(row.get("source_ref", "")),
        )
        expected_entry_id = str(row.get("expected_entry_id", ""))
        if expected_entry_id:
            recall_total += 1
            recall_success += int(
                bool(result.matches)
                and result.matches[0].get("entry_id") == expected_entry_id
            )
        if row.get("expect_abstention"):
            abstention_total += 1
            abstention_success += int(result.abstained)
        max_retrieval_cost = max(max_retrieval_cost, result.event_cost)
        traces.append(
            {
                "profile": profile,
                "kind": "retrieval",
                "case_id": row.get("query_id"),
                **result.to_dict(),
            }
        )

    state = cache.state_dict()
    return {
        "profile": profile,
        "delayed_recall_success": recall_success,
        "delayed_recall_total": recall_total,
        "negative_abstention_success": abstention_success,
        "negative_abstention_total": abstention_total,
        "blocked_decision_integrity": float(blocked_correct)
        / float(max(1, blocked_total)),
        "entry_count": state["entry_count"],
        "max_entry_count": max_entry_count,
        "eviction_count": state["eviction_count"],
        "merge_count": state["merge_count"],
        "max_retrieval_event_cost": max_retrieval_cost,
        "traces": traces,
        "state": state,
    }


def build_report(
    rows: Sequence[Dict[str, Any]],
    *,
    candidate_path: str,
    manifest_path: str,
    trace_path: str,
    state_path: str,
) -> Dict[str, Any]:
    candidates = [row for row in rows if row.get("kind") == "candidate"]
    queries = [row for row in rows if row.get("kind") == "query"]
    write_jsonl(candidate_path, candidates)

    profile_results = {
        profile: _run_profile(profile, candidates, queries)
        for profile in ("none", "fixed", "linear", "logarithmic")
    }
    logarithmic = profile_results["logarithmic"]
    linear = profile_results["linear"]
    fixed = profile_results["fixed"]
    all_traces = [
        trace
        for profile in ("fixed", "linear", "logarithmic")
        for trace in profile_results[profile]["traces"]
    ]
    write_jsonl(trace_path, all_traces)
    write_jsonl(
        manifest_path,
        logarithmic["state"]["entries"] if logarithmic["state"] else [],
    )
    resolved_state = ensure_parent_directory(state_path)
    with open(resolved_state, "w", encoding="utf-8") as handle:
        json.dump(logarithmic["state"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    delayed_total = max(1, int(logarithmic["delayed_recall_total"]))
    logarithmic_recall = float(logarithmic["delayed_recall_success"]) / delayed_total
    fixed_recall = float(fixed["delayed_recall_success"]) / delayed_total
    negative_total = max(1, int(logarithmic["negative_abstention_total"]))
    negative_abstention = (
        float(logarithmic["negative_abstention_success"]) / negative_total
    )
    state_ratio = float(logarithmic["entry_count"]) / float(
        max(1, int(linear["entry_count"]))
    )
    passed = bool(
        logarithmic_recall == 1.0
        and logarithmic_recall > fixed_recall
        and negative_abstention == 1.0
        and logarithmic["blocked_decision_integrity"] == 1.0
        and logarithmic["entry_count"] <= 8
        and logarithmic["entry_count"] < linear["entry_count"]
        and logarithmic["max_retrieval_event_cost"] <= 256
    )
    compact_profiles = {
        profile: {
            key: value
            for key, value in result.items()
            if key not in {"traces", "state"}
        }
        for profile, result in profile_results.items()
    }
    return {
        "schema": "sara-event-state-cache-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "candidate_count": len(candidates),
        "query_count": len(queries),
        "metrics": {
            "logarithmic_delayed_recall": logarithmic_recall,
            "fixed_delayed_recall": fixed_recall,
            "logarithmic_negative_abstention": negative_abstention,
            "blocked_decision_integrity": logarithmic[
                "blocked_decision_integrity"
            ],
            "logarithmic_entry_count": logarithmic["entry_count"],
            "linear_entry_count": linear["entry_count"],
            "logarithmic_to_linear_state_ratio": state_ratio,
            "logarithmic_max_retrieval_event_cost": logarithmic[
                "max_retrieval_event_cost"
            ],
            "logarithmic_eviction_count": logarithmic["eviction_count"],
        },
        "profiles": compact_profiles,
        "outputs": {
            "candidate_path": os.path.abspath(candidate_path),
            "manifest_path": os.path.abspath(manifest_path),
            "trace_path": os.path.abspath(trace_path),
            "state_path": os.path.abspath(state_path),
        },
        "policy_notes": [
            "Only verified observed source-backed states can enter durable memory.",
            "Retrieval uses bounded sparse overlap and metadata agreement without dense matrices.",
            "Logarithmic retention is constrained by hard tier and total-entry budgets.",
            "The benchmark is observed-only and does not alter production memory.",
        ],
    }


def summarize(report: Dict[str, Any]) -> str:
    lines = [
        f"Event-state cache benchmark: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Candidates: {report.get('candidate_count')}",
        f"Queries: {report.get('query_count')}",
    ]
    lines.extend(
        f"- {key}: {value}" for key, value in sorted(report["metrics"].items())
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the verified hierarchical event-state cache benchmark."
    )
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--candidate-path", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fixture_path = ensure_fixture(args.fixture_path)
    report = build_report(
        read_jsonl(fixture_path),
        candidate_path=args.candidate_path,
        manifest_path=args.manifest_path,
        trace_path=args.trace_path,
        state_path=args.state_path,
    )
    report["fixture_path"] = os.path.abspath(fixture_path)
    resolved_report = ensure_parent_directory(args.report_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    resolved_summary = ensure_parent_directory(args.summary_path)
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(summarize(report))
    print(summarize(report), end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
