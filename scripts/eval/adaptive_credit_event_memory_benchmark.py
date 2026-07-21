#!/usr/bin/env python3
"""Run the observed-only adaptive-credit/Event Memory integration benchmark."""

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

from sara_engine.learning.adaptive_credit import summarize_event_memory_credit  # noqa: E402
from sara_engine.memory.event_state_cache import EventStateCandidate, VerifiedHierarchicalEventStateCache  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


DEFAULT_FIXTURE_PATH = processed_data_path(
    "benchmark_fixtures",
    "adaptive_credit_event_memory_cases.jsonl",
)
DEFAULT_REPORT_PATH = workspace_path(
    "evaluation",
    "adaptive_credit_event_memory_benchmark.json",
)
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation",
    "adaptive_credit_event_memory_benchmark_summary.txt",
)
DEFAULT_TRACE_PATH = workspace_path(
    "evaluation",
    "adaptive_credit_event_memory_traces.jsonl",
)


def default_cases() -> List[Dict[str, Any]]:
    return [
        {
            "schema": "sara-adaptive-credit-event-memory-case-v1",
            "entry_id": "baseline_supported",
            "signature": [11, 13, 17],
            "source_ref": "fixture:baseline-supported",
            "time_segment": 1,
            "own_latent_id": "latent:baseline-supported",
            "confidence": 0.8,
            "uncertainty": 0.15,
            "source_reliability": 0.82,
            "resonance_score": 0.8,
            "sequence_support_score": 0.3,
            "source_backed": True,
            "verified": True,
            "observed": True,
            "route_states": [
                {"responsibility": 0.92, "confidence": 0.86, "longevity": 0.8},
                {"responsibility": 0.85, "confidence": 0.84, "longevity": 0.78},
            ],
        },
        {
            "schema": "sara-adaptive-credit-event-memory-case-v1",
            "entry_id": "bundle_supported",
            "signature": [41, 43, 47],
            "source_ref": "bundle::fixture-supported",
            "time_segment": 1,
            "own_latent_id": "bundle:0:123456",
            "confidence": 0.8,
            "uncertainty": 0.15,
            "source_reliability": 0.82,
            "resonance_score": 0.8,
            "sequence_support_score": 0.3,
            "source_backed": True,
            "verified": True,
            "observed": True,
            "route_states": [
                {
                    "responsibility": 0.70,
                    "confidence": 0.72,
                    "longevity": 0.58,
                    "multimodal_bundle_affinity": 1.0,
                }
            ],
        },
        {
            "schema": "sara-adaptive-credit-event-memory-case-v1",
            "entry_id": "baseline_weak",
            "signature": [21, 23, 27],
            "source_ref": "fixture:baseline-weak",
            "time_segment": 2,
            "own_latent_id": "latent:baseline-weak",
            "confidence": 0.8,
            "uncertainty": 0.15,
            "source_reliability": 0.82,
            "resonance_score": 0.8,
            "sequence_support_score": 0.3,
            "source_backed": True,
            "verified": True,
            "observed": True,
            "route_states": [
                {"responsibility": 0.12, "confidence": 0.18, "longevity": 0.1},
            ],
        },
        {
            "schema": "sara-adaptive-credit-event-memory-case-v1",
            "entry_id": "blocked_contradiction",
            "signature": [31, 37],
            "source_ref": "fixture:blocked",
            "time_segment": 3,
            "confidence": 0.9,
            "source_reliability": 0.9,
            "resonance_score": 0.9,
            "source_backed": True,
            "verified": True,
            "observed": True,
            "contradicted": True,
            "route_states": [
                {"responsibility": 0.95, "confidence": 0.9, "longevity": 0.85},
            ],
        },
    ]


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
    required_entry_ids = {row["entry_id"] for row in default_cases()}
    if (
        rows
        and all(row.get("schema") == "sara-adaptive-credit-event-memory-case-v1" for row in rows)
        and required_entry_ids.issubset({str(row.get("entry_id", "")) for row in rows})
    ):
        return path
    return write_jsonl(path, default_cases())


def _candidate_from_row(row: Dict[str, Any], *, with_credit: bool) -> EventStateCandidate:
    credit = summarize_event_memory_credit(row.get("route_states", ())) if with_credit else {}
    return EventStateCandidate.from_verified_evidence(
        verifier_id="adaptive-credit-event-memory-benchmark",
        evidence={"row": row, "with_credit": with_credit},
        entry_id=str(row.get("entry_id", "")),
        signature=tuple(int(value) for value in row.get("signature", ())),
        source_ref=str(row.get("source_ref", "")),
        time_segment=int(row.get("time_segment", 0)),
        own_latent_id=str(row.get("own_latent_id", "")),
        confidence=float(row.get("confidence", 1.0)),
        uncertainty=float(row.get("uncertainty", 0.0)),
        source_reliability=float(row.get("source_reliability", 1.0)),
        resonance_score=float(row.get("resonance_score", 0.0)),
        sequence_support_score=float(row.get("sequence_support_score", 0.0)),
        sequence_support_count=int(row.get("sequence_support_count", 0)),
        credit_score=float(credit.get("credit_score", 0.0)),
        credit_responsibility=float(credit.get("credit_responsibility", 0.0)),
        credit_confidence=float(credit.get("credit_confidence", 0.0)),
        credit_longevity=float(credit.get("credit_longevity", 0.0)),
        metabolic_headroom=float(row.get("metabolic_headroom", 1.0)),
        observed=bool(row.get("observed", True)),
        source_backed=bool(row.get("source_backed", True)),
        verified=bool(row.get("verified", True)),
        contradicted=bool(row.get("contradicted", False)),
        abstained=bool(row.get("abstained", False)),
    )


def build_report(cases: Sequence[Dict[str, Any]], *, trace_path: str) -> Dict[str, Any]:
    baseline = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic", max_entries=1)
    credit = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic", max_entries=1)
    rows: List[Dict[str, Any]] = []
    harmful_block_preserved = 0
    for row in cases:
        baseline_result = baseline.admit(_candidate_from_row(row, with_credit=False))
        credit_result = credit.admit(_candidate_from_row(row, with_credit=True))
        if bool(row.get("contradicted", False)) and not credit_result.accepted:
            harmful_block_preserved += 1
        rows.append(
            {
                "entry_id": row.get("entry_id"),
                "baseline": baseline_result.to_dict(),
                "credit": credit_result.to_dict(),
                "credit_summary": summarize_event_memory_credit(row.get("route_states", ())),
            }
        )
    write_jsonl(trace_path, rows)
    strong_entry_present = any(
        entry.entry_id in {"baseline_supported", "bundle_supported"}
        for entry in credit.entries.values()
    )
    bundle_longevity_bonus_present = any(
        row.get("entry_id") == "bundle_supported"
        and float(row.get("credit_summary", {}).get("credit_longevity", 0.0) or 0.0) > 0.58
        for row in rows
    )
    weak_entry_evicted = all(
        entry.entry_id != "baseline_weak"
        for entry in credit.entries.values()
    )
    passed = bool(
        strong_entry_present
        and weak_entry_evicted
        and harmful_block_preserved >= 1
        and bundle_longevity_bonus_present
    )
    return {
        "schema": "sara-adaptive-credit-event-memory-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "case_count": len(cases),
        "metrics": {
            "harmful_block_preserved_count": harmful_block_preserved,
            "baseline_entry_count": len(baseline.entries),
            "credit_entry_count": len(credit.entries),
            "credit_strong_entry_present": strong_entry_present,
            "credit_weak_entry_evicted": weak_entry_evicted,
            "bundle_longevity_bonus_present": bundle_longevity_bonus_present,
        },
        "rows": rows,
        "outputs": {
            "trace_path": os.path.abspath(trace_path),
        },
        "policy_notes": [
            "Adaptive credit acts as bounded memory-admission pressure rather than replacing verification.",
            "Contradiction and source guards still dominate credit support.",
            "The benchmark is observed-only and keeps production learning unchanged.",
        ],
    }


def summarize(report: Dict[str, Any]) -> str:
    lines = [
        f"Adaptive credit/Event Memory benchmark: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(report["metrics"].items()))
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the adaptive credit/Event Memory integration benchmark.")
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fixture_path = ensure_fixture(args.fixture_path)
    report = build_report(read_jsonl(fixture_path), trace_path=args.trace_path)
    report["fixture_path"] = os.path.abspath(fixture_path)
    resolved_report = ensure_parent_directory(args.report_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    resolved_summary = ensure_parent_directory(args.summary_path)
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(summarize(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
