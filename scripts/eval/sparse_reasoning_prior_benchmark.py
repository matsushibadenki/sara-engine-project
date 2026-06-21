#!/usr/bin/env python3
"""Run the observed-only sparse reasoning-prior benchmark."""

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

from sara_engine.reasoning.sparse_reasoning_prior import evaluate_sparse_reasoning_cases  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_FIXTURE_PATH = processed_data_path(
    "benchmark_fixtures", "sparse_reasoning_prior_cases.jsonl"
)
DEFAULT_REPORT_PATH = workspace_path("evaluation", "sparse_reasoning_prior_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation", "sparse_reasoning_prior_benchmark_summary.txt"
)
DEFAULT_TRACE_PATH = workspace_path("evaluation", "sparse_reasoning_prior_traces.jsonl")


def default_fixture_cases() -> List[Dict[str, Any]]:
    return [
        {
            "schema": "sara-sparse-reasoning-prior-case-v1",
            "case_id": "steady_upward",
            "target": "demand",
            "expected_direction": "up",
            "expected_magnitude": "moderate",
            "evidence": [
                {
                    "source_ref": "fixture://retrieval/demand",
                    "direction": "up",
                    "magnitude": "moderate",
                    "relevance": 0.9,
                }
            ],
        },
        {
            "schema": "sara-sparse-reasoning-prior-case-v1",
            "case_id": "external_shock_downward",
            "target": "availability",
            "sudden_shift": True,
            "expected_direction": "down",
            "expected_magnitude": "large",
            "evidence": [
                {
                    "source_ref": "fixture://external/outage",
                    "direction": "down",
                    "magnitude": "large",
                    "relevance": 1.0,
                    "external_event": True,
                }
            ],
        },
        {
            "schema": "sara-sparse-reasoning-prior-case-v1",
            "case_id": "balanced_flat",
            "target": "load",
            "expected_direction": "flat",
            "expected_magnitude": "small",
            "evidence": [
                {
                    "source_ref": "fixture://claim/up",
                    "direction": "up",
                    "magnitude": "small",
                    "relevance": 0.8,
                },
                {
                    "source_ref": "fixture://counterfactual/down",
                    "direction": "down",
                    "magnitude": "small",
                    "relevance": 0.8,
                },
            ],
        },
        {
            "schema": "sara-sparse-reasoning-prior-case-v1",
            "case_id": "missing_external_context",
            "target": "latency",
            "sudden_shift": True,
            "expected_abstain": True,
            "evidence": [
                {
                    "source_ref": "fixture://history/latency",
                    "direction": "up",
                    "magnitude": "small",
                    "relevance": 0.9,
                    "external_event": False,
                }
            ],
        },
        {
            "schema": "sara-sparse-reasoning-prior-case-v1",
            "case_id": "irrelevant_evidence",
            "target": "energy",
            "expected_abstain": True,
            "evidence": [
                {
                    "source_ref": "fixture://unrelated",
                    "direction": "down",
                    "magnitude": "large",
                    "relevance": 0.1,
                }
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
    if rows and all(
        row.get("schema") == "sara-sparse-reasoning-prior-case-v1" for row in rows
    ):
        return path
    return write_jsonl(path, default_fixture_cases())


def build_report(cases: Sequence[Dict[str, Any]], *, trace_path: str) -> Dict[str, Any]:
    results = evaluate_sparse_reasoning_cases(cases)
    rows = [result.to_dict() for result in results]
    write_jsonl(trace_path, rows)
    case_count = len(results)
    consistency = sum(1 for result in results if result.logic_to_state_consistent)
    expected_abstentions = sum(1 for case in cases if bool(case.get("expected_abstain", False)))
    correct_abstentions = sum(
        1
        for case, result in zip(cases, results)
        if bool(case.get("expected_abstain", False)) and result.abstained
    )
    source_backed_integrity = float(
        all(
            all(not row["accepted"] or bool(row["source_ref"]) for row in result.trace)
            for result in results
        )
    )
    logic_to_state_consistency = float(consistency) / float(max(1, case_count))
    abstention_integrity = float(correct_abstentions) / float(max(1, expected_abstentions))
    event_relevance = sum(result.event_relevance for result in results) / float(max(1, case_count))
    max_event_cost = max((result.event_cost for result in results), default=0)
    max_state_budget = max((result.state_budget_units for result in results), default=0)
    passed = bool(
        case_count > 0
        and logic_to_state_consistency == 1.0
        and abstention_integrity == 1.0
        and source_backed_integrity == 1.0
        and max_event_cost <= 64
        and max_state_budget <= 64
    )
    return {
        "schema": "sara-sparse-reasoning-prior-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "case_count": case_count,
        "metrics": {
            "logic_to_state_consistency": logic_to_state_consistency,
            "external_event_missing_abstention": abstention_integrity,
            "source_backed_integrity": source_backed_integrity,
            "mean_event_relevance": round(event_relevance, 6),
            "max_event_cost": max_event_cost,
            "max_state_budget_units": max_state_budget,
        },
        "rows": rows,
        "trace_path": os.path.abspath(trace_path),
        "policy_notes": [
            "Reasoning priors are sparse source-backed events, not dense embeddings.",
            "Sudden shifts abstain when required external context is missing.",
            "The benchmark is observed-only and does not alter production forecasting.",
            "No runtime backpropagation, GPU, LLM judge, or hidden chain-of-thought is used.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Sparse reasoning prior benchmark: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
    ]
    lines.extend(
        f"- {key}: {value}" for key, value in sorted(report.get("metrics", {}).items())
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sparse reasoning-prior benchmark.")
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
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    resolved_summary = ensure_parent_directory(args.summary_path)
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "observed_only": report["observed_only"],
                "case_count": report["case_count"],
                "report_path": os.path.abspath(args.report_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
