#!/usr/bin/env python3
"""Run an observed-only sparse dendritic feedback gate benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.learning.dendritic_feedback import (  # noqa: E402
    SparseDendriticFeedbackGate,
    precision_at_expected,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REPORT_PATH = workspace_path("evaluation", "dendritic_feedback_gate_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "dendritic_feedback_gate_benchmark_summary.txt")


def build_fixture_cases() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": "noisy_retrieval",
            "active_event_ids": [10, 11, 12, 99],
            "local_potentials": {10: 0.95, 11: 0.94, 12: 0.5, 99: 1.02},
            "recent_output_spikes": [10],
            "neighbor_activity": {11: [10], 12: [99], 99: [12, 77, 78]},
            "expected_events": [10, 11],
            "train_links": [10, 11],
        },
        {
            "case_id": "adversarial_near_miss",
            "active_event_ids": [20, 21, 88],
            "local_potentials": {20: 0.98, 21: 0.93, 88: 1.01},
            "recent_output_spikes": [20],
            "neighbor_activity": {21: [20], 88: [71, 72, 73]},
            "expected_events": [20, 21],
            "train_links": [20, 21],
        },
        {
            "case_id": "contrastive_control",
            "active_event_ids": [30, 31, 32],
            "local_potentials": {30: 0.92, 31: 0.91, 32: 1.03},
            "recent_output_spikes": [30],
            "neighbor_activity": {31: [30], 32: [91, 92, 93]},
            "expected_events": [30, 31],
            "train_links": [30, 31],
        },
        {
            "case_id": "conflicting_material",
            "active_event_ids": [40, 41, 42, 43],
            "local_potentials": {40: 0.9, 41: 0.89, 42: 0.88, 43: 1.0},
            "recent_output_spikes": [40],
            "neighbor_activity": {41: [40], 42: [43], 43: [42, 80, 81]},
            "expected_events": [40, 41],
            "train_links": [40, 41],
        },
    ]


def _baseline_events(case: Dict[str, Any], threshold: float) -> List[int]:
    potentials = case.get("local_potentials", {})
    return sorted(
        int(event_id)
        for event_id in case.get("active_event_ids", [])
        if float(potentials.get(int(event_id), potentials.get(str(event_id), 0.0))) >= threshold
    )


def build_report(*, event_budget: int) -> Dict[str, Any]:
    gate = SparseDendriticFeedbackGate(event_budget=event_budget, max_steps=1)
    cases = build_fixture_cases()
    rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for case in cases:
        gate.update_local_links(case.get("train_links", []), learning_rate=0.12)
        baseline = _baseline_events(case, gate.threshold)
        result = gate.gate(
            active_event_ids=case.get("active_event_ids", []),
            local_potentials=case.get("local_potentials", {}),
            recent_output_spikes=case.get("recent_output_spikes", []),
            neighbor_activity=case.get("neighbor_activity", {}),
        )
        expected = case.get("expected_events", [])
        baseline_precision = precision_at_expected(baseline, expected)
        gated_precision = precision_at_expected(result.gated_events, expected)
        rows.append(
            {
                "case_id": case["case_id"],
                "baseline_events": baseline,
                "gated_events": result.gated_events,
                "expected_events": expected,
                "baseline_precision": round(baseline_precision, 6),
                "gated_precision": round(gated_precision, 6),
                "precision_delta": round(gated_precision - baseline_precision, 6),
                "fallback_used": result.fallback_used,
                "event_cost": result.event_cost,
                "state_budget_units": result.state_budget_units,
                "convergence_steps": result.convergence_steps,
            }
        )
        trace_rows.append({"case_id": case["case_id"], "trace": result.trace})

    baseline_avg = sum(row["baseline_precision"] for row in rows) / float(len(rows))
    gated_avg = sum(row["gated_precision"] for row in rows) / float(len(rows))
    fallback_rate = sum(1 for row in rows if row["fallback_used"]) / float(len(rows))
    max_event_cost = max(row["event_cost"] for row in rows)
    max_state_budget = max(row["state_budget_units"] for row in rows)
    robustness_delta = round(gated_avg - baseline_avg, 6)
    passed = bool(
        robustness_delta >= 0.0
        and fallback_rate == 0.0
        and max_event_cost <= event_budget
        and max_state_budget <= 256
        and all(row["convergence_steps"] <= 1 for row in rows)
    )
    return {
        "schema": "sara-dendritic-feedback-gate-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "case_count": len(rows),
        "baseline_precision": round(baseline_avg, 6),
        "gated_precision": round(gated_avg, 6),
        "robustness_delta": robustness_delta,
        "fallback_rate": round(fallback_rate, 6),
        "max_event_cost": max_event_cost,
        "max_state_budget_units": max_state_budget,
        "event_budget": int(event_budget),
        "rows": rows,
        "trace_samples": trace_rows,
        "policy_notes": [
            "The gate is sparse, CPU-first, bounded-state, and backpropagation-free.",
            "This report is observed-only and does not alter default production inference.",
            "Fallback returns the ungated sparse path when event budget is exceeded.",
            "Generated artifacts are written under workspace/evaluation.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Dendritic feedback gate benchmark: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
        f"Baseline precision: {report.get('baseline_precision')}",
        f"Gated precision: {report.get('gated_precision')}",
        f"Robustness delta: {report.get('robustness_delta')}",
        f"Fallback rate: {report.get('fallback_rate')}",
        f"Max event cost: {report.get('max_event_cost')}/{report.get('event_budget')}",
        f"Max state budget units: {report.get('max_state_budget_units')}",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(report: Dict[str, Any], report_path: str, summary_path: str) -> None:
    resolved_report_path = ensure_parent_directory(report_path)
    with open(resolved_report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    resolved_summary_path = ensure_parent_directory(summary_path)
    with open(resolved_summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sparse dendritic feedback gate benchmark.")
    parser.add_argument("--event-budget", type=int, default=64)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(event_budget=args.event_budget)
    write_outputs(report, args.report_path, args.summary_path)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "observed_only": report["observed_only"],
                "robustness_delta": report["robustness_delta"],
                "report_path": os.path.abspath(args.report_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
