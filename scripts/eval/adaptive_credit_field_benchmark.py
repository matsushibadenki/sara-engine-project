#!/usr/bin/env python3
"""Run the observed-only SARA adaptive credit field benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.learning.adaptive_credit import AdaptiveCreditField  # noqa: E402
from sara_engine.learning.resonance_credit import SparseResonanceCreditAssigner  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


DEFAULT_FIXTURE_PATH = processed_data_path(
    "benchmark_fixtures",
    "adaptive_credit_field_cases.jsonl",
)
DEFAULT_TRACE_PATH = workspace_path("evaluation", "adaptive_credit_field_traces.jsonl")
DEFAULT_STATE_PATH = workspace_path("evaluation", "adaptive_credit_field_state.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "adaptive_credit_field_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation",
    "adaptive_credit_field_benchmark_summary.txt",
)


def default_cases() -> List[Dict[str, Any]]:
    return [
        {
            "schema": "sara-adaptive-credit-case-v1",
            "case_id": "idle_no_learning_event",
            "active_routes": {"1:2": 0.8, "2:3": 0.7},
            "route_regions": {"1:2": "vision", "2:3": "audio"},
            "region_credit": {"vision": 1.0, "audio": 1.0},
            "signals": {
                "prediction_error": 0.12,
                "novelty": 0.08,
                "reward": 0.0,
                "verifier_disagreement": 0.1,
                "contradiction": 0.0,
                "metabolic_headroom": 0.8,
                "source_backed": True,
            },
            "expected_decision": "freeze_no_learning_event",
            "expected_updated_routes": 0,
        },
        {
            "schema": "sara-adaptive-credit-case-v1",
            "case_id": "region_gated_sparse_update",
            "active_routes": {"1:2": 0.9, "2:3": 0.85, "3:4": 0.4},
            "route_regions": {"1:2": "vision", "2:3": "audio", "3:4": "vision"},
            "region_credit": {"vision": 0.92, "audio": 0.05},
            "signals": {
                "prediction_error": 0.74,
                "novelty": 0.61,
                "reward": 0.67,
                "verifier_disagreement": 0.2,
                "contradiction": 0.0,
                "metabolic_headroom": 0.85,
                "source_backed": True,
            },
            "expected_decision": "update",
            "expected_updated_routes": 2,
            "expected_skipped_by_region": 1,
        },
        {
            "schema": "sara-adaptive-credit-case-v1",
            "case_id": "contradiction_freeze",
            "active_routes": {"4:5": 0.95},
            "signals": {
                "prediction_error": 0.8,
                "novelty": 0.8,
                "reward": 0.8,
                "verifier_disagreement": 0.6,
                "contradiction": 0.91,
                "metabolic_headroom": 0.9,
                "source_backed": True,
            },
            "expected_decision": "freeze_contradiction",
            "expected_updated_routes": 0,
            "naive_reward_would_update": True,
        },
        {
            "schema": "sara-adaptive-credit-case-v1",
            "case_id": "source_guard_freeze",
            "active_routes": {"6:7": 0.7},
            "signals": {
                "prediction_error": 0.71,
                "novelty": 0.73,
                "reward": 0.64,
                "verifier_disagreement": 0.4,
                "contradiction": 0.0,
                "metabolic_headroom": 0.9,
                "source_backed": False,
            },
            "expected_decision": "freeze_unverified_source",
            "expected_updated_routes": 0,
            "naive_reward_would_update": True,
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
    if rows and all(row.get("schema") == "sara-adaptive-credit-case-v1" for row in rows):
        return path
    return write_jsonl(path, default_cases())


def _route_map(raw: Any) -> Dict[Tuple[int, int], float]:
    if not isinstance(raw, dict):
        return {}
    result: Dict[Tuple[int, int], float] = {}
    for key, value in raw.items():
        parts = str(key).split(":")
        if len(parts) == 2:
            result[(int(parts[0]), int(parts[1]))] = float(value)
    return result


def _route_regions(raw: Any) -> Dict[Tuple[int, int], str]:
    if not isinstance(raw, dict):
        return {}
    result: Dict[Tuple[int, int], str] = {}
    for key, value in raw.items():
        parts = str(key).split(":")
        if len(parts) == 2:
            result[(int(parts[0]), int(parts[1]))] = str(value)
    return result


def _naive_stdp_updates(case: Mapping[str, Any]) -> int:
    reward = float((case.get("signals") or {}).get("reward", 0.0) or 0.0)
    if reward <= 0.0:
        return 0
    return len(_route_map(case.get("active_routes")))


def _resonance_updates(case: Mapping[str, Any]) -> int:
    assigner = SparseResonanceCreditAssigner(max_links=64)
    signals = case.get("signals", {}) if isinstance(case.get("signals"), dict) else {}
    resonance_signals = {
        "local_coincidence": signals.get("prediction_error", 0.0),
        "prediction_consistency": 1.0 - float(signals.get("verifier_disagreement", 0.0) or 0.0),
        "verifier_confidence": 1.0 - float(signals.get("contradiction", 0.0) or 0.0),
        "cross_modal_agreement": max((case.get("region_credit") or {}).values(), default=1.0),
        "reward_signal": signals.get("reward", 0.0),
        "novelty_signal": signals.get("novelty", 0.0),
        "reward_polarity": 1.0,
        "metabolic_headroom": signals.get("metabolic_headroom", 1.0),
        "source_backed": bool(signals.get("source_backed", False)),
        "abstained": bool(signals.get("abstained", False)),
        "contradiction": signals.get("contradiction", 0.0),
    }
    result = assigner.apply(_route_map(case.get("active_routes")), resonance_signals)
    return result.updated_route_count if hasattr(result, "updated_route_count") else len(result.updates)


def build_report(
    cases: Sequence[Dict[str, Any]],
    *,
    trace_path: str,
    state_path: str,
) -> Dict[str, Any]:
    continuous = AdaptiveCreditField(quantize_credit=False, max_routes=64)
    quantized = AdaptiveCreditField(quantize_credit=True, max_routes=64)
    rows: List[Dict[str, Any]] = []
    correct = 0
    sparse_advantage_count = 0
    harmful_case_count = 0
    harmful_freeze_count = 0
    quantized_match_count = 0

    for case in cases:
        active_routes = _route_map(case.get("active_routes"))
        route_regions = _route_regions(case.get("route_regions"))
        region_credit = case.get("region_credit", {}) if isinstance(case.get("region_credit"), dict) else {}
        signals = case.get("signals", {}) if isinstance(case.get("signals"), dict) else {}
        continuous_result = continuous.apply(
            active_routes=active_routes,
            signals=signals,
            route_regions=route_regions,
            region_credit=region_credit,
        )
        quantized_result = quantized.apply(
            active_routes=active_routes,
            signals=signals,
            route_regions=route_regions,
            region_credit=region_credit,
        )
        expected_decision = str(case.get("expected_decision", ""))
        expected_updated_routes = int(case.get("expected_updated_routes", 0))
        expected_skipped_by_region = int(case.get("expected_skipped_by_region", 0))
        decision_correct = (
            continuous_result.decision == expected_decision
            and continuous_result.updated_route_count == expected_updated_routes
            and continuous_result.skipped_by_region_count == expected_skipped_by_region
        )
        correct += int(decision_correct)
        naive_updates = _naive_stdp_updates(case)
        resonance_updates = _resonance_updates(case)
        sparse_advantage = continuous_result.updated_route_count <= naive_updates
        sparse_advantage_count += int(sparse_advantage)
        harmful = bool(case.get("naive_reward_would_update", False))
        if harmful:
            harmful_case_count += 1
            harmful_freeze_count += int(not continuous_result.update_allowed)
        quantized_match = (
            quantized_result.decision == continuous_result.decision
            and quantized_result.updated_route_count == continuous_result.updated_route_count
        )
        quantized_match_count += int(quantized_match)
        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "expected_decision": expected_decision,
                "decision_correct": decision_correct,
                "sparse_advantage": sparse_advantage,
                "naive_stdp_updates": naive_updates,
                "resonance_updates": resonance_updates,
                "quantized_match": quantized_match,
                "continuous": continuous_result.to_dict(),
                "quantized": quantized_result.to_dict(),
            }
        )

    write_jsonl(trace_path, rows)
    resolved_state = ensure_parent_directory(state_path)
    with open(resolved_state, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "continuous": continuous.state_dict(),
                "quantized": quantized.state_dict(),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    decision_integrity = float(correct) / float(max(1, len(cases)))
    harmful_update_suppression = float(harmful_freeze_count) / float(max(1, harmful_case_count))
    sparse_active_fraction = float(
        sum(row["continuous"]["updated_route_count"] for row in rows)
    ) / float(max(1, sum(max(1, row["naive_stdp_updates"]) for row in rows)))
    quantized_behavior_match = float(quantized_match_count) / float(max(1, len(cases)))
    max_updated_routes = max((int(row["continuous"]["updated_route_count"]) for row in rows), default=0)
    passed = bool(
        cases
        and decision_integrity == 1.0
        and harmful_update_suppression == 1.0
        and quantized_behavior_match == 1.0
        and sparse_advantage_count == len(cases)
        and max_updated_routes <= 2
    )
    return {
        "schema": "sara-adaptive-credit-field-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "case_count": len(cases),
        "metrics": {
            "decision_integrity": decision_integrity,
            "harmful_update_suppression": harmful_update_suppression,
            "sparse_advantage_case_ratio": float(sparse_advantage_count) / float(max(1, len(cases))),
            "sparse_active_fraction_vs_naive": sparse_active_fraction,
            "quantized_behavior_match": quantized_behavior_match,
            "continuous_update_count": continuous.update_count,
            "continuous_freeze_count": continuous.freeze_count,
            "quantized_update_count": quantized.update_count,
            "max_updated_routes": max_updated_routes,
        },
        "rows": rows,
        "outputs": {
            "trace_path": os.path.abspath(trace_path),
            "state_path": os.path.abspath(state_path),
        },
        "policy_notes": [
            "Adaptive Credit Field is event-driven and touches only recently active sparse routes.",
            "Credit behaves as bounded local state rather than a dense backpropagation graph.",
            "Quantized credit is benchmarked as a first-class low-cost mode.",
            "The benchmark is observed-only and keeps dense calibration disabled.",
        ],
    }


def summarize(report: Dict[str, Any]) -> str:
    lines = [
        f"Adaptive credit field benchmark: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(report["metrics"].items()))
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SARA adaptive credit field benchmark.")
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
