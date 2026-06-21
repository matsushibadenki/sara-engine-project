#!/usr/bin/env python3
"""Run the observed-only SARA sparse resonance-credit benchmark."""

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

from sara_engine.learning.resonance_credit import SparseResonanceCreditAssigner  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_FIXTURE_PATH = processed_data_path(
    "benchmark_fixtures", "resonance_credit_cases.jsonl"
)
DEFAULT_REPORT_PATH = workspace_path("evaluation", "resonance_credit_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "resonance_credit_benchmark_summary.txt")
DEFAULT_TRACE_PATH = workspace_path("evaluation", "resonance_credit_traces.jsonl")
DEFAULT_STATE_PATH = workspace_path("evaluation", "resonance_credit_state.json")


def default_cases() -> List[Dict[str, Any]]:
    common = {
        "local_coincidence": 0.9,
        "prediction_consistency": 0.8,
        "verifier_confidence": 0.9,
        "cross_modal_agreement": 0.7,
        "reward_signal": 0.8,
        "novelty_signal": 0.6,
        "reward_polarity": 1.0,
        "metabolic_headroom": 0.8,
        "source_backed": True,
    }
    return [
        {
            "schema": "sara-resonance-credit-case-v1",
            "case_id": "verified_multichannel_success",
            "eligibility": {"1:2": 0.8},
            "signals": common,
            "expected_decision": "reinforce",
        },
        {
            "schema": "sara-resonance-credit-case-v1",
            "case_id": "verified_negative_credit",
            "eligibility": {"2:3": 0.7},
            "signals": {**common, "reward_polarity": -1.0},
            "expected_decision": "reinforce",
        },
        {
            "schema": "sara-resonance-credit-case-v1",
            "case_id": "contradictory_reward",
            "eligibility": {"3:4": 1.0},
            "signals": {**common, "contradiction": 0.95},
            "expected_decision": "freeze_contradiction",
            "naive_reward_would_update": True,
        },
        {
            "schema": "sara-resonance-credit-case-v1",
            "case_id": "reasoning_abstention",
            "eligibility": {"4:5": 1.0},
            "signals": {**common, "abstained": True},
            "expected_decision": "freeze_abstention",
            "naive_reward_would_update": True,
        },
        {
            "schema": "sara-resonance-credit-case-v1",
            "case_id": "metabolic_pressure",
            "eligibility": {"5:6": 1.0},
            "signals": {**common, "metabolic_headroom": 0.1},
            "expected_decision": "freeze_metabolic_budget",
            "naive_reward_would_update": True,
        },
        {
            "schema": "sara-resonance-credit-case-v1",
            "case_id": "single_channel_noise",
            "eligibility": {"6:7": 1.0},
            "signals": {
                "local_coincidence": 0.9,
                "reward_signal": 0.8,
                "reward_polarity": 1.0,
                "metabolic_headroom": 0.8,
                "source_backed": True,
            },
            "expected_decision": "freeze_insufficient_resonance",
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
    if rows and all(row.get("schema") == "sara-resonance-credit-case-v1" for row in rows):
        return path
    return write_jsonl(path, default_cases())


def _eligibility(raw: Any) -> Dict[tuple[int, int], float]:
    if not isinstance(raw, dict):
        return {}
    result: Dict[tuple[int, int], float] = {}
    for key, value in raw.items():
        parts = str(key).split(":")
        if len(parts) == 2:
            result[(int(parts[0]), int(parts[1]))] = float(value)
    return result


def build_report(
    cases: Sequence[Dict[str, Any]],
    *,
    trace_path: str,
    state_path: str,
) -> Dict[str, Any]:
    assigner = SparseResonanceCreditAssigner(max_links=32, weight_clip=1.0)
    rows: List[Dict[str, Any]] = []
    correct = 0
    harmful_case_count = 0
    harmful_freeze_count = 0
    naive_harmful_update_count = 0
    for case in cases:
        result = assigner.apply(
            _eligibility(case.get("eligibility", {})),
            case.get("signals", {}) if isinstance(case.get("signals"), dict) else {},
        )
        expected = str(case.get("expected_decision", ""))
        decision_correct = result.decision == expected
        correct += int(decision_correct)
        harmful = bool(case.get("naive_reward_would_update", False))
        if harmful:
            harmful_case_count += 1
            naive_harmful_update_count += 1
            harmful_freeze_count += int(not result.update_allowed)
        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "expected_decision": expected,
                "decision_correct": decision_correct,
                "naive_reward_would_update": harmful,
                **result.to_dict(),
            }
        )

    write_jsonl(trace_path, rows)
    resolved_state = ensure_parent_directory(state_path)
    with open(resolved_state, "w", encoding="utf-8") as handle:
        json.dump(assigner.state_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    decision_integrity = float(correct) / float(max(1, len(cases)))
    harmful_update_suppression = float(harmful_freeze_count) / float(
        max(1, harmful_case_count)
    )
    max_event_cost = max((int(row["event_cost"]) for row in rows), default=0)
    max_state_budget = max((int(row["state_budget_units"]) for row in rows), default=0)
    passed = bool(
        cases
        and decision_integrity == 1.0
        and harmful_update_suppression == 1.0
        and assigner.update_count == 2
        and max_event_cost <= 32
        and max_state_budget <= 32
    )
    return {
        "schema": "sara-resonance-credit-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "case_count": len(cases),
        "metrics": {
            "decision_integrity": decision_integrity,
            "harmful_update_suppression": harmful_update_suppression,
            "naive_reward_harmful_update_count": naive_harmful_update_count,
            "resonance_update_count": assigner.update_count,
            "resonance_freeze_count": assigner.freeze_count,
            "max_event_cost": max_event_cost,
            "max_state_budget_units": max_state_budget,
        },
        "rows": rows,
        "outputs": {
            "trace_path": os.path.abspath(trace_path),
            "state_path": os.path.abspath(state_path),
        },
        "policy_notes": [
            "The resonance gate coordinates existing local signals rather than adding backpropagation.",
            "Contradiction, abstention, unverified sources, and metabolic pressure freeze plasticity.",
            "All updates are sparse, bounded, local, CPU-first, and auditable.",
            "The benchmark is observed-only and does not alter production learning.",
        ],
    }


def summarize(report: Dict[str, Any]) -> str:
    lines = [
        f"Resonance credit benchmark: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(report["metrics"].items()))
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SARA resonance-credit benchmark.")
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
