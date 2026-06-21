#!/usr/bin/env python3
"""Verify sparse STRIPS-like plan traces and emit managed reports."""

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

from sara_engine.reasoning.sparse_plan_trace import (  # noqa: E402
    build_repair_materials,
    verify_sparse_plan_trace,
)
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


DEFAULT_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "sparse_plan_trace_cases.jsonl")
DEFAULT_REPAIR_PATH = processed_data_path("autobot", "plan_trace_repair_materials.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "sparse_plan_trace_verifier.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "sparse_plan_trace_verifier_summary.txt")


def default_fixture_cases() -> List[Dict[str, Any]]:
    move_actions = {
        "move_a_b": {"pre": ["at_a", "path_a_b"], "add": ["at_b"], "del": ["at_a"]},
        "pickup_key": {"pre": ["at_b", "key_at_b", "hand_empty"], "add": ["has_key"], "del": ["key_at_b", "hand_empty"]},
        "open_door": {"pre": ["at_b", "has_key", "door_closed"], "add": ["door_open"], "del": ["door_closed"]},
    }
    return [
        {
            "schema": "sara-sparse-plan-trace-case-v1",
            "case_id": "valid_key_route",
            "source_ref": "synthetic://sparse-plan/valid-key-route",
            "initial_state": ["at_a", "path_a_b", "key_at_b", "hand_empty", "door_closed"],
            "goal": ["door_open"],
            "actions": move_actions,
            "invariants": [{"name": "no_alarm", "forbids": ["alarm_triggered"]}],
            "plan": [
                {"action": "move_a_b"},
                {"action": "pickup_key"},
                {"action": "open_door"},
            ],
            "expected_valid": True,
        },
        {
            "schema": "sara-sparse-plan-trace-case-v1",
            "case_id": "invalid_missing_precondition",
            "source_ref": "synthetic://sparse-plan/missing-precondition",
            "initial_state": ["at_a", "path_a_b", "key_at_b", "hand_empty", "door_closed"],
            "goal": ["door_open"],
            "actions": move_actions,
            "plan": [{"action": "open_door"}],
            "expected_valid": False,
        },
        {
            "schema": "sara-sparse-plan-trace-case-v1",
            "case_id": "invalid_wrong_effect",
            "source_ref": "synthetic://sparse-plan/wrong-effect",
            "initial_state": ["at_a", "path_a_b", "key_at_b", "hand_empty", "door_closed"],
            "goal": ["door_open"],
            "actions": move_actions,
            "plan": [
                {"action": "move_a_b", "claimed_next_state": ["at_b", "path_a_b", "key_at_b", "hand_empty", "door_closed", "door_open"]}
            ],
            "expected_valid": False,
        },
        {
            "schema": "sara-sparse-plan-trace-case-v1",
            "case_id": "invalid_missing_frame",
            "source_ref": "synthetic://sparse-plan/missing-frame",
            "initial_state": ["at_a", "path_a_b", "key_at_b", "hand_empty", "door_closed"],
            "goal": ["door_open"],
            "actions": move_actions,
            "plan": [{"action": "move_a_b", "claimed_next_state": ["at_b"]}],
            "expected_valid": False,
        },
        {
            "schema": "sara-sparse-plan-trace-case-v1",
            "case_id": "invalid_unmet_goal",
            "source_ref": "synthetic://sparse-plan/unmet-goal",
            "initial_state": ["at_a", "path_a_b", "key_at_b", "hand_empty", "door_closed"],
            "goal": ["door_open"],
            "actions": move_actions,
            "plan": [{"action": "move_a_b"}, {"action": "pickup_key"}],
            "expected_valid": False,
        },
        {
            "schema": "sara-sparse-plan-trace-case-v1",
            "case_id": "invalid_invariant_violation",
            "source_ref": "synthetic://sparse-plan/invariant-violation",
            "initial_state": ["at_a", "path_a_b", "key_at_b", "hand_empty", "door_closed"],
            "goal": ["at_b"],
            "actions": {
                "unsafe_move": {"pre": ["at_a", "path_a_b"], "add": ["at_b", "alarm_triggered"], "del": ["at_a"]}
            },
            "invariants": [{"name": "no_alarm", "forbids": ["alarm_triggered"]}],
            "plan": [{"action": "unsafe_move"}],
            "expected_valid": False,
        },
    ]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
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
    rows = read_jsonl(path) if os.path.exists(path) else []
    if rows and all(row.get("schema") == "sara-sparse-plan-trace-case-v1" for row in rows):
        return path
    return write_jsonl(path, default_fixture_cases())


def build_report(cases: Sequence[Dict[str, Any]], repair_path: str) -> Dict[str, Any]:
    results = [verify_sparse_plan_trace(case) for case in cases]
    repairs = build_repair_materials(cases, results)
    resolved_repair_path = write_jsonl(repair_path, repairs)
    expected_match_count = sum(
        1
        for case, result in zip(cases, results)
        if bool(case.get("expected_valid", False)) == result.valid
    )
    invalid_step_count = sum(result.invalid_step_count for result in results)
    max_event_cost = max((result.event_cost for result in results), default=0)
    max_state_budget = max((result.state_budget_units for result in results), default=0)
    passed = bool(cases) and expected_match_count == len(cases) and repairs and max_event_cost <= 256
    return {
        "schema": "sara-sparse-plan-trace-verifier-v1",
        "passed": passed,
        "observed_only": True,
        "case_count": len(cases),
        "expected_match_count": expected_match_count,
        "valid_case_count": sum(1 for result in results if result.valid),
        "invalid_case_count": sum(1 for result in results if not result.valid),
        "invalid_step_count": invalid_step_count,
        "repair_material_count": len(repairs),
        "repair_material_path": os.path.abspath(resolved_repair_path),
        "max_event_cost": max_event_cost,
        "max_state_budget_units": max_state_budget,
        "fallback_behavior": "abstain_or_emit_repair_material_on_invalid_trace",
        "results": [result.to_dict() for result in results],
        "policy_notes": [
            "Plan traces are sparse, CPU-first, bounded-state, and backpropagation-free.",
            "The verifier uses machine-checkable facts instead of free-form chain-of-thought.",
            "Repair materials are managed under data/processed/autobot.",
            "This evidence is observed-only and does not change production runtime behavior.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Sparse plan trace verifier: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
        f"Expected matches: {report.get('expected_match_count')}/{report.get('case_count')}",
        f"Invalid cases: {report.get('invalid_case_count')}",
        f"Invalid steps: {report.get('invalid_step_count')}",
        f"Repair materials: {report.get('repair_material_count')}",
        f"Max event cost: {report.get('max_event_cost')}",
        f"Max state budget units: {report.get('max_state_budget_units')}",
        f"Fallback: {report.get('fallback_behavior')}",
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
    parser = argparse.ArgumentParser(description="Verify sparse STRIPS-like plan traces.")
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--repair-path", default=DEFAULT_REPAIR_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fixture_path = ensure_fixture(args.fixture_path)
    cases = read_jsonl(fixture_path)
    report = build_report(cases, args.repair_path)
    report["fixture_path"] = os.path.abspath(fixture_path)
    write_outputs(report, args.report_path, args.summary_path)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "observed_only": report["observed_only"],
                "case_count": report["case_count"],
                "repair_material_count": report["repair_material_count"],
                "report_path": os.path.abspath(args.report_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
