#!/usr/bin/env python3
"""Bridge managed SARA evidence reports into verified resonance credit."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.learning.metabolic_budget import (  # noqa: E402
    MetabolicBudgetConfig,
    evaluate_structural_metabolic_budget,
)
from sara_engine.learning.resonance_credit import SparseResonanceCreditAssigner  # noqa: E402
from sara_engine.learning.resonance_evidence import build_resonance_evidence  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REPORT_PATH = workspace_path(
    "evaluation", "resonance_credit_integration_benchmark.json"
)
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation", "resonance_credit_integration_benchmark_summary.txt"
)
DEFAULT_TRACE_PATH = workspace_path(
    "evaluation", "resonance_credit_integration_traces.jsonl"
)
DEFAULT_SOURCE_PATHS = {
    "reasoning_prior": workspace_path(
        "evaluation", "sparse_reasoning_prior_benchmark.json"
    ),
    "plan_verifier": workspace_path("evaluation", "sparse_plan_trace_verifier.json"),
    "multimodal_binding": workspace_path(
        "evaluation", "synesthetic_multimodal_binding_benchmark.json"
    ),
    "dendritic_feedback": workspace_path(
        "evaluation", "dendritic_feedback_gate_benchmark.json"
    ),
    "own_latent": workspace_path("evaluation", "own_latent_learning_benchmark.json"),
}


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_metabolic_report(*, pressure_case: bool = False) -> Dict[str, Any]:
    operations = [
        {
            "kind": "grow",
            "synapse_delta": 1,
            "event_cost": 0.5 if not pressure_case else 0.9,
            "reserve_cost": 0.1 if not pressure_case else 0.45,
            "importance": 0.9,
        }
    ]
    config = MetabolicBudgetConfig(
        max_synapses=4,
        event_budget=4.0,
        plasticity_reserve=1.0,
    )
    report = evaluate_structural_metabolic_budget(operations, config)
    report["schema"] = "sara-structural-metabolic-budget-v1"
    if pressure_case:
        report["resource_pressure"] = 0.95
    return report


def load_source_reports(source_paths: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    reports = {name: load_json(path) for name, path in source_paths.items()}
    reports["metabolic_budget"] = build_metabolic_report()
    return reports


def _case_reports(
    base_reports: Mapping[str, Dict[str, Any]],
    case_id: str,
) -> Dict[str, Dict[str, Any]]:
    reports = copy.deepcopy(dict(base_reports))
    if case_id == "verifier_contradiction":
        reports["plan_verifier"]["expected_match_count"] = 0
    elif case_id == "missing_source":
        reports["own_latent"] = {}
    elif case_id == "abstention_regression":
        reports["reasoning_prior"]["metrics"]["external_event_missing_abstention"] = 0.0
    elif case_id == "metabolic_pressure":
        reports["metabolic_budget"] = build_metabolic_report(pressure_case=True)
    return reports


def build_report(
    source_reports: Mapping[str, Dict[str, Any]],
    *,
    source_paths: Mapping[str, str],
    trace_path: str,
) -> Dict[str, Any]:
    cases = [
        ("live_managed_evidence", "reinforce"),
        ("verifier_contradiction", "freeze_contradiction"),
        ("missing_source", "freeze_unverified_source"),
        ("abstention_regression", "freeze_abstention"),
        ("metabolic_pressure", "freeze_metabolic_budget"),
    ]
    assigner = SparseResonanceCreditAssigner(max_links=16)
    rows = []
    for index, (case_id, expected) in enumerate(cases):
        bundle = build_resonance_evidence(_case_reports(source_reports, case_id))
        result = assigner.apply({(index + 1, index + 2): 0.8}, bundle.signals)
        rows.append(
            {
                "case_id": case_id,
                "expected_decision": expected,
                "decision_correct": result.decision == expected,
                "evidence": bundle.to_dict(),
                "credit_result": result.to_dict(),
            }
        )

    resolved_trace = ensure_parent_directory(trace_path)
    with open(resolved_trace, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    decision_integrity = sum(1 for row in rows if row["decision_correct"]) / float(
        max(1, len(rows))
    )
    live_row = rows[0]
    live_source_backed = float(
        bool(live_row["evidence"]["signals"].get("source_backed", False))
    )
    live_update_allowed = float(
        bool(live_row["credit_result"].get("update_allowed", False))
    )
    max_event_cost = max(
        row["evidence"]["event_cost"] + row["credit_result"]["event_cost"]
        for row in rows
    )
    passed = bool(
        decision_integrity == 1.0
        and live_source_backed == 1.0
        and live_update_allowed == 1.0
        and max_event_cost <= 64
    )
    return {
        "schema": "sara-resonance-credit-integration-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "case_count": len(rows),
        "metrics": {
            "decision_integrity": decision_integrity,
            "live_source_backed_integrity": live_source_backed,
            "live_update_allowed_integrity": live_update_allowed,
            "integration_freeze_case_count": sum(
                1 for row in rows if not row["credit_result"]["update_allowed"]
            ),
            "max_combined_event_cost": max_event_cost,
        },
        "source_paths": {key: os.path.abspath(value) for key, value in source_paths.items()},
        "rows": rows,
        "trace_path": os.path.abspath(trace_path),
        "policy_notes": [
            "Signals are derived from managed SARA evaluator reports, not manually assigned.",
            "Missing or failed evidence cannot be treated as source-backed.",
            "The metabolic signal is recomputed locally for deterministic standalone execution.",
            "This integration remains observed-only and does not mutate production learning.",
        ],
    }


def summarize(report: Mapping[str, Any]) -> str:
    lines = [
        f"Resonance credit integration: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
    ]
    lines.extend(
        f"- {key}: {value}" for key, value in sorted(report.get("metrics", {}).items())
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge managed SARA reports into resonance credit."
    )
    for name, default_path in DEFAULT_SOURCE_PATHS.items():
        parser.add_argument(f"--{name.replace('_', '-')}-path", default=default_path)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source_paths = {
        name: str(getattr(args, f"{name}_path")) for name in DEFAULT_SOURCE_PATHS
    }
    report = build_report(
        load_source_reports(source_paths),
        source_paths=source_paths,
        trace_path=args.trace_path,
    )
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
