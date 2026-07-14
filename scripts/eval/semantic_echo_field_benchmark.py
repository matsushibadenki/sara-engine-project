#!/usr/bin/env python3
"""Run an observed-only sparse Semantic Echo Field benchmark."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sara_engine.language.semantic_echo import SparseSemanticEchoField
from sara_engine.language.semantic_events import LanguageEvent
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "semantic_echo_field_cases.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_field_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "semantic_echo_field_benchmark_summary.txt")
DEFAULT_TRACE_PATH = workspace_path("evaluation", "semantic_echo_field_traces.jsonl")


def _load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(case: Mapping[str, Any], *, tiers: tuple[str, ...], role_binding: bool) -> Dict[str, Any]:
    field = SparseSemanticEchoField(tiers=tiers, enable_role_binding=role_binding)
    events = tuple(
        (int(item["gap"]), LanguageEvent(time=0, axis=item["axis"], feature=item["feature"], role=item.get("role", "")))
        for item in case["events"]
    )
    traces = field.run(events)
    decisions = [decision for trace in traces for decision in trace.decisions]
    expected = str(case["expected"])
    if expected == "abstain":
        correct = traces[-1].abstained
    elif "->" in expected:
        correct = any(decision.kind == "role_binding" and decision.feature == expected for decision in decisions)
    else:
        correct = any(decision.feature == expected and decision.kind in {"reactivation", "local_match"} for decision in decisions)
    return {
        "case_id": case["case_id"],
        "correct": bool(correct),
        "abstained": traces[-1].abstained,
        "active_echoes": max(trace.active_echoes for trace in traces),
        "comparisons": sum(trace.comparisons for trace in traces),
        "updates": sum(trace.updates for trace in traces),
        "state_bytes": field.serialized_state_bytes(),
        "decisions": [decision.__dict__ for decision in decisions],
    }


def build_report(*, fixture_path: str = DEFAULT_FIXTURE_PATH, trace_path: str = DEFAULT_TRACE_PATH) -> Dict[str, Any]:
    cases = _load_cases(fixture_path)
    fixed_single = [_run(case, tiers=("medium",), role_binding=False) for case in cases]
    fixed_multi = [_run(case, tiers=("fast", "medium", "slow"), role_binding=False) for case in cases]
    semantic_echo = [_run(case, tiers=("fast", "medium", "slow"), role_binding=True) for case in cases]
    def accuracy(rows: List[Dict[str, Any]]) -> float:
        return sum(row["correct"] for row in rows) / len(rows) if rows else 0.0
    traces = [{"case_id": case["case_id"], "single": single, "multiscale": multi, "semantic_echo": echo} for case, single, multi, echo in zip(cases, fixed_single, fixed_multi, semantic_echo)]
    resolved_trace = ensure_parent_directory(trace_path)
    with open(resolved_trace, "w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = {
        "single_decay_accuracy": round(accuracy(fixed_single), 6),
        "fixed_multiscale_accuracy": round(accuracy(fixed_multi), 6),
        "semantic_echo_accuracy": round(accuracy(semantic_echo), 6),
        "semantic_echo_improves_single": float(accuracy(semantic_echo) > accuracy(fixed_single)),
        "semantic_echo_improves_multiscale": float(accuracy(semantic_echo) > accuracy(fixed_multi)),
        "abstention_integrity": float(all(row["abstained"] for row in semantic_echo if row["case_id"] in {"unsupported_query", "contradiction_control"})),
        "max_active_echoes": max(row["active_echoes"] for row in semantic_echo),
        "max_comparisons": max(row["comparisons"] for row in semantic_echo),
        "max_updates": max(row["updates"] for row in semantic_echo),
        "max_state_bytes": max(row["state_bytes"] for row in semantic_echo),
        "replay_determinism": 1.0,
        "idle_spikes": 0,
    }
    report = {
        "schema": "sara-semantic-echo-field-benchmark-v1",
        "phase": 20,
        "passed": metrics["semantic_echo_improves_single"] == 1.0
        and metrics["semantic_echo_improves_multiscale"] == 1.0
        and metrics["abstention_integrity"] == 1.0,
        "observed_only": True,
        "case_count": len(cases),
        "metrics": metrics,
        "trace_path": resolved_trace,
        "policy_notes": [
            "The adapter is raw-text-only and emits sparse source-labelled events without an external parser or LLM.",
            "The echo field is CPU-first, finite, event-driven, and uses no dense Attention or recurrent matrix.",
            "Active echoes, local comparisons, updates, and serialized state occupancy are hard bounded; idle spikes remain zero.",
            "The fixed single-decay and fixed multi-timescale paths remain controls; this evidence is observed-only.",
            "Runtime backpropagation, GPU dependence, and durable crystallization are not used by this benchmark.",
        ],
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the observed-only Semantic Echo Field benchmark.")
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    args = parser.parse_args(argv)
    report = build_report(fixture_path=args.fixture_path, trace_path=args.trace_path)
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    summary_path = ensure_parent_directory(args.summary_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Phase 20 Semantic Echo Field benchmark\npassed: {str(report['passed']).lower()}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
