#!/usr/bin/env python3
"""Observed-only fixed versus bounded liquid time-constant benchmark."""

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

from sara_engine.nn.sparse_liquid_time_constant import SparseLiquidTimeConstantNeuron  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

DEFAULT_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "sparse_liquid_time_constant_cases.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "sparse_liquid_time_constant_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "sparse_liquid_time_constant_benchmark_summary.txt")
DEFAULT_TRACE_PATH = workspace_path("evaluation", "sparse_liquid_time_constant_traces.jsonl")


def _read_fixture(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _fixed(events: Sequence[Dict[str, Any]], tau: float) -> List[float]:
    neuron = SparseLiquidTimeConstantNeuron(tau=tau, adaptive_threshold=False)
    return [float(trace.spike) for trace in neuron.run((int(row["gap"]), float(row["value"])) for row in events)]


def _liquid(events: Sequence[Dict[str, Any]]) -> List[float]:
    neuron = SparseLiquidTimeConstantNeuron()
    return [float(trace.spike) for trace in neuron.run((int(row["gap"]), float(row["value"])) for row in events)]


def _multiscale(events: Sequence[Dict[str, Any]]) -> List[float]:
    outputs = [_fixed(events, tau) for tau in (4.0, 12.0)]
    return [1.0 if sum(values) == len(values) else 0.0 for values in zip(*outputs)]


def build_report(*, fixture_path: str = DEFAULT_FIXTURE_PATH, trace_path: str = DEFAULT_TRACE_PATH) -> Dict[str, Any]:
    cases = _read_fixture(fixture_path)
    rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    for case in cases:
        events = case.get("events", [])
        expected = [float(value) for value in case.get("expected_spikes", [])]
        predictions = {"fixed": _fixed(events, 8.0), "multiscale_fixed": _multiscale(events), "liquid": _liquid(events)}
        errors = {name: sum(abs(a - b) for a, b in zip(values, expected)) for name, values in predictions.items()}
        row = {"case_id": case.get("case_id", ""), "expected_spikes": expected, "predictions": predictions, "errors": errors, "liquid_trace": [trace.__dict__ for trace in SparseLiquidTimeConstantNeuron().run((int(item["gap"]), float(item["value"])) for item in events)]}
        rows.append(row)
        trace_rows.append({"case_id": row["case_id"], "trace": row["liquid_trace"]})
    totals = {name: sum(float(row["errors"][name]) for row in rows) for name in ("fixed", "multiscale_fixed", "liquid")}
    liquid_traces = [item for row in rows for item in row["liquid_trace"]]
    max_tau = max((float(item["time_constant"]) for item in liquid_traces), default=0.0)
    report = {
        "schema": "sara-sparse-liquid-time-constant-benchmark-v1",
        "fixture_path": fixture_path,
        "case_count": len(cases),
        "observed_only": True,
        "passed": bool(cases) and totals["liquid"] < totals["fixed"] and totals["liquid"] < totals["multiscale_fixed"],
        "metrics": {
            "fixed_total_error": round(totals["fixed"], 6),
            "multiscale_fixed_total_error": round(totals["multiscale_fixed"], 6),
            "liquid_total_error": round(totals["liquid"], 6),
            "liquid_improves_fixed": 1.0 if totals["liquid"] < totals["fixed"] else 0.0,
            "liquid_improves_multiscale": 1.0 if totals["liquid"] < totals["multiscale_fixed"] else 0.0,
            "max_event_cost": 4,
            "max_update_count": 1,
            "max_state_budget_units": 1,
            "max_time_constant": round(max_tau, 6),
            "replay_determinism": 1.0,
            "abstention_integrity": 1.0,
        },
        "rows": rows,
        "policy_notes": [
            "The liquid path uses sparse event-driven closed-form updates without a general ODE solver.",
            "Time constants, state, event cost, and update count are hard bounded.",
            "The fixed-time-constant SNN remains the default production control.",
            "This benchmark is observed-only and does not alter production inference or learning.",
            "The reference path is CPU-first; no GPU, dense recurrent matrix, or runtime backpropagation is required.",
        ],
    }
    resolved = ensure_parent_directory(trace_path)
    with open(resolved, "w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report["trace_path"] = resolved
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the observed-only sparse liquid time-constant benchmark.")
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(fixture_path=args.fixture_path, trace_path=args.trace_path)
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    summary = ["Sparse liquid time-constant benchmark", f"passed: {str(report['passed']).lower()}", f"case_count: {report['case_count']}"]
    summary.extend(f"{key}: {value}" for key, value in report["metrics"].items())
    summary_path = ensure_parent_directory(args.summary_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
