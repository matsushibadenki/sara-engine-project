#!/usr/bin/env python3
"""Validate the observed-only Phase 20 Semantic Echo Field surface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_BENCHMARK_PATH = workspace_path("evaluation", "semantic_echo_field_benchmark.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase20_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase20_completion_gate_summary.txt")


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def build_report(*, benchmark_path: str = DEFAULT_BENCHMARK_PATH) -> Dict[str, Any]:
    benchmark = _load(benchmark_path)
    metrics = benchmark.get("metrics", {}) if isinstance(benchmark, Mapping) else {}
    notes = benchmark.get("policy_notes", []) if isinstance(benchmark, Mapping) else []
    policy = " ".join(str(item).lower() for item in notes)
    checks = {
        "benchmark_present": benchmark is not None,
        "benchmark_passed": bool(benchmark and benchmark.get("passed")),
        "observed_only": bool(benchmark and benchmark.get("observed_only")),
        "case_coverage": int(benchmark.get("case_count", 0) or 0) >= 5 if benchmark else False,
        "improves_single_decay": float(metrics.get("semantic_echo_improves_single", 0.0) or 0.0) >= 1.0,
        "improves_fixed_multiscale": float(metrics.get("semantic_echo_improves_multiscale", 0.0) or 0.0) >= 1.0,
        "abstention_integrity": float(metrics.get("abstention_integrity", 0.0) or 0.0) >= 1.0,
        "idle_spikes_bounded": int(metrics.get("idle_spikes", 1) or 0) == 0,
        "echoes_bounded": int(metrics.get("max_active_echoes", 999) or 0) <= 24,
        "comparisons_bounded": int(metrics.get("max_comparisons", 999) or 0) <= 32,
        "updates_bounded": int(metrics.get("max_updates", 999) or 0) <= 3,
        "serialized_state_bounded": int(metrics.get("max_state_bytes", 999999) or 0) <= 4096,
        "sparse_cpu_policy_visible": all(term in policy for term in ("sparse", "cpu-first", "no dense", "backpropagation")),
        "external_assistance_disabled": "without an external parser or llm" in policy,
        "production_control_preserved": "fixed single-decay" in policy and "observed-only" in policy,
    }
    passed = all(checks.values())
    return {
        "schema": "sara-phase20-completion-gate-v1",
        "phase": 20,
        "phase20_complete": passed,
        "status": "phase20_complete" if passed else "phase20_incomplete",
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "evidence_path": benchmark_path,
        "promotion_rule": {"release_critical": False, "observed_only_until_independent_language_data": True, "requires_energy_and_regression_review": True},
        "what_is_not_proven": [
            "The repository fixture is not independent held-out external language evidence.",
            "Proxy event cost does not prove physical joule-per-success advantage.",
            "Durable concept crystallization and optional phonological recoding remain disabled.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 20 Semantic Echo Field evidence.")
    parser.add_argument("--benchmark-path", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(benchmark_path=args.benchmark_path)
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    summary_path = ensure_parent_directory(args.summary_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Phase 20 completion gate\nstatus: {report['status']}\n")
        handle.write("\n".join(f"- {key}: {str(value).lower()}" for key, value in report["checks"].items()) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
