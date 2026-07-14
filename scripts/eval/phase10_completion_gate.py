#!/usr/bin/env python3
"""Validate the managed Phase 10 Rust sparse-runtime hardening evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_READINESS_PATH = workspace_path("evaluation", "rust_core_readiness.json")
DEFAULT_BENCHMARK_PATH = workspace_path("evaluation", "rust_core_benchmark.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase10_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase10_completion_gate_summary.txt")


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _check_readiness(path: str) -> Dict[str, Any]:
    report = _load_json(path)
    errors: List[str] = []
    if report is None:
        return {"passed": False, "errors": [f"Missing or invalid Rust readiness report: {path}"]}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), Mapping) else {}
    required_checks = (
        "versions_match",
        "cargo_feature_split_ready",
        "pymodule_exports_registered",
        "rust_core_comments_english",
        "batch_sdr_parallelized",
        "python_extension_available",
        "python_exports_complete",
    )
    failed = [name for name in required_checks if not bool(checks.get(name))]
    if not bool(report.get("source_readiness_passed")):
        errors.append("Rust source readiness is not passed.")
    if not bool(report.get("built_extension_readiness_passed")):
        errors.append("Built Python extension readiness is not passed.")
    if failed:
        errors.append("Rust readiness checks failed: " + ", ".join(failed))
    cargo_test = report.get("cargo_test", {}) if isinstance(report.get("cargo_test"), Mapping) else {}
    if cargo_test.get("status") == "not_run":
        errors.append("cargo test was not run for the Phase 10 evidence cycle.")
    if not bool(cargo_test.get("passed")):
        errors.append("cargo test did not pass.")
    test_count = report.get("cargo_test_test_count")
    if not isinstance(test_count, int) or test_count <= 0:
        errors.append("cargo test did not report a meaningful positive test count.")
    return {
        "passed": not errors,
        "errors": errors,
        "test_count": test_count,
        "required_check_count": len(required_checks),
        "readiness_path": os.path.abspath(path),
    }


def _check_benchmark(path: str) -> Dict[str, Any]:
    report = _load_json(path)
    errors: List[str] = []
    if report is None:
        return {"passed": False, "errors": [f"Missing or invalid Rust benchmark report: {path}"]}
    cases = report.get("cases", []) if isinstance(report.get("cases"), list) else []
    expected = {"sdr_overlap", "sparse_propagate_threshold", "build_direct_synapses", "batch_tokens_to_sdr"}
    observed = {str(item.get("name", "")) for item in cases if isinstance(item, Mapping)}
    missing = sorted(expected - observed)
    if missing:
        errors.append("Rust benchmark cases are missing: " + ", ".join(missing))
    if not bool(report.get("rust_extension_available")):
        errors.append("Rust extension was not available for the comparison benchmark.")
    if int(report.get("comparable_case_count", 0) or 0) < len(expected):
        errors.append("Rust benchmark did not compare every required primitive.")
    if not bool(report.get("output_equivalence_passed")):
        errors.append("Rust/Python output equivalence did not pass.")
    return {
        "passed": not errors,
        "errors": errors,
        "case_count": len(cases),
        "comparable_case_count": int(report.get("comparable_case_count", 0) or 0),
        "min_speedup_vs_python": report.get("min_speedup_vs_python"),
        "benchmark_path": os.path.abspath(path),
    }


def build_report(*, readiness_path: str, benchmark_path: str) -> Dict[str, Any]:
    checks = {
        "rust_readiness": _check_readiness(readiness_path),
        "rust_python_benchmark": _check_benchmark(benchmark_path),
    }
    passed = all(bool(item.get("passed")) for item in checks.values())
    return {
        "schema": "sara-phase10-completion-gate-v1",
        "phase": 10,
        "phase10_complete": passed,
        "status": "phase10_complete" if passed else "phase10_incomplete",
        "passed": passed,
        "checks": checks,
        "policy": {
            "cpu_first": True,
            "sparse_event_runtime": True,
            "no_runtime_backpropagation": True,
            "no_dense_matrix_first_runtime": True,
            "python_fallback_remains_optional": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 10 Rust runtime hardening evidence.")
    parser.add_argument("--readiness-path", default=DEFAULT_READINESS_PATH)
    parser.add_argument("--benchmark-path", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(readiness_path=args.readiness_path, benchmark_path=args.benchmark_path)
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    lines = [
        "Phase 10 completion gate",
        f"status: {report['status']}",
        f"phase10_complete: {str(report['phase10_complete']).lower()}",
    ]
    for name, check in report["checks"].items():
        lines.append(f"{name}: {str(check['passed']).lower()}")
        for error in check.get("errors", []):
            lines.append(f"error: {error}")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
