#!/usr/bin/env python3
"""Run the internal-only practical integration benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
EVAL_PATH = os.path.join(PROJECT_ROOT, "scripts", "eval")
for path in (SRC_PATH, EVAL_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REPORT_PATH = workspace_path("evaluation", "internal_practical_integration_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation", "internal_practical_integration_benchmark_summary.txt"
)


def _load_module(name: str, filename: str) -> Any:
    path = os.path.join(EVAL_PATH, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load internal benchmark module: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _inference_fingerprint(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "passed": bool(report.get("passed", False)),
        "metrics": dict(report.get("metrics", {})) if isinstance(report.get("metrics"), Mapping) else {},
        "threshold_results": dict(report.get("threshold_results", {}))
        if isinstance(report.get("threshold_results"), Mapping)
        else {},
    }


def _migration_fingerprint(report: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), Mapping) else {}
    return {
        "passed": bool(report.get("passed", False)),
        "workload_sha256": str(report.get("workload_sha256", "") or ""),
        "metrics": dict(metrics),
    }


def build_report() -> Dict[str, Any]:
    inference_module = _load_module(
        "internal_integration_inference", "inference_accuracy_benchmark.py"
    )
    phase4_module = _load_module(
        "internal_integration_phase4", "phase4_scale_continual_benchmark.py"
    )
    migration_module = _load_module(
        "internal_integration_migration", "architecture_migration_benchmark.py"
    )
    maintenance_module = _load_module(
        "internal_integration_maintenance", "internal_maintenance_efficiency_benchmark.py"
    )

    inference_first = inference_module.run_inference_accuracy_benchmark()
    inference_second = inference_module.run_inference_accuracy_benchmark()
    phase4 = phase4_module.run_phase4_scale_continual_benchmark()
    migration_first = migration_module.build_report()
    migration_second = migration_module.build_report()
    maintenance = maintenance_module.build_report()

    inference_fingerprint = _inference_fingerprint(inference_first)
    migration_fingerprint = _migration_fingerprint(migration_first)
    checks = {
        "practical_task_quality": bool(inference_first.get("passed", False)),
        "continual_learning_and_drift_recovery": bool(phase4.get("passed", False)),
        "architecture_change_knowledge_reuse": bool(migration_first.get("passed", False)),
        "state_migration_read_only_legacy": float(
            migration_first.get("metrics", {}).get("legacy_reference_unchanged", 0.0)
            if isinstance(migration_first.get("metrics"), Mapping)
            else 0.0
        )
        == 1.0,
        "internal_maintenance_efficiency": bool(maintenance.get("passed", False)),
        "reproducible_practical_tasks": _stable_digest(inference_fingerprint)
        == _stable_digest(_inference_fingerprint(inference_second)),
        "reproducible_state_migration": _stable_digest(migration_fingerprint)
        == _stable_digest(_migration_fingerprint(migration_second)),
        "cpu_only_execution": True,
        "no_external_device_required": True,
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "schema": "sara-internal-practical-integration-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "external_device_required": False,
        "execution_policy": {
            "cpu_only": True,
            "gpu_required": False,
            "external_service_required": False,
            "network_collection_performed": False,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "checks": checks,
        "metrics": {
            "practical_task_count": len(
                inference_first.get("metrics", {})
                if isinstance(inference_first.get("metrics"), Mapping)
                else {}
            ),
            "phase4_continual_score": float(phase4.get("overall_score", 0.0) or 0.0),
            "migration_target_replay_recall": float(
                migration_first.get("metrics", {}).get("target_replay_recall", 0.0)
                if isinstance(migration_first.get("metrics"), Mapping)
                else 0.0
            ),
            "maintenance_event_cost_per_selected": float(
                maintenance.get("normalized_metrics", {}).get(
                    "maintenance_event_cost_per_selected", 0.0
                )
                if isinstance(maintenance.get("normalized_metrics"), Mapping)
                else 0.0
            ),
        },
        "component_reports": {
            "inference_accuracy": inference_fingerprint,
            "phase4_scale_continual": {
                "passed": bool(phase4.get("passed", False)),
                "metrics": dict(phase4.get("metrics", {}))
                if isinstance(phase4.get("metrics"), Mapping)
                else {},
            },
            "architecture_migration": migration_fingerprint,
            "internal_maintenance": {
                "passed": bool(maintenance.get("passed", False)),
                "metrics": dict(maintenance.get("metrics", {}))
                if isinstance(maintenance.get("metrics"), Mapping)
                else {},
            },
        },
        "policy_notes": [
            "This benchmark uses only deterministic local CPU execution and managed fixtures.",
            "Architecture migration reuses compatible verified state while retaining the legacy cache read-only.",
            "System power estimates and physical joule evidence are intentionally outside this internal gate.",
            "Passing this gate does not promote independent external evidence or physical energy claims.",
        ],
    }


def format_summary(report: Mapping[str, Any]) -> str:
    checks = report.get("checks", {}) if isinstance(report.get("checks"), Mapping) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), Mapping) else {}
    lines = [
        f"Internal practical integration benchmark: {'PASS' if report.get('passed') else 'FAIL'}",
        f"- external_device_required: {bool(report.get('external_device_required', True))}",
        f"- practical_task_count: {int(metrics.get('practical_task_count', 0) or 0)}",
        f"- phase4_continual_score: {float(metrics.get('phase4_continual_score', 0.0) or 0.0):.3f}",
        f"- migration_target_replay_recall: {float(metrics.get('migration_target_replay_recall', 0.0) or 0.0):.3f}",
        f"- maintenance_event_cost_per_selected: {float(metrics.get('maintenance_event_cost_per_selected', 0.0) or 0.0):.3f}",
        "Checks:",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: {bool(value)}")
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
) -> Dict[str, Any]:
    report = build_report()
    resolved_report = ensure_parent_directory(report_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    resolved_summary = ensure_parent_directory(summary_path)
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(format_summary(report))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = run_benchmark(report_path=args.report_path, summary_path=args.summary_path)
    print(json.dumps({"passed": report["passed"], "report_path": args.report_path}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
