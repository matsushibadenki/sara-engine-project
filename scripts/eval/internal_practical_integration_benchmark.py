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
from sara_engine.rag.rag_pipeline import SNNRAGPipeline  # noqa: E402


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


def _run_revision_uptake_case(inference_module: Any) -> Dict[str, Any]:
    """Compare bounded revision uptake with a frozen initial-state control."""
    frozen = inference_module._build_engine()
    continued = inference_module._build_engine()
    initial_sequence = [10, 20, 30]
    revised_sequence = [10, 20, 31]
    frozen.learn_sequence(initial_sequence)
    continued.learn_sequence(initial_sequence)
    initial_prediction = inference_module._predict_next_token(continued, [10, 20])
    for _ in range(2):
        continued.learn_sequence(revised_sequence)
    frozen_prediction = inference_module._predict_next_token(frozen, [10, 20])
    revised_prediction = inference_module._predict_next_token(continued, [10, 20])
    return {
        "passed": bool(
            initial_prediction == 30
            and frozen_prediction == 30
            and revised_prediction == 31
        ),
        "initial_prediction": initial_prediction,
        "frozen_prediction": frozen_prediction,
        "revised_prediction": revised_prediction,
        "revision_uptake": bool(revised_prediction == 31),
        "frozen_control_preserved": bool(frozen_prediction == 30),
    }


def _run_source_grounding_case() -> Dict[str, Any]:
    rag = SNNRAGPipeline(sdr_size=256, max_chunk_size=200)
    rag.add_document(
        "Release gate passes after pytest evidence is complete.",
        source="release_notes",
    )
    rag.add_document(
        "Release should not proceed without pytest evidence.",
        source="risk_notes",
    )
    trace = rag.query_with_rerank("release gate pytest evidence", top_k=1, candidate_k=2)
    selected = trace.get("selected", []) if isinstance(trace, Mapping) else []
    top = selected[0] if selected and isinstance(selected[0], Mapping) else {}
    metrics = trace.get("metrics", {}) if isinstance(trace, Mapping) else {}
    agreement = float(metrics.get("sparse_rag_rerank_source_agreement_observed", 0.0) or 0.0)
    return {
        "passed": bool(
            agreement == 1.0
            and str(top.get("citation_id", "") or "") == "release_notes#0"
            and float(top.get("source_agreement", 0.0) or 0.0) > 0.0
        ),
        "source_agreement": agreement,
        "citation_id": str(top.get("citation_id", "") or ""),
        "selected_source_count": int(trace.get("selected_source_count", 0) or 0),
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
    revision_uptake = _run_revision_uptake_case(inference_module)
    source_grounding = _run_source_grounding_case()

    inference_fingerprint = _inference_fingerprint(inference_first)
    migration_fingerprint = _migration_fingerprint(migration_first)
    checks = {
        "practical_task_quality": bool(inference_first.get("passed", False)),
        "continual_learning_and_drift_recovery": bool(phase4.get("passed", False)),
        "revision_uptake_against_frozen_control": bool(revision_uptake["passed"]),
        "source_grounding_and_citation": bool(source_grounding["passed"]),
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
            "revision_uptake": 1.0 if revision_uptake["revision_uptake"] else 0.0,
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
            "revision_uptake": revision_uptake,
            "source_grounding": source_grounding,
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
