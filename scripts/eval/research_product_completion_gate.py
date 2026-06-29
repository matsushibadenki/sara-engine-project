#!/usr/bin/env python3
"""Validate SARA's research-product completion surface from managed artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.core.hal import MockNeuromorphicBackend, PythonBackend  # noqa: E402
from sara_engine.evaluation.report_artifacts import artifact_state, format_artifact_state_line  # noqa: E402
from sara_engine.utils.project_paths import ensure_allowed_output_path, ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_PHASE3_REPORT_PATH = workspace_path("evaluation", "phase3_accuracy_suite.json")
DEFAULT_PHASE4_REPORT_PATH = workspace_path("evaluation", "phase4_scale_continual_benchmark.json")
DEFAULT_PHASE5_COMPLETION_REPORT_PATH = workspace_path("evaluation", "phase5_completion_gate_report.json")
DEFAULT_OPERATIONAL_REPORT_PATH = workspace_path("release", "operational_readiness_report.json")
DEFAULT_ANN_EFFICIENCY_ROADMAP_REPORT_PATH = workspace_path("evaluation", "ann_efficiency_roadmap_gate.json")
DEFAULT_SPARSE_DIFFUSION_BLOCK_REPORT_PATH = workspace_path("evaluation", "sparse_diffusion_block_readiness.json")
DEFAULT_ENERGY_MEASUREMENT_SESSION_PLAN_PATH = workspace_path("evaluation", "energy_measurement_session_plan.json")
DEFAULT_ADAPTIVE_CREDIT_FIELD_REPORT_PATH = workspace_path("evaluation", "adaptive_credit_field_benchmark.json")
DEFAULT_ADAPTIVE_CREDIT_EVENT_MEMORY_REPORT_PATH = workspace_path(
    "evaluation", "adaptive_credit_event_memory_benchmark.json"
)
DEFAULT_RUST_CORE_READINESS_REPORT_PATH = workspace_path("evaluation", "rust_core_readiness.json")
DEFAULT_RESEARCH_FIXTURE_READINESS_REPORT_PATH = workspace_path("evaluation", "research_fixture_readiness.json")
DEFAULT_AUTOBOT_GAP_LOOP_READINESS_REPORT_PATH = workspace_path("evaluation", "autobot_gap_loop_readiness.json")
DEFAULT_OUTPUT_REPORT_PATH = workspace_path("evaluation", "research_product_completion_gate_report.json")
DEFAULT_OUTPUT_SUMMARY_PATH = workspace_path("evaluation", "research_product_completion_gate_summary.txt")


def _load_module_from_path(module_name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _passed_check(details: Dict[str, Any]) -> Dict[str, Any]:
    return {"passed": True, "errors": [], "details": details}


def _failed_check(errors: List[str], details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {"passed": False, "errors": errors, "details": details or {}}


def _check_policy_text(policy_text: str) -> Dict[str, Any]:
    required = [
        "Do not make runtime learning depend on backpropagation.",
        "Do not make dense matrix operations the primary runtime design.",
        "Do not require GPUs for correctness or normal operation.",
        "Generated files must stay in managed directories",
        "Prefer `src/sara_engine/utils/project_paths.py`",
    ]
    missing = [item for item in required if item not in policy_text]
    if missing:
        return _failed_check(["policy.md is missing required research-product constraints."], {"missing": missing})
    return _passed_check({"required_constraint_count": len(required)})


def _check_roadmap_audit(roadmap_report: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    if not bool(roadmap_report.get("passed", False)):
        errors.append("ROADMAP closure audit is not passed.")
    if int(roadmap_report.get("closure_done_count", 0) or 0) < 4:
        errors.append("ROADMAP closure audit does not include all required DONE markers.")
    if int(roadmap_report.get("unchecked_marker_count", 0) or 0) > 0:
        errors.append("ROADMAP still contains unchecked completion markers.")
    if errors:
        return _failed_check(errors, dict(roadmap_report))
    return _passed_check(
        {
            "closure_done_count": int(roadmap_report.get("closure_done_count", 0) or 0),
            "candidate_line_count": int(roadmap_report.get("candidate_line_count", 0) or 0),
        }
    )


def _check_phase3_completion(phase3_report: Mapping[str, Any]) -> Dict[str, Any]:
    phase3_completion = phase3_report.get("phase3_completion", {})
    if not isinstance(phase3_completion, Mapping):
        return _failed_check(["Phase 3 report is missing phase3_completion."])
    errors: List[str] = []
    if not bool(phase3_completion.get("passed", False)):
        errors.append("Phase 3 completion did not pass.")
    if float(phase3_completion.get("completion_score", 0.0) or 0.0) < 1.0:
        errors.append("Phase 3 completion_score is below 1.0.")
    checks = phase3_completion.get("checks", {})
    if isinstance(checks, Mapping):
        failed = sorted(str(name) for name, value in checks.items() if not bool(value))
        if failed:
            errors.append("Phase 3 completion contains failed checks: " + ", ".join(failed))
    if errors:
        return _failed_check(errors, dict(phase3_completion))
    return _passed_check(
        {
            "completion_score": float(phase3_completion.get("completion_score", 0.0) or 0.0),
            "check_count": len(checks) if isinstance(checks, Mapping) else 0,
        }
    )


def _check_phase4_completion(phase4_report: Mapping[str, Any]) -> Dict[str, Any]:
    required_metrics = [
        "structural_plasticity_stability",
        "hippocampal_transfer_integrity",
        "scale_out_retention_integrity",
        "continual_drift_recovery_integrity",
    ]
    metrics = phase4_report.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    errors: List[str] = []
    if not bool(phase4_report.get("passed", False)):
        errors.append("Phase 4 benchmark did not pass.")
    missing_or_failed = [
        name for name in required_metrics if float(metrics.get(name, 0.0) or 0.0) < 1.0
    ]
    if missing_or_failed:
        errors.append("Phase 4 required metrics failed: " + ", ".join(missing_or_failed))
    quality_metrics = phase4_report.get("quality_metrics", {})
    if not isinstance(quality_metrics, Mapping):
        errors.append("Phase 4 report is missing quality_metrics.")
    if errors:
        return _failed_check(errors, {"required_metrics": required_metrics})
    return _passed_check(
        {
            "overall_score": float(phase4_report.get("overall_score", 0.0) or 0.0),
            "required_metric_count": len(required_metrics),
        }
    )


def _check_phase5_completion(phase5_report: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    if not bool(phase5_report.get("passed", False)):
        errors.append("Phase 5 completion gate did not pass.")
    failed_checks = phase5_report.get("failed_checks", [])
    if isinstance(failed_checks, list) and failed_checks:
        errors.append("Phase 5 completion failed checks: " + ", ".join(str(item) for item in failed_checks))
    check_count = int(phase5_report.get("check_count", 0) or 0)
    pass_count = int(phase5_report.get("pass_count", 0) or 0)
    if check_count > 0 and pass_count < check_count:
        errors.append(f"Phase 5 completion pass_count is incomplete ({pass_count}/{check_count}).")
    if errors:
        return _failed_check(errors, {"check_count": check_count, "pass_count": pass_count})
    return _passed_check(
        {
            "phase5_overall_score": float(phase5_report.get("phase5_overall_score", 0.0) or 0.0),
            "check_count": check_count,
            "pass_count": pass_count,
        }
    )


def _check_operational_readiness(operational_report: Mapping[str, Any]) -> Dict[str, Any]:
    required_checks = [
        "phase3_accuracy",
        "phase3_completion",
        "phase4_completion",
        "phase5_entry_gate",
        "phase5_completion_gate",
        "external_validity",
        "external_validity_ladder",
        "release_gate",
        "production_profile",
    ]
    checks = operational_report.get("checks", {})
    checks = checks if isinstance(checks, Mapping) else {}
    errors: List[str] = []
    if not bool(operational_report.get("passed", False)):
        errors.append("Operational readiness did not pass.")
    if not bool(operational_report.get("strict_production", False)):
        errors.append("Operational readiness is not marked strict_production=true.")
    failed = [
        name
        for name in required_checks
        if not isinstance(checks.get(name), Mapping) or not bool(checks.get(name, {}).get("passed", False))
    ]
    if failed:
        errors.append("Operational readiness required checks failed or are missing: " + ", ".join(failed))
    if errors:
        return _failed_check(errors, {"required_checks": required_checks})
    return _passed_check(
        {
            "readiness_score": float(operational_report.get("readiness_score", 0.0) or 0.0),
            "required_check_count": len(required_checks),
        }
    )


def _check_ann_efficiency_roadmap(roadmap_report: Mapping[str, Any]) -> Dict[str, Any]:
    stages = roadmap_report.get("stages", [])
    stages = stages if isinstance(stages, list) else []
    errors: List[str] = []
    if not bool(roadmap_report.get("passed", False)):
        errors.append("ANN efficiency roadmap gate did not pass.")
    if float(roadmap_report.get("completion_score", 0.0) or 0.0) < 1.0:
        errors.append("ANN efficiency roadmap completion_score is below 1.0.")
    failed_stages = [
        str(stage.get("name", ""))
        for stage in stages
        if isinstance(stage, Mapping) and not bool(stage.get("passed", False))
    ]
    if failed_stages:
        errors.append("ANN efficiency roadmap failed stages: " + ", ".join(failed_stages))
    stage_names = {str(stage.get("name", "")) for stage in stages if isinstance(stage, Mapping)}
    required_stages = {
        "stage_1_instrumented_sparse_proxy",
        "stage_2_limited_real_data_advantage",
        "stage_3_scale_ladder_advantage",
        "stage_4_production_regression_guard",
        "stage_5_neuromorphic_transfer_readiness",
        "stage_6_real_joule_measurement_readiness",
    }
    missing_stages = sorted(required_stages - stage_names)
    if missing_stages:
        errors.append("ANN efficiency roadmap is missing stages: " + ", ".join(missing_stages))
    if errors:
        return _failed_check(errors, dict(roadmap_report))
    return _passed_check(
        {
            "completion_score": float(roadmap_report.get("completion_score", 0.0) or 0.0),
            "stage_count": int(roadmap_report.get("stage_count", len(stages)) or 0),
            "passed_stage_count": int(roadmap_report.get("passed_stage_count", 0) or 0),
        }
    )


def _check_sparse_diffusion_block_readiness(report: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    thresholds = report.get("threshold_results", {})
    thresholds = thresholds if isinstance(thresholds, Mapping) else {}
    required_metrics = {
        "sparse_diffusion_partition_integrity": 1.0,
        "sparse_diffusion_independent_block_integrity": 1.0,
        "sparse_diffusion_denoise_accuracy": 1.0,
        "sparse_diffusion_event_cost_advantage": 2.0,
        "sparse_diffusion_block_ablation_integrity": 1.0,
        "sparse_diffusion_single_pass_recurrent_integrity": 1.0,
        "sparse_diffusion_policy_compatibility": 1.0,
    }
    errors: List[str] = []
    if str(report.get("suite_name", "")) != "SparseDiffusionBlockReadiness":
        errors.append("Sparse diffusion block readiness report has an unexpected suite name.")
    if not bool(report.get("passed", False)):
        errors.append("Sparse diffusion block readiness did not pass.")
    failed_thresholds = sorted(str(name) for name, passed in thresholds.items() if not bool(passed))
    if failed_thresholds:
        errors.append("Sparse diffusion block thresholds failed: " + ", ".join(failed_thresholds))
    missing_or_failed = [
        name
        for name, required_min in required_metrics.items()
        if float(metrics.get(name, 0.0) or 0.0) < required_min
    ]
    if missing_or_failed:
        errors.append("Sparse diffusion block required metrics failed: " + ", ".join(missing_or_failed))
    if errors:
        return _failed_check(errors, {"required_metrics": required_metrics})
    return _passed_check(
        {
            "overall_score": float(report.get("overall_score", 0.0) or 0.0),
            "block_count": int(report.get("block_count", 0) or 0),
            "required_metric_count": len(required_metrics),
        }
    )


def _check_energy_measurement_session_plan(session_plan: Mapping[str, Any]) -> Dict[str, Any]:
    planned_runs = (
        session_plan.get("planned_runs", [])
        if isinstance(session_plan.get("planned_runs"), list)
        else []
    )
    pairing_matrix = (
        session_plan.get("pairing_matrix", {})
        if isinstance(session_plan.get("pairing_matrix"), Mapping)
        else {}
    )
    errors: List[str] = []
    if str(session_plan.get("schema", "")) != "sara-energy-measurement-session-plan-v2":
        errors.append("Energy measurement session plan has an unexpected schema.")
    if str(session_plan.get("status", "")) not in {"pending_measurement", "ready_for_real_joule_claim"}:
        errors.append("Energy measurement session plan has an unexpected status.")
    planned_run_count = int(session_plan.get("planned_run_count", -1) or 0)
    if planned_run_count != len(planned_runs):
        errors.append("Energy measurement session plan planned_run_count does not match planned_runs.")
    systems = pairing_matrix.get("systems", []) if isinstance(pairing_matrix.get("systems"), list) else []
    if sorted(str(system) for system in systems) != ["ann", "sara"]:
        errors.append("Energy measurement session plan pairing matrix must include sara and ann systems.")
    if int(pairing_matrix.get("required_rows_per_task", 0) or 0) < 2:
        errors.append("Energy measurement session plan must require paired rows per task.")
    if int(pairing_matrix.get("required_paired_replicates_per_task", 0) or 0) < 3:
        errors.append("Energy measurement session plan must require repeated paired runs.")
    fairness = (
        session_plan.get("fair_comparison_contract", {})
        if isinstance(session_plan.get("fair_comparison_contract"), Mapping)
        else {}
    )
    if str(fairness.get("protocol_version", "")) != "sara-energy-fair-comparison-v2":
        errors.append("Energy measurement session plan is missing the v2 fairness contract.")
    if str(fairness.get("aggregation", "")) != "per-task median joule_per_success with MAD":
        errors.append("Energy measurement session plan must use median and MAD aggregation.")
    if str(session_plan.get("status", "")) == "pending_measurement" and planned_run_count <= 0:
        errors.append("Pending energy measurement session plan has no planned runs.")
    for index, run in enumerate(planned_runs):
        if not isinstance(run, Mapping):
            errors.append(f"Energy measurement planned run {index} is not an object.")
            continue
        system = str(run.get("system", "") or "")
        task = str(run.get("task", "") or "")
        command = str(run.get("command_template", "") or "")
        run_id_template = str(run.get("run_id_template", "") or "")
        if system not in {"sara", "ann"}:
            errors.append(f"Energy measurement planned run {index} has invalid system.")
        if not task:
            errors.append(f"Energy measurement planned run {index} is missing task.")
        if "<replicate>" not in run_id_template:
            errors.append(f"Energy measurement planned run {index} is missing replicate placeholder.")
        if "record-energy-measurement" not in command or "--source real_energy_session" not in command:
            errors.append(f"Energy measurement planned run {index} command is not a real-energy recording command.")
        for required_option in (
            "--pair-id",
            "--environment-fingerprint",
            "--task-fixture-hash",
            "--success-criterion-id",
            "--measurement-boundary",
            "--measurement-tool",
            "--trial-count",
            "--run-order",
        ):
            if required_option not in command:
                errors.append(
                    f"Energy measurement planned run {index} is missing {required_option}."
                )
    if errors:
        return _failed_check(errors, dict(session_plan))
    return _passed_check(
        {
            "status": str(session_plan.get("status", "")),
            "session_id": str(session_plan.get("session_id", "")),
            "planned_run_count": planned_run_count,
            "task_count": len(pairing_matrix.get("tasks", [])) if isinstance(pairing_matrix.get("tasks"), list) else 0,
        }
    )


def _check_rust_core_readiness(report: Mapping[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks", {})
    checks = checks if isinstance(checks, Mapping) else {}
    export_contract = report.get("export_contract", {})
    export_contract = export_contract if isinstance(export_contract, Mapping) else {}
    benchmark_report = report.get("benchmark_report", {})
    benchmark_report = benchmark_report if isinstance(benchmark_report, Mapping) else {}
    errors: List[str] = []
    if str(report.get("schema", "")) != "sara-rust-core-readiness-v1":
        errors.append("Rust core readiness report has an unexpected schema.")
    if not bool(report.get("source_readiness_passed", False)):
        errors.append("Rust core source readiness did not pass.")
    required_checks = [
        "versions_match",
        "cargo_feature_split_ready",
        "pymodule_exports_registered",
        "rust_core_comments_english",
        "batch_sdr_parallelized",
        "benchmark_report_present",
    ]
    failed = [name for name in required_checks if not bool(checks.get(name, False))]
    if failed:
        errors.append("Rust core readiness required checks failed: " + ", ".join(failed))
    cargo_test = checks.get("cargo_test_passed")
    if cargo_test is False:
        errors.append("Rust core cargo test failed in readiness report.")
    missing_exports = export_contract.get("missing_from_pymodule_registration", [])
    if isinstance(missing_exports, list) and missing_exports:
        errors.append("Rust core missing PyO3 exports: " + ", ".join(str(item) for item in missing_exports))
    if not bool(benchmark_report.get("present", False)):
        errors.append("Rust core benchmark report is missing.")
    if errors:
        return _failed_check(errors, dict(report))
    return _passed_check(
        {
            "status": str(report.get("status", "")),
            "source_readiness_passed": bool(report.get("source_readiness_passed", False)),
            "built_extension_readiness_passed": bool(report.get("built_extension_readiness_passed", False)),
            "cargo_test_passed": cargo_test,
            "benchmark_report_present": bool(benchmark_report.get("present", False)),
            "batch_sdr_parallelized": bool(checks.get("batch_sdr_parallelized", False)),
        }
    )


def _check_research_fixture_readiness(report: Mapping[str, Any]) -> Dict[str, Any]:
    coverage = report.get("coverage", {})
    coverage = coverage if isinstance(coverage, Mapping) else {}
    errors: List[str] = []
    if str(report.get("schema", "")) != "sara-research-fixture-readiness-v1":
        errors.append("Research fixture readiness report has an unexpected schema.")
    if not bool(report.get("passed", False)):
        errors.append("Research fixture readiness did not pass.")
    required_task_types = {"qa", "negative", "partial", "contrastive", "noisy", "adversarial", "delayed"}
    task_types = {
        str(item)
        for item in report.get("task_types", [])
        if isinstance(item, str)
    }
    missing_task_types = sorted(required_task_types - task_types)
    if missing_task_types:
        errors.append("Research fixtures are missing task types: " + ", ".join(missing_task_types))
    required_coverage = [
        "has_repository_safe_fixture",
        "has_noisy_case",
        "has_adversarial_case",
        "has_delayed_recall_case",
        "has_abstention_cases",
        "has_retrieval_cases",
    ]
    failed_coverage = [name for name in required_coverage if not bool(coverage.get(name, False))]
    if failed_coverage:
        errors.append("Research fixture coverage failed: " + ", ".join(failed_coverage))
    if int(report.get("case_count", 0) or 0) < 8:
        errors.append("Research fixtures must include at least 8 cases.")
    if errors:
        return _failed_check(errors, dict(report))
    return _passed_check(
        {
            "case_count": int(report.get("case_count", 0) or 0),
            "task_type_count": len(task_types),
            "fixture_path": str(report.get("fixture_path", "")),
        }
    )


def _check_autobot_gap_loop_readiness(report: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    checks = report.get("checks", {})
    checks = checks if isinstance(checks, Mapping) else {}
    errors: List[str] = []
    if str(report.get("schema", "")) != "sara-autobot-gap-loop-readiness-v1":
        errors.append("Autobot gap-loop readiness report has an unexpected schema.")
    if not bool(report.get("passed", False)):
        errors.append("Autobot gap-loop readiness did not pass.")
    required_checks = [
        "loop_report_present",
        "dataset_report_present",
        "gap_report_present",
        "enqueue_report_present",
        "collection_targets_present",
        "loop_passed",
        "accepted_materials_ready",
        "gap_material_coverage_ready",
        "gap_enqueue_ready",
        "repair_curriculum_present",
    ]
    failed = [
        name
        for name in required_checks
        if not isinstance(checks.get(name), Mapping) or not bool(checks.get(name, {}).get("passed", False))
    ]
    if failed:
        errors.append("Autobot gap-loop readiness required checks failed: " + ", ".join(failed))
    if float(metrics.get("gap_build_coverage", 0.0) or 0.0) <= 0.0:
        errors.append("Autobot gap-loop readiness has zero gap-build coverage.")
    if float(metrics.get("gap_enqueue_coverage", 0.0) or 0.0) <= 0.0:
        errors.append("Autobot gap-loop readiness has zero gap-enqueue coverage.")
    if int(metrics.get("requested_slot_count", 0) or 0) <= 0:
        errors.append("Autobot gap-loop readiness did not record any requested gap slots.")
    if errors:
        return _failed_check(errors, dict(report))
    return _passed_check(
        {
            "requested_slot_count": int(metrics.get("requested_slot_count", 0) or 0),
            "gap_build_coverage": float(metrics.get("gap_build_coverage", 0.0) or 0.0),
            "gap_enqueue_coverage": float(metrics.get("gap_enqueue_coverage", 0.0) or 0.0),
            "gap_skip_ratio": float(metrics.get("gap_skip_ratio", 0.0) or 0.0),
            "repair_curriculum_share": float(metrics.get("repair_curriculum_share", 0.0) or 0.0),
            "replay_curriculum_share": float(metrics.get("replay_curriculum_share", 0.0) or 0.0),
        }
    )


def _check_adaptive_credit_field(report: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    errors: List[str] = []
    if str(report.get("schema", "")) != "sara-adaptive-credit-field-benchmark-v1":
        errors.append("Adaptive credit field report has an unexpected schema.")
    if not bool(report.get("passed", False)):
        errors.append("Adaptive credit field benchmark did not pass.")
    if not bool(report.get("observed_only", False)):
        errors.append("Adaptive credit field benchmark must remain observed-only.")
    if float(metrics.get("decision_integrity", 0.0) or 0.0) < 1.0:
        errors.append("Adaptive credit field decision_integrity is below 1.0.")
    if float(metrics.get("harmful_update_suppression", 0.0) or 0.0) < 1.0:
        errors.append("Adaptive credit field harmful_update_suppression is below 1.0.")
    if float(metrics.get("quantized_behavior_match", 0.0) or 0.0) < 1.0:
        errors.append("Adaptive credit field quantized_behavior_match is below 1.0.")
    if errors:
        return _failed_check(errors, dict(report))
    return _passed_check(
        {
            "decision_integrity": float(metrics.get("decision_integrity", 0.0) or 0.0),
            "harmful_update_suppression": float(metrics.get("harmful_update_suppression", 0.0) or 0.0),
            "quantized_behavior_match": float(metrics.get("quantized_behavior_match", 0.0) or 0.0),
            "sparse_active_fraction_vs_naive": float(metrics.get("sparse_active_fraction_vs_naive", 0.0) or 0.0),
            "max_updated_routes": int(metrics.get("max_updated_routes", 0) or 0),
        }
    )


def _check_adaptive_credit_event_memory(report: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    errors: List[str] = []
    if str(report.get("schema", "")) != "sara-adaptive-credit-event-memory-benchmark-v1":
        errors.append("Adaptive credit/Event Memory report has an unexpected schema.")
    if not bool(report.get("passed", False)):
        errors.append("Adaptive credit/Event Memory benchmark did not pass.")
    if not bool(report.get("observed_only", False)):
        errors.append("Adaptive credit/Event Memory benchmark must remain observed-only.")
    if not bool(metrics.get("credit_strong_entry_present", False)):
        errors.append("Adaptive credit/Event Memory failed to preserve the strong supported entry.")
    if not bool(metrics.get("credit_weak_entry_evicted", False)):
        errors.append("Adaptive credit/Event Memory failed to evict the weak entry.")
    if int(metrics.get("harmful_block_preserved_count", 0) or 0) < 1:
        errors.append("Adaptive credit/Event Memory did not preserve any harmful contradiction block.")
    if errors:
        return _failed_check(errors, dict(report))
    return _passed_check(
        {
            "harmful_block_preserved_count": int(metrics.get("harmful_block_preserved_count", 0) or 0),
            "credit_strong_entry_present": bool(metrics.get("credit_strong_entry_present", False)),
            "credit_weak_entry_evicted": bool(metrics.get("credit_weak_entry_evicted", False)),
            "credit_entry_count": int(metrics.get("credit_entry_count", 0) or 0),
        }
    )


def _check_memory_operations(source_texts: Mapping[str, str]) -> Dict[str, Any]:
    fix_memory_text = source_texts.get("fix_memory", "")
    sara_cli_text = source_texts.get("sara_cli", "")
    tools_text = source_texts.get("tools", "")
    required_pairs = {
        "fix_inference_memory": fix_memory_text,
        "dry_run": fix_memory_text,
        "ensure_parent_directory": fix_memory_text,
        "fix-memory": sara_cli_text + tools_text,
        "memory_fix_report": fix_memory_text,
    }
    missing = [needle for needle, haystack in required_pairs.items() if needle not in haystack]
    if missing:
        return _failed_check(["Memory repair command surface is incomplete."], {"missing": missing})
    return _passed_check({"memory_repair_contract_terms": sorted(required_pairs.keys())})


def _check_managed_output_boundary() -> Dict[str, Any]:
    errors: List[str] = []
    try:
        ensure_allowed_output_path(workspace_path("evaluation", "research_product_completion_probe.json"))
    except ValueError as exc:
        errors.append(f"Managed workspace output was rejected unexpectedly: {exc}")
    try:
        ensure_allowed_output_path("README.md")
        errors.append("Repository-root output path was accepted unexpectedly.")
    except ValueError:
        pass
    if errors:
        return _failed_check(errors)
    return _passed_check({"root_output_rejected": True, "workspace_output_accepted": True})


def _check_neuromorphic_hal_smoke() -> Dict[str, Any]:
    weights = [
        {0: 0.75, 1: 0.25},
        {1: 0.75},
    ]
    python_backend = PythonBackend()
    mock_backend = MockNeuromorphicBackend()
    python_backend.set_weights(weights)
    mock_backend.set_weights(weights)
    python_output = python_backend.propagate([0, 1], threshold=0.5, max_out=2)
    mock_output = mock_backend.propagate([0, 1], threshold=0.5, max_out=2)
    mapping_report = mock_backend.mapping_report()
    if mock_output != python_output:
        return _failed_check(
            ["Neuromorphic HAL smoke output diverged from PythonBackend."],
            {"python_output": python_output, "mock_output": mock_output, "mapping_report": mapping_report},
        )
    if mapping_report.get("last_event_cost", 0.0) <= 0.0:
        return _failed_check(["Neuromorphic HAL did not report event cost."], mapping_report)
    return _passed_check(
        {
            "backend": mock_backend.get_name(),
            "output": mock_output,
            "mapping_report": mapping_report,
        }
    )


def build_research_product_completion_report(
    *,
    policy_text: str,
    roadmap_report: Mapping[str, Any],
    phase3_report: Mapping[str, Any],
    phase4_report: Mapping[str, Any],
    phase5_completion_report: Mapping[str, Any],
    operational_report: Mapping[str, Any],
    ann_efficiency_roadmap_report: Mapping[str, Any],
    sparse_diffusion_block_report: Mapping[str, Any],
    energy_measurement_session_plan: Mapping[str, Any],
    adaptive_credit_field_report: Mapping[str, Any],
    adaptive_credit_event_memory_report: Mapping[str, Any],
    rust_core_readiness_report: Mapping[str, Any],
    research_fixture_readiness_report: Mapping[str, Any],
    autobot_gap_loop_readiness_report: Mapping[str, Any],
    source_texts: Mapping[str, str],
) -> Dict[str, Any]:
    checks = {
        "policy_core_constraints": _check_policy_text(policy_text),
        "roadmap_closure_audit": _check_roadmap_audit(roadmap_report),
        "managed_output_boundary": _check_managed_output_boundary(),
        "phase3_completion": _check_phase3_completion(phase3_report),
        "phase4_completion": _check_phase4_completion(phase4_report),
        "phase5_completion": _check_phase5_completion(phase5_completion_report),
        "operational_strict_production": _check_operational_readiness(operational_report),
        "ann_efficiency_roadmap": _check_ann_efficiency_roadmap(ann_efficiency_roadmap_report),
        "sparse_diffusion_block_readiness": _check_sparse_diffusion_block_readiness(sparse_diffusion_block_report),
        "energy_measurement_session_plan": _check_energy_measurement_session_plan(energy_measurement_session_plan),
        "adaptive_credit_field": _check_adaptive_credit_field(adaptive_credit_field_report),
        "adaptive_credit_event_memory": _check_adaptive_credit_event_memory(adaptive_credit_event_memory_report),
        "rust_core_readiness": _check_rust_core_readiness(rust_core_readiness_report),
        "research_fixture_readiness": _check_research_fixture_readiness(research_fixture_readiness_report),
        "autobot_gap_loop_readiness": _check_autobot_gap_loop_readiness(autobot_gap_loop_readiness_report),
        "memory_repair_operations": _check_memory_operations(source_texts),
        "neuromorphic_hal_smoke": _check_neuromorphic_hal_smoke(),
    }
    failed_checks = [name for name, check in checks.items() if not bool(check.get("passed", False))]
    pass_count = len(checks) - len(failed_checks)
    completion_score = pass_count / max(len(checks), 1)
    return {
        "schema": "sara-research-product-completion-gate-v1",
        "passed": not failed_checks,
        "completion_score": completion_score,
        "check_count": len(checks),
        "pass_count": pass_count,
        "failed_checks": failed_checks,
        "checks": checks,
        "artifact_state": {
            "roadmap_closure_audit": artifact_state(roadmap_report),
            "phase3_accuracy_suite": artifact_state(phase3_report, pass_field=None),
            "phase4_scale_continual_benchmark": artifact_state(phase4_report),
            "phase5_completion_gate": artifact_state(phase5_completion_report),
            "operational_readiness": artifact_state(operational_report),
            "ann_efficiency_roadmap_gate": artifact_state(ann_efficiency_roadmap_report),
            "sparse_diffusion_block_readiness": artifact_state(sparse_diffusion_block_report),
            "energy_measurement_session_plan": artifact_state(
                energy_measurement_session_plan, pass_field=None
            ),
            "adaptive_credit_field": artifact_state(adaptive_credit_field_report),
            "adaptive_credit_event_memory": artifact_state(adaptive_credit_event_memory_report),
            "rust_core_readiness": artifact_state(
                rust_core_readiness_report, pass_field="source_readiness_passed"
            ),
            "research_fixture_readiness": artifact_state(research_fixture_readiness_report),
            "autobot_gap_loop_readiness": artifact_state(autobot_gap_loop_readiness_report),
        },
        "status": "complete" if not failed_checks else "needs_repair",
    }


def format_research_product_completion_summary(report: Mapping[str, Any]) -> str:
    checks = report.get("checks", {})
    checks = checks if isinstance(checks, Mapping) else {}
    artifact_state = report.get("artifact_state", {})
    artifact_state = artifact_state if isinstance(artifact_state, Mapping) else {}
    ann_check = (
        checks.get("ann_efficiency_roadmap", {})
        if isinstance(checks.get("ann_efficiency_roadmap"), Mapping)
        else {}
    )
    ann_details = (
        ann_check.get("details", {})
        if isinstance(ann_check.get("details"), Mapping)
        else {}
    )
    energy_check = (
        checks.get("energy_measurement_session_plan", {})
        if isinstance(checks.get("energy_measurement_session_plan"), Mapping)
        else {}
    )
    energy_details = (
        energy_check.get("details", {})
        if isinstance(energy_check.get("details"), Mapping)
        else {}
    )
    fixture_check = (
        checks.get("research_fixture_readiness", {})
        if isinstance(checks.get("research_fixture_readiness"), Mapping)
        else {}
    )
    fixture_details = (
        fixture_check.get("details", {})
        if isinstance(fixture_check.get("details"), Mapping)
        else {}
    )
    gap_loop_check = (
        checks.get("autobot_gap_loop_readiness", {})
        if isinstance(checks.get("autobot_gap_loop_readiness"), Mapping)
        else {}
    )
    adaptive_credit_field_check = (
        checks.get("adaptive_credit_field", {})
        if isinstance(checks.get("adaptive_credit_field"), Mapping)
        else {}
    )
    adaptive_credit_field_details = (
        adaptive_credit_field_check.get("details", {})
        if isinstance(adaptive_credit_field_check.get("details"), Mapping)
        else {}
    )
    adaptive_credit_event_memory_check = (
        checks.get("adaptive_credit_event_memory", {})
        if isinstance(checks.get("adaptive_credit_event_memory"), Mapping)
        else {}
    )
    adaptive_credit_event_memory_details = (
        adaptive_credit_event_memory_check.get("details", {})
        if isinstance(adaptive_credit_event_memory_check.get("details"), Mapping)
        else {}
    )
    gap_loop_details = (
        gap_loop_check.get("details", {})
        if isinstance(gap_loop_check.get("details"), Mapping)
        else {}
    )
    lines = [
        "# SARA Research Product Completion Gate",
        f"- passed: {bool(report.get('passed', False))}",
        f"- status: {str(report.get('status', ''))}",
        f"- completion_score: {float(report.get('completion_score', 0.0) or 0.0):.3f}",
        f"- pass_count: {int(report.get('pass_count', 0) or 0)}",
        f"- check_count: {int(report.get('check_count', 0) or 0)}",
        format_artifact_state_line(
            "- artifact_state",
            [
                ("phase6", artifact_state.get("energy_measurement_session_plan")),
                ("phase8", artifact_state.get("research_fixture_readiness")),
                ("phase7", artifact_state.get("autobot_gap_loop_readiness")),
            ],
        ),
        (
            "- phase6_energy_metrics: "
            f"status={energy_details.get('status', '')}, "
            f"session_id={energy_details.get('session_id', '')}, "
            f"planned_runs={int(energy_details.get('planned_run_count', 0) or 0)}, "
            f"task_count={int(energy_details.get('task_count', 0) or 0)}"
        ),
        (
            "- phase8_baseline_metrics: "
            f"roadmap_completion={float(ann_details.get('completion_score', 0.0) or 0.0):.3f}, "
            f"passed_stages={int(ann_details.get('passed_stage_count', 0) or 0)}/{int(ann_details.get('stage_count', 0) or 0)}, "
            f"fixture_cases={int(fixture_details.get('case_count', 0) or 0)}, "
            f"fixture_task_types={int(fixture_details.get('task_type_count', 0) or 0)}"
        ),
        (
            "- autobot_gap_loop_metrics: "
            f"requested_slots={int(gap_loop_details.get('requested_slot_count', 0) or 0)}, "
            f"build_coverage={float(gap_loop_details.get('gap_build_coverage', 0.0) or 0.0):.3f}, "
            f"enqueue_coverage={float(gap_loop_details.get('gap_enqueue_coverage', 0.0) or 0.0):.3f}, "
            f"skip_ratio={float(gap_loop_details.get('gap_skip_ratio', 0.0) or 0.0):.3f}, "
            f"repair_share={float(gap_loop_details.get('repair_curriculum_share', 0.0) or 0.0):.3f}, "
            f"replay_share={float(gap_loop_details.get('replay_curriculum_share', 0.0) or 0.0):.3f}"
        ),
        (
            "- adaptive_credit_metrics: "
            f"decision_integrity={float(adaptive_credit_field_details.get('decision_integrity', 0.0) or 0.0):.3f}, "
            f"harmful_update_suppression={float(adaptive_credit_field_details.get('harmful_update_suppression', 0.0) or 0.0):.3f}, "
            f"quantized_behavior_match={float(adaptive_credit_field_details.get('quantized_behavior_match', 0.0) or 0.0):.3f}, "
            f"sparse_active_fraction_vs_naive={float(adaptive_credit_field_details.get('sparse_active_fraction_vs_naive', 0.0) or 0.0):.3f}"
        ),
        (
            "- adaptive_credit_event_memory_metrics: "
            f"harmful_block_preserved_count={int(adaptive_credit_event_memory_details.get('harmful_block_preserved_count', 0) or 0)}, "
            f"credit_strong_entry_present={bool(adaptive_credit_event_memory_details.get('credit_strong_entry_present', False))}, "
            f"credit_weak_entry_evicted={bool(adaptive_credit_event_memory_details.get('credit_weak_entry_evicted', False))}, "
            f"credit_entry_count={int(adaptive_credit_event_memory_details.get('credit_entry_count', 0) or 0)}"
        ),
    ]
    failed_checks = report.get("failed_checks", [])
    lines.append("- failed_checks: " + (", ".join(str(item) for item in failed_checks) if failed_checks else "none"))
    if isinstance(checks, Mapping):
        for name in sorted(checks):
            check = checks[name] if isinstance(checks[name], Mapping) else {}
            lines.append(f"- {name}: {'PASS' if bool(check.get('passed', False)) else 'FAIL'}")
            for error in check.get("errors", []) if isinstance(check.get("errors", []), list) else []:
                lines.append(f"  - error: {error}")
    return "\n".join(lines) + "\n"


def _build_roadmap_report(roadmap_path: str) -> Dict[str, Any]:
    module_path = os.path.join(PROJECT_ROOT, "scripts", "eval", "roadmap_completion_audit.py")
    module = _load_module_from_path("roadmap_completion_audit", module_path)
    return module.audit_roadmap_path(Path(roadmap_path))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SARA research-product completion.")
    parser.add_argument("--policy-path", default=os.path.join(PROJECT_ROOT, "doc", "policy.md"))
    parser.add_argument("--roadmap-path", default=os.path.join(PROJECT_ROOT, "doc", "ROADMAP.md"))
    parser.add_argument("--phase3-report-path", default=DEFAULT_PHASE3_REPORT_PATH)
    parser.add_argument("--phase4-report-path", default=DEFAULT_PHASE4_REPORT_PATH)
    parser.add_argument("--phase5-completion-report-path", default=DEFAULT_PHASE5_COMPLETION_REPORT_PATH)
    parser.add_argument("--operational-report-path", default=DEFAULT_OPERATIONAL_REPORT_PATH)
    parser.add_argument("--ann-efficiency-roadmap-report-path", default=DEFAULT_ANN_EFFICIENCY_ROADMAP_REPORT_PATH)
    parser.add_argument("--sparse-diffusion-block-report-path", default=DEFAULT_SPARSE_DIFFUSION_BLOCK_REPORT_PATH)
    parser.add_argument("--energy-measurement-session-plan-path", default=DEFAULT_ENERGY_MEASUREMENT_SESSION_PLAN_PATH)
    parser.add_argument("--adaptive-credit-field-report-path", default=DEFAULT_ADAPTIVE_CREDIT_FIELD_REPORT_PATH)
    parser.add_argument(
        "--adaptive-credit-event-memory-report-path",
        default=DEFAULT_ADAPTIVE_CREDIT_EVENT_MEMORY_REPORT_PATH,
    )
    parser.add_argument("--rust-core-readiness-report-path", default=DEFAULT_RUST_CORE_READINESS_REPORT_PATH)
    parser.add_argument("--research-fixture-readiness-report-path", default=DEFAULT_RESEARCH_FIXTURE_READINESS_REPORT_PATH)
    parser.add_argument("--autobot-gap-loop-readiness-report-path", default=DEFAULT_AUTOBOT_GAP_LOOP_READINESS_REPORT_PATH)
    parser.add_argument("--output-report-path", default=DEFAULT_OUTPUT_REPORT_PATH)
    parser.add_argument("--output-summary-path", default=DEFAULT_OUTPUT_SUMMARY_PATH)
    args = parser.parse_args(argv)

    required_paths = [
        args.policy_path,
        args.roadmap_path,
        args.phase3_report_path,
        args.phase4_report_path,
        args.phase5_completion_report_path,
        args.operational_report_path,
        args.ann_efficiency_roadmap_report_path,
        args.sparse_diffusion_block_report_path,
        args.energy_measurement_session_plan_path,
        args.adaptive_credit_field_report_path,
        args.adaptive_credit_event_memory_report_path,
        args.rust_core_readiness_report_path,
        args.research_fixture_readiness_report_path,
        args.autobot_gap_loop_readiness_report_path,
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        print("Research product completion gate failed: missing artifacts")
        for path in missing_paths:
            print(f"- {path}")
        return 1

    report = build_research_product_completion_report(
        policy_text=Path(args.policy_path).read_text(encoding="utf-8"),
        roadmap_report=_build_roadmap_report(args.roadmap_path),
        phase3_report=_load_json(args.phase3_report_path),
        phase4_report=_load_json(args.phase4_report_path),
        phase5_completion_report=_load_json(args.phase5_completion_report_path),
        operational_report=_load_json(args.operational_report_path),
        ann_efficiency_roadmap_report=_load_json(args.ann_efficiency_roadmap_report_path),
        sparse_diffusion_block_report=_load_json(args.sparse_diffusion_block_report_path),
        energy_measurement_session_plan=_load_json(args.energy_measurement_session_plan_path),
        adaptive_credit_field_report=_load_json(args.adaptive_credit_field_report_path),
        adaptive_credit_event_memory_report=_load_json(args.adaptive_credit_event_memory_report_path),
        rust_core_readiness_report=_load_json(args.rust_core_readiness_report_path),
        research_fixture_readiness_report=_load_json(args.research_fixture_readiness_report_path),
        autobot_gap_loop_readiness_report=_load_json(args.autobot_gap_loop_readiness_report_path),
        source_texts={
            "fix_memory": Path(os.path.join(PROJECT_ROOT, "scripts", "utils", "fix_memory.py")).read_text(encoding="utf-8"),
            "sara_cli": Path(os.path.join(PROJECT_ROOT, "scripts", "sara_cli.py")).read_text(encoding="utf-8"),
            "tools": Path(os.path.join(PROJECT_ROOT, "doc", "TOOLS.md")).read_text(encoding="utf-8"),
        },
    )
    output_report_path = ensure_parent_directory(args.output_report_path)
    output_summary_path = ensure_parent_directory(args.output_summary_path)
    with open(output_report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with open(output_summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_research_product_completion_summary(report))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
