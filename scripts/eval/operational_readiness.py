# Directory Path: scripts/eval/operational_readiness.py
# English Title: Operational Readiness Orchestrator
# Purpose/Content: Executes and validates Phase 3/4 and release soak gates, then writes a single managed operational readiness report for practical deployment decisions.

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SCRIPT_PATH not in sys.path:
    sys.path.insert(0, SCRIPT_PATH)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", "cache"))


from sara_engine.evaluation.phase3_tracking import (
    COGNITIVE_DELTA_MEMORY_METRIC_NAMES,
    COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES,
    COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES,
    COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES,
    compact_neuromorphic_profile_trend,
    extract_cognitive_delta_memory_metrics,
    extract_cognitive_linear_snn_fusion_metrics,
    extract_cognitive_manifold_trace_metrics,
    extract_cognitive_plastic_submodel_metrics,
)
from sara_engine.evaluation.stage_d_contract import (
    STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS,
    STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS,
    STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_NEXT_STEP_HINT,
    STAGE_D_DELTA_MEMORY_PROMOTION_CHECKS,
)
from sara_engine.evaluation.stage_e_contract import STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_CHECKS
from phase3_completion_gate import validate_phase3_completion
from phase4_completion_gate import validate_phase4_completion
from release_gate import validate_phase3_accuracy_report, validate_release_report
from research_automation_benchmark import (
    DEFAULT_RESEARCH_JOURNAL_PATH,
    RESEARCH_JOURNAL_ALTERNATIVE_BENCHMARK_ACTIONS,
    STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID,
    attach_alternative_probe_results_to_research_journal_entries,
    attach_remeasure_results_to_research_journal_entries,
    attach_roadmap_patch_evidence_collection_completions_to_research_journal_entries,
    attach_research_planner_task_completions_to_research_journal_entries,
    attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_entries,
    build_experiment_status_priority_plan,
    build_experiment_promotion_target_plan,
    build_roadmap_patch_suggestion,
    build_research_review_report,
    compact_research_review_report,
    load_research_journal_entries,
    summarize_completed_roadmap_patch_evidence_review,
    summarize_research_journal_entries,
    write_research_journal_entries,
)


def _load_module_from_path(module_name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_stage_d_contract() -> List[str]:
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "evaluation", "stage_d_contract.py")
    module = _load_module_from_path("sara_eval_stage_d_contract", module_path)
    return list(getattr(module, "STAGE_D_MINIMUM_METRIC_NAMES"))


def _load_phase5_contract_metrics() -> List[str]:
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "evaluation", "phase5_contract.py")
    module = _load_module_from_path("sara_eval_phase5_contract", module_path)
    return list(getattr(module, "PHASE5_ENTRY_METRIC_NAMES"))


def _load_project_paths_helpers() -> tuple[Any, Any]:
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "utils", "project_paths.py")
    module = _load_module_from_path("sara_project_paths", module_path)
    ensure_parent = getattr(module, "ensure_parent_directory", None)
    workspace = getattr(module, "workspace_path", None)
    if not callable(ensure_parent) or not callable(workspace):
        raise RuntimeError("project_paths helper is missing required callables.")
    return ensure_parent, workspace


STAGE_D_MINIMUM_METRIC_NAMES = _load_stage_d_contract()
STAGE_E_RECOVERY_REVIEW_STALE_SECONDS = 3.0 * 24.0 * 60.0 * 60.0


def _stage_d_candidate_failure_description(failure: Dict[str, Any]) -> str:
    description = str(failure.get("description", "") or "").strip()
    if description:
        return description
    check_name = str(failure.get("check", "") or "").strip()
    metric_name = str(failure.get("metric", "") or "").strip()
    if not check_name and metric_name:
        check_name = f"metric.{metric_name}"
    return str(
        STAGE_D_DELTA_MEMORY_PROMOTION_CHECKS.get(
            check_name,
            STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS.get(check_name, ""),
        )
        or ""
    )


def _stage_e_observed_candidate_failure_description(failure: Dict[str, Any]) -> str:
    description = str(failure.get("description", "") or "").strip()
    if description:
        return description
    check_name = str(failure.get("check", "") or "").strip()
    metric_name = str(failure.get("metric", "") or "").strip()
    if not check_name and metric_name:
        check_name = f"metric.{metric_name}"
    return str(STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_CHECKS.get(check_name, "") or "")


STAGE_D_STRUCTURAL_OBSERVED_METRIC_NAMES = (
    "synaptic_tag_integrity_observed",
    "synaptic_tag_importance_score_observed",
    "synaptic_tag_replay_priority_observed",
    "synaptic_tag_pruning_candidate_observed",
    "synaptic_tag_state_budget_observed",
    "memory_phase_transition_integrity_observed",
    "memory_phase_retention_protection_observed",
    "memory_phase_plasticity_guard_observed",
    "memory_phase_overfixation_guard_observed",
    "memory_phase_state_budget_observed",
    "metabolic_budget_integrity_observed",
    "plasticity_reserve_integrity_observed",
    "structural_growth_bounded_observed",
    "pruning_reason_trace_observed",
    "resource_pressure_observed",
    "sleep_consolidation_retention_observed",
    "latent_replay_noise_resilience_observed",
    "sleep_consolidation_memory_health_observed",
    "latent_replay_counterfactual_branch_observed",
    "sleep_consolidation_energy_budget_observed",
    "astro_structural_unlock_observed",
    "astro_structural_lock_observed",
    "astro_bounded_stdp_fallback_observed",
    "world_model_replay_policy_trace_observed",
    "astro_policy_state_budget_observed",
    "delta_memory_phase_retention_policy_observed",
    "delta_memory_crystal_retention_observed",
    "delta_memory_liquid_forget_observed",
    "delta_memory_astro_gate_alignment_observed",
    "delta_memory_policy_state_budget_observed",
    "delta_memory_multi_history_recall_observed",
    "delta_memory_multi_history_noise_resilience_observed",
    "delta_memory_multi_history_health_observed",
    "delta_memory_multi_history_manifold_guard_observed",
    "delta_memory_erase_write_decoupling_observed",
    "delta_memory_erase_preserves_stable_memory_observed",
    "delta_memory_write_commits_residual_observed",
)
PHASE5_ENTRY_METRIC_NAMES = _load_phase5_contract_metrics()
ensure_parent_directory, workspace_path = _load_project_paths_helpers()


DEFAULT_PHASE3_REPORT_PATH = workspace_path("evaluation", "phase3_accuracy_suite.json")
DEFAULT_PHASE4_REPORT_PATH = workspace_path("evaluation", "phase4_scale_continual_benchmark.json")
DEFAULT_PHASE5_ENTRY_GATE_REPORT_PATH = workspace_path("evaluation", "phase5_entry_gate_report.json")
DEFAULT_PHASE5_COMPLETION_GATE_REPORT_PATH = workspace_path("evaluation", "phase5_completion_gate_report.json")
DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH = workspace_path("evaluation", "real_data_external_validity.json")
DEFAULT_EXTERNAL_VALIDITY_LADDER_REPORT_PATH = workspace_path(
    "evaluation", "real_data_external_validity_ladder.json"
)
DEFAULT_ANN_EFFICIENCY_ROADMAP_REPORT_PATH = workspace_path(
    "evaluation", "ann_efficiency_roadmap_gate.json"
)
DEFAULT_SARA_ANN_COMPARISON_REPORT_PATH = workspace_path(
    "evaluation", "sara_ann_comparison_report.json"
)
DEFAULT_RELEASE_SOAK_REPORT_PATH = workspace_path("release", "release_soak_report.json")
DEFAULT_OPERATIONAL_REPORT_PATH = workspace_path("release", "operational_readiness_report.json")
DEFAULT_OPERATIONAL_SUMMARY_PATH = workspace_path("release", "operational_readiness_summary.txt")
DEFAULT_OPERATIONAL_RUNBOOK_PATH = workspace_path("release", "operational_readiness_runbook.md")
DEFAULT_OPERATIONAL_RUNBOOK_ACTIONS_PATH = workspace_path("release", "operational_readiness_runbook_actions.json")
DEFAULT_V1_RELEASE_ACTIONS_PATH = workspace_path("release", "v1_release_gate_actions.json")
DEFAULT_OPERATIONAL_REPAIR_LOG_PATH = workspace_path("release", "operational_repair_execution_log.json")
DEFAULT_OPERATIONAL_REPAIR_PLAN_PATH = workspace_path("release", "operational_repair_plan.json")
DEFAULT_OPERATIONAL_RESEARCH_JOURNAL_PATH = DEFAULT_RESEARCH_JOURNAL_PATH
DEFAULT_TOOL_VERIFICATION_TRACE_PATH = workspace_path("evaluation", "tool_verification_trace.json")


def _load_json_object(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"JSON list expected: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _load_recent_v1_actions(
    path: str,
    *,
    max_age_seconds: float,
    now_timestamp: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    snapshot = {
        "path": os.path.abspath(path),
        "configured_max_age_seconds": float(max(max_age_seconds, 0.0)),
        "loaded_count": 0,
        "accepted_count": 0,
        "rejected_stale_count": 0,
        "rejected_missing_timestamp_count": 0,
    }
    loaded_actions = _load_json_list(path)
    snapshot["loaded_count"] = len(loaded_actions)
    if max_age_seconds <= 0:
        snapshot["accepted_count"] = len(loaded_actions)
        snapshot["age_filter_active"] = False
        return loaded_actions, snapshot

    now = float(now_timestamp) if isinstance(now_timestamp, (int, float)) else time.time()
    accepted: List[Dict[str, Any]] = []
    stale_count = 0
    missing_timestamp_count = 0
    for item in loaded_actions:
        raw_generated_at = item.get("generated_at")
        if not isinstance(raw_generated_at, (int, float)):
            missing_timestamp_count += 1
            continue
        age = max(now - float(raw_generated_at), 0.0)
        if age > float(max_age_seconds):
            stale_count += 1
            continue
        accepted.append(dict(item))

    snapshot["age_filter_active"] = True
    snapshot["accepted_count"] = len(accepted)
    snapshot["rejected_stale_count"] = int(stale_count)
    snapshot["rejected_missing_timestamp_count"] = int(missing_timestamp_count)
    return accepted, snapshot


def _run_command(command: List[str]) -> Dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration = time.time() - started
    return {
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "duration_seconds": float(duration),
        "stdout_tail": str(completed.stdout or "")[-4000:],
        "stderr_tail": str(completed.stderr or "")[-4000:],
        "passed": completed.returncode == 0,
    }


def _validate_phase5_entry_gate_report(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(report.get("suite_name", "")) != "Phase5EntryGate":
        errors.append("Phase 5 entry gate report has an unexpected suite name.")
    if not bool(report.get("passed", False)):
        errors.append("Phase 5 entry gate report did not pass.")

    failed_checks = report.get("failed_checks", [])
    if isinstance(failed_checks, list) and failed_checks:
        errors.append("Phase 5 entry gate has failed checks: " + ", ".join(str(item) for item in failed_checks))
    elif not isinstance(failed_checks, list):
        errors.append("Phase 5 entry gate report is missing failed_checks.")

    gate_errors = report.get("errors", [])
    if isinstance(gate_errors, list) and gate_errors:
        errors.append("Phase 5 entry gate reported errors: " + " | ".join(str(item) for item in gate_errors))
    elif not isinstance(gate_errors, list):
        errors.append("Phase 5 entry gate report is missing errors.")

    checks = report.get("checks", {})
    if not isinstance(checks, dict):
        errors.append("Phase 5 entry gate report is missing checks.")
    else:
        failed_named_checks = sorted(
            name
            for name, check in checks.items()
            if not (isinstance(check, dict) and bool(check.get("passed", False)))
        )
        if failed_named_checks:
            errors.append("Phase 5 entry gate check map contains failed checks: " + ", ".join(failed_named_checks))
    if float(report.get("phase5_overall_score", 0.0) or 0.0) < 1.0:
        errors.append(
            "Phase 5 entry gate overall score is below the required threshold "
            f"(value={float(report.get('phase5_overall_score', 0.0) or 0.0):.3f}, required>=1.000)."
        )
    return errors


def _validate_phase5_completion_gate_report(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(report.get("suite_name", "")) != "Phase5CompletionGate":
        errors.append("Phase 5 completion gate report has an unexpected suite name.")
    if not bool(report.get("passed", False)):
        errors.append("Phase 5 completion gate report did not pass.")
    failed_checks = report.get("failed_checks", [])
    if isinstance(failed_checks, list) and failed_checks:
        errors.append(
            "Phase 5 completion gate has failed checks: " + ", ".join(str(item) for item in failed_checks)
        )
    elif not isinstance(failed_checks, list):
        errors.append("Phase 5 completion gate report is missing failed_checks.")
    gate_errors = report.get("errors", [])
    if isinstance(gate_errors, list) and gate_errors:
        errors.append("Phase 5 completion gate reported errors: " + " | ".join(str(item) for item in gate_errors))
    elif not isinstance(gate_errors, list):
        errors.append("Phase 5 completion gate report is missing errors.")
    checks = report.get("checks", {})
    if not isinstance(checks, dict):
        errors.append("Phase 5 completion gate report is missing checks.")
    else:
        required_check_names = {
            "phase5_entry_gate_passed",
            "multi_step_trace_complete",
            "counterfactual_branch_separable",
            "macro_step_reduction",
            "macro_cost_reduction",
            "subgoal_coverage_ratio",
            "micro_es_low_rank_trace_complete",
            "micro_es_fitness_improvement",
            "micro_es_event_cost_reduction",
            "micro_es_population_event_budget",
            "sparse_diffusion_block_readiness_passed",
            "sparse_diffusion.sparse_diffusion_partition_integrity",
            "sparse_diffusion.sparse_diffusion_independent_block_integrity",
            "sparse_diffusion.sparse_diffusion_denoise_accuracy",
            "sparse_diffusion.sparse_diffusion_event_cost_advantage",
            "sparse_diffusion.sparse_diffusion_block_ablation_integrity",
            "sparse_diffusion.sparse_diffusion_single_pass_recurrent_integrity",
            "sparse_diffusion.sparse_diffusion_policy_compatibility",
        }
        required_check_names.update({f"metric.{name}" for name in PHASE5_ENTRY_METRIC_NAMES})
        required_check_names.update({f"threshold.{name}" for name in PHASE5_ENTRY_METRIC_NAMES})
        missing_required_checks = sorted(name for name in required_check_names if name not in checks)
        if missing_required_checks:
            errors.append(
                "Phase 5 completion gate check map is missing required checks: "
                + ", ".join(missing_required_checks)
            )
        failed_named_checks = sorted(
            name
            for name, check in checks.items()
            if not (isinstance(check, dict) and bool(check.get("passed", False)))
        )
        if failed_named_checks:
            errors.append(
                "Phase 5 completion gate check map contains failed checks: " + ", ".join(failed_named_checks)
            )
    if float(report.get("phase5_overall_score", 0.0) or 0.0) < 1.0:
        errors.append(
            "Phase 5 completion gate overall score is below the required threshold "
            f"(value={float(report.get('phase5_overall_score', 0.0) or 0.0):.3f}, required>=1.000)."
        )
    return errors


def _extract_phase5_completion_detail_values(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    detail_check_names = [
        "macro_step_reduction",
        "macro_cost_reduction",
        "subgoal_coverage_ratio",
        "micro_es_fitness_improvement",
        "micro_es_event_cost_reduction",
        "micro_es_population_event_budget",
    ]
    values: Dict[str, Dict[str, float]] = {}
    for check_name in detail_check_names:
        check_data = checks.get(check_name, {})
        if not isinstance(check_data, dict) or not isinstance(check_data.get("details"), dict):
            continue
        detail = check_data["details"]
        if "value" not in detail:
            continue
        item = {"value": float(detail.get("value", 0.0) or 0.0)}
        if "required_min" in detail:
            item["required_min"] = float(detail.get("required_min", 0.0) or 0.0)
        if "required_gt" in detail:
            item["required_gt"] = float(detail.get("required_gt", 0.0) or 0.0)
        if "event_budget" in detail:
            item["event_budget"] = float(detail.get("event_budget", 0.0) or 0.0)
        values[check_name] = item
    return values


def _validate_external_validity_report(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(report.get("suite_name", "")) != "RealDataExternalValidity":
        errors.append("Real-data external validity report has an unexpected suite name.")
    if not bool(report.get("passed", False)):
        errors.append("Real-data external validity report did not pass.")

    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    check_details = report.get("check_details", {}) if isinstance(report.get("check_details"), dict) else {}
    thresholds = report.get("thresholds", {}) if isinstance(report.get("thresholds"), dict) else {}
    required_checks = {
        "real_data_task_count",
        "sparse_accuracy_floor",
        "sparse_matches_dense_accuracy",
        "summary_keyword_coverage_floor",
        "continual_memory_hit_rate_floor",
        "ann_cost_advantage_proxy",
        "performance_energy_ratio_proxy",
        "trend.no_regressions",
    }
    if not isinstance(checks, dict):
        errors.append("Real-data external validity report is missing checks.")
    else:
        missing_checks = sorted(required_checks.difference(checks.keys()))
        if missing_checks:
            errors.append(
                "Real-data external validity report is missing required checks: "
                + ", ".join(missing_checks)
            )
        failed_checks = sorted(name for name, passed in checks.items() if not bool(passed))
        if failed_checks:
            errors.append(
                "Real-data external validity report has failed checks: "
                + ", ".join(failed_checks)
            )
            for check_name in failed_checks:
                detail = check_details.get(check_name, {}) if isinstance(check_details.get(check_name), dict) else {}
                value = detail.get("value")
                required_min = detail.get("required_min")
                required_max = detail.get("required_max")
                if isinstance(value, (int, float)) and isinstance(required_min, (int, float)):
                    errors.append(
                        "Real-data external validity check detail: "
                        f"{check_name} value={float(value):.3f} required>={float(required_min):.3f}."
                    )
                elif isinstance(value, (int, float)) and isinstance(required_max, (int, float)):
                    errors.append(
                        "Real-data external validity check detail: "
                        f"{check_name} value={float(value):.3f} required<={float(required_max):.3f}."
                    )
    if check_details:
        return errors

    metric_thresholds = {
        "real_data_qa_accuracy": float(thresholds.get("min_real_data_qa_accuracy", 0.80) or 0.80),
        "real_data_summary_keyword_coverage": float(thresholds.get("min_summary_keyword_coverage", 0.60) or 0.60),
        "continual_memory_hit_rate": float(thresholds.get("min_continual_memory_hit_rate", 0.80) or 0.80),
        "performance_energy_ratio_proxy": float(thresholds.get("min_performance_energy_ratio_proxy", 2.0) or 2.0),
        "ann_cost_advantage_proxy": float(thresholds.get("min_ann_cost_advantage_proxy", 2.0) or 2.0),
    }
    for metric_name, threshold in metric_thresholds.items():
        value = float(metrics.get(metric_name, 0.0) or 0.0)
        if value < threshold:
            errors.append(
                "Real-data external validity metric is below threshold "
                f"({metric_name}, value={value:.3f}, required>={threshold:.3f})."
            )
    sparse_accuracy = float(metrics.get("real_data_qa_accuracy", 0.0) or 0.0)
    dense_accuracy = float(metrics.get("ann_proxy_qa_accuracy", 0.0) or 0.0)
    dense_tolerance = float(thresholds.get("dense_accuracy_tolerance", 0.05) or 0.05)
    if sparse_accuracy < max(dense_accuracy - dense_tolerance, 0.0):
        errors.append(
            "Real-data sparse retrieval accuracy trails ANN proxy beyond tolerance "
            f"(sara={sparse_accuracy:.3f}, ann_proxy={dense_accuracy:.3f}, tolerance={dense_tolerance:.3f})."
        )
    return errors


def _validate_external_validity_ladder_report(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(report.get("suite_name", "")) != "RealDataExternalValidityLadder":
        errors.append("Real-data external validity ladder report has an unexpected suite name.")
    if not bool(report.get("passed", False)):
        errors.append("Real-data external validity ladder report did not pass.")

    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    required_checks = {
        "all_profiles_passed",
        "profile_count_matches_plan",
        "scale_doc_counts_monotonic",
        "large_profile_present",
        "ann_cost_advantage_all_profiles",
        "performance_energy_ratio_all_profiles",
        "no_trend_regressions_all_profiles",
    }
    if not checks:
        errors.append("Real-data external validity ladder report is missing checks.")
    else:
        missing_checks = sorted(required_checks.difference(checks.keys()))
        if missing_checks:
            errors.append(
                "Real-data external validity ladder report is missing required checks: "
                + ", ".join(missing_checks)
            )
        failed_checks = sorted(name for name, passed in checks.items() if not bool(passed))
        if failed_checks:
            errors.append(
                "Real-data external validity ladder report has failed checks: "
                + ", ".join(failed_checks)
            )

    profile_count = int(metrics.get("profile_count", 0) or 0)
    passed_profile_count = int(metrics.get("passed_profile_count", 0) or 0)
    min_ann_advantage = float(metrics.get("min_ann_cost_advantage_proxy", 0.0) or 0.0)
    min_performance_ratio = float(metrics.get("min_performance_energy_ratio_proxy", 0.0) or 0.0)
    min_qa = float(metrics.get("min_real_data_qa_accuracy", 0.0) or 0.0)
    if profile_count < 3:
        errors.append(
            "Real-data external validity ladder has too few scale profiles "
            f"(value={profile_count}, required>=3)."
        )
    if passed_profile_count < profile_count:
        errors.append(
            "Real-data external validity ladder did not pass every profile "
            f"(passed={passed_profile_count}, total={profile_count})."
        )
    if min_qa < 0.80:
        errors.append(
            "Real-data external validity ladder QA floor is below threshold "
            f"(value={min_qa:.3f}, required>=0.800)."
        )
    if min_ann_advantage < 2.0:
        errors.append(
            "Real-data external validity ladder ANN cost advantage is below threshold "
            f"(value={min_ann_advantage:.3f}, required>=2.000)."
        )
    if min_performance_ratio < 2.0:
        errors.append(
            "Real-data external validity ladder performance-energy ratio is below threshold "
            f"(value={min_performance_ratio:.3f}, required>=2.000)."
        )
    return errors


def load_operational_repair_execution_log(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
        if isinstance(entries, list):
            return [dict(item) for item in entries if isinstance(item, dict)]
    return []


def save_operational_repair_execution_log(path: str, entries: List[Dict[str, Any]]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)
    return resolved


def _parse_repair_checks_csv(text: str) -> List[str]:
    if not text:
        return []
    return [token.strip() for token in str(text).split(",") if token.strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _finalize_pending_operational_repair_entries(
    entries: List[Dict[str, Any]],
    *,
    command: str,
    status: str,
    covered_checks: Optional[List[str]] = None,
    source: str = "manual_completion",
) -> int:
    cmd = str(command).strip()
    state = str(status).strip().lower()
    if not cmd or state not in {"success", "failed", "skipped"}:
        return 0
    checks = sorted({str(item).strip() for item in (covered_checks or []) if str(item).strip()})
    updated = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("command", "")).strip() != cmd:
            continue
        if str(entry.get("status", "")).strip().lower() != "pending":
            continue
        previous_checks = (
            {str(item).strip() for item in entry.get("covered_checks", []) if str(item).strip()}
            if isinstance(entry.get("covered_checks"), list)
            else set()
        )
        entry["status"] = state
        entry["covered_checks"] = sorted(previous_checks.union(set(checks)))
        entry["source"] = str(source).strip() or "manual_completion"
        entry["resolved_timestamp"] = time.time()
        updated += 1
    return updated


ROADMAP_PATCH_REVIEW_COMMAND = "python scripts/eval/research_automation_benchmark.py --append-journal"


def _is_roadmap_patch_review_entry(entry: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    source = str(entry.get("source", "")).strip()
    command = str(entry.get("command", "")).strip()
    checks = entry.get("covered_checks", [])
    check_set = (
        {str(item).strip() for item in checks if str(item).strip()}
        if isinstance(checks, list)
        else set()
    )
    return bool(
        "roadmap_patch_review" in source
        or command == ROADMAP_PATCH_REVIEW_COMMAND
        or "roadmap_patch_suggestion" in check_set
    )


def latest_roadmap_patch_review_decision(
    entries: List[Dict[str, Any]],
    *,
    review_generated_at: float = 0.0,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not _is_roadmap_patch_review_entry(entry):
            continue
        status = str(entry.get("status", "")).strip().lower()
        decision = str(entry.get("roadmap_patch_review_decision", "")).strip().lower()
        if status not in {"success", "skipped"} and decision not in {"approved", "rejected"}:
            continue
        resolved_at = _safe_float(
            entry.get("resolved_timestamp", entry.get("timestamp", 0.0)),
            0.0,
        )
        if review_generated_at > 0 and resolved_at > 0 and resolved_at < review_generated_at:
            continue
        candidates.append(dict(entry))
    candidates.sort(
        key=lambda item: _safe_float(
            item.get("resolved_timestamp", item.get("timestamp", 0.0)),
            0.0,
        ),
        reverse=True,
    )
    if not candidates:
        return {"available": False}
    latest = candidates[0]
    return {
        "available": True,
        "decision": str(latest.get("roadmap_patch_review_decision", "") or "").strip().lower(),
        "status": str(latest.get("status", "") or "").strip().lower(),
        "reason": str(latest.get("roadmap_patch_review_reason", "") or ""),
        "timestamp": _safe_float(
            latest.get("resolved_timestamp", latest.get("timestamp", 0.0)),
            0.0,
        ),
    }


def record_roadmap_patch_review_decision(
    entries: List[Dict[str, Any]],
    *,
    decision: str,
    reason: str = "",
    command: str = ROADMAP_PATCH_REVIEW_COMMAND,
) -> int:
    normalized = str(decision).strip().lower()
    if normalized not in {"approved", "rejected"}:
        return 0
    status = "success" if normalized == "approved" else "skipped"
    source = "roadmap_patch_review"
    checks = ["research_review", "roadmap_patch_suggestion"]
    completed = _finalize_pending_operational_repair_entries(
        entries,
        command=command,
        status=status,
        covered_checks=checks,
        source=source,
    )
    now = time.time()
    updated = 0
    if completed == 0:
        append_operational_repair_execution_entry(
            entries,
            command=command,
            status=status,
            covered_checks=checks,
            source=source,
        )
        completed = 1
    for entry in entries:
        if not isinstance(entry, dict) or not _is_roadmap_patch_review_entry(entry):
            continue
        entry_status = str(entry.get("status", "")).strip().lower()
        if entry_status != status:
            continue
        if str(entry.get("roadmap_patch_review_decision", "")).strip():
            continue
        entry["roadmap_patch_review_decision"] = normalized
        entry["roadmap_patch_review_reason"] = str(reason).strip()
        entry.setdefault("resolved_timestamp", now)
        updated += 1
    return int(max(completed, updated))


def append_operational_repair_execution_entry(
    entries: List[Dict[str, Any]],
    *,
    command: str,
    status: str,
    covered_checks: Optional[List[str]] = None,
    source: str = "manual",
) -> bool:
    cmd = str(command).strip()
    state = str(status).strip().lower()
    checks = sorted({str(item).strip() for item in (covered_checks or []) if str(item).strip()})
    if not cmd or not state:
        return False
    if state in {"success", "failed", "skipped"}:
        finalized = _finalize_pending_operational_repair_entries(
            entries,
            command=cmd,
            status=state,
            covered_checks=checks,
            source=source,
        )
        if finalized > 0:
            return True
    entries.append(
        {
            "command": cmd,
            "status": state,
            "covered_checks": checks,
            "source": str(source).strip() or "manual",
            "timestamp": time.time(),
        }
    )
    return True


def build_tool_verification_trace(
    *,
    command: str,
    status: str,
    covered_checks: Optional[List[str]] = None,
    source: str = "tool_verification",
    summary: str = "",
    stdout_excerpt: str = "",
    stderr_excerpt: str = "",
    artifact_path: str = "",
) -> Dict[str, Any]:
    cmd = str(command).strip()
    state = str(status).strip().lower()
    checks = sorted({str(item).strip() for item in (covered_checks or []) if str(item).strip()})
    passed = state in {"success", "skipped"}
    return {
        "schema": "sara-tool-verification-trace-v1",
        "command": cmd,
        "status": state,
        "passed": bool(passed),
        "covered_checks": checks,
        "source": str(source).strip() or "tool_verification",
        "summary": str(summary).strip(),
        "stdout_excerpt": str(stdout_excerpt)[:2000],
        "stderr_excerpt": str(stderr_excerpt)[:2000],
        "artifact_path": os.path.abspath(artifact_path) if artifact_path else "",
        "managed_output_policy": {
            "trace_path_managed": True,
            "unmanaged_output_allowed": False,
            "records_result_only": True,
        },
        "timestamp": time.time(),
    }


def append_tool_verification_trace(
    traces: List[Dict[str, Any]],
    repair_entries: List[Dict[str, Any]],
    *,
    command: str,
    status: str,
    covered_checks: Optional[List[str]] = None,
    source: str = "tool_verification",
    summary: str = "",
    stdout_excerpt: str = "",
    stderr_excerpt: str = "",
    artifact_path: str = "",
) -> Dict[str, Any]:
    trace = build_tool_verification_trace(
        command=command,
        status=status,
        covered_checks=covered_checks,
        source=source,
        summary=summary,
        stdout_excerpt=stdout_excerpt,
        stderr_excerpt=stderr_excerpt,
        artifact_path=artifact_path,
    )
    if not trace["command"] or not trace["status"]:
        return {"appended": False, "trace": trace}
    traces.append(trace)
    append_operational_repair_execution_entry(
        repair_entries,
        command=str(trace["command"]),
        status=str(trace["status"]),
        covered_checks=[*list(trace.get("covered_checks", [])), "tool_verification_trace"],
        source=str(trace.get("source", "") or "tool_verification"),
    )
    for entry in reversed(repair_entries):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("command", "")).strip() != str(trace["command"]):
            continue
        if str(entry.get("source", "")).strip() != str(trace["source"]):
            continue
        entry["tool_verification_trace"] = {
            "schema": trace["schema"],
            "status": trace["status"],
            "passed": trace["passed"],
            "summary": trace["summary"],
            "artifact_path": trace["artifact_path"],
        }
        break
    return {"appended": True, "trace": trace}


def load_tool_verification_traces(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        traces = payload.get("traces", [])
        if isinstance(traces, list):
            return [dict(item) for item in traces if isinstance(item, dict)]
    return []


def save_tool_verification_traces(path: str, traces: List[Dict[str, Any]]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump({"schema": "sara-tool-verification-trace-log-v1", "traces": traces}, handle, indent=2, ensure_ascii=False)
    return resolved


def expire_pending_operational_repair_entries(
    entries: List[Dict[str, Any]],
    *,
    ttl_seconds: float,
    now_timestamp: Optional[float] = None,
    source: str = "pending_ttl_timeout",
) -> int:
    if ttl_seconds <= 0:
        return 0
    now = float(now_timestamp) if isinstance(now_timestamp, (int, float)) else time.time()
    expired = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status", "")).strip().lower() != "pending":
            continue
        created_raw = entry.get("timestamp", 0.0)
        try:
            created = float(created_raw)
        except (TypeError, ValueError):
            created = 0.0
        age = max(now - created, 0.0)
        if age < float(ttl_seconds):
            continue
        entry["status"] = "timeout"
        entry["source"] = str(source).strip() or "pending_ttl_timeout"
        entry["resolved_timestamp"] = now
        entry["timeout_after_seconds"] = float(ttl_seconds)
        expired += 1
    return expired


def append_operational_iterative_next_actions_to_repair_log(
    entries: List[Dict[str, Any]],
    iterative_plan: Dict[str, Any],
) -> int:
    next_actions = (
        iterative_plan.get("next_actions", [])
        if isinstance(iterative_plan.get("next_actions"), list)
        else []
    )
    if not next_actions:
        return 0
    existing_pending = {
        str(item.get("command", "")).strip()
        for item in entries
        if isinstance(item, dict)
        and str(item.get("status", "")).strip().lower() == "pending"
        and str(item.get("source", "")).strip() == "iterative_next_action"
        and str(item.get("command", "")).strip()
    }
    appended = 0
    for action in next_actions:
        if not isinstance(action, dict):
            continue
        command = str(action.get("command", "")).strip()
        if not command or command in existing_pending:
            continue
        append_operational_repair_execution_entry(
            entries,
            command=command,
            status="pending",
            covered_checks=(
                [str(item) for item in action.get("affected_checks", []) if str(item).strip()]
                if isinstance(action.get("affected_checks"), list)
                else []
            ),
            source="iterative_next_action",
        )
        existing_pending.add(command)
        appended += 1
    return appended


def append_operational_runbook_actions_to_repair_log(
    entries: List[Dict[str, Any]],
    runbook_actions: List[Dict[str, Any]],
    *,
    max_append: int = 0,
    min_priority: str = "low",
) -> int:
    if not isinstance(runbook_actions, list) or not runbook_actions:
        return 0
    priority_rank = {"low": 0, "medium": 1, "high": 2}
    threshold = priority_rank.get(str(min_priority).strip().lower(), 0)
    budget = int(max_append) if isinstance(max_append, int) else 0
    if budget < 0:
        budget = 0
    existing_pending = {
        str(item.get("command", "")).strip()
        for item in entries
        if isinstance(item, dict)
        and str(item.get("status", "")).strip().lower() == "pending"
        and str(item.get("command", "")).strip()
    }
    appended = 0
    for action in runbook_actions:
        if budget > 0 and appended >= budget:
            break
        if not isinstance(action, dict):
            continue
        command = str(action.get("command", "")).strip()
        if not command or command in existing_pending:
            continue
        priority = str(action.get("priority", "low")).strip().lower()
        if priority_rank.get(priority, 0) < threshold:
            continue
        covered_checks = (
            [str(item) for item in action.get("affected_checks", []) if str(item).strip()]
            if isinstance(action.get("affected_checks"), list)
            else []
        )
        source = str(action.get("source", "")).strip() or "runbook_action"
        append_operational_repair_execution_entry(
            entries,
            command=command,
            status="pending",
            covered_checks=covered_checks,
            source=f"runbook_action:{source}",
        )
        existing_pending.add(command)
        appended += 1
    return appended


def append_efficiency_incident_repair_shortcut(
    entries: List[Dict[str, Any]],
    *,
    source: str = "efficiency_incident_shortcut",
) -> int:
    appended = 0
    commands_with_checks = [
        (
            "python scripts/eval/energy_efficiency_benchmark.py",
            [
                "focus.efficiency_readiness.passed",
                "energy_efficiency.performance_energy_ratio_proxy",
                "energy_efficiency.ann_cost_advantage_proxy",
                "energy_efficiency.brain_efficiency_alignment_proxy",
            ],
        ),
        (
            "python scripts/eval/phase3_accuracy_suite.py",
            [
                "phase3_accuracy",
                "phase3_completion",
                "focus.efficiency_readiness.passed",
            ],
        ),
        (
            "python scripts/eval/release_gate.py",
            [
                "release_gate",
                "focus.efficiency_readiness.passed",
            ],
        ),
    ]
    for command, checks in commands_with_checks:
        updated = append_operational_repair_execution_entry(
            entries,
            command=command,
            status="pending",
            covered_checks=checks,
            source=source,
        )
        if updated:
            appended += 1
    return appended


def append_efficiency_incident_runbook_actions(
    actions: List[Dict[str, Any]],
    *,
    source: str = "efficiency_incident_shortcut",
    priority: str = "high",
) -> int:
    appended = 0
    command_specs = [
        (
            "python scripts/eval/energy_efficiency_benchmark.py",
            [
                "focus.efficiency_readiness.passed",
                "energy_efficiency.performance_energy_ratio_proxy",
                "energy_efficiency.ann_cost_advantage_proxy",
                "energy_efficiency.brain_efficiency_alignment_proxy",
            ],
        ),
        (
            "python scripts/eval/phase3_accuracy_suite.py",
            [
                "phase3_accuracy",
                "phase3_completion",
                "focus.efficiency_readiness.passed",
            ],
        ),
        (
            "python scripts/eval/release_gate.py",
            [
                "release_gate",
                "focus.efficiency_readiness.passed",
            ],
        ),
    ]
    existing_commands = {
        str(item.get("command", "")).strip()
        for item in actions
        if isinstance(item, dict) and str(item.get("command", "")).strip()
    }
    for command, checks in command_specs:
        if command in existing_commands:
            continue
        actions.append(
            {
                "command": command,
                "priority": str(priority).strip().lower() or "high",
                "source": str(source).strip() or "efficiency_incident_shortcut",
                "affected_checks": sorted({str(item).strip() for item in checks if str(item).strip()}),
            }
        )
        existing_commands.add(command)
        appended += 1
    return appended


def _operational_repair_entry_event_timestamp(entry: Dict[str, Any]) -> float:
    raw = entry.get("resolved_timestamp", entry.get("timestamp", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _group_operational_repair_entries_by_command(
    entries: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_command: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        command = str(entry.get("command", "")).strip()
        if not command:
            continue
        by_command.setdefault(command, []).append(entry)
    return by_command


def build_operational_retry_queue_from_repair_log(
    entries: List[Dict[str, Any]],
    *,
    max_attempts: int = 2,
    cooldown_seconds: float = 0.0,
    now_timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if max_attempts < 1:
        max_attempts = 1
    if cooldown_seconds < 0:
        cooldown_seconds = 0.0
    now = float(now_timestamp) if isinstance(now_timestamp, (int, float)) else time.time()
    by_command = _group_operational_repair_entries_by_command(entries)
    queue: List[Dict[str, Any]] = []
    for command, command_entries in by_command.items():
        sorted_entries = sorted(command_entries, key=_operational_repair_entry_event_timestamp)
        latest = sorted_entries[-1]
        latest_status = str(latest.get("status", "")).strip().lower()
        if latest_status not in {"failed", "timeout", "error"}:
            continue
        attempts = sum(
            1
            for item in sorted_entries
            if str(item.get("status", "")).strip().lower() in {"failed", "timeout", "error"}
        )
        if attempts >= max_attempts:
            continue
        latest_timestamp = _operational_repair_entry_event_timestamp(latest)
        elapsed_since_latest = max(now - latest_timestamp, 0.0) if latest_timestamp > 0 else cooldown_seconds
        if cooldown_seconds > 0 and latest_timestamp > 0 and elapsed_since_latest < cooldown_seconds:
            continue
        covered_checks = (
            [str(item) for item in latest.get("covered_checks", []) if str(item).strip()]
            if isinstance(latest.get("covered_checks"), list)
            else []
        )
        queue.append(
            {
                "command": command,
                "reason": latest_status,
                "covered_checks": sorted(set(covered_checks)),
                "attempts_used": int(attempts),
                "max_attempts": int(max_attempts),
                "next_attempt": int(attempts + 1),
                "last_attempt_timestamp": float(latest_timestamp),
            }
        )
    queue.sort(key=lambda item: (str(item.get("reason", "")), str(item.get("command", ""))))
    return queue


def build_operational_retry_cooldown_blocked_from_repair_log(
    entries: List[Dict[str, Any]],
    *,
    max_attempts: int = 2,
    cooldown_seconds: float = 0.0,
    now_timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if max_attempts < 1:
        max_attempts = 1
    if cooldown_seconds <= 0:
        return []
    now = float(now_timestamp) if isinstance(now_timestamp, (int, float)) else time.time()
    by_command = _group_operational_repair_entries_by_command(entries)
    blocked: List[Dict[str, Any]] = []
    for command, command_entries in by_command.items():
        sorted_entries = sorted(command_entries, key=_operational_repair_entry_event_timestamp)
        latest = sorted_entries[-1]
        latest_status = str(latest.get("status", "")).strip().lower()
        if latest_status not in {"failed", "timeout", "error"}:
            continue
        attempts = sum(
            1
            for item in sorted_entries
            if str(item.get("status", "")).strip().lower() in {"failed", "timeout", "error"}
        )
        if attempts >= max_attempts:
            continue
        latest_timestamp = _operational_repair_entry_event_timestamp(latest)
        if latest_timestamp <= 0:
            continue
        elapsed_since_latest = max(now - latest_timestamp, 0.0)
        if elapsed_since_latest >= cooldown_seconds:
            continue
        covered_checks = (
            [str(item) for item in latest.get("covered_checks", []) if str(item).strip()]
            if isinstance(latest.get("covered_checks"), list)
            else []
        )
        blocked.append(
            {
                "command": command,
                "reason": latest_status,
                "covered_checks": sorted(set(covered_checks)),
                "attempts_used": int(attempts),
                "max_attempts": int(max_attempts),
                "next_attempt": int(attempts + 1),
                "last_attempt_timestamp": float(latest_timestamp),
                "cooldown_remaining_seconds": float(max(float(cooldown_seconds) - elapsed_since_latest, 0.0)),
            }
        )
    blocked.sort(key=lambda item: (str(item.get("reason", "")), str(item.get("command", ""))))
    return blocked


def prioritize_operational_retry_queue(
    retry_queue: List[Dict[str, Any]],
    *,
    iterative_plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not retry_queue:
        return []
    remaining_checks = set()
    if isinstance(iterative_plan, dict):
        values = iterative_plan.get("remaining_checks", [])
        if isinstance(values, list):
            remaining_checks = {str(item) for item in values if str(item).strip()}
    scored: List[Dict[str, Any]] = []
    reason_base = {"timeout": 3.0, "failed": 2.0, "error": 2.0}
    for item in retry_queue:
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        reason = str(payload.get("reason", "")).strip().lower()
        checks = payload.get("covered_checks", [])
        checks_set = {str(v) for v in checks if str(v).strip()} if isinstance(checks, list) else set()
        overlap = len(checks_set.intersection(remaining_checks))
        attempts_used = int(payload.get("attempts_used", 0) or 0)
        max_attempts = max(int(payload.get("max_attempts", 1) or 1), 1)
        pressure = float(attempts_used) / float(max_attempts)
        score = reason_base.get(reason, 1.0) + float(overlap) * 2.0 + pressure
        if score >= 5.0:
            tier = "high"
        elif score >= 3.5:
            tier = "medium"
        else:
            tier = "low"
        payload["priority_score"] = round(float(score), 3)
        payload["priority_tier"] = tier
        payload["priority_overlap_checks"] = int(overlap)
        scored.append(payload)
    scored.sort(
        key=lambda value: (
            -float(value.get("priority_score", 0.0) or 0.0),
            str(value.get("command", "")),
        )
    )
    return scored


def select_operational_retry_dispatch_batch(
    retry_queue: List[Dict[str, Any]],
    *,
    max_dispatch: int,
    min_priority_tier: str = "low",
    diversify_checks: bool = False,
    max_per_check: int = 0,
) -> Dict[str, Any]:
    priority_rank = {"low": 0, "medium": 1, "high": 2}
    normalized_tier = str(min_priority_tier).strip().lower()
    if normalized_tier not in priority_rank:
        normalized_tier = "low"
    threshold = priority_rank[normalized_tier]
    per_check_limit = int(max_per_check) if isinstance(max_per_check, int) else 0
    if per_check_limit < 0:
        per_check_limit = 0
    allowed: List[Dict[str, Any]] = []
    skipped_low_priority_commands: List[str] = []
    for item in retry_queue:
        if not isinstance(item, dict):
            continue
        tier = str(item.get("priority_tier", "low")).strip().lower()
        if priority_rank.get(tier, 0) < threshold:
            command = str(item.get("command", "")).strip()
            if command:
                skipped_low_priority_commands.append(command)
            continue
        allowed.append(item)
    selection_budget = max(int(max_dispatch), 0)
    selected: List[Dict[str, Any]] = []
    skipped_check_quota_commands: List[str] = []
    check_counts: Dict[str, int] = {}

    def _covered_checks(payload: Dict[str, Any]) -> List[str]:
        checks = payload.get("covered_checks", [])
        if not isinstance(checks, list):
            return []
        return sorted({str(value) for value in checks if str(value).strip()})

    def _violates_quota(checks: List[str]) -> bool:
        if per_check_limit <= 0:
            return False
        return any(int(check_counts.get(check, 0)) >= per_check_limit for check in checks)

    def _apply_quota(checks: List[str]) -> None:
        if per_check_limit <= 0:
            return
        for check in checks:
            check_counts[check] = int(check_counts.get(check, 0)) + 1

    if not diversify_checks:
        for item in allowed:
            if len(selected) >= selection_budget:
                break
            checks = _covered_checks(item)
            if _violates_quota(checks):
                command = str(item.get("command", "")).strip()
                if command:
                    skipped_check_quota_commands.append(command)
                continue
            selected.append(item)
            _apply_quota(checks)
    else:
        remaining = list(allowed)
        selected_checks: set[str] = set()
        while remaining and len(selected) < selection_budget:
            best_index = -1
            best_gain = -1
            for index, item in enumerate(remaining):
                checks_set = set(_covered_checks(item))
                if _violates_quota(sorted(checks_set)):
                    continue
                gain = len(checks_set.difference(selected_checks))
                if gain > best_gain:
                    best_gain = gain
                    best_index = index
            if best_index < 0:
                break
            chosen = remaining.pop(best_index)
            selected.append(chosen)
            chosen_checks = _covered_checks(chosen)
            selected_checks.update(set(chosen_checks))
            _apply_quota(chosen_checks)
        if per_check_limit > 0:
            selected_commands = {
                str(item.get("command", "")).strip()
                for item in selected
                if isinstance(item, dict) and str(item.get("command", "")).strip()
            }
            for item in allowed:
                command = str(item.get("command", "")).strip()
                if not command or command in selected_commands:
                    continue
                checks = _covered_checks(item)
                if _violates_quota(checks):
                    skipped_check_quota_commands.append(command)
    selected_unique_checks = set()
    for item in selected:
        selected_unique_checks.update(set(_covered_checks(item)))
    return {
        "min_priority_tier": normalized_tier,
        "selection_mode": "priority_diversified" if diversify_checks else "priority",
        "max_per_check": int(per_check_limit),
        "eligible_count": int(len(allowed)),
        "selected": selected,
        "selected_count": int(len(selected)),
        "selected_unique_check_count": int(len(selected_unique_checks)),
        "skipped_low_priority_commands": skipped_low_priority_commands,
        "skipped_low_priority_count": int(len(skipped_low_priority_commands)),
        "skipped_check_quota_commands": skipped_check_quota_commands,
        "skipped_check_quota_count": int(len(skipped_check_quota_commands)),
    }


def dispatch_operational_retry_queue_to_pending_with_report(
    entries: List[Dict[str, Any]],
    retry_queue: List[Dict[str, Any]],
    *,
    max_dispatch: int = 1,
) -> Dict[str, Any]:
    if max_dispatch < 0:
        max_dispatch = 0
    if max_dispatch < 1:
        return {
            "requested": int(max_dispatch),
            "candidate_count": 0,
            "dispatched": 0,
            "dispatched_commands": [],
            "skipped_pending_commands": [],
            "skipped_limit_commands": [],
        }
    dispatched_commands: List[str] = []
    skipped_pending_commands: List[str] = []
    skipped_limit_commands: List[str] = []
    existing_pending = {
        str(item.get("command", "")).strip()
        for item in entries
        if isinstance(item, dict)
        and str(item.get("status", "")).strip().lower() == "pending"
        and str(item.get("command", "")).strip()
    }
    dispatched = 0
    candidate_count = 0
    for retry in retry_queue:
        if not isinstance(retry, dict):
            continue
        command = str(retry.get("command", "")).strip()
        if not command:
            continue
        candidate_count += 1
        if command in existing_pending:
            skipped_pending_commands.append(command)
            continue
        if dispatched >= max_dispatch:
            skipped_limit_commands.append(command)
            continue
        append_operational_repair_execution_entry(
            entries,
            command=command,
            status="pending",
            covered_checks=(
                [str(item) for item in retry.get("covered_checks", []) if str(item).strip()]
                if isinstance(retry.get("covered_checks"), list)
                else []
            ),
            source="operational_retry_queue_dispatch",
        )
        existing_pending.add(command)
        dispatched_commands.append(command)
        dispatched += 1
    return {
        "requested": int(max_dispatch),
        "candidate_count": int(candidate_count),
        "dispatched": int(dispatched),
        "dispatched_commands": dispatched_commands,
        "skipped_pending_commands": skipped_pending_commands,
        "skipped_limit_commands": skipped_limit_commands,
    }


def _evaluate_operational_readiness(
    phase3_report: Dict[str, Any],
    phase4_report: Dict[str, Any],
    release_report: Dict[str, Any],
    *,
    phase5_entry_gate_report: Optional[Dict[str, Any]] = None,
    phase5_completion_gate_report: Optional[Dict[str, Any]] = None,
    external_validity_report: Optional[Dict[str, Any]] = None,
    external_validity_ladder_report: Optional[Dict[str, Any]] = None,
    execution_log: Optional[List[Dict[str, Any]]] = None,
    strict_production: bool = False,
    retry_max_attempts: int = 2,
    retry_cooldown_seconds: float = 0.0,
) -> Tuple[bool, Dict[str, Any]]:
    checks: Dict[str, Any] = {}

    phase3_accuracy_errors = validate_phase3_accuracy_report(phase3_report)
    checks["phase3_accuracy"] = {
        "passed": len(phase3_accuracy_errors) == 0,
        "errors": phase3_accuracy_errors,
    }

    phase3_completion_errors = validate_phase3_completion(phase3_report)
    checks["phase3_completion"] = {
        "passed": len(phase3_completion_errors) == 0,
        "errors": phase3_completion_errors,
    }

    phase4_completion_errors = validate_phase4_completion(phase3_report, phase4_report)
    checks["phase4_completion"] = {
        "passed": len(phase4_completion_errors) == 0,
        "errors": phase4_completion_errors,
    }

    if phase5_entry_gate_report is not None:
        phase5_gate_errors = _validate_phase5_entry_gate_report(phase5_entry_gate_report)
        checks["phase5_entry_gate"] = {
            "passed": len(phase5_gate_errors) == 0,
            "errors": phase5_gate_errors,
        }
    if phase5_completion_gate_report is not None:
        phase5_completion_errors = _validate_phase5_completion_gate_report(phase5_completion_gate_report)
        checks["phase5_completion_gate"] = {
            "passed": len(phase5_completion_errors) == 0,
            "errors": phase5_completion_errors,
            "detail_values": _extract_phase5_completion_detail_values(phase5_completion_gate_report),
        }
    if external_validity_report is not None:
        external_validity_errors = _validate_external_validity_report(external_validity_report)
        checks["external_validity"] = {
            "passed": len(external_validity_errors) == 0,
            "errors": external_validity_errors,
            "metrics": (
                external_validity_report.get("metrics", {})
                if isinstance(external_validity_report.get("metrics"), dict)
                else {}
            ),
        }
    if external_validity_ladder_report is not None:
        ladder_errors = _validate_external_validity_ladder_report(external_validity_ladder_report)
        checks["external_validity_ladder"] = {
            "passed": len(ladder_errors) == 0,
            "errors": ladder_errors,
            "metrics": (
                external_validity_ladder_report.get("metrics", {})
                if isinstance(external_validity_ladder_report.get("metrics"), dict)
                else {}
            ),
        }

    release_gate_errors = validate_release_report(release_report, skip_embedded_accuracy=True)
    checks["release_gate"] = {
        "passed": len(release_gate_errors) == 0,
        "errors": release_gate_errors,
    }

    production_profile_errors: List[str] = []
    criteria = release_report.get("criteria", {})
    if not isinstance(criteria, dict):
        criteria = {}
    if strict_production:
        shipping_ready = bool(criteria.get("shipping_ready", False))
        min_agent_turns = int(criteria.get("min_agent_turns", 0) or 0)
        min_inference_iterations = int(criteria.get("min_inference_iterations", 0) or 0)
        min_duration_seconds = float(criteria.get("min_duration_seconds", 0.0) or 0.0)
        if not shipping_ready:
            production_profile_errors.append(
                "Strict production mode requires shipping_ready=true (extended soak profile)."
            )
        if min_agent_turns < 60:
            production_profile_errors.append(
                "Strict production mode requires min_agent_turns>=60."
            )
        if min_inference_iterations < 96:
            production_profile_errors.append(
                "Strict production mode requires min_inference_iterations>=96."
            )
        if min_duration_seconds < 30.0:
            production_profile_errors.append(
                "Strict production mode requires min_duration_seconds>=30.0."
            )
    checks["production_profile"] = {
        "passed": len(production_profile_errors) == 0,
        "errors": production_profile_errors,
    }

    passed = all(bool(section.get("passed", False)) for section in checks.values())
    error_count = sum(len(section.get("errors", [])) for section in checks.values())
    passed_count = sum(1 for section in checks.values() if bool(section.get("passed", False)))
    readiness_score = float(passed_count) / max(len(checks), 1)
    stage_b_promotion = _extract_stage_b_promotion_snapshot(phase3_report, release_report)
    stage_d_readiness = _extract_stage_d_operational_snapshot(phase3_report)
    neuromorphic_profile = _extract_neuromorphic_profile_operational_snapshot(phase3_report)
    artifacts = collect_operational_readiness_artifacts(
        checks,
        stage_b_promotion=stage_b_promotion,
        stage_d_readiness=stage_d_readiness,
        neuromorphic_profile=neuromorphic_profile,
        execution_log=execution_log,
        strict_production=bool(strict_production),
        retry_max_attempts=retry_max_attempts,
        retry_cooldown_seconds=retry_cooldown_seconds,
    )
    stage_e_readiness = _extract_stage_e_operational_snapshot(phase3_report)
    phase5_entry = _extract_phase5_operational_snapshot(phase3_report)
    summary = {
        "passed": bool(passed),
        "error_count": int(error_count),
        "readiness_score": float(readiness_score),
        "strict_production": bool(strict_production),
        "checks": checks,
        "stage_b_promotion": stage_b_promotion,
        "stage_d_readiness": stage_d_readiness,
        "stage_e_readiness": stage_e_readiness,
        "phase5_entry_readiness": phase5_entry,
        "neuromorphic_profile_readiness": neuromorphic_profile,
        "recovery_actions": artifacts.get("recovery_actions", []),
        "repair_plan": artifacts.get("repair_plan", {}),
        "error_details": artifacts.get("error_details", []),
        "error_details_summary": artifacts.get("error_details_summary", {}),
        "failure_focus": artifacts.get("failure_focus", {}),
        "iterative_repair_plan": artifacts.get("iterative_repair_plan", {}),
        "repair_retry_queue": artifacts.get("repair_retry_queue", []),
        "repair_retry_queue_count": int(artifacts.get("repair_retry_queue_count", 0) or 0),
        "repair_retry_cooldown_seconds": float(artifacts.get("repair_retry_cooldown_seconds", 0.0) or 0.0),
        "repair_retry_cooldown_blocked": artifacts.get("repair_retry_cooldown_blocked", []),
        "repair_retry_cooldown_blocked_count": int(artifacts.get("repair_retry_cooldown_blocked_count", 0) or 0),
        "repair_pending_count": int(
            sum(
                1
                for item in (execution_log if isinstance(execution_log, list) else [])
                if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "pending"
            )
        ),
        "repair_timeout_count": int(
            sum(
                1
                for item in (execution_log if isinstance(execution_log, list) else [])
                if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "timeout"
            )
        ),
    }
    return passed, summary


def build_operational_research_review(
    *,
    phase3_report: Dict[str, Any],
    release_report: Dict[str, Any],
    operational_report: Dict[str, Any],
    research_journal_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    release_snapshot = dict(release_report) if isinstance(release_report, dict) else {}
    checks = operational_report.get("checks", {}) if isinstance(operational_report.get("checks"), dict) else {}
    release_gate_check = checks.get("release_gate", {}) if isinstance(checks.get("release_gate"), dict) else {}
    if "passed" not in release_snapshot and isinstance(release_gate_check, dict):
        release_snapshot["passed"] = bool(release_gate_check.get("passed", False))
    review = build_research_review_report(
        phase3_report=phase3_report,
        release_soak_report=release_snapshot,
        operational_report=operational_report,
        input_snapshots=[
            {
                "path": "embedded:operational.phase3",
                "exists": True,
                "loaded": True,
                "error": "",
            },
            {
                "path": "embedded:operational.release",
                "exists": True,
                "loaded": True,
                "error": "",
            },
            {
                "path": "embedded:operational",
                "exists": True,
                "loaded": True,
                "error": "",
            },
        ],
        generated_at=float(operational_report.get("generated_at", time.time()) or time.time()),
        require_operational_readiness=True,
        research_journal_summary=research_journal_summary,
    )
    return {
        "report": review,
        "compact": compact_research_review_report(review),
    }


def _extract_stage_b_promotion_snapshot(
    phase3_report: Dict[str, Any],
    release_report: Dict[str, Any],
) -> Dict[str, Any]:
    stage_b = (
        phase3_report.get("stage_b_readiness", {})
        if isinstance(phase3_report.get("stage_b_readiness"), dict)
        else {}
    )
    promotion_readiness = (
        stage_b.get("promotion_readiness", {})
        if isinstance(stage_b.get("promotion_readiness"), dict)
        else {}
    )
    gate_feedback = (
        release_report.get("gate_feedback", {})
        if isinstance(release_report.get("gate_feedback"), dict)
        else {}
    )
    actions = (
        gate_feedback.get("stage_b_promotion_actions", [])
        if isinstance(gate_feedback.get("stage_b_promotion_actions"), list)
        else []
    )
    rlm_actions = (
        gate_feedback.get("stage_b_rlm_observation_actions", [])
        if isinstance(gate_feedback.get("stage_b_rlm_observation_actions"), list)
        else []
    )
    rlm_readiness = (
        stage_b.get("rlm_observation_promotion_readiness", {})
        if isinstance(stage_b.get("rlm_observation_promotion_readiness"), dict)
        else {}
    )
    return {
        "stage_b_passed": bool(stage_b.get("passed", False)),
        "minimum_requirements_passed": bool(stage_b.get("minimum_requirements_passed", False)),
        "minimum_failure_count": int(stage_b.get("minimum_failure_count", 0) or 0),
        "readiness_score": float(stage_b.get("readiness_score", 0.0) or 0.0),
        "promotion_candidate_ready": bool(stage_b.get("promotion_candidate_ready", False)),
        "promotion_candidate_failure_count": int(stage_b.get("promotion_candidate_failure_count", 0) or 0),
        "promotion_candidate_promoted": bool(stage_b.get("promotion_candidate_promoted", False)),
        "promotion_consecutive_passes": int(promotion_readiness.get("consecutive_passes", 0) or 0),
        "promotion_required_streak": int(promotion_readiness.get("required_streak", 3) or 3),
        "promotion_recommended": bool(promotion_readiness.get("recommended", False)),
        "promotion_next_step_hint": str(gate_feedback.get("stage_b_promotion_next_step_hint", "") or ""),
        "promotion_actions": [str(item) for item in actions if str(item).strip()],
        "rlm_observation_candidate_ready": bool(stage_b.get("rlm_observation_candidate_ready", False)),
        "rlm_observation_candidate_failure_count": int(stage_b.get("rlm_observation_candidate_failure_count", 0) or 0),
        "rlm_observation_candidate_promoted": bool(stage_b.get("rlm_observation_candidate_promoted", False)),
        "rlm_observation_consecutive_passes": int(rlm_readiness.get("consecutive_passes", 0) or 0),
        "rlm_observation_required_streak": int(rlm_readiness.get("required_streak", 3) or 3),
        "rlm_observation_promotion_recommended": bool(rlm_readiness.get("recommended", False)),
        "rlm_observation_next_step_hint": str(gate_feedback.get("stage_b_rlm_observation_next_step_hint", "") or ""),
        "rlm_observation_actions": [str(item) for item in rlm_actions if str(item).strip()],
    }


def _extract_stage_d_operational_snapshot(phase3_report: Dict[str, Any]) -> Dict[str, Any]:
    stage_d = (
        phase3_report.get("stage_d_readiness", {})
        if isinstance(phase3_report.get("stage_d_readiness"), dict)
        else {}
    )
    metrics = stage_d.get("metrics", {}) if isinstance(stage_d.get("metrics"), dict) else {}
    component_reports = (
        phase3_report.get("component_reports", {})
        if isinstance(phase3_report.get("component_reports"), dict)
        else {}
    )
    continual_component = (
        component_reports.get("continual_consolidation", {})
        if isinstance(component_reports.get("continual_consolidation"), dict)
        else {}
    )
    continual_metrics = (
        continual_component.get("metrics", {})
        if isinstance(continual_component.get("metrics"), dict)
        else {}
    )
    minimum_failures = (
        stage_d.get("minimum_failures", [])
        if isinstance(stage_d.get("minimum_failures"), list)
        else []
    )
    acceptance_candidate_stability = (
        stage_d.get("acceptance_candidate_stability", {})
        if isinstance(stage_d.get("acceptance_candidate_stability"), dict)
        else {}
    )
    acceptance_candidates = [
        dict(item)
        for item in stage_d.get("acceptance_candidates", [])
        if isinstance(item, dict)
    ] if isinstance(stage_d.get("acceptance_candidates", []), list) else []
    acceptance_candidate_failures = [
        dict(item)
        for item in stage_d.get("acceptance_candidate_failures", [])
        if isinstance(item, dict)
    ] if isinstance(stage_d.get("acceptance_candidate_failures", []), list) else []
    if not acceptance_candidate_failures:
        acceptance_candidate_failures = [
            dict(item)
            for item in acceptance_candidates
            if not bool(item.get("ready", False))
        ]
    acceptance_candidate_actions: List[str] = []
    acceptance_candidate_next_step_hint = ""
    if bool(acceptance_candidate_stability.get("recommended", False)):
        acceptance_candidate_next_step_hint = STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_NEXT_STEP_HINT
        acceptance_candidate_actions = list(STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS)
    snapshot: Dict[str, Any] = {
        "passed": bool(stage_d.get("passed", False)),
        "minimum_requirements_passed": bool(stage_d.get("minimum_requirements_passed", False)),
        "minimum_failure_count": int(stage_d.get("minimum_failure_count", len(minimum_failures)) or 0),
        "readiness_score": float(stage_d.get("readiness_score", 0.0) or 0.0),
        "acceptance_candidate_count": int(stage_d.get("acceptance_candidate_count", 0) or 0),
        "acceptance_candidate_ready_count": int(stage_d.get("acceptance_candidate_ready_count", 0) or 0),
        "acceptance_candidates_ready": bool(stage_d.get("acceptance_candidates_ready", False)),
        "acceptance_candidate_failure_count": int(stage_d.get("acceptance_candidate_failure_count", 0) or 0),
        "acceptance_candidates": acceptance_candidates,
        "acceptance_candidate_failures": acceptance_candidate_failures,
        "acceptance_candidate_stability": (
            dict(acceptance_candidate_stability)
        ),
        "acceptance_candidate_next_step_hint": acceptance_candidate_next_step_hint,
        "acceptance_candidate_actions": acceptance_candidate_actions,
        "acceptance_candidate_action_count": int(len(acceptance_candidate_actions)),
        "delta_memory_candidate_ready": bool(stage_d.get("delta_memory_candidate_ready", False)),
        "delta_memory_candidate_failure_count": int(
            stage_d.get("delta_memory_candidate_failure_count", 0) or 0
        ),
        "delta_memory_candidate_failures": [
            dict(item)
            for item in stage_d.get("delta_memory_candidate_failures", [])
            if isinstance(item, dict)
        ] if isinstance(stage_d.get("delta_memory_candidate_failures", []), list) else [],
        "delta_memory_candidate_promoted": bool(stage_d.get("delta_memory_candidate_promoted", False)),
        "delta_memory_promotion_readiness": (
            dict(stage_d.get("delta_memory_promotion_readiness", {}))
            if isinstance(stage_d.get("delta_memory_promotion_readiness"), dict)
            else {}
        ),
        "minimum_failures": [
            dict(item) for item in minimum_failures if isinstance(item, dict)
        ],
    }
    for metric_name in STAGE_D_MINIMUM_METRIC_NAMES:
        snapshot[metric_name] = float(metrics.get(metric_name, 0.0) or 0.0)
    snapshot.update(
        {
            "manifold_continual_retention_observed": float(
                continual_metrics.get("manifold_continual_retention_observed", 0.0) or 0.0
            ),
            "manifold_trajectory_case_coverage_observed": float(
                continual_metrics.get("manifold_trajectory_case_coverage_observed", 0.0) or 0.0
            ),
            "manifold_average_case_recall_observed": float(
                continual_metrics.get("manifold_average_case_recall_observed", 0.0) or 0.0
            ),
            "manifold_scan_budget_integrity_observed": float(
                continual_metrics.get("manifold_scan_budget_integrity_observed", 0.0) or 0.0
            ),
            "manifold_indexed_candidate_integrity_observed": float(
                continual_metrics.get("manifold_indexed_candidate_integrity_observed", 0.0) or 0.0
            ),
            "manifold_index_scan_reduction_observed": float(
                continual_metrics.get("manifold_index_scan_reduction_observed", 0.0) or 0.0
            ),
            "manifold_capacity_pressure_recall_observed": float(
                continual_metrics.get("manifold_capacity_pressure_recall_observed", 0.0) or 0.0
            ),
            "manifold_capacity_pressure_scan_reduction_observed": float(
                continual_metrics.get("manifold_capacity_pressure_scan_reduction_observed", 0.0) or 0.0
            ),
            "manifold_replay_refresh_retention_observed": float(
                continual_metrics.get("manifold_replay_refresh_retention_observed", 0.0) or 0.0
            ),
            "manifold_replay_refresh_eviction_integrity_observed": float(
                continual_metrics.get("manifold_replay_refresh_eviction_integrity_observed", 0.0) or 0.0
            ),
            **{
                metric_name: float(continual_metrics.get(metric_name, 0.0) or 0.0)
                for metric_name in STAGE_D_STRUCTURAL_OBSERVED_METRIC_NAMES
            },
        }
    )
    return snapshot


def _extract_neuromorphic_profile_operational_snapshot(
    phase3_report: Dict[str, Any],
) -> Dict[str, Any]:
    component_reports = (
        phase3_report.get("component_reports", {})
        if isinstance(phase3_report.get("component_reports"), dict)
        else {}
    )
    energy_component = (
        component_reports.get("energy_efficiency", {})
        if isinstance(component_reports.get("energy_efficiency"), dict)
        else {}
    )
    metrics = (
        energy_component.get("metrics", {})
        if isinstance(energy_component.get("metrics"), dict)
        else {}
    )
    trend = (
        energy_component.get("neuromorphic_profile_trend", {})
        if isinstance(energy_component.get("neuromorphic_profile_trend"), dict)
        else {}
    )
    regression_count = int(trend.get("regression_count", 0) or 0)
    policy_change_count = int(trend.get("policy_change_count", 0) or 0)
    compact_trend = compact_neuromorphic_profile_trend(trend)
    missing_profiles = [
        str(profile)
        for profile in compact_trend.get("missing_profiles", [])
        if str(profile)
    ] if isinstance(compact_trend.get("missing_profiles", []), list) else []
    detail_items = [
        str(item)
        for item in compact_trend.get("regression_details", [])
        if str(item)
    ] if isinstance(compact_trend.get("regression_details", []), list) else []
    policy_detail_items = [
        str(item)
        for item in compact_trend.get("policy_change_details", [])
        if str(item)
    ] if isinstance(compact_trend.get("policy_change_details", []), list) else []
    if missing_profiles:
        recovery_hint = "Re-run edge neuromorphic export validation and inspect missing backend profiles."
    elif regression_count > 0:
        recovery_hint = "Re-run energy efficiency benchmark and inspect neuromorphic profile compatibility checks."
    elif policy_change_count > 0:
        recovery_hint = "Review neuromorphic adapter policy changes before release promotion."
    else:
        recovery_hint = "No neuromorphic profile repair required."
    return {
        "history_regression_observed": float(
            metrics.get("neuromorphic_profile_history_regression_observed", 0.0) or 0.0
        ),
        "profile_report_integrity_observed": float(
            metrics.get("neuromorphic_profile_report_integrity_observed", 0.0) or 0.0
        ),
        "backend_profile_compatibility_observed": float(
            metrics.get("neuromorphic_backend_profile_compatibility_observed", 0.0) or 0.0
        ),
        "stage_e_state_trace_ir_observed": float(
            metrics.get("neuromorphic_stage_e_state_trace_ir_observed", 0.0) or 0.0
        ),
        "stage_e_routing_hint_coverage_observed": float(
            metrics.get("neuromorphic_stage_e_routing_hint_coverage_observed", 0.0) or 0.0
        ),
        "stage_e_online_update_policy_observed": float(
            metrics.get("neuromorphic_stage_e_online_update_policy_observed", 0.0) or 0.0
        ),
        "stage_e_event_budget_observed": float(
            metrics.get("neuromorphic_stage_e_event_budget_observed", 0.0) or 0.0
        ),
        "trend_has_previous": bool(trend.get("has_previous", False)),
        "trend_regression_count": regression_count,
        "trend_policy_change_count": policy_change_count,
        "trend_new_profiles": [
            str(profile) for profile in trend.get("new_profiles", []) if str(profile)
        ]
        if isinstance(trend.get("new_profiles", []), list)
        else [],
        "trend_missing_profiles": missing_profiles,
        "trend_regression_details": detail_items,
        "trend_policy_change_details": policy_detail_items,
        "trend_regression_detail_line": str(
            compact_trend.get("regression_detail_line", "none") or "none"
        ),
        "trend_policy_change_detail_line": str(
            compact_trend.get("policy_change_detail_line", "none") or "none"
        ),
        "recovery_hint": recovery_hint,
    }


def _extract_stage_e_operational_snapshot(phase3_report: Dict[str, Any]) -> Dict[str, Any]:
    stage_e = (
        phase3_report.get("stage_e_readiness", {})
        if isinstance(phase3_report.get("stage_e_readiness"), dict)
        else {}
    )
    metrics = stage_e.get("metrics", {}) if isinstance(stage_e.get("metrics"), dict) else {}
    minimum_failures = (
        stage_e.get("minimum_failures", [])
        if isinstance(stage_e.get("minimum_failures"), list)
        else []
    )
    observed_acceptance_candidate_failures = (
        [
            dict(item)
            for item in stage_e.get("observed_acceptance_candidate_failures", [])
            if isinstance(item, dict)
        ]
        if isinstance(stage_e.get("observed_acceptance_candidate_failures", []), list)
        else []
    )
    if not observed_acceptance_candidate_failures and isinstance(
        stage_e.get("observed_acceptance_candidates", []), list
    ):
        observed_acceptance_candidate_failures = [
            dict(item)
            for item in stage_e.get("observed_acceptance_candidates", [])
            if isinstance(item, dict) and not bool(item.get("ready", False))
        ]
    cognitive_manifold_trace_metrics = extract_cognitive_manifold_trace_metrics(phase3_report)
    cognitive_delta_memory_metrics = extract_cognitive_delta_memory_metrics(phase3_report)
    cognitive_linear_snn_fusion_metrics = extract_cognitive_linear_snn_fusion_metrics(phase3_report)
    cognitive_plastic_submodel_metrics = extract_cognitive_plastic_submodel_metrics(phase3_report)
    linear_snn_fusion_observed_trend = (
        phase3_report.get("linear_snn_fusion_observed_trend", {})
        if isinstance(phase3_report.get("linear_snn_fusion_observed_trend"), dict)
        else {}
    )
    stage_e_architecture_integration_observed_trend = (
        phase3_report.get("stage_e_architecture_integration_observed_trend", {})
        if isinstance(phase3_report.get("stage_e_architecture_integration_observed_trend"), dict)
        else {}
    )
    return {
        "passed": bool(stage_e.get("passed", False)),
        "minimum_requirements_passed": bool(stage_e.get("minimum_requirements_passed", False)),
        "minimum_failure_count": int(stage_e.get("minimum_failure_count", len(minimum_failures)) or 0),
        "readiness_score": float(stage_e.get("readiness_score", 0.0) or 0.0),
        "observed_acceptance_candidate_count": int(
            stage_e.get("observed_acceptance_candidate_count", 0) or 0
        ),
        "observed_acceptance_candidate_ready_count": int(
            stage_e.get("observed_acceptance_candidate_ready_count", 0) or 0
        ),
        "observed_acceptance_candidates_ready": bool(
            stage_e.get("observed_acceptance_candidates_ready", False)
        ),
        "observed_acceptance_candidate_failure_count": int(
            stage_e.get("observed_acceptance_candidate_failure_count", 0) or 0
        ),
        "observed_acceptance_candidate_failures": observed_acceptance_candidate_failures,
        "observed_acceptance_candidate_consecutive_passes": int(
            stage_e.get("observed_acceptance_candidate_stability", {}).get("consecutive_passes", 0)
            if isinstance(stage_e.get("observed_acceptance_candidate_stability"), dict)
            else 0
        ),
        "observed_acceptance_candidate_required_streak": int(
            stage_e.get("observed_acceptance_candidate_stability", {}).get("required_streak", 3)
            if isinstance(stage_e.get("observed_acceptance_candidate_stability"), dict)
            else 3
        ),
        "observed_acceptance_candidate_stability_recommended": bool(
            stage_e.get("observed_acceptance_candidate_stability", {}).get("recommended", False)
            if isinstance(stage_e.get("observed_acceptance_candidate_stability"), dict)
            else False
        ),
        "common_spike_space_integrity": float(metrics.get("common_spike_space_integrity", 0.0) or 0.0),
        "temporal_compression_efficiency": float(metrics.get("temporal_compression_efficiency", 0.0) or 0.0),
        "modality_temporal_budget_integrity": float(
            metrics.get("modality_temporal_budget_integrity", 0.0) or 0.0
        ),
        "dendritic_context_gate_stability": float(metrics.get("dendritic_context_gate_stability", 0.0) or 0.0),
        "spiking_hjepa_latent_transition": float(metrics.get("spiking_hjepa_latent_transition", 0.0) or 0.0),
        "reverse_reasoning_trace_integrity": float(metrics.get("reverse_reasoning_trace_integrity", 0.0) or 0.0),
        "causal_candidate_trace_integrity": float(metrics.get("causal_candidate_trace_integrity", 0.0) or 0.0),
        "module_orchestration_integrity": float(metrics.get("module_orchestration_integrity", 0.0) or 0.0),
        "counterfactual_lane_integrity": float(metrics.get("counterfactual_lane_integrity", 0.0) or 0.0),
        "action_trace_observability": float(metrics.get("action_trace_observability", 0.0) or 0.0),
        "runtime_trace_replay_consistency": float(metrics.get("runtime_trace_replay_consistency", 0.0) or 0.0),
        **{
            metric_name: float(cognitive_manifold_trace_metrics[metric_name])
            for metric_name in COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES
        },
        **{
            metric_name: float(cognitive_delta_memory_metrics[metric_name])
            for metric_name in COGNITIVE_DELTA_MEMORY_METRIC_NAMES
        },
        "linear_snn_fusion_observed_policy": "excluded_from_score_and_release_gate",
        "linear_snn_fusion_trend_has_previous": bool(
            linear_snn_fusion_observed_trend.get("has_previous", False)
        ),
        "linear_snn_fusion_trend_regression_count": int(
            linear_snn_fusion_observed_trend.get("regression_count", 0) or 0
        ),
        "linear_snn_fusion_trend_release_gate_blocking": bool(
            linear_snn_fusion_observed_trend.get("release_gate_blocking", False)
        ),
        "architecture_integration_observed_policy": "excluded_from_score_and_release_gate",
        "architecture_integration_trend_has_previous": bool(
            stage_e_architecture_integration_observed_trend.get("has_previous", False)
        ),
        "architecture_integration_trend_regression_count": int(
            stage_e_architecture_integration_observed_trend.get("regression_count", 0) or 0
        ),
        "architecture_integration_trend_release_gate_blocking": bool(
            stage_e_architecture_integration_observed_trend.get("release_gate_blocking", False)
        ),
        **{
            metric_name: float(cognitive_linear_snn_fusion_metrics[metric_name])
            for metric_name in COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES
        },
        **{
            metric_name: float(cognitive_plastic_submodel_metrics[metric_name])
            for metric_name in COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES
        },
        "minimum_failures": [
            dict(item) for item in minimum_failures if isinstance(item, dict)
        ],
    }


def _extract_phase5_operational_snapshot(phase3_report: Dict[str, Any]) -> Dict[str, Any]:
    focus_summary = (
        phase3_report.get("focus_summary", {})
        if isinstance(phase3_report.get("focus_summary"), dict)
        else {}
    )
    phase5_entry = (
        focus_summary.get("phase5_entry_readiness", {})
        if isinstance(focus_summary.get("phase5_entry_readiness"), dict)
        else {}
    )
    metrics = phase5_entry.get("metrics", {}) if isinstance(phase5_entry.get("metrics"), dict) else {}
    component_reports = (
        phase3_report.get("component_reports", {})
        if isinstance(phase3_report.get("component_reports"), dict)
        else {}
    )
    phase5_component = (
        component_reports.get("phase5_predictive_coding", {})
        if isinstance(component_reports.get("phase5_predictive_coding"), dict)
        else {}
    )
    phase5_component_metrics = (
        phase5_component.get("metrics", {})
        if isinstance(phase5_component.get("metrics"), dict)
        else {}
    )
    return {
        "passed": bool(phase5_entry.get("passed", False)),
        "readiness_score": float(phase5_entry.get("score", 0.0) or 0.0),
        "latent_transition_alignment": float(
            metrics.get("phase5_predictive_coding.latent_transition_alignment", 0.0) or 0.0
        ),
        "prediction_error_observability": float(
            metrics.get("phase5_predictive_coding.prediction_error_observability", 0.0) or 0.0
        ),
        "correction_event_coverage": float(
            metrics.get("phase5_predictive_coding.correction_event_coverage", 0.0) or 0.0
        ),
        "anti_collapse_event_diversity": float(
            metrics.get("phase5_predictive_coding.anti_collapse_event_diversity", 0.0) or 0.0
        ),
        "counterfactual_transition_separation": float(
            metrics.get("phase5_predictive_coding.counterfactual_transition_separation", 0.0) or 0.0
        ),
        "multi_step_latent_chain_integrity": float(
            metrics.get("phase5_predictive_coding.multi_step_latent_chain_integrity", 0.0) or 0.0
        ),
        "long_horizon_error_correction_convergence": float(
            metrics.get("phase5_predictive_coding.long_horizon_error_correction_convergence", 0.0) or 0.0
        ),
        "horizon_bucket_stability": float(
            metrics.get("phase5_predictive_coding.horizon_bucket_stability", 0.0) or 0.0
        ),
        "macro_action_effectiveness": float(
            metrics.get("phase5_predictive_coding.macro_action_effectiveness", 0.0) or 0.0
        ),
        "subgoal_decomposition_integrity": float(
            metrics.get("phase5_predictive_coding.subgoal_decomposition_integrity", 0.0) or 0.0
        ),
        "depth_selective_routing_integrity": float(
            metrics.get("phase5_predictive_coding.depth_selective_routing_integrity", 0.0) or 0.0
        ),
        "micro_es_policy_refinement_integrity": float(
            metrics.get("phase5_predictive_coding.micro_es_policy_refinement_integrity", 0.0) or 0.0
        ),
        "manifold_transition_locality_observed": float(
            phase5_component_metrics.get("manifold_transition_locality", 0.0) or 0.0
        ),
        "manifold_rollout_stability_observed": float(
            phase5_component_metrics.get("manifold_rollout_stability", 0.0) or 0.0
        ),
        "causal_route_sparsity_observed": float(
            phase5_component_metrics.get("causal_route_sparsity", 0.0) or 0.0
        ),
        "withheld_trajectory_recall_observed": float(
            phase5_component_metrics.get("withheld_trajectory_recall", 0.0) or 0.0
        ),
        "manifold_trajectory_case_coverage_observed": float(
            phase5_component_metrics.get("manifold_trajectory_case_coverage", 0.0) or 0.0
        ),
        "manifold_average_case_recall_observed": float(
            phase5_component_metrics.get("manifold_average_case_recall", 0.0) or 0.0
        ),
        "manifold_scan_budget_integrity_observed": float(
            phase5_component_metrics.get("manifold_scan_budget_integrity", 0.0) or 0.0
        ),
        "manifold_indexed_candidate_integrity_observed": float(
            phase5_component_metrics.get("manifold_indexed_candidate_integrity", 0.0) or 0.0
        ),
        "manifold_index_scan_reduction_observed": float(
            phase5_component_metrics.get("manifold_index_scan_reduction", 0.0) or 0.0
        ),
        "manifold_candidate_miss_guard_observed": float(
            phase5_component_metrics.get("manifold_candidate_miss_guard", 0.0) or 0.0
        ),
    }


def _append_recovery_action(
    actions: List[Dict[str, str]],
    *,
    title: str,
    command: str,
    priority: str,
    expected_effect: str,
    affected_checks: List[str],
) -> None:
    normalized_command = str(command).strip()
    if not normalized_command:
        return
    for item in actions:
        if str(item.get("command", "")).strip() == normalized_command:
            return
    actions.append(
        {
            "title": str(title).strip(),
            "command": normalized_command,
            "priority": str(priority).strip(),
            "expected_effect": str(expected_effect).strip(),
            "affected_checks": [str(name).strip() for name in affected_checks if str(name).strip()],
        }
    )


def _is_efficiency_kpi_error_text(text: str) -> bool:
    normalized = str(text).strip().lower()
    if not normalized:
        return False
    return any(
        token in normalized
        for token in [
            "efficiency_readiness",
            "performance-per-energy ratio proxy",
            "ann-reference cost advantage proxy",
            "brain-efficiency alignment proxy",
            "energy_per_success_proxy",
            "performance_energy_ratio_proxy",
            "ann_cost_advantage_proxy",
            "sparse_event_cost_score",
            "brain_efficiency_alignment_proxy",
        ]
    )


def _build_recovery_actions(
    checks: Dict[str, Any],
    *,
    stage_b_promotion: Dict[str, Any],
    stage_d_readiness: Optional[Dict[str, Any]] = None,
    neuromorphic_profile: Optional[Dict[str, Any]] = None,
    strict_production: bool,
) -> List[Dict[str, str]]:
    actions: List[Dict[str, str]] = []
    stage_d_snapshot = stage_d_readiness if isinstance(stage_d_readiness, dict) else {}
    phase3_accuracy_errors = (
        checks.get("phase3_accuracy", {}).get("errors", [])
        if isinstance(checks.get("phase3_accuracy"), dict)
        else []
    )
    phase3_completion_errors = (
        checks.get("phase3_completion", {}).get("errors", [])
        if isinstance(checks.get("phase3_completion"), dict)
        else []
    )
    has_efficiency_failure = any(
        _is_efficiency_kpi_error_text(item)
        for item in (
            list(phase3_accuracy_errors if isinstance(phase3_accuracy_errors, list) else [])
            + list(phase3_completion_errors if isinstance(phase3_completion_errors, list) else [])
        )
    )

    if has_efficiency_failure:
        _append_recovery_action(
            actions,
            title="Re-run Energy Efficiency Benchmark",
            command="python scripts/eval/energy_efficiency_benchmark.py",
            priority="high",
            expected_effect="Recomputes performance-per-energy and ANN-cost advantage proxy metrics for efficiency readiness.",
            affected_checks=["phase3_accuracy", "phase3_completion"],
        )

    neuromorphic_profile = (
        neuromorphic_profile if isinstance(neuromorphic_profile, dict) else {}
    )
    neuromorphic_regression_count = int(
        neuromorphic_profile.get("trend_regression_count", 0) or 0
    )
    neuromorphic_policy_change_count = int(
        neuromorphic_profile.get("trend_policy_change_count", 0) or 0
    )
    neuromorphic_missing_profiles = (
        neuromorphic_profile.get("trend_missing_profiles", [])
        if isinstance(neuromorphic_profile.get("trend_missing_profiles", []), list)
        else []
    )
    if neuromorphic_regression_count > 0 or neuromorphic_missing_profiles:
        _append_recovery_action(
            actions,
            title="Inspect Neuromorphic Profile Regression",
            command="python scripts/eval/energy_efficiency_benchmark.py --no-history-update",
            priority="high",
            expected_effect=(
                "Recomputes neuromorphic profile compatibility without updating history so "
                "missing profiles, event budgets, low-precision checks, and policy regressions can be inspected."
            ),
            affected_checks=["phase3_accuracy", "phase3_completion"],
        )
        _append_recovery_action(
            actions,
            title="Refresh Phase 3 Summary After Neuromorphic Profile Check",
            command="python scripts/eval/phase3_accuracy_suite.py",
            priority="medium",
            expected_effect=(
                "Refreshes the human-readable Phase 3 summary after neuromorphic profile checks are corrected."
            ),
            affected_checks=["phase3_accuracy", "phase3_completion"],
        )
    elif neuromorphic_policy_change_count > 0:
        _append_recovery_action(
            actions,
            title="Review Neuromorphic Adapter Policy Change",
            command="python scripts/eval/energy_efficiency_benchmark.py --no-history-update",
            priority="medium",
            expected_effect=(
                "Recomputes profile reports so adapter policy changes can be reviewed before release promotion."
            ),
            affected_checks=["phase3_accuracy", "phase3_completion"],
        )

    if not bool(checks.get("phase3_accuracy", {}).get("passed", False)):
        _append_recovery_action(
            actions,
            title="Re-run Phase 3 Accuracy Suite",
            command="python scripts/eval/phase3_accuracy_suite.py",
            priority="high",
            expected_effect="Refreshes phase3 metrics and stage readiness checks for gate re-evaluation.",
            affected_checks=["phase3_accuracy", "phase3_completion"],
        )
    if not bool(checks.get("phase4_completion", {}).get("passed", False)):
        _append_recovery_action(
            actions,
            title="Re-run Phase 4 Scale-Out Benchmark",
            command="python scripts/eval/phase4_scale_continual_benchmark.py",
            priority="high",
            expected_effect="Recomputes phase4 scale/continual signals required by completion gate.",
            affected_checks=["phase4_completion"],
        )
    if not bool(checks.get("phase5_entry_gate", {}).get("passed", False)):
        _append_recovery_action(
            actions,
            title="Re-run Phase 5 Predictive Coding Benchmark",
            command="python scripts/eval/phase5_predictive_coding_benchmark.py",
            priority="high",
            expected_effect="Rebuilds Phase 5 predictive-coding metrics used by entry/completion gates.",
            affected_checks=["phase5_entry_gate", "phase5_completion_gate"],
        )
        _append_recovery_action(
            actions,
            title="Re-run Phase 5 Entry Gate",
            command="python scripts/eval/phase5_entry_gate.py",
            priority="high",
            expected_effect="Revalidates Phase 5 entry-gate checks from the refreshed predictive-coding report.",
            affected_checks=["phase5_entry_gate"],
        )
    if not bool(checks.get("phase5_completion_gate", {}).get("passed", False)):
        _append_recovery_action(
            actions,
            title="Re-run Phase 5 Completion Gate",
            command="python scripts/eval/phase5_completion_gate.py",
            priority="high",
            expected_effect="Revalidates Phase 5 completion-level checks across Phase 4/Phase 5 artifacts.",
            affected_checks=["phase5_completion_gate"],
        )
    if not bool(checks.get("external_validity", {}).get("passed", False)):
        _append_recovery_action(
            actions,
            title="Re-run Real-Data External Validity Benchmark",
            command="python scripts/eval/real_data_external_validity.py",
            priority="high",
            expected_effect="Recomputes real-corpus QA, summary, continual-memory, and ANN-cost advantage evidence.",
            affected_checks=["external_validity"],
        )
    if not bool(checks.get("external_validity_ladder", {}).get("passed", False)):
        _append_recovery_action(
            actions,
            title="Re-run Real-Data External Validity Ladder",
            command="python scripts/eval/real_data_external_validity_ladder.py",
            priority="high",
            expected_effect="Recomputes small/medium/large real-corpus ANN-ratio and performance-energy evidence.",
            affected_checks=["external_validity_ladder"],
        )
    if not bool(checks.get("release_gate", {}).get("passed", False)):
        _append_recovery_action(
            actions,
            title="Re-run Extended Release Soak with Accuracy",
            command="python scripts/eval/release_soak.py --profile extended --include-accuracy",
            priority="high",
            expected_effect="Rebuilds soak artifacts and embedded accuracy needed by release gate.",
            affected_checks=["release_gate", "production_profile"],
        )
        _append_recovery_action(
            actions,
            title="Re-run Release Gate",
            command="python scripts/eval/release_gate.py",
            priority="medium",
            expected_effect="Validates refreshed artifacts against release-gate contract.",
            affected_checks=["release_gate"],
        )
    if strict_production and not bool(checks.get("production_profile", {}).get("passed", False)):
        _append_recovery_action(
            actions,
            title="Run Operational Readiness in Strict Mode with Artifact Refresh",
            command="python scripts/eval/operational_readiness.py --refresh-artifacts --soak-profile extended --strict-production",
            priority="high",
            expected_effect="Forces extended-profile thresholds and revalidates operational promotion criteria.",
            affected_checks=["production_profile", "release_gate"],
        )

    if bool(stage_b_promotion.get("promotion_recommended", False)):
        next_step_hint = str(stage_b_promotion.get("promotion_next_step_hint", "") or "").strip()
        if next_step_hint:
            _append_recovery_action(
                actions,
                title="Apply Stage B Promotion Follow-up",
                command=f"python scripts/eval/release_soak.py --record-repair-command \"{next_step_hint}\" --record-repair-status pending --record-repair-source stage_b_promotion",
                priority="medium",
                expected_effect="Records Stage B promotion follow-up so contract updates are tracked in repair history.",
                affected_checks=["stage_b_promotion"],
            )
        for idx, action_text in enumerate(stage_b_promotion.get("promotion_actions", [])[:3], start=1):
            action_label = str(action_text).strip()
            if not action_label:
                continue
            _append_recovery_action(
                actions,
                title=f"Execute Stage B Promotion Action {idx}",
                command=f"python scripts/eval/release_soak.py --record-repair-command \"{action_label}\" --record-repair-status pending --record-repair-source stage_b_promotion",
                priority="low",
                expected_effect="Captures manual Stage B promotion work items in the same operational trail.",
                affected_checks=["stage_b_promotion"],
            )

    if bool(stage_b_promotion.get("rlm_observation_promotion_recommended", False)):
        next_step_hint = str(stage_b_promotion.get("rlm_observation_next_step_hint", "") or "").strip()
        if next_step_hint:
            _append_recovery_action(
                actions,
                title="Apply Stage B RLM Observation Promotion Follow-up",
                command=f"python scripts/eval/release_soak.py --record-repair-command \"{next_step_hint}\" --record-repair-status pending --record-repair-source stage_b_rlm_observation_promotion",
                priority="medium",
                expected_effect="Records the long-context and branch-consistency promotion follow-up in repair history.",
                affected_checks=["stage_b_rlm_observation_promotion"],
            )
        for idx, action_text in enumerate(stage_b_promotion.get("rlm_observation_actions", [])[:3], start=1):
            action_label = str(action_text).strip()
            if not action_label:
                continue
            _append_recovery_action(
                actions,
                title=f"Execute Stage B RLM Observation Promotion Action {idx}",
                command=f"python scripts/eval/release_soak.py --record-repair-command \"{action_label}\" --record-repair-status pending --record-repair-source stage_b_rlm_observation_promotion",
                priority="low",
                expected_effect="Captures manual RLM-observation promotion work items in the operational trail.",
                affected_checks=["stage_b_rlm_observation_promotion"],
            )

    delta_memory_readiness = (
        stage_d_snapshot.get("delta_memory_promotion_readiness", {})
        if isinstance(stage_d_snapshot.get("delta_memory_promotion_readiness"), dict)
        else {}
    )
    if bool(delta_memory_readiness.get("recommended", False)):
        next_step_hint = "promote_stage_d_delta_memory_metrics_to_minimum_gate"
        _append_recovery_action(
            actions,
            title="Apply Stage D Delta Memory Promotion Follow-up",
            command=f"python scripts/eval/release_soak.py --record-repair-command \"{next_step_hint}\" --record-repair-status pending --record-repair-source stage_d_delta_memory_promotion",
            priority="medium",
            expected_effect="Records Stage D delta-memory promotion follow-up so contract updates are tracked in repair history.",
            affected_checks=["stage_d_delta_memory_promotion"],
        )
        for idx, action_text in enumerate(
            [
                "review stage_d_contract minimum list and add the delta-memory promotion metrics",
                "run python scripts/eval/phase3_accuracy_suite.py and verify Stage D remains green",
                "run python scripts/eval/release_gate.py --skip-accuracy to verify release gate compatibility",
            ],
            start=1,
        ):
            _append_recovery_action(
                actions,
                title=f"Execute Stage D Delta Memory Promotion Action {idx}",
                command=f"python scripts/eval/release_soak.py --record-repair-command \"{action_text}\" --record-repair-status pending --record-repair-source stage_d_delta_memory_promotion",
                priority="low",
                expected_effect="Captures manual Stage D delta-memory promotion work items in the operational trail.",
                affected_checks=["stage_d_delta_memory_promotion"],
            )

    acceptance_candidate_stability = (
        stage_d_snapshot.get("acceptance_candidate_stability", {})
        if isinstance(stage_d_snapshot.get("acceptance_candidate_stability"), dict)
        else {}
    )
    if bool(acceptance_candidate_stability.get("recommended", False)):
        next_step_hint = STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_NEXT_STEP_HINT
        _append_recovery_action(
            actions,
            title="Apply Stage D Acceptance Candidate Stability Follow-up",
            command=f"python scripts/eval/release_soak.py --record-repair-command \"{next_step_hint}\" --record-repair-status pending --record-repair-source stage_d_acceptance_candidate_stability",
            priority="medium",
            expected_effect="Records Stage D acceptance-candidate stability follow-up before minimum contract promotion.",
            affected_checks=["stage_d_acceptance_candidate_stability"],
        )
        for idx, action_text in enumerate(
            STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS,
            start=1,
        ):
            _append_recovery_action(
                actions,
                title=f"Execute Stage D Acceptance Candidate Stability Action {idx}",
                command=f"python scripts/eval/release_soak.py --record-repair-command \"{action_text}\" --record-repair-status pending --record-repair-source stage_d_acceptance_candidate_stability",
                priority="low",
                expected_effect="Captures manual Stage D acceptance-candidate promotion review work items in the operational trail.",
                affected_checks=["stage_d_acceptance_candidate_stability"],
            )

    acceptance_candidate_failures = (
        stage_d_snapshot.get("acceptance_candidate_failures", [])
        if isinstance(stage_d_snapshot.get("acceptance_candidate_failures"), list)
        else []
    )
    if acceptance_candidate_failures:
        failed_metrics = [
            str(item.get("metric", item.get("check", ""))).strip()
            for item in acceptance_candidate_failures
            if isinstance(item, dict) and str(item.get("metric", item.get("check", ""))).strip()
        ]
        failed_label = ",".join(failed_metrics[:5])
        repair_command = f"repair_stage_d_acceptance_candidates:{failed_label}" if failed_label else "repair_stage_d_acceptance_candidates"
        affected = ["stage_d_acceptance_candidate_repair", *failed_metrics[:5]]
        _append_recovery_action(
            actions,
            title="Repair Stage D Acceptance Candidate Failures",
            command=f"python scripts/eval/release_soak.py --record-repair-command \"{repair_command}\" --record-repair-status pending --record-repair-source stage_d_acceptance_candidate_repair",
            priority="medium",
            expected_effect="Records Stage D acceptance-candidate repair work for failed observed-only promotion candidates.",
            affected_checks=affected,
        )

    return actions


def _build_iterative_operational_repair_plan(
    checks: Dict[str, Any],
    recovery_actions: List[Dict[str, Any]],
    *,
    execution_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    failed_checks = sorted(
        name
        for name, payload in checks.items()
        if isinstance(payload, dict) and not bool(payload.get("passed", False))
    )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    ranked_actions = sorted(
        [dict(item) for item in recovery_actions if isinstance(item, dict)],
        key=lambda item: (
            priority_order.get(str(item.get("priority", "low")).lower(), 2),
            str(item.get("title", "")),
        ),
    )
    successful_commands = {
        str(item.get("command", "")).strip()
        for item in (execution_log if isinstance(execution_log, list) else [])
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "success"
    }
    executed_steps = sum(
        1
        for item in (execution_log if isinstance(execution_log, list) else [])
        if isinstance(item, dict)
    )
    successful_steps = sum(
        1
        for item in (execution_log if isinstance(execution_log, list) else [])
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "success"
    )
    failed_steps = sum(
        1
        for item in (execution_log if isinstance(execution_log, list) else [])
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() in {"failed", "timeout", "error"}
    )

    filtered_actions: List[Dict[str, Any]] = []
    for action in ranked_actions:
        command = str(action.get("command", "")).strip()
        if command and command in successful_commands:
            continue
        affected_checks = (
            {
                str(item).strip()
                for item in action.get("affected_checks", [])
                if str(item).strip()
            }
            if isinstance(action.get("affected_checks"), list)
            else set()
        )
        if failed_checks and affected_checks and not affected_checks.intersection(set(failed_checks)):
            continue
        filtered_actions.append(action)

    next_actions: List[Dict[str, Any]] = []
    for step, action in enumerate(filtered_actions, start=1):
        payload = dict(action)
        payload["step"] = int(step)
        next_actions.append(payload)
    completed = len(failed_checks) == 0
    stalled = (not completed) and (len(next_actions) == 0)
    if completed:
        stop_reason = "auto_stopped_completed"
        next_step_hint = "No further action required. Re-run operational_readiness only for verification."
        next_actions = []
    elif stalled:
        stop_reason = "stalled_no_candidate_actions"
        next_step_hint = "python scripts/eval/operational_readiness.py --refresh-artifacts --soak-profile extended"
    else:
        stop_reason = "pending_actions"
        next_step_hint = str(next_actions[0].get("command", "python scripts/eval/operational_readiness.py")).strip()
    return {
        "iteration": int(executed_steps + 1),
        "executed_steps": int(executed_steps),
        "successful_steps": int(successful_steps),
        "failed_steps": int(failed_steps),
        "completed": bool(completed),
        "auto_stopped": bool(completed),
        "stalled": bool(stalled),
        "stop_reason": str(stop_reason),
        "failed_checks": failed_checks,
        "remaining_checks": list(failed_checks),
        "next_actions": next_actions,
        "next_step_hint": next_step_hint,
    }


def _build_operational_repair_plan(
    recovery_actions: List[Dict[str, Any]],
    failed_checks: List[str],
) -> Dict[str, Any]:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    targets = sorted({str(item).strip() for item in failed_checks if str(item).strip()})
    ranked = sorted(
        [dict(item) for item in recovery_actions if isinstance(item, dict)],
        key=lambda item: (
            priority_order.get(str(item.get("priority", "low")).lower(), 2),
            str(item.get("title", "")),
        ),
    )
    selected: List[Dict[str, Any]] = []
    covered: set[str] = set()
    for idx, action in enumerate(ranked, start=1):
        affected = (
            sorted(
                {
                    str(item).strip()
                    for item in action.get("affected_checks", [])
                    if str(item).strip()
                }
            )
            if isinstance(action.get("affected_checks"), list)
            else []
        )
        if targets and affected and not set(affected).intersection(set(targets)):
            continue
        payload = dict(action)
        payload["step"] = int(idx)
        payload["affected_checks"] = affected
        selected.append(payload)
        covered.update(set(affected).intersection(set(targets)))
    uncovered = sorted(set(targets).difference(covered))
    fallback_actions: List[Dict[str, Any]] = []
    if uncovered:
        fallback_actions.append(
            {
                "step": 1,
                "title": "Re-run Operational Readiness with Artifact Refresh",
                "command": "python scripts/eval/operational_readiness.py --refresh-artifacts --soak-profile extended --include-accuracy",
                "priority": "medium",
                "expected_effect": "Refreshes all upstream artifacts and recomputes operational checks.",
                "affected_checks": uncovered,
            }
        )
    coverage_ratio = float(len(covered) / max(len(targets), 1)) if targets else 1.0
    return {
        "selected_actions": selected,
        "covered_checks": sorted(covered),
        "uncovered_checks": uncovered,
        "fallback_actions": fallback_actions,
        "coverage_ratio": float(coverage_ratio),
        "estimated_steps": int(len(selected)),
    }


def _build_operational_error_details(checks: Dict[str, Any]) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    if not isinstance(checks, dict):
        return details
    index = 0
    for section_name, section in checks.items():
        if not isinstance(section, dict):
            continue
        errors = section.get("errors", [])
        if not isinstance(errors, list):
            continue
        for error in errors:
            index += 1
            category = str(section_name)
            if _is_efficiency_kpi_error_text(error):
                if str(section_name) in {"phase3_accuracy", "phase3_completion"}:
                    category = "phase3_efficiency_kpi"
                elif str(section_name) == "release_gate":
                    category = "release_gate_efficiency_kpi"
                else:
                    category = "efficiency_kpi"
            details.append(
                {
                    "index": int(index),
                    "type": "check_failure",
                    "category": category,
                    "error": str(error),
                }
            )
    return details


def _build_operational_error_details_summary(
    error_details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(error_details, list):
        return {
            "total": 0,
            "by_type": {},
            "by_category": {},
            "top_types": [],
            "top_categories": [],
        }
    by_type: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    total = 0
    for item in error_details:
        if not isinstance(item, dict):
            continue
        total += 1
        detail_type = str(item.get("type", "check_failure"))
        category = str(item.get("category", "operational_readiness"))
        by_type[detail_type] = int(by_type.get(detail_type, 0)) + 1
        by_category[category] = int(by_category.get(category, 0)) + 1
    sorted_types = sorted(by_type.items(), key=lambda pair: (-int(pair[1]), str(pair[0])))
    sorted_categories = sorted(by_category.items(), key=lambda pair: (-int(pair[1]), str(pair[0])))
    return {
        "total": int(total),
        "by_type": dict(sorted(by_type.items())),
        "by_category": dict(sorted(by_category.items())),
        "top_types": [{"name": str(name), "count": int(count)} for name, count in sorted_types[:5]],
        "top_categories": [{"name": str(name), "count": int(count)} for name, count in sorted_categories[:5]],
    }


def _build_operational_failure_focus(
    error_details_summary: Dict[str, Any],
    repair_plan: Dict[str, Any],
) -> Dict[str, Any]:
    summary = error_details_summary if isinstance(error_details_summary, dict) else {}
    plan = repair_plan if isinstance(repair_plan, dict) else {}
    top_categories = summary.get("top_categories", [])
    selected_actions = plan.get("selected_actions", [])
    primary_category = ""
    secondary_category = ""
    if isinstance(top_categories, list) and top_categories and isinstance(top_categories[0], dict):
        primary_category = str(top_categories[0].get("name", "")).strip()
    if isinstance(top_categories, list) and len(top_categories) > 1 and isinstance(top_categories[1], dict):
        secondary_category = str(top_categories[1].get("name", "")).strip()
    primary_action: Dict[str, str] = {}
    if isinstance(selected_actions, list) and selected_actions and isinstance(selected_actions[0], dict):
        primary_action = {
            "title": str(selected_actions[0].get("title", "")).strip(),
            "command": str(selected_actions[0].get("command", "")).strip(),
            "priority": str(selected_actions[0].get("priority", "")).strip(),
        }
    total = int(summary.get("total", 0) or 0)
    primary_count = 0
    if isinstance(top_categories, list) and top_categories and isinstance(top_categories[0], dict):
        primary_count = int(top_categories[0].get("count", 0) or 0)
    concentration = (float(primary_count) / float(total)) if total > 0 else 0.0
    confidence = min(max(concentration + (0.15 if primary_action else 0.0), 0.0), 1.0)
    return {
        "primary_category": primary_category,
        "secondary_category": secondary_category,
        "primary_action": primary_action,
        "confidence": float(round(confidence, 3)),
    }


def collect_operational_readiness_artifacts(
    checks: Dict[str, Any],
    *,
    stage_b_promotion: Dict[str, Any],
    stage_d_readiness: Optional[Dict[str, Any]] = None,
    neuromorphic_profile: Optional[Dict[str, Any]] = None,
    execution_log: Optional[List[Dict[str, Any]]] = None,
    strict_production: bool = False,
    retry_max_attempts: int = 2,
    retry_cooldown_seconds: float = 0.0,
) -> Dict[str, Any]:
    recovery_actions = _build_recovery_actions(
        checks,
        stage_b_promotion=stage_b_promotion,
        stage_d_readiness=stage_d_readiness,
        neuromorphic_profile=neuromorphic_profile,
        strict_production=bool(strict_production),
    )
    failed_checks = sorted(
        name
        for name, payload in checks.items()
        if isinstance(payload, dict) and not bool(payload.get("passed", False))
    )
    repair_plan = _build_operational_repair_plan(recovery_actions, failed_checks)
    error_details = _build_operational_error_details(checks)
    error_details_summary = _build_operational_error_details_summary(error_details)
    failure_focus = _build_operational_failure_focus(error_details_summary, repair_plan)
    iterative_repair_plan = _build_iterative_operational_repair_plan(
        checks,
        recovery_actions,
        execution_log=execution_log,
    )
    retry_queue = build_operational_retry_queue_from_repair_log(
        execution_log if isinstance(execution_log, list) else [],
        max_attempts=retry_max_attempts,
        cooldown_seconds=retry_cooldown_seconds,
    )
    retry_cooldown_blocked = build_operational_retry_cooldown_blocked_from_repair_log(
        execution_log if isinstance(execution_log, list) else [],
        max_attempts=retry_max_attempts,
        cooldown_seconds=retry_cooldown_seconds,
    )
    prioritized_retry_queue = prioritize_operational_retry_queue(
        retry_queue,
        iterative_plan=iterative_repair_plan,
    )
    prioritized_cooldown_blocked = prioritize_operational_retry_queue(
        retry_cooldown_blocked,
        iterative_plan=iterative_repair_plan,
    )
    return {
        "recovery_actions": recovery_actions,
        "repair_plan": repair_plan,
        "error_details": error_details,
        "error_details_summary": error_details_summary,
        "failure_focus": failure_focus,
        "iterative_repair_plan": iterative_repair_plan,
        "repair_retry_queue": prioritized_retry_queue,
        "repair_retry_queue_count": int(len(prioritized_retry_queue)),
        "repair_retry_cooldown_seconds": float(max(retry_cooldown_seconds, 0.0)),
        "repair_retry_cooldown_blocked": prioritized_cooldown_blocked,
        "repair_retry_cooldown_blocked_count": int(len(prioritized_cooldown_blocked)),
    }


def format_operational_summary(report: Dict[str, Any]) -> str:
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    stage_b = (
        report.get("stage_b_promotion", {})
        if isinstance(report.get("stage_b_promotion"), dict)
        else {}
    )
    stage_e = (
        report.get("stage_e_readiness", {})
        if isinstance(report.get("stage_e_readiness"), dict)
        else {}
    )
    stage_d = (
        report.get("stage_d_readiness", {})
        if isinstance(report.get("stage_d_readiness"), dict)
        else {}
    )
    stage_d_delta_readiness = (
        stage_d.get("delta_memory_promotion_readiness", {})
        if isinstance(stage_d.get("delta_memory_promotion_readiness"), dict)
        else {}
    )
    phase5_entry = (
        report.get("phase5_entry_readiness", {})
        if isinstance(report.get("phase5_entry_readiness"), dict)
        else {}
    )
    neuromorphic_profile = (
        report.get("neuromorphic_profile_readiness", {})
        if isinstance(report.get("neuromorphic_profile_readiness"), dict)
        else {}
    )
    iterative_plan = (
        report.get("iterative_repair_plan", {})
        if isinstance(report.get("iterative_repair_plan"), dict)
        else {}
    )
    repair_plan = (
        report.get("repair_plan", {})
        if isinstance(report.get("repair_plan"), dict)
        else {}
    )
    failure_focus = (
        report.get("failure_focus", {})
        if isinstance(report.get("failure_focus"), dict)
        else {}
    )
    retry_queue = (
        report.get("repair_retry_queue", [])
        if isinstance(report.get("repair_retry_queue"), list)
        else []
    )
    retry_cooldown_blocked = (
        report.get("repair_retry_cooldown_blocked", [])
        if isinstance(report.get("repair_retry_cooldown_blocked"), list)
        else []
    )
    auto_dispatch = (
        report.get("repair_auto_dispatch", {})
        if isinstance(report.get("repair_auto_dispatch"), dict)
        else {}
    )
    error_details = (
        report.get("error_details", [])
        if isinstance(report.get("error_details"), list)
        else []
    )
    error_details_summary = (
        report.get("error_details_summary", {})
        if isinstance(report.get("error_details_summary"), dict)
        else {}
    )
    checklist = (
        report.get("operational_checklist", {})
        if isinstance(report.get("operational_checklist"), dict)
        else {}
    )
    research_journal_summary = (
        report.get("research_journal_summary", {})
        if isinstance(report.get("research_journal_summary"), dict)
        else {}
    )
    research_review = (
        report.get("research_review", {})
        if isinstance(report.get("research_review"), dict)
        else {}
    )
    research_review_compact = (
        research_review.get("compact", {})
        if isinstance(research_review.get("compact"), dict)
        else {}
    )
    research_journal_summary = (
        report.get("research_journal_summary", {})
        if isinstance(report.get("research_journal_summary"), dict)
        else {}
    )
    research_planner_task_status = summarize_research_planner_task_status(
        research_review_compact,
        research_journal_summary,
        cleanup_threshold=int(report.get("research_planner_task_cleanup_threshold", 2) or 2),
    )
    execution_log = (
        report.get("execution_log", [])
        if isinstance(report.get("execution_log"), list)
        else []
    )
    research_journal_summary = attach_roadmap_patch_refresh_policy_followups_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    research_journal_summary = attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    roadmap_patch_refresh_policy = summarize_roadmap_patch_refresh_policy(
        research_journal_summary
    )
    completed_evidence_review = summarize_completed_roadmap_patch_evidence_review(
        research_journal_summary
    )
    research_journal_experiment_priority_plan = (
        research_journal_summary.get("experiment_priority_plan", {})
        if isinstance(research_journal_summary.get("experiment_priority_plan"), dict)
        else {}
    )
    research_journal_experiment_promotion_target_plan = (
        research_journal_summary.get("experiment_promotion_target_plan", {})
        if isinstance(research_journal_summary.get("experiment_promotion_target_plan"), dict)
        else {}
    )
    stage_e_observed_candidate_repair_loop = (
        research_journal_summary.get("stage_e_observed_acceptance_candidate_repair_loop", {})
        if isinstance(
            research_journal_summary.get("stage_e_observed_acceptance_candidate_repair_loop"),
            dict,
        )
        else {}
    )
    stage_e_readiness = (
        report.get("stage_e_readiness", {})
        if isinstance(report.get("stage_e_readiness"), dict)
        else {}
    )
    external_validity = (
        checks.get("external_validity", {})
        if isinstance(checks.get("external_validity"), dict)
        else {}
    )
    external_metrics = (
        external_validity.get("metrics", {})
        if isinstance(external_validity.get("metrics"), dict)
        else {}
    )
    external_ladder = (
        checks.get("external_validity_ladder", {})
        if isinstance(checks.get("external_validity_ladder"), dict)
        else {}
    )
    external_ladder_metrics = (
        external_ladder.get("metrics", {})
        if isinstance(external_ladder.get("metrics"), dict)
        else {}
    )

    def _status(name: str) -> str:
        section = checks.get(name, {}) if isinstance(checks.get(name), dict) else {}
        return "PASS" if bool(section.get("passed", False)) else "FAIL"

    phase5_completion = checks.get("phase5_completion_gate", {}) if isinstance(checks.get("phase5_completion_gate"), dict) else {}
    phase5_completion_detail_values = (
        phase5_completion.get("detail_values", {})
        if isinstance(phase5_completion.get("detail_values", {}), dict)
        else {}
    )
    phase5_completion_detail_lines = []
    for check_name in [
        "macro_step_reduction",
        "macro_cost_reduction",
        "subgoal_coverage_ratio",
        "micro_es_fitness_improvement",
        "micro_es_event_cost_reduction",
        "micro_es_population_event_budget",
    ]:
        item = phase5_completion_detail_values.get(check_name, {})
        if not isinstance(item, dict) or "value" not in item:
            continue
        line = f"- phase5_completion_{check_name}_value: {float(item.get('value', 0.0) or 0.0):.3f}"
        if "required_min" in item:
            line += f" required_min={float(item.get('required_min', 0.0) or 0.0):.3f}"
        if "required_gt" in item:
            line += f" required_gt={float(item.get('required_gt', 0.0) or 0.0):.3f}"
        if "event_budget" in item:
            line += f" event_budget={float(item.get('event_budget', 0.0) or 0.0):.3f}"
        phase5_completion_detail_lines.append(line)

    lines = [
        "SARA Engine Operational Readiness Summary",
        f"- operational_status: {'PASS' if bool(report.get('passed', False)) else 'FAIL'}",
        f"- total_error_count: {int(report.get('error_count', 0))}",
        f"- readiness_score: {float(report.get('readiness_score', 0.0)):.3f}",
        f"- strict_production: {bool(report.get('strict_production', False))}",
        f"- phase3_accuracy: {_status('phase3_accuracy')}",
        f"- phase3_completion: {_status('phase3_completion')}",
        f"- phase4_completion: {_status('phase4_completion')}",
        f"- phase5_entry_gate: {_status('phase5_entry_gate')}",
        f"- phase5_completion_gate: {_status('phase5_completion_gate')}",
        *phase5_completion_detail_lines,
        f"- external_validity: {'PASS' if bool(external_validity.get('passed', False)) else ('FAIL' if 'external_validity' in checks else 'SKIP')}",
        f"- external_validity_real_data_qa_accuracy: {float(external_metrics.get('real_data_qa_accuracy', 0.0) or 0.0):.3f}",
        f"- external_validity_ann_cost_advantage_proxy: {float(external_metrics.get('ann_cost_advantage_proxy', 0.0) or 0.0):.3f}",
        f"- external_validity_performance_energy_ratio_proxy: {float(external_metrics.get('performance_energy_ratio_proxy', 0.0) or 0.0):.3f}",
        f"- external_validity_ladder: {'PASS' if bool(external_ladder.get('passed', False)) else ('FAIL' if 'external_validity_ladder' in checks else 'SKIP')}",
        f"- external_validity_ladder_profile_count: {int(external_ladder_metrics.get('profile_count', 0) or 0)}",
        f"- external_validity_ladder_min_ann_cost_advantage_proxy: {float(external_ladder_metrics.get('min_ann_cost_advantage_proxy', 0.0) or 0.0):.3f}",
        f"- external_validity_ladder_min_performance_energy_ratio_proxy: {float(external_ladder_metrics.get('min_performance_energy_ratio_proxy', 0.0) or 0.0):.3f}",
        f"- release_gate: {_status('release_gate')}",
        f"- production_profile: {_status('production_profile')}",
        f"- research_review_passed: {bool(research_review_compact.get('passed', False))}",
        f"- research_review_score: {float(research_review_compact.get('review_score', 0.0) or 0.0):.3f}",
        f"- research_review_release_gate_blocking: {bool(research_review_compact.get('release_gate_blocking', False))}",
        f"- research_review_requires_human_approval: {bool(research_review_compact.get('requires_human_approval', True))}",
        f"- research_review_next_hypothesis_count: {int(research_review_compact.get('next_hypothesis_count', 0) or 0)}",
        f"- research_review_regression_watchlist_count: {int(research_review_compact.get('regression_watchlist_count', 0) or 0)}",
        f"- research_review_negative_result_count: {int(research_review_compact.get('negative_result_count', 0) or 0)}",
        f"- research_review_bounded_experiment_graph_node_count: {int(research_review_compact.get('bounded_experiment_graph_node_count', 0) or 0)}",
        f"- research_review_bounded_experiment_graph_edge_count: {int(research_review_compact.get('bounded_experiment_graph_edge_count', 0) or 0)}",
        f"- research_review_sara_policy_dimension_count: {int(research_review_compact.get('sara_policy_dimension_count', 0) or 0)}",
        f"- research_review_sara_policy_needs_review_count: {int(research_review_compact.get('sara_policy_needs_review_count', 0) or 0)}",
        f"- research_review_experiment_adoption_candidate_count: {int(research_review_compact.get('experiment_adoption_candidate_count', 0) or 0)}",
        f"- research_review_experiment_regressing_item_count: {int(research_review_compact.get('experiment_regressing_item_count', 0) or 0)}",
        f"- research_review_experiment_falsified_item_count: {int(research_review_compact.get('experiment_falsified_item_count', 0) or 0)}",
        f"- research_review_experiment_human_review_pending_count: {int(research_review_compact.get('experiment_human_review_pending_count', 0) or 0)}",
        f"- research_review_experiment_priority_action_count: {int(research_review_compact.get('experiment_priority_action_count', 0) or 0)}",
        f"- research_review_experiment_top_priority_source: {str(research_review_compact.get('experiment_top_priority_source', '') or '')}",
        f"- research_review_experiment_top_priority_category: {str(research_review_compact.get('experiment_top_priority_category', '') or '')}",
        f"- research_review_experiment_promotion_target_candidate_count: {int(research_review_compact.get('experiment_promotion_target_candidate_count', 0) or 0)}",
        f"- research_review_experiment_promotion_target_review_action_count: {int(research_review_compact.get('experiment_promotion_target_review_action_count', 0) or 0)}",
        f"- research_review_roadmap_patch_rejection_suppressed_count: {int(research_review_compact.get('roadmap_patch_rejection_suppressed_count', 0) or 0)}",
        f"- research_review_roadmap_patch_rejection_refreshed_count: {int(research_review_compact.get('roadmap_patch_rejection_refreshed_count', 0) or 0)}",
        f"- research_planner_task_pending_count: {int(research_planner_task_status.get('pending_count', 0) or 0)}",
        f"- research_planner_task_completed_count: {int(research_planner_task_status.get('completed_count', 0) or 0)}",
        f"- research_planner_task_completion_ratio: {float(research_planner_task_status.get('completion_ratio', 1.0) or 0.0):.3f}",
        f"- research_planner_task_cleanup_needed: {bool(research_planner_task_status.get('cleanup_needed', False))}",
        f"- research_planner_task_cleanup_pending_count: {int(research_planner_task_status.get('cleanup_pending_count', 0) or 0)}",
        f"- research_planner_task_cleanup_stalled: {bool(research_planner_task_status.get('cleanup_stalled', False))}",
        f"- research_planner_task_cleanup_stalled_reason: {str(research_planner_task_status.get('cleanup_stalled_reason', '') or '')}",
        f"- research_review_next_hypothesis_ids: {', '.join(research_review_compact.get('next_hypothesis_ids', [])) if isinstance(research_review_compact.get('next_hypothesis_ids', []), list) else ''}",
        f"- research_review_regression_watchlist_ids: {', '.join(research_review_compact.get('regression_watchlist_ids', [])) if isinstance(research_review_compact.get('regression_watchlist_ids', []), list) else ''}",
        f"- research_journal_entry_count: {int(research_journal_summary.get('entry_count', 0) or 0)}",
        f"- research_journal_total_seen_count: {int(research_journal_summary.get('total_seen_count', 0) or 0)}",
        f"- research_journal_stale_age_seconds: {float(research_journal_summary.get('stale_age_seconds', 0.0) or 0.0):.1f}",
        f"- research_journal_remeasure_result_count: {int(research_journal_summary.get('remeasure_result_count', 0) or 0)}",
        f"- research_journal_remeasure_success_count: {int((research_journal_summary.get('remeasure_status_counts', {}) if isinstance(research_journal_summary.get('remeasure_status_counts'), dict) else {}).get('success', 0) or 0)}",
        f"- research_journal_remeasure_failed_count: {int((research_journal_summary.get('remeasure_status_counts', {}) if isinstance(research_journal_summary.get('remeasure_status_counts'), dict) else {}).get('failed', 0) or 0)}",
        f"- research_journal_alternative_probe_result_count: {int(research_journal_summary.get('alternative_probe_result_count', 0) or 0)}",
        f"- research_journal_remeasure_quota_hold_count: {int(research_journal_summary.get('remeasure_quota_hold_count', 0) or 0)}",
        f"- research_journal_completed_research_planner_task_count: {int(research_journal_summary.get('completed_research_planner_task_count', 0) or 0)}",
        f"- research_journal_completed_roadmap_patch_evidence_collection_count: {int(research_journal_summary.get('completed_roadmap_patch_evidence_collection_count', 0) or 0)}",
        f"- research_journal_roadmap_patch_approved_count: {int(research_journal_summary.get('roadmap_patch_review_approved_count', 0) or 0)}",
        f"- research_journal_roadmap_patch_rejected_count: {int(research_journal_summary.get('roadmap_patch_review_rejected_count', 0) or 0)}",
        f"- research_journal_roadmap_patch_rejected_item_count: {int(research_journal_summary.get('roadmap_patch_rejected_item_count', 0) or 0)}",
        f"- research_journal_roadmap_patch_refreshed_item_count: {int(research_journal_summary.get('roadmap_patch_refreshed_item_count', 0) or 0)}",
        f"- research_journal_roadmap_patch_refresh_to_rejection_ratio: {float(research_journal_summary.get('roadmap_patch_refresh_to_rejection_ratio', 0.0) or 0.0):.3f}",
        f"- research_journal_roadmap_patch_refresh_policy_status: {str(roadmap_patch_refresh_policy.get('status', '') or '')}",
        f"- research_journal_roadmap_patch_refresh_policy_needs_followup: {bool(roadmap_patch_refresh_policy.get('needs_followup', False))}",
        f"- research_journal_roadmap_patch_refresh_policy_followup_success_count: {int(research_journal_summary.get('roadmap_patch_refresh_policy_followup_success_count', 0) or 0)}",
        f"- research_journal_roadmap_patch_refresh_policy_followup_pending_count: {int(research_journal_summary.get('roadmap_patch_refresh_policy_followup_pending_count', 0) or 0)}",
        f"- research_journal_roadmap_patch_evidence_collection_success_count: {int(research_journal_summary.get('roadmap_patch_evidence_collection_success_count', 0) or 0)}",
        f"- research_journal_roadmap_patch_evidence_collection_latest_status: {str(research_journal_summary.get('roadmap_patch_evidence_collection_latest_status', '') or '')}",
        f"- research_journal_roadmap_patch_evidence_collection_latest_kind: {str(research_journal_summary.get('roadmap_patch_evidence_collection_latest_kind', '') or '')}",
        f"- research_journal_roadmap_patch_evidence_collection_next_required_kind: {str(research_journal_summary.get('roadmap_patch_evidence_collection_next_required_kind', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_needs_followup: {bool(stage_e_observed_candidate_repair_loop.get('needs_followup', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_remeasure_recommended: {bool(stage_e_observed_candidate_repair_loop.get('remeasure_recommended', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_remeasure_suppressed: {bool(stage_e_observed_candidate_repair_loop.get('remeasure_suppressed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_alternative_probe_recommended: {bool(stage_e_observed_candidate_repair_loop.get('alternative_probe_recommended', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_recovery_confirmed: {bool(stage_e_observed_candidate_repair_loop.get('recovery_confirmed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_recommended: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_recommended', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_completed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_in_progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_in_progress', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_stale: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_stale', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_in_progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_in_progress', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_completed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_failed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_latest_status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_latest_status', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_in_progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_in_progress', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_completed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_failed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_latest_status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_latest_status', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_escalation_in_progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_escalation_in_progress', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_escalation_completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_escalation_completed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_escalation_failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_escalation_failed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_escalation_latest_status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_escalation_latest_status', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_collection_in_progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_collection_in_progress', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_collection_completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_collection_completed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_collection_failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_collection_failed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_collection_latest_status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_collection_latest_status', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_recheck_in_progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_recheck_in_progress', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_recheck_completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_recheck_completed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_recheck_failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_recheck_failed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_recheck_latest_status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_recheck_latest_status', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_in_progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_in_progress', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_completed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_failed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_latest_status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_latest_status', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_recheck_in_progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_recheck_in_progress', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_recheck_completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_recheck_completed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_recheck_failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_recheck_failed', False))}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_recheck_latest_status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_recheck_latest_status', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_latest_status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_latest_status', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_latest_age_seconds: {float(stage_e_observed_candidate_repair_loop.get('promotion_review_latest_age_seconds', 0.0) or 0.0):.1f}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_recovery_source: {str(stage_e_observed_candidate_repair_loop.get('recovery_source', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_next_review_action: {str(stage_e_observed_candidate_repair_loop.get('next_review_action', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_latest_remeasure_trend: {str(stage_e_observed_candidate_repair_loop.get('latest_remeasure_trend', '') or '')}",
        f"- research_journal_stage_e_observed_acceptance_candidate_repair_latest_alternative_probe_trend: {str(stage_e_observed_candidate_repair_loop.get('latest_alternative_probe_trend', '') or '')}",
        f"- research_journal_experiment_priority_action_count: {int(research_journal_experiment_priority_plan.get('action_count', 0) or 0)}",
        f"- research_journal_experiment_top_priority_source: {str(research_journal_experiment_priority_plan.get('top_priority_source', '') or '')}",
        f"- research_journal_experiment_top_priority_category: {str(research_journal_experiment_priority_plan.get('top_priority_category', '') or '')}",
        f"- research_journal_experiment_promotion_target_candidate_count: {int(research_journal_experiment_promotion_target_plan.get('candidate_count', 0) or 0)}",
        f"- research_journal_experiment_promotion_target_review_action_count: {int(research_journal_experiment_promotion_target_plan.get('review_action_count', 0) or 0)}",
        f"- stage_b_passed: {bool(stage_b.get('stage_b_passed', False))}",
        f"- stage_b_minimum_requirements_passed: {bool(stage_b.get('minimum_requirements_passed', False))}",
        f"- stage_b_minimum_failure_count: {int(stage_b.get('minimum_failure_count', 0) or 0)}",
        f"- stage_b_readiness_score: {float(stage_b.get('readiness_score', 0.0) or 0.0):.3f}",
        f"- stage_b_promotion_candidate_ready: {bool(stage_b.get('promotion_candidate_ready', False))}",
        f"- stage_b_promotion_candidate_failure_count: {int(stage_b.get('promotion_candidate_failure_count', 0) or 0)}",
        f"- stage_b_promotion_candidate_promoted: {bool(stage_b.get('promotion_candidate_promoted', False))}",
        f"- stage_b_promotion_consecutive_passes: {int(stage_b.get('promotion_consecutive_passes', 0) or 0)}",
        f"- stage_b_promotion_required_streak: {int(stage_b.get('promotion_required_streak', 3) or 3)}",
        f"- stage_b_promotion_recommended: {bool(stage_b.get('promotion_recommended', False))}",
        f"- stage_b_promotion_next_step_hint: {str(stage_b.get('promotion_next_step_hint', '') or '')}",
        f"- stage_b_rlm_observation_candidate_ready: {bool(stage_b.get('rlm_observation_candidate_ready', False))}",
        f"- stage_b_rlm_observation_candidate_failure_count: {int(stage_b.get('rlm_observation_candidate_failure_count', 0) or 0)}",
        f"- stage_b_rlm_observation_candidate_promoted: {bool(stage_b.get('rlm_observation_candidate_promoted', False))}",
        f"- stage_b_rlm_observation_consecutive_passes: {int(stage_b.get('rlm_observation_consecutive_passes', 0) or 0)}",
        f"- stage_b_rlm_observation_required_streak: {int(stage_b.get('rlm_observation_required_streak', 3) or 3)}",
        f"- stage_b_rlm_observation_promotion_recommended: {bool(stage_b.get('rlm_observation_promotion_recommended', False))}",
        f"- stage_b_rlm_observation_next_step_hint: {str(stage_b.get('rlm_observation_next_step_hint', '') or '')}",
        f"- stage_d_passed: {bool(stage_d.get('passed', False))}",
        f"- stage_d_minimum_requirements_passed: {bool(stage_d.get('minimum_requirements_passed', False))}",
        f"- stage_d_minimum_failure_count: {int(stage_d.get('minimum_failure_count', 0) or 0)}",
        f"- stage_d_readiness_score: {float(stage_d.get('readiness_score', 0.0) or 0.0):.3f}",
        f"- stage_d_acceptance_candidate_count: {int(stage_d.get('acceptance_candidate_count', 0) or 0)}",
        f"- stage_d_acceptance_candidate_ready_count: {int(stage_d.get('acceptance_candidate_ready_count', 0) or 0)}",
        f"- stage_d_acceptance_candidates_ready: {bool(stage_d.get('acceptance_candidates_ready', False))}",
        f"- stage_d_acceptance_candidate_failure_count: {int(stage_d.get('acceptance_candidate_failure_count', 0) or 0)}",
        f"- stage_d_acceptance_candidate_consecutive_passes: {int(stage_d.get('acceptance_candidate_stability', {}).get('consecutive_passes', 0) if isinstance(stage_d.get('acceptance_candidate_stability'), dict) else 0)}",
        f"- stage_d_acceptance_candidate_required_streak: {int(stage_d.get('acceptance_candidate_stability', {}).get('required_streak', 3) if isinstance(stage_d.get('acceptance_candidate_stability'), dict) else 3)}",
        f"- stage_d_acceptance_candidate_stability_recommended: {bool(stage_d.get('acceptance_candidate_stability', {}).get('recommended', False) if isinstance(stage_d.get('acceptance_candidate_stability'), dict) else False)}",
        f"- stage_d_acceptance_candidate_next_step_hint: {str(stage_d.get('acceptance_candidate_next_step_hint', '') or '')}",
        f"- stage_d_acceptance_candidate_action_count: {int(stage_d.get('acceptance_candidate_action_count', 0) or 0)}",
        f"- stage_d_delta_memory_candidate_ready: {bool(stage_d.get('delta_memory_candidate_ready', False))}",
        f"- stage_d_delta_memory_candidate_failure_count: {int(stage_d.get('delta_memory_candidate_failure_count', 0) or 0)}",
        f"- stage_d_delta_memory_candidate_promoted: {bool(stage_d.get('delta_memory_candidate_promoted', False))}",
        f"- stage_d_delta_memory_consecutive_passes: {int(stage_d_delta_readiness.get('consecutive_passes', 0) or 0)}",
        f"- stage_d_delta_memory_required_streak: {int(stage_d_delta_readiness.get('required_streak', 3) or 3)}",
        f"- stage_d_delta_memory_promotion_recommended: {bool(stage_d_delta_readiness.get('recommended', False))}",
        f"- stage_d_replay_recovery_integrity: {float(stage_d.get('replay_recovery_integrity', 0.0) or 0.0):.3f}",
        f"- stage_d_replay_noise_resilience_integrity: {float(stage_d.get('replay_noise_resilience_integrity', 0.0) or 0.0):.3f}",
        f"- stage_d_astro_modulation_stability: {float(stage_d.get('astro_modulation_stability', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_continual_retention_observed: {float(stage_d.get('manifold_continual_retention_observed', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_trajectory_case_coverage_observed: {float(stage_d.get('manifold_trajectory_case_coverage_observed', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_average_case_recall_observed: {float(stage_d.get('manifold_average_case_recall_observed', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_scan_budget_integrity_observed: {float(stage_d.get('manifold_scan_budget_integrity_observed', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_indexed_candidate_integrity_observed: {float(stage_d.get('manifold_indexed_candidate_integrity_observed', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_index_scan_reduction_observed: {float(stage_d.get('manifold_index_scan_reduction_observed', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_capacity_pressure_recall_observed: {float(stage_d.get('manifold_capacity_pressure_recall_observed', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_capacity_pressure_scan_reduction_observed: {float(stage_d.get('manifold_capacity_pressure_scan_reduction_observed', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_replay_refresh_retention_observed: {float(stage_d.get('manifold_replay_refresh_retention_observed', 0.0) or 0.0):.3f}",
        f"- stage_d_manifold_replay_refresh_eviction_integrity_observed: {float(stage_d.get('manifold_replay_refresh_eviction_integrity_observed', 0.0) or 0.0):.3f}",
        *[
            f"- stage_d_{metric_name}: {float(stage_d.get(metric_name, 0.0) or 0.0):.3f}"
            for metric_name in STAGE_D_STRUCTURAL_OBSERVED_METRIC_NAMES
        ],
        f"- stage_e_passed: {bool(stage_e.get('passed', False))}",
        f"- stage_e_minimum_requirements_passed: {bool(stage_e.get('minimum_requirements_passed', False))}",
        f"- stage_e_minimum_failure_count: {int(stage_e.get('minimum_failure_count', 0) or 0)}",
        f"- stage_e_readiness_score: {float(stage_e.get('readiness_score', 0.0) or 0.0):.3f}",
        f"- stage_e_observed_acceptance_candidate_count: {int(stage_e.get('observed_acceptance_candidate_count', 0) or 0)}",
        f"- stage_e_observed_acceptance_candidate_ready_count: {int(stage_e.get('observed_acceptance_candidate_ready_count', 0) or 0)}",
        f"- stage_e_observed_acceptance_candidates_ready: {bool(stage_e.get('observed_acceptance_candidates_ready', False))}",
        f"- stage_e_observed_acceptance_candidate_failure_count: {int(stage_e.get('observed_acceptance_candidate_failure_count', 0) or 0)}",
        f"- stage_e_observed_acceptance_candidate_consecutive_passes: {int(stage_e.get('observed_acceptance_candidate_consecutive_passes', 0) or 0)}",
        f"- stage_e_observed_acceptance_candidate_required_streak: {int(stage_e.get('observed_acceptance_candidate_required_streak', 3) or 3)}",
        f"- stage_e_observed_acceptance_candidate_stability_recommended: {bool(stage_e.get('observed_acceptance_candidate_stability_recommended', False))}",
        f"- stage_e_causal_candidate_trace_integrity: {float(stage_e.get('causal_candidate_trace_integrity', 0.0) or 0.0):.3f}",
        f"- stage_e_module_orchestration_integrity: {float(stage_e.get('module_orchestration_integrity', 0.0) or 0.0):.3f}",
        f"- stage_e_counterfactual_lane_integrity: {float(stage_e.get('counterfactual_lane_integrity', 0.0) or 0.0):.3f}",
        f"- stage_e_action_trace_observability: {float(stage_e.get('action_trace_observability', 0.0) or 0.0):.3f}",
        f"- stage_e_runtime_trace_replay_consistency: {float(stage_e.get('runtime_trace_replay_consistency', 0.0) or 0.0):.3f}",
        *[
            f"- stage_e_{metric_name}: {float(stage_e.get(metric_name, 0.0) or 0.0):.3f}"
            for metric_name in COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES
        ],
        *[
            f"- stage_e_{metric_name}: {float(stage_e.get(metric_name, 0.0) or 0.0):.3f}"
            for metric_name in COGNITIVE_DELTA_MEMORY_METRIC_NAMES
        ],
        f"- stage_e_linear_snn_fusion_observed_policy: {str(stage_e.get('linear_snn_fusion_observed_policy', 'excluded_from_score_and_release_gate') or 'excluded_from_score_and_release_gate')}",
        f"- stage_e_linear_snn_fusion_trend_has_previous: {bool(stage_e.get('linear_snn_fusion_trend_has_previous', False))}",
        f"- stage_e_linear_snn_fusion_trend_regression_count: {int(stage_e.get('linear_snn_fusion_trend_regression_count', 0) or 0)}",
        f"- stage_e_linear_snn_fusion_trend_release_gate_blocking: {bool(stage_e.get('linear_snn_fusion_trend_release_gate_blocking', False))}",
        f"- stage_e_architecture_integration_observed_policy: {str(stage_e.get('architecture_integration_observed_policy', 'excluded_from_score_and_release_gate') or 'excluded_from_score_and_release_gate')}",
        f"- stage_e_architecture_integration_trend_has_previous: {bool(stage_e.get('architecture_integration_trend_has_previous', False))}",
        f"- stage_e_architecture_integration_trend_regression_count: {int(stage_e.get('architecture_integration_trend_regression_count', 0) or 0)}",
        f"- stage_e_architecture_integration_trend_release_gate_blocking: {bool(stage_e.get('architecture_integration_trend_release_gate_blocking', False))}",
        *[
            f"- stage_e_{metric_name}: {float(stage_e.get(metric_name, 0.0) or 0.0):.3f}"
            for metric_name in COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES
        ],
        *[
            f"- stage_e_{metric_name}: {float(stage_e.get(metric_name, 0.0) or 0.0):.3f}"
            for metric_name in COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES
        ],
        f"- neuromorphic_profile_history_regression_observed: {float(neuromorphic_profile.get('history_regression_observed', 0.0) or 0.0):.3f}",
        f"- neuromorphic_profile_report_integrity_observed: {float(neuromorphic_profile.get('profile_report_integrity_observed', 0.0) or 0.0):.3f}",
        f"- neuromorphic_backend_profile_compatibility_observed: {float(neuromorphic_profile.get('backend_profile_compatibility_observed', 0.0) or 0.0):.3f}",
        f"- neuromorphic_stage_e_state_trace_ir_observed: {float(neuromorphic_profile.get('stage_e_state_trace_ir_observed', 0.0) or 0.0):.3f}",
        f"- neuromorphic_stage_e_routing_hint_coverage_observed: {float(neuromorphic_profile.get('stage_e_routing_hint_coverage_observed', 0.0) or 0.0):.3f}",
        f"- neuromorphic_stage_e_online_update_policy_observed: {float(neuromorphic_profile.get('stage_e_online_update_policy_observed', 0.0) or 0.0):.3f}",
        f"- neuromorphic_stage_e_event_budget_observed: {float(neuromorphic_profile.get('stage_e_event_budget_observed', 0.0) or 0.0):.3f}",
        f"- neuromorphic_profile_trend_has_previous: {bool(neuromorphic_profile.get('trend_has_previous', False))}",
        f"- neuromorphic_profile_trend_regression_count: {int(neuromorphic_profile.get('trend_regression_count', 0) or 0)}",
        f"- neuromorphic_profile_trend_policy_change_count: {int(neuromorphic_profile.get('trend_policy_change_count', 0) or 0)}",
        f"- neuromorphic_profile_trend_regression_details: {str(neuromorphic_profile.get('trend_regression_detail_line', 'none') or 'none')}",
        f"- neuromorphic_profile_trend_policy_change_details: {str(neuromorphic_profile.get('trend_policy_change_detail_line', 'none') or 'none')}",
        f"- neuromorphic_profile_recovery_hint: {str(neuromorphic_profile.get('recovery_hint', '') or '')}",
        f"- phase5_entry_passed: {bool(phase5_entry.get('passed', False))}",
        f"- phase5_entry_readiness_score: {float(phase5_entry.get('readiness_score', 0.0) or 0.0):.3f}",
        f"- phase5_latent_transition_alignment: {float(phase5_entry.get('latent_transition_alignment', 0.0) or 0.0):.3f}",
        f"- phase5_prediction_error_observability: {float(phase5_entry.get('prediction_error_observability', 0.0) or 0.0):.3f}",
        f"- phase5_correction_event_coverage: {float(phase5_entry.get('correction_event_coverage', 0.0) or 0.0):.3f}",
        f"- phase5_anti_collapse_event_diversity: {float(phase5_entry.get('anti_collapse_event_diversity', 0.0) or 0.0):.3f}",
        f"- phase5_counterfactual_transition_separation: {float(phase5_entry.get('counterfactual_transition_separation', 0.0) or 0.0):.3f}",
        f"- phase5_multi_step_latent_chain_integrity: {float(phase5_entry.get('multi_step_latent_chain_integrity', 0.0) or 0.0):.3f}",
        f"- phase5_long_horizon_error_correction_convergence: {float(phase5_entry.get('long_horizon_error_correction_convergence', 0.0) or 0.0):.3f}",
        f"- phase5_horizon_bucket_stability: {float(phase5_entry.get('horizon_bucket_stability', 0.0) or 0.0):.3f}",
        f"- phase5_macro_action_effectiveness: {float(phase5_entry.get('macro_action_effectiveness', 0.0) or 0.0):.3f}",
        f"- phase5_subgoal_decomposition_integrity: {float(phase5_entry.get('subgoal_decomposition_integrity', 0.0) or 0.0):.3f}",
        f"- phase5_depth_selective_routing_integrity: {float(phase5_entry.get('depth_selective_routing_integrity', 0.0) or 0.0):.3f}",
        f"- phase5_micro_es_policy_refinement_integrity: {float(phase5_entry.get('micro_es_policy_refinement_integrity', 0.0) or 0.0):.3f}",
        f"- phase5_manifold_transition_locality_observed: {float(phase5_entry.get('manifold_transition_locality_observed', 0.0) or 0.0):.3f}",
        f"- phase5_manifold_rollout_stability_observed: {float(phase5_entry.get('manifold_rollout_stability_observed', 0.0) or 0.0):.3f}",
        f"- phase5_causal_route_sparsity_observed: {float(phase5_entry.get('causal_route_sparsity_observed', 0.0) or 0.0):.3f}",
        f"- phase5_withheld_trajectory_recall_observed: {float(phase5_entry.get('withheld_trajectory_recall_observed', 0.0) or 0.0):.3f}",
        f"- phase5_manifold_trajectory_case_coverage_observed: {float(phase5_entry.get('manifold_trajectory_case_coverage_observed', 0.0) or 0.0):.3f}",
        f"- phase5_manifold_average_case_recall_observed: {float(phase5_entry.get('manifold_average_case_recall_observed', 0.0) or 0.0):.3f}",
        f"- phase5_manifold_scan_budget_integrity_observed: {float(phase5_entry.get('manifold_scan_budget_integrity_observed', 0.0) or 0.0):.3f}",
        f"- phase5_manifold_indexed_candidate_integrity_observed: {float(phase5_entry.get('manifold_indexed_candidate_integrity_observed', 0.0) or 0.0):.3f}",
        f"- phase5_manifold_index_scan_reduction_observed: {float(phase5_entry.get('manifold_index_scan_reduction_observed', 0.0) or 0.0):.3f}",
        f"- phase5_manifold_candidate_miss_guard_observed: {float(phase5_entry.get('manifold_candidate_miss_guard_observed', 0.0) or 0.0):.3f}",
        f"- iterative_completed: {bool(iterative_plan.get('completed', False))}",
        f"- iterative_stalled: {bool(iterative_plan.get('stalled', False))}",
        f"- iterative_stop_reason: {str(iterative_plan.get('stop_reason', ''))}",
        f"- iterative_next_step_hint: {str(iterative_plan.get('next_step_hint', ''))}",
    ]
    minimum_failures = (
        stage_e.get("minimum_failures", [])
        if isinstance(stage_e.get("minimum_failures"), list)
        else []
    )
    for failure in minimum_failures[:5]:
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_e_minimum_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f}"
        )
    for failure in (
        stage_d.get("delta_memory_candidate_failures", [])
        if isinstance(stage_d.get("delta_memory_candidate_failures"), list)
        else []
    )[:5]:
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_d_delta_memory_candidate_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f} "
            f"description={_stage_d_candidate_failure_description(failure)}"
        )
    for failure in (
        stage_d.get("acceptance_candidate_failures", [])
        if isinstance(stage_d.get("acceptance_candidate_failures"), list)
        else []
    )[:5]:
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_d_acceptance_candidate_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f} "
            f"description={_stage_d_candidate_failure_description(failure)}"
        )
    for failure in (
        stage_e.get("observed_acceptance_candidate_failures", [])
        if isinstance(stage_e.get("observed_acceptance_candidate_failures"), list)
        else []
    )[:5]:
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_e_observed_acceptance_candidate_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f} "
            f"description={_stage_e_observed_candidate_failure_description(failure)}"
        )
    promotion_actions = (
        stage_b.get("promotion_actions", [])
        if isinstance(stage_b.get("promotion_actions"), list)
        else []
    )
    for action in promotion_actions[:5]:
        lines.append(f"- stage_b_promotion_action: {str(action)}")
    rlm_observation_actions = (
        stage_b.get("rlm_observation_actions", [])
        if isinstance(stage_b.get("rlm_observation_actions"), list)
        else []
    )
    for action in rlm_observation_actions[:5]:
        lines.append(f"- stage_b_rlm_observation_action: {str(action)}")
    stage_d_acceptance_candidate_actions = (
        stage_d.get("acceptance_candidate_actions", [])
        if isinstance(stage_d.get("acceptance_candidate_actions"), list)
        else []
    )
    for action in stage_d_acceptance_candidate_actions[:5]:
        lines.append(f"- stage_d_acceptance_candidate_action: {str(action)}")
    lines.append(f"- repair_plan_steps: {int(repair_plan.get('estimated_steps', 0) or 0)}")
    lines.append(
        f"- repair_plan_coverage: "
        f"{len(repair_plan.get('covered_checks', []) if isinstance(repair_plan.get('covered_checks', []), list) else [])}/"
        f"{len(repair_plan.get('covered_checks', []) if isinstance(repair_plan.get('covered_checks', []), list) else []) + len(repair_plan.get('uncovered_checks', []) if isinstance(repair_plan.get('uncovered_checks', []), list) else [])}"
    )
    lines.append(f"- failure_focus_primary_category: {str(failure_focus.get('primary_category', ''))}")
    lines.append(f"- failure_focus_secondary_category: {str(failure_focus.get('secondary_category', ''))}")
    lines.append(f"- failure_focus_confidence: {float(failure_focus.get('confidence', 0.0) or 0.0):.3f}")
    lines.append(f"- repair_retry_queue_count: {int(report.get('repair_retry_queue_count', 0) or 0)}")
    lines.append(f"- repair_retry_cooldown_seconds: {float(report.get('repair_retry_cooldown_seconds', 0.0) or 0.0):.1f}")
    lines.append(f"- repair_retry_cooldown_blocked_count: {int(report.get('repair_retry_cooldown_blocked_count', 0) or 0)}")
    v1_actions_snapshot = (
        report.get("v1_actions_snapshot", {})
        if isinstance(report.get("v1_actions_snapshot"), dict)
        else {}
    )
    operational_checklist = (
        report.get("operational_checklist", {})
        if isinstance(report.get("operational_checklist"), dict)
        else {}
    )
    execution_log = (
        report.get("execution_log", [])
        if isinstance(report.get("execution_log"), list)
        else []
    )
    research_journal_summary = attach_roadmap_patch_refresh_policy_followups_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    research_journal_summary = attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    research_review = (
        report.get("research_review", {})
        if isinstance(report.get("research_review"), dict)
        else {}
    )
    research_review_compact = (
        research_review.get("compact", {})
        if isinstance(research_review.get("compact"), dict)
        else {}
    )
    runbook_drop_rate_threshold = float(report.get("runbook_drop_rate_threshold", 0.9) or 0.9)
    lines.append(f"- v1_actions_loaded_count: {int(v1_actions_snapshot.get('loaded_count', 0) or 0)}")
    lines.append(f"- v1_actions_accepted_count: {int(v1_actions_snapshot.get('accepted_count', 0) or 0)}")
    lines.append(f"- v1_actions_rejected_stale_count: {int(v1_actions_snapshot.get('rejected_stale_count', 0) or 0)}")
    lines.append(f"- v1_actions_rejected_missing_timestamp_count: {int(v1_actions_snapshot.get('rejected_missing_timestamp_count', 0) or 0)}")
    lines.append(f"- repair_pending_count: {int(report.get('repair_pending_count', 0) or 0)}")
    lines.append(f"- repair_timeout_count: {int(report.get('repair_timeout_count', 0) or 0)}")
    lines.append(f"- runbook_max_actions: {int(report.get('runbook_max_actions', 50) or 50)}")
    lines.append(f"- runbook_max_per_source: {int(report.get('runbook_max_per_source', 0) or 0)}")
    runbook_action_summary = (
        report.get("runbook_action_summary", {})
        if isinstance(report.get("runbook_action_summary"), dict)
        else {}
    )
    lines.append(f"- runbook_action_total: {int(runbook_action_summary.get('total_actions', 0) or 0)}")
    source_counts = (
        runbook_action_summary.get("source_counts", {})
        if isinstance(runbook_action_summary.get("source_counts"), dict)
        else {}
    )
    efficiency_shortcut_action_count = int(source_counts.get("efficiency_incident_shortcut", 0) or 0)
    lines.append(f"- efficiency_incident_shortcut_action_count: {efficiency_shortcut_action_count}")
    for source, count in list(source_counts.items())[:8]:
        lines.append(f"- runbook_action_source_count: {source}={int(count or 0)}")
    runbook_action_build_stats = (
        report.get("runbook_action_build_stats", {})
        if isinstance(report.get("runbook_action_build_stats"), dict)
        else {}
    )
    runbook_action_build_rates = (
        report.get("runbook_action_build_rates", {})
        if isinstance(report.get("runbook_action_build_rates"), dict)
        else {}
    )
    runbook_action_build_rates = (
        report.get("runbook_action_build_rates", {})
        if isinstance(report.get("runbook_action_build_rates"), dict)
        else {}
    )
    runbook_action_build_rates = (
        report.get("runbook_action_build_rates", {})
        if isinstance(report.get("runbook_action_build_rates"), dict)
        else {}
    )
    runbook_action_build_rates = (
        report.get("runbook_action_build_rates", {})
        if isinstance(report.get("runbook_action_build_rates"), dict)
        else {}
    )
    lines.append(
        f"- runbook_action_considered_count: {int(runbook_action_build_stats.get('considered_count', 0) or 0)}"
    )
    lines.append(
        f"- runbook_action_skipped_duplicate_count: {int(runbook_action_build_stats.get('skipped_duplicate_count', 0) or 0)}"
    )
    lines.append(
        f"- runbook_action_skipped_empty_command_count: {int(runbook_action_build_stats.get('skipped_empty_command_count', 0) or 0)}"
    )
    skipped_empty_command_by_source = (
        runbook_action_build_stats.get("skipped_empty_command_by_source", {})
        if isinstance(runbook_action_build_stats.get("skipped_empty_command_by_source"), dict)
        else {}
    )
    for source, count in list(skipped_empty_command_by_source.items())[:8]:
        lines.append(f"- runbook_action_skipped_empty_command_by_source: {source}={int(count or 0)}")
    skipped_duplicate_by_source = (
        runbook_action_build_stats.get("skipped_duplicate_by_source", {})
        if isinstance(runbook_action_build_stats.get("skipped_duplicate_by_source"), dict)
        else {}
    )
    for source, count in list(skipped_duplicate_by_source.items())[:8]:
        lines.append(f"- runbook_action_skipped_duplicate_by_source: {source}={int(count or 0)}")
    lines.append(
        f"- runbook_action_skipped_source_cap_count: {int(runbook_action_build_stats.get('skipped_source_cap_count', 0) or 0)}"
    )
    skipped_source_cap_by_source = (
        runbook_action_build_stats.get("skipped_source_cap_by_source", {})
        if isinstance(runbook_action_build_stats.get("skipped_source_cap_by_source"), dict)
        else {}
    )
    for source, count in list(skipped_source_cap_by_source.items())[:8]:
        lines.append(f"- runbook_action_skipped_source_cap_by_source: {source}={int(count or 0)}")
    lines.append(
        f"- runbook_action_skipped_max_actions_count: {int(runbook_action_build_stats.get('skipped_max_actions_count', 0) or 0)}"
    )
    skipped_max_actions_by_source = (
        runbook_action_build_stats.get("skipped_max_actions_by_source", {})
        if isinstance(runbook_action_build_stats.get("skipped_max_actions_by_source"), dict)
        else {}
    )
    for source, count in list(skipped_max_actions_by_source.items())[:8]:
        lines.append(f"- runbook_action_skipped_max_actions_by_source: {source}={int(count or 0)}")
    runbook_action_build_rates = (
        report.get("runbook_action_build_rates", {})
        if isinstance(report.get("runbook_action_build_rates"), dict)
        else {}
    )
    lines.append(f"- runbook_action_drop_rate: {float(runbook_action_build_rates.get('drop_rate', 0.0) or 0.0):.3f}")
    lines.append(
        f"- runbook_action_duplicate_drop_rate: {float(runbook_action_build_rates.get('duplicate_drop_rate', 0.0) or 0.0):.3f}"
    )
    lines.append(
        f"- runbook_action_empty_drop_rate: {float(runbook_action_build_rates.get('empty_drop_rate', 0.0) or 0.0):.3f}"
    )
    lines.append(
        f"- runbook_action_source_cap_drop_rate: {float(runbook_action_build_rates.get('source_cap_drop_rate', 0.0) or 0.0):.3f}"
    )
    lines.append(
        f"- runbook_action_max_actions_drop_rate: {float(runbook_action_build_rates.get('max_actions_drop_rate', 0.0) or 0.0):.3f}"
    )
    lines.append(f"- auto_dispatch_requested: {int(auto_dispatch.get('requested', 0) or 0)}")
    lines.append(f"- auto_dispatch_candidates: {int(auto_dispatch.get('candidate_count', 0) or 0)}")
    lines.append(f"- auto_dispatch_eligible: {int(auto_dispatch.get('eligible_count', 0) or 0)}")
    lines.append(f"- auto_dispatch_selected: {int(auto_dispatch.get('selected_count', 0) or 0)}")
    lines.append(f"- auto_dispatch_selected_unique_checks: {int(auto_dispatch.get('selected_unique_check_count', 0) or 0)}")
    lines.append(f"- auto_dispatch_min_priority_tier: {str(auto_dispatch.get('min_priority_tier', 'low'))}")
    lines.append(f"- auto_dispatch_selection_mode: {str(auto_dispatch.get('selection_mode', 'priority'))}")
    lines.append(f"- auto_dispatch_max_per_check: {int(auto_dispatch.get('max_per_check', 0) or 0)}")
    lines.append(f"- auto_dispatch_dispatched: {int(auto_dispatch.get('dispatched', 0) or 0)}")
    lines.append(f"- auto_dispatch_skipped_pending: {len(auto_dispatch.get('skipped_pending_commands', [])) if isinstance(auto_dispatch.get('skipped_pending_commands', []), list) else 0}")
    lines.append(f"- auto_dispatch_skipped_limit: {len(auto_dispatch.get('skipped_limit_commands', [])) if isinstance(auto_dispatch.get('skipped_limit_commands', []), list) else 0}")
    lines.append(f"- auto_dispatch_skipped_low_priority: {int(auto_dispatch.get('skipped_low_priority_count', 0) or 0)}")
    lines.append(f"- auto_dispatch_skipped_check_quota: {int(auto_dispatch.get('skipped_check_quota_count', 0) or 0)}")
    lines.append(f"- error_detail_count: {len(error_details)}")
    efficiency_kpi_failure_count = int(
        sum(
            1
            for item in error_details
            if isinstance(item, dict)
            and str(item.get("category", "")).strip().lower().endswith("efficiency_kpi")
        )
    )
    lines.append(f"- efficiency_kpi_failure_count: {efficiency_kpi_failure_count}")
    lines.append(
        f"- efficiency_shortcut_overuse_event_count: {int(report.get('efficiency_shortcut_overuse_event_count', 0) or 0)}"
    )
    if error_details_summary:
        lines.append(f"- error_detail_total: {int(error_details_summary.get('total', len(error_details)) or 0)}")
        top_types = (
            error_details_summary.get("top_types", [])
            if isinstance(error_details_summary.get("top_types"), list)
            else []
        )
        if top_types:
            for item in top_types[:5]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- error_detail_type_count: {item.get('name', '')}={int(item.get('count', 0) or 0)}"
                )
        top_categories = (
            error_details_summary.get("top_categories", [])
            if isinstance(error_details_summary.get("top_categories"), list)
            else []
        )
        if top_categories:
            for item in top_categories[:5]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- error_detail_category_count: {item.get('name', '')}={int(item.get('count', 0) or 0)}"
                )

    for section_name in ("phase3_accuracy", "phase3_completion", "phase4_completion", "release_gate", "production_profile"):
        section = checks.get(section_name, {}) if isinstance(checks.get(section_name), dict) else {}
        errors = section.get("errors", []) if isinstance(section.get("errors"), list) else []
        lines.append(f"- {section_name}_error_count: {len(errors)}")
        for index, error in enumerate(errors[:5], start=1):
            lines.append(f"  {section_name}.error[{index}]: {str(error)}")
    recovery_actions = report.get("recovery_actions", []) if isinstance(report.get("recovery_actions"), list) else []
    lines.append(f"- recovery_action_count: {len(recovery_actions)}")
    for index, action in enumerate(recovery_actions[:5], start=1):
        if not isinstance(action, dict):
            continue
        lines.append(
            f"  recovery_action[{index}]: {str(action.get('title', ''))} -> {str(action.get('command', ''))}"
        )
    iterative_actions = (
        iterative_plan.get("next_actions", [])
        if isinstance(iterative_plan.get("next_actions"), list)
        else []
    )
    lines.append(f"- iterative_action_count: {len(iterative_actions)}")
    for action in iterative_actions[:5]:
        if not isinstance(action, dict):
            continue
        lines.append(
            f"  iterative_action[{int(action.get('step', 0) or 0)}]: "
            f"{str(action.get('title', ''))} ({str(action.get('priority', 'low'))}) -> {str(action.get('command', ''))}"
        )
    fallback_actions = (
        repair_plan.get("fallback_actions", [])
        if isinstance(repair_plan.get("fallback_actions"), list)
        else []
    )
    for action in fallback_actions[:3]:
        if not isinstance(action, dict):
            continue
        lines.append(
            f"  fallback_action[{int(action.get('step', 0) or 0)}]: "
            f"{str(action.get('title', ''))} -> {str(action.get('command', ''))}"
        )
    for retry in retry_queue[:5]:
        if not isinstance(retry, dict):
            continue
        checks_text = ", ".join(retry.get("covered_checks", [])) if isinstance(retry.get("covered_checks"), list) else ""
        lines.append(
            "- retry_queue_entry: "
            f"{str(retry.get('command', ''))} "
            f"(reason={str(retry.get('reason', ''))}, attempt={int(retry.get('next_attempt', 0) or 0)}/{int(retry.get('max_attempts', 0) or 0)}, "
            f"priority={str(retry.get('priority_tier', ''))}, score={float(retry.get('priority_score', 0.0) or 0.0):.3f}, checks={checks_text})"
        )
    for blocked in retry_cooldown_blocked[:5]:
        if not isinstance(blocked, dict):
            continue
        checks_text = ", ".join(blocked.get("covered_checks", [])) if isinstance(blocked.get("covered_checks"), list) else ""
        lines.append(
            "- retry_cooldown_blocked_entry: "
            f"{str(blocked.get('command', ''))} "
            f"(reason={str(blocked.get('reason', ''))}, attempt={int(blocked.get('next_attempt', 0) or 0)}/{int(blocked.get('max_attempts', 0) or 0)}, "
            f"priority={str(blocked.get('priority_tier', ''))}, score={float(blocked.get('priority_score', 0.0) or 0.0):.3f}, "
            f"cooldown_remaining_seconds={float(blocked.get('cooldown_remaining_seconds', 0.0) or 0.0):.1f}, checks={checks_text})"
        )
    dispatched_commands = (
        auto_dispatch.get("dispatched_commands", [])
        if isinstance(auto_dispatch.get("dispatched_commands"), list)
        else []
    )
    for command in dispatched_commands[:5]:
        lines.append(f"- auto_dispatch_command: {command}")
    skipped_pending_commands = (
        auto_dispatch.get("skipped_pending_commands", [])
        if isinstance(auto_dispatch.get("skipped_pending_commands"), list)
        else []
    )
    for command in skipped_pending_commands[:5]:
        lines.append(f"- auto_dispatch_skipped_pending_command: {command}")
    skipped_limit_commands = (
        auto_dispatch.get("skipped_limit_commands", [])
        if isinstance(auto_dispatch.get("skipped_limit_commands"), list)
        else []
    )
    for command in skipped_limit_commands[:5]:
        lines.append(f"- auto_dispatch_skipped_limit_command: {command}")
    skipped_low_priority_commands = (
        auto_dispatch.get("skipped_low_priority_commands", [])
        if isinstance(auto_dispatch.get("skipped_low_priority_commands"), list)
        else []
    )
    for command in skipped_low_priority_commands[:5]:
        lines.append(f"- auto_dispatch_skipped_low_priority_command: {command}")
    skipped_check_quota_commands = (
        auto_dispatch.get("skipped_check_quota_commands", [])
        if isinstance(auto_dispatch.get("skipped_check_quota_commands"), list)
        else []
    )
    for command in skipped_check_quota_commands[:5]:
        lines.append(f"- auto_dispatch_skipped_check_quota_command: {command}")
    if checklist:
        lines.extend(
            [
                "Checklist",
                f"- status: {'PASS' if bool(checklist.get('passed', False)) else 'FAIL'}",
                f"- managed_output_paths_ok: {bool(checklist.get('managed_output_paths_ok', False))}",
                f"- report_summary_review_ready: {bool(checklist.get('report_summary_review_ready', False))}",
                f"- runbook_manifest_hygiene_ok: {bool(checklist.get('runbook_manifest_hygiene_ok', False))}",
                f"- runbook_drop_rate_ok: {bool(checklist.get('runbook_drop_rate_ok', False))}",
                f"- runbook_drop_rate_threshold: {float(checklist.get('runbook_drop_rate_threshold', 0.9) or 0.9):.3f}",
                f"- efficiency_shortcut_action_ok: {bool(checklist.get('efficiency_shortcut_action_ok', True))}",
                f"- efficiency_shortcut_action_count: {int(checklist.get('efficiency_shortcut_action_count', 0) or 0)}",
                f"- efficiency_shortcut_action_threshold: {int(checklist.get('efficiency_shortcut_action_threshold', 0) or 0)}",
                f"- efficiency_shortcut_overuse_rate_ok: {bool(checklist.get('efficiency_shortcut_overuse_rate_ok', True))}",
                f"- efficiency_shortcut_overuse_window: {int(checklist.get('efficiency_shortcut_overuse_window', 0) or 0)}",
                f"- efficiency_shortcut_overuse_rate_threshold: {float(checklist.get('efficiency_shortcut_overuse_rate_threshold', 0.0) or 0.0):.3f}",
                f"- efficiency_shortcut_overuse_rate: {float(checklist.get('efficiency_shortcut_overuse_rate', 0.0) or 0.0):.3f}",
            ]
        )
    return "\n".join(lines)


def build_operational_runbook(report: Dict[str, Any]) -> str:
    failure_focus = (
        report.get("failure_focus", {})
        if isinstance(report.get("failure_focus"), dict)
        else {}
    )
    iterative_plan = (
        report.get("iterative_repair_plan", {})
        if isinstance(report.get("iterative_repair_plan"), dict)
        else {}
    )
    retry_queue = (
        report.get("repair_retry_queue", [])
        if isinstance(report.get("repair_retry_queue"), list)
        else []
    )
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    failed_checks = sorted(
        name
        for name, payload in checks.items()
        if isinstance(payload, dict) and not bool(payload.get("passed", False))
    )
    runbook_actions = (
        report.get("runbook_actions", [])
        if isinstance(report.get("runbook_actions"), list)
        else build_operational_runbook_actions(report)
    )
    efficiency_shortcut_action_count = int(
        sum(
            1
            for item in runbook_actions
            if isinstance(item, dict) and str(item.get("source", "")).strip() == "efficiency_incident_shortcut"
        )
    )
    runbook_action_build_stats = (
        report.get("runbook_action_build_stats", {})
        if isinstance(report.get("runbook_action_build_stats"), dict)
        else {}
    )
    runbook_action_build_rates = (
        report.get("runbook_action_build_rates", {})
        if isinstance(report.get("runbook_action_build_rates"), dict)
        else {}
    )
    error_details = (
        report.get("error_details", [])
        if isinstance(report.get("error_details"), list)
        else []
    )
    efficiency_kpi_details = [
        dict(item)
        for item in error_details
        if isinstance(item, dict)
        and str(item.get("category", "")).strip().lower().endswith("efficiency_kpi")
    ]
    checklist = (
        report.get("operational_checklist", {})
        if isinstance(report.get("operational_checklist"), dict)
        else {}
    )
    research_review = (
        report.get("research_review", {})
        if isinstance(report.get("research_review"), dict)
        else {}
    )
    research_review_report = (
        research_review.get("report", {})
        if isinstance(research_review.get("report"), dict)
        else {}
    )
    research_review_compact = (
        research_review.get("compact", {})
        if isinstance(research_review.get("compact"), dict)
        else {}
    )
    execution_log = (
        report.get("execution_log", [])
        if isinstance(report.get("execution_log"), list)
        else []
    )
    research_journal_summary = (
        report.get("research_journal_summary", {})
        if isinstance(report.get("research_journal_summary"), dict)
        else {}
    )
    research_journal_summary = attach_roadmap_patch_refresh_policy_followups_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    research_journal_summary = attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    research_planner_task_status = summarize_research_planner_task_status(
        research_review_compact,
        research_journal_summary,
        cleanup_threshold=int(report.get("research_planner_task_cleanup_threshold", 2) or 2),
    )
    roadmap_patch_refresh_policy = summarize_roadmap_patch_refresh_policy(
        research_journal_summary
    )
    lines = [
        "# SARA Engine Operational Runbook",
        "",
        "## Status Snapshot",
        f"- Operational status: {'PASS' if bool(report.get('passed', False)) else 'FAIL'}",
        f"- Readiness score: {float(report.get('readiness_score', 0.0) or 0.0):.3f}",
        f"- Error count: {int(report.get('error_count', 0) or 0)}",
        f"- Strict production: {bool(report.get('strict_production', False))}",
        f"- Efficiency incident shortcut actions: {efficiency_shortcut_action_count}",
        "",
        "## Failure Focus",
        f"- Primary category: {str(failure_focus.get('primary_category', '') or '')}",
        f"- Secondary category: {str(failure_focus.get('secondary_category', '') or '')}",
        f"- Primary action: {str(failure_focus.get('primary_action', '') or '')}",
        f"- Confidence: {float(failure_focus.get('confidence', 0.0) or 0.0):.3f}",
    ]
    if efficiency_kpi_details:
        lines.extend(
            [
                "",
                "## Efficiency KPI Incident",
                f"- Failure count: {len(efficiency_kpi_details)}",
            ]
        )
        categories = sorted(
            {
                str(item.get("category", "")).strip()
                for item in efficiency_kpi_details
                if str(item.get("category", "")).strip()
            }
        )
        if categories:
            lines.append(f"- Categories: {', '.join(categories)}")
        for item in efficiency_kpi_details[:3]:
            lines.append(f"- detail: {str(item.get('error', '')).strip()}")
        lines.extend(
            [
                "- Immediate commands:",
                "  1. `python scripts/eval/energy_efficiency_benchmark.py`",
                "  2. `python scripts/eval/phase3_accuracy_suite.py`",
                "  3. `python scripts/eval/release_gate.py`",
            ]
        )
    if not bool(checklist.get("efficiency_shortcut_action_ok", True)):
        overuse_count = int(checklist.get("efficiency_shortcut_action_count", 0) or 0)
        overuse_threshold = int(checklist.get("efficiency_shortcut_action_threshold", 0) or 0)
        lines.extend(
            [
                "",
                "## Efficiency Shortcut Overuse Incident",
                f"- Shortcut action count: {overuse_count}",
                f"- Threshold: {overuse_threshold}",
                "- Interpretation: recurring shortcut dependence is above steady-state operational budget.",
                "- Immediate commands:",
                "  1. `python scripts/eval/energy_efficiency_benchmark.py`",
                "  2. `python scripts/eval/phase3_accuracy_suite.py`",
                "  3. `python scripts/eval/operational_readiness.py --strict-production`",
            ]
        )
    if research_review_report or research_review_compact:
        patch_suggestion = (
            build_roadmap_patch_suggestion(research_review_report)
            if research_review_report
            else {"suggestions": []}
        )
        suggestions = (
            patch_suggestion.get("suggestions", [])
            if isinstance(patch_suggestion.get("suggestions"), list)
            else []
        )
        review_decision = latest_roadmap_patch_review_decision(
            execution_log,
            review_generated_at=_safe_float(research_review_report.get("generated_at", 0.0), 0.0),
        )
        lines.extend(
            [
                "",
                "## Roadmap Patch Review",
                f"- Review status: {'PASS' if bool(research_review_compact.get('passed', False)) else 'NEEDS_REVIEW'}",
                f"- Review score: {float(research_review_compact.get('review_score', 0.0) or 0.0):.3f}",
                f"- Requires human approval: {bool(research_review_compact.get('requires_human_approval', True))}",
                f"- Apply automatically: {bool(patch_suggestion.get('apply_automatically', False))}",
                f"- Next hypothesis count: {int(research_review_compact.get('next_hypothesis_count', 0) or 0)}",
                f"- Regression watchlist count: {int(research_review_compact.get('regression_watchlist_count', 0) or 0)}",
                f"- Negative result count: {int(research_review_compact.get('negative_result_count', 0) or 0)}",
                f"- Cause boundary documentation count: {int(research_review_compact.get('cause_boundary_documentation_count', 0) or 0)}",
                f"- Targeted fixture repair count: {int(research_review_compact.get('targeted_fixture_repair_count', 0) or 0)}",
                f"- Roadmap patch rejection suppressed count: {int(research_review_compact.get('roadmap_patch_rejection_suppressed_count', 0) or 0)}",
                f"- Roadmap patch rejection refreshed count: {int(research_review_compact.get('roadmap_patch_rejection_refreshed_count', 0) or 0)}",
                f"- Experiment priority action count: {int(research_review_compact.get('experiment_priority_action_count', 0) or 0)}",
                f"- Experiment top priority source: {str(research_review_compact.get('experiment_top_priority_source', '') or '')}",
                f"- Experiment top priority category: {str(research_review_compact.get('experiment_top_priority_category', '') or '')}",
                f"- Experiment promotion target candidate count: {int(research_review_compact.get('experiment_promotion_target_candidate_count', 0) or 0)}",
                f"- Experiment promotion target review action count: {int(research_review_compact.get('experiment_promotion_target_review_action_count', 0) or 0)}",
                f"- Planner task pending count: {int(research_planner_task_status.get('pending_count', 0) or 0)}",
                f"- Planner task completed count: {int(research_planner_task_status.get('completed_count', 0) or 0)}",
                f"- Planner task completion ratio: {float(research_planner_task_status.get('completion_ratio', 1.0) or 0.0):.3f}",
                f"- Planner task cleanup needed: {bool(research_planner_task_status.get('cleanup_needed', False))}",
                f"- Planner task cleanup pending count: {int(research_planner_task_status.get('cleanup_pending_count', 0) or 0)}",
                f"- Planner task cleanup stalled: {bool(research_planner_task_status.get('cleanup_stalled', False))}",
                f"- Planner task cleanup stalled reason: {str(research_planner_task_status.get('cleanup_stalled_reason', '') or '')}",
                f"- Review decision recorded: {bool(review_decision.get('available', False))}",
                f"- Review decision: {str(review_decision.get('decision', '') or '')}",
                f"- Review decision reason: {str(review_decision.get('reason', '') or '')}",
            ]
        )
        if suggestions:
            lines.append("- Suggested roadmap review items:")
            for index, suggestion in enumerate(suggestions[:5], start=1):
                lines.append(f"  {index}. {str(suggestion)}")
        else:
            lines.append("- Suggested roadmap review items: none")
    if research_journal_summary:
        completed_evidence_review = summarize_completed_roadmap_patch_evidence_review(
            research_journal_summary
        )
        journal_experiment_status = (
            research_journal_summary.get("experiment_status_summary", {})
            if isinstance(research_journal_summary.get("experiment_status_summary"), dict)
            else {}
        )
        journal_experiment_priority_plan = (
            research_journal_summary.get("experiment_priority_plan", {})
            if isinstance(research_journal_summary.get("experiment_priority_plan"), dict)
            else {}
        )
        journal_experiment_promotion_target_plan = (
            research_journal_summary.get("experiment_promotion_target_plan", {})
            if isinstance(research_journal_summary.get("experiment_promotion_target_plan"), dict)
            else {}
        )
        stage_e_observed_candidate_repair_loop = (
            research_journal_summary.get("stage_e_observed_acceptance_candidate_repair_loop", {})
            if isinstance(
                research_journal_summary.get("stage_e_observed_acceptance_candidate_repair_loop"),
                dict,
            )
            else {}
        )
        lines.extend(
            [
                "",
                "## Research Journal Summary",
                f"- Entry count: {int(research_journal_summary.get('entry_count', 0) or 0)}",
                f"- Total seen count: {int(research_journal_summary.get('total_seen_count', 0) or 0)}",
                f"- Stale age seconds: {float(research_journal_summary.get('stale_age_seconds', 0.0) or 0.0):.1f}",
                f"- Remeasure result count: {int(research_journal_summary.get('remeasure_result_count', 0) or 0)}",
                f"- Alternative probe result count: {int(research_journal_summary.get('alternative_probe_result_count', 0) or 0)}",
                f"- Remeasure quota hold count: {int(research_journal_summary.get('remeasure_quota_hold_count', 0) or 0)}",
                f"- Completed research planner task count: {int(research_journal_summary.get('completed_research_planner_task_count', 0) or 0)}",
                f"- Completed roadmap patch evidence collection count: {int(research_journal_summary.get('completed_roadmap_patch_evidence_collection_count', 0) or 0)}",
                f"- Roadmap patch approved count: {int(research_journal_summary.get('roadmap_patch_review_approved_count', 0) or 0)}",
                f"- Roadmap patch rejected count: {int(research_journal_summary.get('roadmap_patch_review_rejected_count', 0) or 0)}",
                f"- Roadmap patch rejected item count: {int(research_journal_summary.get('roadmap_patch_rejected_item_count', 0) or 0)}",
                f"- Roadmap patch refreshed item count: {int(research_journal_summary.get('roadmap_patch_refreshed_item_count', 0) or 0)}",
                f"- Roadmap patch refresh to rejection ratio: {float(research_journal_summary.get('roadmap_patch_refresh_to_rejection_ratio', 0.0) or 0.0):.3f}",
                f"- Roadmap patch refresh policy status: {str(roadmap_patch_refresh_policy.get('status', '') or '')}",
                f"- Roadmap patch refresh policy needs followup: {bool(roadmap_patch_refresh_policy.get('needs_followup', False))}",
                f"- Roadmap patch refresh policy followup success count: {int(research_journal_summary.get('roadmap_patch_refresh_policy_followup_success_count', 0) or 0)}",
                f"- Roadmap patch refresh policy followup pending count: {int(research_journal_summary.get('roadmap_patch_refresh_policy_followup_pending_count', 0) or 0)}",
                f"- Roadmap patch evidence collection success count: {int(research_journal_summary.get('roadmap_patch_evidence_collection_success_count', 0) or 0)}",
                f"- Roadmap patch evidence collection latest status: {str(research_journal_summary.get('roadmap_patch_evidence_collection_latest_status', '') or '')}",
                f"- Roadmap patch evidence collection latest kind: {str(research_journal_summary.get('roadmap_patch_evidence_collection_latest_kind', '') or '')}",
                f"- Roadmap patch evidence collection next required kind: {str(research_journal_summary.get('roadmap_patch_evidence_collection_next_required_kind', '') or '')}",
                f"- Stage E observed acceptance candidate repair needs followup: {bool(stage_e_observed_candidate_repair_loop.get('needs_followup', False))}",
                f"- Stage E observed acceptance candidate repair remeasure recommended: {bool(stage_e_observed_candidate_repair_loop.get('remeasure_recommended', False))}",
                f"- Stage E observed acceptance candidate repair remeasure suppressed: {bool(stage_e_observed_candidate_repair_loop.get('remeasure_suppressed', False))}",
                f"- Stage E observed acceptance candidate repair alternative probe recommended: {bool(stage_e_observed_candidate_repair_loop.get('alternative_probe_recommended', False))}",
                f"- Stage E observed acceptance candidate repair recovery confirmed: {bool(stage_e_observed_candidate_repair_loop.get('recovery_confirmed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review recommended: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_recommended', False))}",
                f"- Stage E observed acceptance candidate repair promotion review completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_completed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review in progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_in_progress', False))}",
                f"- Stage E observed acceptance candidate repair promotion review stale: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_stale', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup in progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_in_progress', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_completed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_failed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup latest status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_latest_status', '') or '')}",
                f"- Stage E observed acceptance candidate repair promotion review followup retry in progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_in_progress', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup retry completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_completed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup retry failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_failed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup retry latest status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_latest_status', '') or '')}",
                f"- Stage E observed acceptance candidate repair promotion review followup retry escalation in progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_escalation_in_progress', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup retry escalation completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_escalation_completed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup retry escalation failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_escalation_failed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review followup retry escalation latest status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_followup_retry_escalation_latest_status', '') or '')}",
                f"- Stage E observed acceptance candidate repair promotion review evidence collection in progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_collection_in_progress', False))}",
                f"- Stage E observed acceptance candidate repair promotion review evidence collection completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_collection_completed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review evidence collection failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_collection_failed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review evidence collection latest status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_collection_latest_status', '') or '')}",
                f"- Stage E observed acceptance candidate repair promotion review evidence recheck in progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_recheck_in_progress', False))}",
                f"- Stage E observed acceptance candidate repair promotion review evidence recheck completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_recheck_completed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review evidence recheck failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_recheck_failed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review evidence recheck latest status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_evidence_recheck_latest_status', '') or '')}",
                f"- Stage E observed acceptance candidate repair promotion review targeted probe in progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_in_progress', False))}",
                f"- Stage E observed acceptance candidate repair promotion review targeted probe completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_completed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review targeted probe failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_failed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review targeted probe latest status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_latest_status', '') or '')}",
                f"- Stage E observed acceptance candidate repair promotion review targeted probe recheck in progress: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_recheck_in_progress', False))}",
                f"- Stage E observed acceptance candidate repair promotion review targeted probe recheck completed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_recheck_completed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review targeted probe recheck failed: {bool(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_recheck_failed', False))}",
                f"- Stage E observed acceptance candidate repair promotion review targeted probe recheck latest status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_targeted_probe_recheck_latest_status', '') or '')}",
                f"- Stage E observed acceptance candidate repair promotion review latest status: {str(stage_e_observed_candidate_repair_loop.get('promotion_review_latest_status', '') or '')}",
                f"- Stage E observed acceptance candidate repair promotion review latest age seconds: {float(stage_e_observed_candidate_repair_loop.get('promotion_review_latest_age_seconds', 0.0) or 0.0):.1f}",
                f"- Stage E observed acceptance candidate repair recovery source: {str(stage_e_observed_candidate_repair_loop.get('recovery_source', '') or '')}",
                f"- Stage E observed acceptance candidate repair next review action: {str(stage_e_observed_candidate_repair_loop.get('next_review_action', '') or '')}",
                f"- Stage E observed acceptance candidate repair latest remeasure trend: {str(stage_e_observed_candidate_repair_loop.get('latest_remeasure_trend', '') or '')}",
                f"- Stage E observed acceptance candidate repair latest alternative probe trend: {str(stage_e_observed_candidate_repair_loop.get('latest_alternative_probe_trend', '') or '')}",
                f"- Completed evidence collection pending review count: {int(completed_evidence_review.get('pending_review_count', 0) or 0)}",
                f"- Experiment adoption candidate count: {int(journal_experiment_status.get('adoption_candidate_count', 0) or 0)}",
                f"- Experiment regressing item count: {int(journal_experiment_status.get('regressing_item_count', 0) or 0)}",
                f"- Experiment falsified item count: {int(journal_experiment_status.get('falsified_item_count', 0) or 0)}",
                f"- Experiment human review pending count: {int(journal_experiment_status.get('human_review_pending_count', 0) or 0)}",
                f"- Experiment priority action count: {int(journal_experiment_priority_plan.get('action_count', 0) or 0)}",
                f"- Experiment top priority source: {str(journal_experiment_priority_plan.get('top_priority_source', '') or '')}",
                f"- Experiment top priority category: {str(journal_experiment_priority_plan.get('top_priority_category', '') or '')}",
                f"- Experiment promotion target candidate count: {int(journal_experiment_promotion_target_plan.get('candidate_count', 0) or 0)}",
                f"- Experiment promotion target review action count: {int(journal_experiment_promotion_target_plan.get('review_action_count', 0) or 0)}",
            ]
        )
        for target in journal_experiment_promotion_target_plan.get("targets", [])[:5]:
            if isinstance(target, dict):
                lines.append(
                    "- experiment_promotion_target: "
                    f"{str(target.get('id', ''))} "
                    f"stage={str(target.get('target_stage', ''))} "
                    f"surface={str(target.get('target_surface', ''))} "
                    f"path={str(target.get('promotion_path', ''))}"
                )
        for action in journal_experiment_priority_plan.get("actions", [])[:5]:
            if isinstance(action, dict):
                lines.append(
                    "- experiment_priority_action: "
                    f"{str(action.get('source', ''))} "
                    f"priority={str(action.get('priority', ''))} "
                    f"category={str(action.get('category', ''))} "
                    f"count={int(action.get('count', 0) or 0)}"
                )
        for item_id in journal_experiment_status.get("adoption_candidate_ids", [])[:5]:
            lines.append(f"- experiment_adoption_candidate: {str(item_id)}")
        for item_id in journal_experiment_status.get("regressing_item_ids", [])[:5]:
            lines.append(f"- experiment_regressing_item: {str(item_id)}")
        for item_id in journal_experiment_status.get("falsified_item_ids", [])[:5]:
            lines.append(f"- experiment_falsified_item: {str(item_id)}")
        for item_id in journal_experiment_status.get("human_review_pending_ids", [])[:5]:
            lines.append(f"- experiment_human_review_pending: {str(item_id)}")
        for key in completed_evidence_review.get("pending_review_keys", [])[:5]:
            lines.append(f"- completed_evidence_pending_review_key: {str(key)}")
        status_counts = (
            research_journal_summary.get("remeasure_status_counts", {})
            if isinstance(research_journal_summary.get("remeasure_status_counts"), dict)
            else {}
        )
        for status, count in sorted(status_counts.items()):
            lines.append(f"- remeasure_status_count: {str(status)}={int(count or 0)}")
        for item in research_journal_summary.get("remeasure_trends", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- remeasure_trend: "
                    f"{str(item.get('id', ''))} "
                    f"trend={str(item.get('trend', ''))} "
                    f"latest_status={str(item.get('latest_status', ''))}"
                )
        for item in research_journal_summary.get("alternative_probe_trends", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- alternative_probe_trend: "
                    f"{str(item.get('id', ''))} "
                    f"trend={str(item.get('trend', ''))} "
                    f"latest_status={str(item.get('latest_status', ''))}"
                )
        for item in research_journal_summary.get("roadmap_patch_rejected_items", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- roadmap_patch_rejected_item: "
                    f"{str(item.get('id', ''))} "
                    f"count={int(item.get('count', 0) or 0)} "
                    f"reason={str(item.get('latest_reason', '') or '')}"
                )
        for item in research_journal_summary.get("roadmap_patch_refreshed_items", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- roadmap_patch_refreshed_item: "
                    f"{str(item.get('id', ''))} "
                    f"count={int(item.get('count', 0) or 0)} "
                    f"reason={str(item.get('latest_reason', '') or '')}"
                )
        for item in research_journal_summary.get("roadmap_patch_evidence_collection_entries", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- roadmap_patch_evidence_collection: "
                    f"status={str(item.get('status', '') or '')} "
                    f"kind={str(item.get('evidence_kind', '') or '')} "
                    f"source={str(item.get('source', '') or '')}"
                )
        for item in research_journal_summary.get("remeasure_quota_holds", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- remeasure_quota_hold: "
                    f"{str(item.get('id', ''))} "
                    f"command={str(item.get('command', ''))} "
                    f"history={int(item.get('history_count', 0) or 0)}/"
                    f"{int(item.get('quota', 0) or 0)}"
                )
        for item in research_journal_summary.get("completed_research_planner_tasks", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- completed_research_planner_task: "
                    f"{str(item.get('id', ''))} "
                    f"type={str(item.get('task_type', ''))} "
                    f"status={str(item.get('status', ''))}"
                )
        for item in research_journal_summary.get("top_negative_results", [])[:5]:
            if isinstance(item, dict):
                lines.append(f"- top_negative_result: {str(item.get('id', ''))} count={int(item.get('count', 0) or 0)}")
        for item in research_journal_summary.get("top_next_hypotheses", [])[:5]:
            if isinstance(item, dict):
                lines.append(f"- top_next_hypothesis: {str(item.get('id', ''))} count={int(item.get('count', 0) or 0)}")
        for item in research_journal_summary.get("recommended_benchmark_actions", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- recommended_benchmark_action: "
                    f"{str(item.get('command', ''))} "
                    f"(id={str(item.get('id', ''))}, priority={str(item.get('priority', ''))})"
                )
        for item in research_journal_summary.get("suppressed_benchmark_actions", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- suppressed_benchmark_action: "
                    f"{str(item.get('command', ''))} "
                    f"(id={str(item.get('id', ''))}, trend={str(item.get('remeasure_trend', ''))}, "
                    f"retry_after={float(item.get('seconds_until_next_remeasure', 0.0) or 0.0):.1f}s)"
                )
        for item in research_journal_summary.get("alternative_benchmark_actions", [])[:5]:
            if isinstance(item, dict):
                lines.append(
                    "- alternative_benchmark_action: "
                    f"{str(item.get('command', ''))} "
                    f"(id={str(item.get('id', ''))}, priority={str(item.get('priority', ''))})"
                )
    lines.extend(
        [
            "",
            "## Iterative Next Actions",
        ]
    )
    next_actions = (
        iterative_plan.get("next_actions", [])
        if isinstance(iterative_plan.get("next_actions"), list)
        else []
    )
    if next_actions:
        for action in next_actions[:10]:
            if not isinstance(action, dict):
                continue
            title = str(action.get("title", "repair_action") or "repair_action")
            command = str(action.get("command", "") or "")
            checks_text = (
                ", ".join(str(item) for item in action.get("affected_checks", []) if str(item).strip())
                if isinstance(action.get("affected_checks"), list)
                else ""
            )
            lines.append(f"- {title}: `{command}`")
            if checks_text:
                lines.append(f"  - Covers: {checks_text}")
    else:
        lines.append(f"- Next step hint: {str(iterative_plan.get('next_step_hint', '') or '')}")
    lines.extend(
        [
            "",
            "## Retry Queue",
            f"- Retry candidate count: {int(report.get('repair_retry_queue_count', 0) or 0)}",
        ]
    )
    if retry_queue:
        for retry in retry_queue[:10]:
            if not isinstance(retry, dict):
                continue
            lines.append(
                "- "
                f"`{str(retry.get('command', '') or '')}` "
                f"(reason={str(retry.get('reason', '') or '')}, "
                f"attempt={int(retry.get('next_attempt', 0) or 0)}/{int(retry.get('max_attempts', 0) or 0)}, "
                f"priority={str(retry.get('priority_tier', '') or '')}, "
                f"score={float(retry.get('priority_score', 0.0) or 0.0):.3f})"
            )
    else:
        lines.append("- No retry candidate available.")
    lines.extend(
        [
            "",
            "## Failed Checks",
        ]
    )
    if failed_checks:
        for name in failed_checks:
            lines.append(f"- {name}")
    else:
        lines.append("- No failed checks.")
    lines.extend(
        [
            "",
            "## Execution Manifest",
            f"- Planned command count: {len(runbook_actions)}",
            f"- Configured max actions: {int(report.get('runbook_max_actions', runbook_action_build_stats.get('max_actions', 50)) or 50)}",
            f"- Configured max per source: {int(report.get('runbook_max_per_source', runbook_action_build_stats.get('max_per_source', 0)) or 0)}",
            f"- Considered candidates: {int(runbook_action_build_stats.get('considered_count', 0) or 0)}",
            f"- Skipped by duplicate: {int(runbook_action_build_stats.get('skipped_duplicate_count', 0) or 0)}",
            f"- Skipped by empty command: {int(runbook_action_build_stats.get('skipped_empty_command_count', 0) or 0)}",
            f"- Skipped by source cap: {int(runbook_action_build_stats.get('skipped_source_cap_count', 0) or 0)}",
            f"- Skipped by max actions: {int(runbook_action_build_stats.get('skipped_max_actions_count', 0) or 0)}",
            f"- Skipped by remeasure command history quota: {int(runbook_action_build_stats.get('skipped_remeasure_command_history_quota_count', 0) or 0)}",
            f"- Drop rate: {float(runbook_action_build_rates.get('drop_rate', 0.0) or 0.0):.3f}",
        ]
    )
    skipped_empty_command_by_source = (
        runbook_action_build_stats.get("skipped_empty_command_by_source", {})
        if isinstance(runbook_action_build_stats.get("skipped_empty_command_by_source"), dict)
        else {}
    )
    for source, count in list(skipped_empty_command_by_source.items())[:8]:
        lines.append(f"- Skipped by empty command ({source}): {int(count or 0)}")
    skipped_duplicate_by_source = (
        runbook_action_build_stats.get("skipped_duplicate_by_source", {})
        if isinstance(runbook_action_build_stats.get("skipped_duplicate_by_source"), dict)
        else {}
    )
    for source, count in list(skipped_duplicate_by_source.items())[:8]:
        lines.append(f"- Skipped by duplicate ({source}): {int(count or 0)}")
    skipped_source_cap_by_source = (
        runbook_action_build_stats.get("skipped_source_cap_by_source", {})
        if isinstance(runbook_action_build_stats.get("skipped_source_cap_by_source"), dict)
        else {}
    )
    for source, count in list(skipped_source_cap_by_source.items())[:8]:
        lines.append(f"- Skipped by source cap ({source}): {int(count or 0)}")
    skipped_max_actions_by_source = (
        runbook_action_build_stats.get("skipped_max_actions_by_source", {})
        if isinstance(runbook_action_build_stats.get("skipped_max_actions_by_source"), dict)
        else {}
    )
    for source, count in list(skipped_max_actions_by_source.items())[:8]:
        lines.append(f"- Skipped by max actions ({source}): {int(count or 0)}")
    skipped_remeasure_command_quota = (
        runbook_action_build_stats.get("skipped_remeasure_command_history_quota_by_command", {})
        if isinstance(runbook_action_build_stats.get("skipped_remeasure_command_history_quota_by_command"), dict)
        else {}
    )
    for command, count in list(skipped_remeasure_command_quota.items())[:8]:
        lines.append(f"- Skipped by remeasure command history quota ({command}): {int(count or 0)}")
    if runbook_actions:
        for action in runbook_actions[:20]:
            if not isinstance(action, dict):
                continue
            lines.append(
                "- "
                f"`{str(action.get('command', '') or '')}` "
                f"(source={str(action.get('source', '') or '')}, priority={str(action.get('priority', '') or '')})"
            )
    else:
        lines.append("- No executable action in manifest.")
    return "\n".join(lines)


def build_operational_runbook_actions(
    report: Dict[str, Any],
    *,
    max_actions: int = 50,
    max_per_source: int = 0,
    return_metadata: bool = False,
    external_actions: Optional[List[Dict[str, Any]]] = None,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if max_actions < 1:
        return []
    iterative_plan = (
        report.get("iterative_repair_plan", {})
        if isinstance(report.get("iterative_repair_plan"), dict)
        else {}
    )
    retry_queue = (
        report.get("repair_retry_queue", [])
        if isinstance(report.get("repair_retry_queue"), list)
        else []
    )
    error_details = (
        report.get("error_details", [])
        if isinstance(report.get("error_details"), list)
        else []
    )
    repair_plan = (
        report.get("repair_plan", {})
        if isinstance(report.get("repair_plan"), dict)
        else {}
    )
    v1_actions_snapshot = (
        report.get("v1_actions_snapshot", {})
        if isinstance(report.get("v1_actions_snapshot"), dict)
        else {}
    )
    ann_efficiency_roadmap = (
        report.get("ann_efficiency_roadmap", {})
        if isinstance(report.get("ann_efficiency_roadmap"), dict)
        else {}
    )
    sara_ann_comparison = (
        report.get("sara_ann_comparison", {})
        if isinstance(report.get("sara_ann_comparison"), dict)
        else {}
    )
    operational_checklist = (
        report.get("operational_checklist", {})
        if isinstance(report.get("operational_checklist"), dict)
        else {}
    )
    research_review = (
        report.get("research_review", {})
        if isinstance(report.get("research_review"), dict)
        else {}
    )
    research_review_compact = (
        research_review.get("compact", {})
        if isinstance(research_review.get("compact"), dict)
        else {}
    )
    execution_log = (
        report.get("execution_log", [])
        if isinstance(report.get("execution_log"), list)
        else []
    )
    research_journal_summary = (
        report.get("research_journal_summary", {})
        if isinstance(report.get("research_journal_summary"), dict)
        else {}
    )
    research_journal_summary = attach_roadmap_patch_refresh_policy_followups_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    research_journal_summary = attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    research_planner_task_status = summarize_research_planner_task_status(
        research_review_compact,
        research_journal_summary,
        cleanup_threshold=int(report.get("research_planner_task_cleanup_threshold", 2) or 2),
    )
    roadmap_patch_refresh_policy = summarize_roadmap_patch_refresh_policy(
        research_journal_summary
    )
    completed_evidence_review = summarize_completed_roadmap_patch_evidence_review(
        research_journal_summary
    )
    stage_e_readiness = (
        report.get("stage_e_readiness", {})
        if isinstance(report.get("stage_e_readiness"), dict)
        else {}
    )
    runbook_drop_rate_threshold = float(report.get("runbook_drop_rate_threshold", 0.9) or 0.9)
    remeasure_command_history_quota = int(report.get("runbook_remeasure_command_history_quota", 2) or 2)
    actions: List[Dict[str, Any]] = []
    seen_commands: set[str] = set()
    source_counts: Dict[str, int] = {}
    remeasure_history_command_counts: Dict[str, int] = {}
    for entry in execution_log:
        if not isinstance(entry, dict):
            continue
        command = str(entry.get("command", "") or "").strip()
        if not command:
            continue
        source = str(entry.get("source", "") or "").strip()
        covered_checks = (
            [str(item).strip() for item in entry.get("covered_checks", []) if str(item).strip()]
            if isinstance(entry.get("covered_checks"), list)
            else []
        )
        if "research_journal_remeasure" not in source and "research_journal_summary" not in covered_checks:
            continue
        status = str(entry.get("status", "") or "").strip().lower()
        if status not in {"pending", "success", "failed", "timeout", "error"}:
            continue
        remeasure_history_command_counts[command] = int(remeasure_history_command_counts.get(command, 0)) + 1
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    stats: Dict[str, Any] = {
        "considered_count": 0,
        "appended_count": 0,
        "skipped_duplicate_count": 0,
        "skipped_duplicate_by_source": {},
        "skipped_max_actions_count": 0,
        "skipped_max_actions_by_source": {},
        "skipped_source_cap_count": 0,
        "skipped_empty_command_count": 0,
        "skipped_empty_command_by_source": {},
        "skipped_source_cap_by_source": {},
        "skipped_remeasure_command_history_quota_count": 0,
        "skipped_remeasure_command_history_quota_by_command": {},
        "skipped_remeasure_command_history_quota_items": [],
    }

    def _append_action(
        *,
        command: str,
        source: str,
        priority: str,
        reason: str,
        affected_checks: Optional[List[str]] = None,
    ) -> None:
        cmd = str(command).strip()
        src = str(source).strip() or "unknown"
        stats["considered_count"] = int(stats["considered_count"]) + 1
        if not cmd:
            stats["skipped_empty_command_count"] = int(stats["skipped_empty_command_count"]) + 1
            skipped_empty_by_source = (
                stats.get("skipped_empty_command_by_source", {})
                if isinstance(stats.get("skipped_empty_command_by_source"), dict)
                else {}
            )
            skipped_empty_by_source[src] = int(skipped_empty_by_source.get(src, 0)) + 1
            stats["skipped_empty_command_by_source"] = skipped_empty_by_source
            return
        if cmd in seen_commands:
            stats["skipped_duplicate_count"] = int(stats["skipped_duplicate_count"]) + 1
            skipped_duplicate_by_source = (
                stats.get("skipped_duplicate_by_source", {})
                if isinstance(stats.get("skipped_duplicate_by_source"), dict)
                else {}
            )
            skipped_duplicate_by_source[src] = int(skipped_duplicate_by_source.get(src, 0)) + 1
            stats["skipped_duplicate_by_source"] = skipped_duplicate_by_source
            return
        if len(actions) >= max_actions:
            stats["skipped_max_actions_count"] = int(stats["skipped_max_actions_count"]) + 1
            skipped_max_by_source = (
                stats.get("skipped_max_actions_by_source", {})
                if isinstance(stats.get("skipped_max_actions_by_source"), dict)
                else {}
            )
            skipped_max_by_source[src] = int(skipped_max_by_source.get(src, 0)) + 1
            stats["skipped_max_actions_by_source"] = skipped_max_by_source
            return
        if max_per_source > 0 and int(source_counts.get(src, 0)) >= int(max_per_source):
            stats["skipped_source_cap_count"] = int(stats["skipped_source_cap_count"]) + 1
            skipped_by_source = (
                stats.get("skipped_source_cap_by_source", {})
                if isinstance(stats.get("skipped_source_cap_by_source"), dict)
                else {}
            )
            skipped_by_source[src] = int(skipped_by_source.get(src, 0)) + 1
            stats["skipped_source_cap_by_source"] = skipped_by_source
            return
        p = str(priority).strip().lower()
        if p not in {"high", "medium", "low"}:
            p = "medium"
        checks = sorted({str(item).strip() for item in (affected_checks or []) if str(item).strip()})
        seen_commands.add(cmd)
        source_counts[src] = int(source_counts.get(src, 0)) + 1
        actions.append(
            {
                "step": int(len(actions) + 1),
                "source": src,
                "priority": p,
                "command": cmd,
                "reason": str(reason).strip(),
                "affected_checks": checks,
            }
        )
        stats["appended_count"] = int(stats["appended_count"]) + 1

    next_actions = (
        iterative_plan.get("next_actions", [])
        if isinstance(iterative_plan.get("next_actions"), list)
        else []
    )
    for action in next_actions:
        if not isinstance(action, dict):
            continue
        _append_action(
            command=str(action.get("command", "")),
            source="iterative_next_action",
            priority=str(action.get("priority", "high")),
            reason=str(action.get("title", "iterative_next_action")),
            affected_checks=(
                [str(item) for item in action.get("affected_checks", []) if str(item).strip()]
                if isinstance(action.get("affected_checks"), list)
                else []
            ),
        )

    retry_candidates = [dict(item) for item in retry_queue if isinstance(item, dict)]
    retry_candidates.sort(
        key=lambda item: (
            priority_rank.get(str(item.get("priority_tier", "low")).strip().lower(), 2),
            -float(item.get("priority_score", 0.0) or 0.0),
            str(item.get("command", "")),
        )
    )
    for retry in retry_candidates:
        _append_action(
            command=str(retry.get("command", "")),
            source="retry_queue",
            priority=str(retry.get("priority_tier", "medium")),
            reason=f"retry_reason={str(retry.get('reason', '')).strip()}",
            affected_checks=(
                [str(item) for item in retry.get("covered_checks", []) if str(item).strip()]
                if isinstance(retry.get("covered_checks"), list)
                else []
            ),
        )
    has_efficiency_kpi_incident = any(
        isinstance(item, dict) and str(item.get("category", "")).strip().lower().endswith("efficiency_kpi")
        for item in error_details
    )
    if has_efficiency_kpi_incident:
        shortcut_actions: List[Dict[str, Any]] = []
        append_efficiency_incident_runbook_actions(
            shortcut_actions,
            source="efficiency_incident_shortcut",
            priority="high",
        )
        for shortcut in shortcut_actions:
            if not isinstance(shortcut, dict):
                continue
            _append_action(
                command=str(shortcut.get("command", "")),
                source=str(shortcut.get("source", "efficiency_incident_shortcut")),
                priority=str(shortcut.get("priority", "high")),
                reason="efficiency_kpi_incident_repair_shortcut",
                affected_checks=(
                    [str(item) for item in shortcut.get("affected_checks", []) if str(item).strip()]
                    if isinstance(shortcut.get("affected_checks"), list)
                    else []
                ),
            )

    fallback_actions = (
        repair_plan.get("fallback_actions", [])
        if isinstance(repair_plan.get("fallback_actions"), list)
        else []
    )
    for fallback in fallback_actions:
        if not isinstance(fallback, dict):
            continue
        _append_action(
            command=str(fallback.get("command", "")),
            source="fallback_action",
            priority="medium",
            reason=str(fallback.get("title", "fallback_action")),
            affected_checks=(
                [str(item) for item in fallback.get("affected_checks", []) if str(item).strip()]
                if isinstance(fallback.get("affected_checks"), list)
                else []
            ),
        )
    ann_next_evidence_actions = (
        ann_efficiency_roadmap.get("next_evidence_actions", [])
        if isinstance(ann_efficiency_roadmap.get("next_evidence_actions"), list)
        else []
    )
    for evidence_action in ann_next_evidence_actions:
        if not isinstance(evidence_action, dict):
            continue
        category = str(evidence_action.get("category", "") or "").strip()
        task = str(evidence_action.get("task", "") or "").strip()
        affected_checks = ["ann_efficiency_roadmap"]
        if category in {"pending_joule_pair", "weak_joule_pair", "partial_pair", "invalid_pair", "missing_pair"}:
            affected_checks.append("energy_measurement")
        elif "internal_maintenance" in category or task == "phase6_maintenance_efficiency":
            affected_checks.append("internal_maintenance_efficiency")
        elif "reference" in category or task == "phase8_reference_strength":
            affected_checks.append("external_validity")
        _append_action(
            command=str(evidence_action.get("command", "")),
            source="ann_efficiency_next_evidence",
            priority=str(evidence_action.get("priority", "medium")),
            reason=f"category={category}; task={task}",
            affected_checks=affected_checks,
        )
    comparison_next_actions = (
        sara_ann_comparison.get("next_actions", [])
        if isinstance(sara_ann_comparison.get("next_actions"), list)
        else []
    )
    for comparison_action in comparison_next_actions:
        if not isinstance(comparison_action, dict):
            continue
        category = str(comparison_action.get("category", "") or "").strip()
        affected_checks = ["sara_ann_comparison"]
        if "maintenance" in category:
            affected_checks.append("internal_maintenance_efficiency")
        if "event_memory" in category or "compression" in category:
            affected_checks.append("event_memory_ingest_pipeline")
        if "physical" in category:
            affected_checks.append("energy_measurement")
        if "reference" in category or "bm25" in category:
            affected_checks.append("external_validity")
        _append_action(
            command=str(comparison_action.get("command", "")),
            source="sara_ann_comparison_next_action",
            priority=str(comparison_action.get("priority", "medium")),
            reason=f"category={category}",
            affected_checks=affected_checks,
        )
    for external in external_actions if isinstance(external_actions, list) else []:
        if not isinstance(external, dict):
            continue
        _append_action(
            command=str(external.get("command", "")),
            source="v1_recovery_action",
            priority=str(external.get("priority", "medium")),
            reason=str(external.get("expected_effect", external.get("category", "v1_recovery_action"))),
            affected_checks=(
                [str(item) for item in external.get("affected_checks", []) if str(item).strip()]
                if isinstance(external.get("affected_checks"), list)
                else []
            ),
        )
    rejected_stale = int(v1_actions_snapshot.get("rejected_stale_count", 0) or 0)
    rejected_missing = int(v1_actions_snapshot.get("rejected_missing_timestamp_count", 0) or 0)
    if rejected_stale > 0 or rejected_missing > 0:
        hygiene_reason = (
            "v1_action_manifest_hygiene: "
            f"stale={rejected_stale}, missing_timestamp={rejected_missing}"
        )
        _append_action(
            command="python scripts/eval/v1_release_gate.py",
            source="v1_action_hygiene",
            priority="high",
            reason=hygiene_reason,
            affected_checks=["v1_action_manifest_hygiene"],
        )
    if bool(roadmap_patch_refresh_policy.get("needs_followup", False)):
        policy_source = str(
            roadmap_patch_refresh_policy.get("action_source", "")
            or "roadmap_patch_refresh_policy_followup"
        )
        command_label = (
            "roadmap_patch_refresh_evidence_collection"
            if policy_source == "roadmap_patch_refresh_evidence_collection_fallback"
            else "roadmap_patch_refresh_policy_review"
        )
        covered_checks = (
            "roadmap_patch_refresh_policy,evidence_collection,research_journal_summary"
            if policy_source == "roadmap_patch_refresh_evidence_collection_fallback"
            else "roadmap_patch_refresh_policy,research_journal_summary"
        )
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                f"--record-repair-command \"{command_label}\" "
                "--record-repair-status pending "
                f"--record-repair-source {policy_source} "
                f"--record-repair-checks \"{covered_checks}\""
            ),
            source=policy_source,
            priority="medium",
            reason=(
                "roadmap_patch_refresh_policy_status="
                f"{str(roadmap_patch_refresh_policy.get('status', '') or '')};"
                "ratio="
                f"{float(roadmap_patch_refresh_policy.get('refresh_to_rejection_ratio', 0.0) or 0.0):.3f};"
                "bounds="
                f"{float(roadmap_patch_refresh_policy.get('low_ratio_threshold', 0.0) or 0.0):.3f}-"
                f"{float(roadmap_patch_refresh_policy.get('high_ratio_threshold', 0.0) or 0.0):.3f}"
            ),
            affected_checks=(
                ["evidence_collection", "research_journal_summary", "roadmap_patch_refresh_policy"]
                if policy_source == "roadmap_patch_refresh_evidence_collection_fallback"
                else ["roadmap_patch_refresh_policy", "research_journal_summary"]
            ),
        )
    if bool(completed_evidence_review.get("needs_review", False)):
        pending_keys = [
            str(item)
            for item in completed_evidence_review.get("pending_review_keys", [])
            if str(item).strip()
        ]
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command \"roadmap_patch_evidence_review\" "
                "--record-repair-status pending "
                "--record-repair-source roadmap_patch_evidence_review "
                "--record-repair-checks \"roadmap_patch_evidence_collection,research_journal_summary\""
            ),
            source="roadmap_patch_evidence_review",
            priority="medium",
            reason=(
                "completed_evidence_pending_review_count="
                f"{int(completed_evidence_review.get('pending_review_count', 0) or 0)};"
                "keys="
                f"{','.join(pending_keys[:5])}"
            ),
            affected_checks=[
                "roadmap_patch_evidence_collection",
                "research_journal_summary",
                *pending_keys[:5],
            ],
        )
    if stage_e_readiness:
        linear_needs_long_run = not bool(
            stage_e_readiness.get("linear_snn_fusion_trend_has_previous", False)
        )
        architecture_needs_long_run = not bool(
            stage_e_readiness.get("architecture_integration_trend_has_previous", False)
        )
        architecture_regression_count = int(
            stage_e_readiness.get("architecture_integration_trend_regression_count", 0) or 0
        )
        architecture_needs_review = bool(
            architecture_needs_long_run or architecture_regression_count > 0
        )
        delta_metric_names = [
            "delta_memory_steering_integrity_observed",
            "delta_memory_counterfactual_isolation_observed",
            "delta_memory_trace_observability_observed",
        ]
        delta_metrics_available = any(name in stage_e_readiness for name in delta_metric_names)
        delta_needs_long_run = bool(
            delta_metrics_available
            and not bool(stage_e_readiness.get("delta_memory_observed_long_run_validated", False))
        )
        if linear_needs_long_run or delta_needs_long_run or architecture_needs_review:
            affected = ["real_data_external_validity"]
            if linear_needs_long_run:
                affected.append("linear_snn_fusion_observed_trend")
            if architecture_needs_review:
                affected.append("stage_e_architecture_integration_observed_trend")
            if delta_needs_long_run:
                affected.extend(delta_metric_names)
            _append_action(
                command=(
                    "python scripts/eval/operational_readiness.py "
                    "--strict-production --refresh-artifacts --include-accuracy --soak-profile extended"
                ),
                source="observed_trend_long_run_validation",
                priority="medium",
                reason=(
                    "linear_snn_fusion_needs_history="
                    f"{linear_needs_long_run};"
                    "stage_e_architecture_integration_needs_history="
                    f"{architecture_needs_long_run};"
                    "stage_e_architecture_integration_regression_count="
                    f"{architecture_regression_count};"
                    "delta_memory_needs_history="
                    f"{delta_needs_long_run}"
                ),
                affected_checks=affected,
            )
        if bool(stage_e_readiness.get("observed_acceptance_candidate_stability_recommended", False)):
            _append_action(
                command=(
                    "python scripts/eval/operational_readiness.py "
                    "--record-repair-command \"review_stage_e_observed_acceptance_candidates_for_minimum_promotion\" "
                    "--record-repair-status pending "
                    "--record-repair-source stage_e_observed_acceptance_candidate_stability "
                    "--record-repair-checks \"stage_e_observed_acceptance_candidate_stability,stage_e_readiness\""
                ),
                source="stage_e_observed_acceptance_candidate_stability",
                priority="medium",
                reason=(
                    "stage_e_observed_acceptance_candidate_stability_recommended=True;"
                    "consecutive_passes="
                    f"{int(stage_e_readiness.get('observed_acceptance_candidate_consecutive_passes', 0) or 0)};"
                    "required_streak="
                    f"{int(stage_e_readiness.get('observed_acceptance_candidate_required_streak', 3) or 3)}"
                ),
                affected_checks=[
                    "stage_e_observed_acceptance_candidate_stability",
                    "stage_e_readiness",
                ],
            )
        stage_e_candidate_failures = (
            stage_e_readiness.get("observed_acceptance_candidate_failures", [])
            if isinstance(stage_e_readiness.get("observed_acceptance_candidate_failures", []), list)
            else []
        )
        if int(stage_e_readiness.get("observed_acceptance_candidate_failure_count", 0) or 0) > 0:
            failed_metrics = [
                str(item.get("metric", item.get("check", ""))).strip()
                for item in stage_e_candidate_failures
                if isinstance(item, dict) and str(item.get("metric", item.get("check", ""))).strip()
            ]
            failed_label = ",".join(failed_metrics[:5])
            repair_command = (
                f"repair_stage_e_observed_acceptance_candidates:{failed_label}"
                if failed_label
                else "repair_stage_e_observed_acceptance_candidates"
            )
            _append_action(
                command=(
                    "python scripts/eval/operational_readiness.py "
                    f"--record-repair-command \"{repair_command}\" "
                    "--record-repair-status pending "
                    "--record-repair-source stage_e_observed_acceptance_candidate_repair "
                    "--record-repair-checks \"stage_e_observed_acceptance_candidate_repair,stage_e_readiness\""
                ),
                source="stage_e_observed_acceptance_candidate_repair",
                priority="medium",
                reason=(
                    "stage_e_observed_acceptance_candidate_failure_count="
                    f"{int(stage_e_readiness.get('observed_acceptance_candidate_failure_count', 0) or 0)};"
                    f"metrics={failed_label}"
                ),
                affected_checks=[
                    "stage_e_observed_acceptance_candidate_repair",
                    "stage_e_readiness",
                    *failed_metrics[:5],
                ],
            )
    stage_e_repair_loop = (
        research_journal_summary.get("stage_e_observed_acceptance_candidate_repair_loop", {})
        if isinstance(
            research_journal_summary.get("stage_e_observed_acceptance_candidate_repair_loop", {}),
            dict,
        )
        else {}
    )
    if (
        bool(stage_e_repair_loop.get("promotion_review_stale", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_completed", False))
    ):
        latest_age_seconds = float(
            stage_e_repair_loop.get("promotion_review_latest_age_seconds", 0.0) or 0.0
        )
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command "
                "\"followup_stage_e_observed_acceptance_candidate_recovery_review\" "
                "--record-repair-status pending "
                "--record-repair-source stage_e_observed_acceptance_candidate_recovery_review_followup "
                "--record-repair-checks "
                "\"stage_e_observed_acceptance_candidate_repair_recovery,"
                "stage_e_observed_acceptance_candidate_stability,research_journal_summary\""
            ),
            source="stage_e_observed_acceptance_candidate_recovery_review_followup",
            priority="medium",
            reason=(
                "stage_e_observed_acceptance_candidate_recovery_review_stale=True;"
                f"latest_status={str(stage_e_repair_loop.get('promotion_review_latest_status', '') or '')};"
                f"age_seconds={latest_age_seconds:.1f};"
                "stale_after_seconds="
                f"{float(stage_e_repair_loop.get('promotion_review_stale_after_seconds', STAGE_E_RECOVERY_REVIEW_STALE_SECONDS) or STAGE_E_RECOVERY_REVIEW_STALE_SECONDS):.1f}"
            ),
            affected_checks=[
                "stage_e_observed_acceptance_candidate_repair_recovery",
                "stage_e_observed_acceptance_candidate_stability",
                "research_journal_summary",
            ],
        )
    if (
        bool(stage_e_repair_loop.get("promotion_review_followup_failed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_retry_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_retry_completed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_retry_failed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_retry_escalation_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_retry_escalation_completed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_evidence_collection_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_evidence_collection_completed", False))
    ):
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command "
                "\"retry_stage_e_observed_acceptance_candidate_recovery_review_followup\" "
                "--record-repair-status pending "
                "--record-repair-source stage_e_observed_acceptance_candidate_recovery_review_followup_retry "
                "--record-repair-checks "
                "\"stage_e_observed_acceptance_candidate_repair_recovery,"
                "stage_e_observed_acceptance_candidate_stability,research_journal_summary\""
            ),
            source="stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
            priority="medium",
            reason=(
                "stage_e_observed_acceptance_candidate_recovery_review_followup_failed=True;"
                "latest_followup_status="
                f"{str(stage_e_repair_loop.get('promotion_review_followup_latest_status', '') or '')}"
            ),
            affected_checks=[
                "stage_e_observed_acceptance_candidate_repair_recovery",
                "stage_e_observed_acceptance_candidate_stability",
                "research_journal_summary",
            ],
        )
    if (
        bool(stage_e_repair_loop.get("promotion_review_followup_retry_failed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_retry_escalation_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_retry_escalation_completed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_followup_retry_escalation_failed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_evidence_collection_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_evidence_collection_completed", False))
    ):
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command "
                "\"escalate_stage_e_observed_acceptance_candidate_recovery_review_followup_retry\" "
                "--record-repair-status pending "
                "--record-repair-source stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation "
                "--record-repair-checks "
                "\"stage_e_observed_acceptance_candidate_repair_recovery,"
                "stage_e_observed_acceptance_candidate_stability,research_journal_summary\""
            ),
            source="stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation",
            priority="high",
            reason=(
                "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_failed=True;"
                "latest_retry_status="
                f"{str(stage_e_repair_loop.get('promotion_review_followup_retry_latest_status', '') or '')}"
            ),
            affected_checks=[
                "stage_e_observed_acceptance_candidate_repair_recovery",
                "stage_e_observed_acceptance_candidate_stability",
                "research_journal_summary",
            ],
        )
    if (
        bool(stage_e_repair_loop.get("promotion_review_followup_retry_escalation_failed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_evidence_collection_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_evidence_collection_completed", False))
    ):
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command "
                "\"collect_stage_e_observed_acceptance_candidate_recovery_review_evidence\" "
                "--record-repair-status pending "
                "--record-repair-source stage_e_observed_acceptance_candidate_recovery_review_evidence_collection "
                "--record-repair-checks "
                "\"stage_e_observed_acceptance_candidate_repair_recovery,"
                "stage_e_observed_acceptance_candidate_stability,research_journal_summary\""
            ),
            source="stage_e_observed_acceptance_candidate_recovery_review_evidence_collection",
            priority="high",
            reason=(
                "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_failed=True;"
                "latest_escalation_status="
                f"{str(stage_e_repair_loop.get('promotion_review_followup_retry_escalation_latest_status', '') or '')}"
            ),
            affected_checks=[
                "stage_e_observed_acceptance_candidate_repair_recovery",
                "stage_e_observed_acceptance_candidate_stability",
                "research_journal_summary",
            ],
        )
    if (
        bool(stage_e_repair_loop.get("promotion_review_evidence_collection_completed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_evidence_recheck_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_evidence_recheck_completed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_evidence_recheck_failed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_targeted_probe_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_targeted_probe_completed", False))
    ):
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command "
                "\"recheck_stage_e_observed_acceptance_candidate_recovery_review_evidence\" "
                "--record-repair-status pending "
                "--record-repair-source stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck "
                "--record-repair-checks "
                "\"stage_e_observed_acceptance_candidate_repair_recovery,"
                "stage_e_observed_acceptance_candidate_stability,research_journal_summary\""
            ),
            source="stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck",
            priority="medium",
            reason=(
                "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_completed=True;"
                "latest_evidence_collection_status="
                f"{str(stage_e_repair_loop.get('promotion_review_evidence_collection_latest_status', '') or '')}"
            ),
            affected_checks=[
                "stage_e_observed_acceptance_candidate_repair_recovery",
                "stage_e_observed_acceptance_candidate_stability",
                "research_journal_summary",
            ],
        )
    if (
        bool(stage_e_repair_loop.get("promotion_review_evidence_recheck_failed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_targeted_probe_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_targeted_probe_completed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_targeted_probe_failed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_targeted_probe_recheck_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_targeted_probe_recheck_completed", False))
    ):
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command "
                "\"probe_stage_e_observed_acceptance_candidate_recovery_review_evidence\" "
                "--record-repair-status pending "
                "--record-repair-source stage_e_observed_acceptance_candidate_recovery_review_targeted_probe "
                "--record-repair-checks "
                "\"stage_e_observed_acceptance_candidate_repair_recovery,"
                "stage_e_observed_acceptance_candidate_stability,research_journal_summary\""
            ),
            source="stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
            priority="high",
            reason=(
                "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_failed=True;"
                "latest_evidence_recheck_status="
                f"{str(stage_e_repair_loop.get('promotion_review_evidence_recheck_latest_status', '') or '')}"
            ),
            affected_checks=[
                "stage_e_observed_acceptance_candidate_repair_recovery",
                "stage_e_observed_acceptance_candidate_stability",
                "research_journal_summary",
            ],
        )
    if (
        bool(stage_e_repair_loop.get("promotion_review_targeted_probe_completed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_targeted_probe_recheck_in_progress", False))
        and not bool(stage_e_repair_loop.get("promotion_review_targeted_probe_recheck_completed", False))
    ):
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command "
                "\"recheck_stage_e_observed_acceptance_candidate_recovery_review_targeted_probe\" "
                "--record-repair-status pending "
                "--record-repair-source stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck "
                "--record-repair-checks "
                "\"stage_e_observed_acceptance_candidate_repair_recovery,"
                "stage_e_observed_acceptance_candidate_stability,research_journal_summary\""
            ),
            source="stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck",
            priority="medium",
            reason=(
                "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_completed=True;"
                "latest_targeted_probe_status="
                f"{str(stage_e_repair_loop.get('promotion_review_targeted_probe_latest_status', '') or '')}"
            ),
            affected_checks=[
                "stage_e_observed_acceptance_candidate_repair_recovery",
                "stage_e_observed_acceptance_candidate_stability",
                "research_journal_summary",
            ],
        )
    if (
        bool(stage_e_repair_loop.get("promotion_review_recommended", False))
        and not bool(stage_e_repair_loop.get("promotion_review_completed", False))
        and not bool(stage_e_repair_loop.get("promotion_review_in_progress", False))
    ):
        next_review_action = str(
            stage_e_repair_loop.get("next_review_action", "")
            or "stage_e_observed_acceptance_candidate_stability"
        )
        recovery_source = str(stage_e_repair_loop.get("recovery_source", "") or "")
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command "
                "\"review_stage_e_observed_acceptance_candidate_recovery_for_stability\" "
                "--record-repair-status pending "
                "--record-repair-source stage_e_observed_acceptance_candidate_recovery_review "
                "--record-repair-checks "
                "\"stage_e_observed_acceptance_candidate_repair_recovery,"
                "stage_e_observed_acceptance_candidate_stability,research_journal_summary\""
            ),
            source="stage_e_observed_acceptance_candidate_recovery_review",
            priority="medium",
            reason=(
                "stage_e_observed_acceptance_candidate_repair_recovery_confirmed=True;"
                f"recovery_source={recovery_source};"
                f"next_review_action={next_review_action};"
                "latest_remeasure_trend="
                f"{str(stage_e_repair_loop.get('latest_remeasure_trend', '') or '')};"
                "latest_alternative_probe_trend="
                f"{str(stage_e_repair_loop.get('latest_alternative_probe_trend', '') or '')}"
            ),
            affected_checks=[
                "stage_e_observed_acceptance_candidate_repair_recovery",
                "stage_e_observed_acceptance_candidate_stability",
                "research_journal_summary",
                next_review_action,
            ],
        )
    research_review_experiment_priority_plan = (
        research_review_compact.get("experiment_priority_plan", {})
        if isinstance(research_review_compact.get("experiment_priority_plan"), dict)
        else build_experiment_status_priority_plan(
            research_review_compact.get("experiment_status_summary", {})
            if isinstance(research_review_compact.get("experiment_status_summary"), dict)
            else {}
        )
    )
    research_journal_experiment_priority_plan = (
        research_journal_summary.get("experiment_priority_plan", {})
        if isinstance(research_journal_summary.get("experiment_priority_plan"), dict)
        else build_experiment_status_priority_plan(
            research_journal_summary.get("experiment_status_summary", {})
            if isinstance(research_journal_summary.get("experiment_status_summary"), dict)
            else {}
        )
    )
    for plan_source, priority_plan in (
        ("research_review", research_review_experiment_priority_plan),
        ("research_journal_summary", research_journal_experiment_priority_plan),
    ):
        plan_actions = (
            priority_plan.get("actions", [])
            if isinstance(priority_plan.get("actions", []), list)
            else []
        )
        for plan_action in plan_actions:
            if not isinstance(plan_action, dict):
                continue
            category = str(plan_action.get("category", "") or "").strip()
            ids = [
                str(item).strip()
                for item in (plan_action.get("ids", []) if isinstance(plan_action.get("ids", []), list) else [])
                if str(item).strip()
            ]
            command_label = str(plan_action.get("command_label", "") or plan_action.get("source", "") or "").strip()
            if ids:
                command_label = f"{command_label}:{','.join(ids[:5])}"
            _append_action(
                command=(
                    "python scripts/eval/operational_readiness.py "
                    f"--record-repair-command \"{command_label}\" "
                    "--record-repair-status pending "
                    f"--record-repair-source {str(plan_action.get('source', 'experiment_priority_plan') or 'experiment_priority_plan')} "
                    f"--record-repair-checks \"experiment_priority_plan,{plan_source},{category}\""
                ),
                source=str(plan_action.get("source", "experiment_priority_plan")),
                priority=str(plan_action.get("priority", "medium")),
                reason=(
                    f"{plan_source}_experiment_priority:"
                    f"category={category};"
                    f"count={int(plan_action.get('count', 0) or 0)};"
                    f"policy={str(plan_action.get('policy', '') or '')}"
                ),
                affected_checks=[
                    "experiment_priority_plan",
                    plan_source,
                    category,
                    *ids[:5],
                ],
            )
    research_review_promotion_target_plan = (
        research_review_compact.get("experiment_promotion_target_plan", {})
        if isinstance(research_review_compact.get("experiment_promotion_target_plan"), dict)
        else build_experiment_promotion_target_plan(
            research_review_compact.get("experiment_status_summary", {})
            if isinstance(research_review_compact.get("experiment_status_summary"), dict)
            else {}
        )
    )
    research_journal_promotion_target_plan = (
        research_journal_summary.get("experiment_promotion_target_plan", {})
        if isinstance(research_journal_summary.get("experiment_promotion_target_plan"), dict)
        else build_experiment_promotion_target_plan(
            research_journal_summary.get("experiment_status_summary", {})
            if isinstance(research_journal_summary.get("experiment_status_summary"), dict)
            else {}
        )
    )
    for plan_source, promotion_plan in (
        ("research_review", research_review_promotion_target_plan),
        ("research_journal_summary", research_journal_promotion_target_plan),
    ):
        review_actions = (
            promotion_plan.get("review_actions", [])
            if isinstance(promotion_plan.get("review_actions", []), list)
            else []
        )
        for review_action in review_actions:
            if not isinstance(review_action, dict):
                continue
            item_id = str(review_action.get("id", "") or "").strip()
            target_stage = str(review_action.get("target_stage", "") or "").strip()
            target_surface = str(review_action.get("target_surface", "") or "").strip()
            promotion_path = str(review_action.get("promotion_path", "") or "").strip()
            command_label = f"experiment_promotion_target_review:{item_id}:{target_stage}:{target_surface}"
            _append_action(
                command=(
                    "python scripts/eval/operational_readiness.py "
                    f"--record-repair-command \"{command_label}\" "
                    "--record-repair-status pending "
                    "--record-repair-source experiment_promotion_target_review "
                    f"--record-repair-checks \"experiment_promotion_target_plan,{plan_source},{target_stage},{target_surface}\""
                ),
                source="experiment_promotion_target_review",
                priority=str(review_action.get("priority", "medium")),
                reason=(
                    f"{plan_source}_promotion_target:"
                    f"id={item_id};"
                    f"target_stage={target_stage};"
                    f"target_surface={target_surface};"
                    f"promotion_path={promotion_path};"
                    f"policy={str(review_action.get('policy', '') or '')}"
                ),
                affected_checks=[
                    "experiment_promotion_target_plan",
                    plan_source,
                    item_id,
                    target_stage,
                    target_surface,
                    promotion_path,
                ],
            )
    research_review_needs_action = bool(
        research_review_compact
        and (
            not bool(research_review_compact.get("passed", True))
            or int(research_review_compact.get("next_hypothesis_count", 0) or 0) > 0
            or int(research_review_compact.get("regression_watchlist_count", 0) or 0) > 0
            or int(research_review_compact.get("negative_result_count", 0) or 0) > 0
        )
    )
    research_review_report = (
        research_review.get("report", {})
        if isinstance(research_review.get("report"), dict)
        else {}
    )
    review_decision = latest_roadmap_patch_review_decision(
        execution_log,
        review_generated_at=_safe_float(research_review_report.get("generated_at", 0.0), 0.0),
    )
    if bool(research_planner_task_status.get("cleanup_needed", False)):
        affected = ["research_review", "research_planner_task_cleanup"]
        cause_ids = (
            research_review_compact.get("cause_boundary_documentation_ids", [])
            if isinstance(research_review_compact.get("cause_boundary_documentation_ids", []), list)
            else []
        )
        fixture_ids = (
            research_review_compact.get("targeted_fixture_repair_ids", [])
            if isinstance(research_review_compact.get("targeted_fixture_repair_ids", []), list)
            else []
        )
        affected.extend(str(item) for item in [*cause_ids, *fixture_ids] if str(item).strip())
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command \"research_planner_task_cleanup\" "
                "--record-repair-status pending "
                "--record-repair-source research_planner_task_cleanup "
                "--record-repair-checks \"cause_boundary_documentation,targeted_fixture_repair\""
            ),
            source="research_planner_task_cleanup",
            priority="high",
            reason=(
                "research_planner_task_pending_count="
                f"{int(research_planner_task_status.get('pending_count', 0) or 0)};"
                "completion_ratio="
                f"{float(research_planner_task_status.get('completion_ratio', 0.0) or 0.0):.3f}"
            ),
            affected_checks=affected,
        )
    elif bool(research_planner_task_status.get("cleanup_stalled", False)):
        stalled_source = str(
            research_planner_task_status.get(
                "cleanup_stalled_action_source",
                "research_planner_task_cleanup_stalled",
            )
            or "research_planner_task_cleanup_stalled"
        )
        stalled_reason = str(research_planner_task_status.get("cleanup_stalled_reason", "") or "")
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--record-repair-command \"research_planner_task_cleanup\" "
                "--record-repair-status failed "
                f"--record-repair-source {stalled_source} "
                "--record-repair-checks \"research_planner_task_cleanup\""
            ),
            source=stalled_source,
            priority="high",
            reason=(
                "research_planner_task_cleanup_pending_count="
                f"{int(research_planner_task_status.get('cleanup_pending_count', 0) or 0)};"
                "pending_task_count="
                f"{int(research_planner_task_status.get('pending_count', 0) or 0)};"
                f"stalled_reason={stalled_reason}"
            ),
            affected_checks=["research_planner_task_cleanup"],
        )
    if (
        research_review_needs_action
        and not bool(review_decision.get("available", False))
        and not bool(research_planner_task_status.get("cleanup_needed", False))
        and not bool(research_planner_task_status.get("cleanup_stalled", False))
    ):
        priority = (
            "high"
            if int(research_review_compact.get("regression_watchlist_count", 0) or 0) > 0
            or int(research_review_compact.get("negative_result_count", 0) or 0) > 0
            else "medium"
        )
        _append_action(
            command=ROADMAP_PATCH_REVIEW_COMMAND,
            source="roadmap_patch_review",
            priority=priority,
            reason="review roadmap_patch_suggestion before any ROADMAP update",
            affected_checks=["research_review", "roadmap_patch_suggestion"],
        )
    recommended_benchmark_actions = (
        research_journal_summary.get("recommended_benchmark_actions", [])
        if isinstance(research_journal_summary.get("recommended_benchmark_actions"), list)
        else []
    )
    for recommended in recommended_benchmark_actions:
        if not isinstance(recommended, dict):
            continue
        command = str(recommended.get("command", "") or "").strip()
        if (
            command
            and remeasure_command_history_quota > 0
            and int(remeasure_history_command_counts.get(command, 0)) >= remeasure_command_history_quota
        ):
            item_id = str(recommended.get("id", "") or "")
            alternative = (
                RESEARCH_JOURNAL_ALTERNATIVE_BENCHMARK_ACTIONS.get(item_id, {})
                if isinstance(RESEARCH_JOURNAL_ALTERNATIVE_BENCHMARK_ACTIONS, dict)
                else {}
            )
            alternative_command = str(alternative.get("command", "") or "").strip() if isinstance(alternative, dict) else ""
            alternative_reason = str(alternative.get("reason", "") or "").strip() if isinstance(alternative, dict) else ""
            stats["considered_count"] = int(stats["considered_count"]) + 1
            stats["skipped_remeasure_command_history_quota_count"] = (
                int(stats["skipped_remeasure_command_history_quota_count"]) + 1
            )
            skipped_by_command = (
                stats.get("skipped_remeasure_command_history_quota_by_command", {})
                if isinstance(stats.get("skipped_remeasure_command_history_quota_by_command"), dict)
                else {}
            )
            skipped_by_command[command] = int(skipped_by_command.get(command, 0)) + 1
            stats["skipped_remeasure_command_history_quota_by_command"] = skipped_by_command
            skipped_items = (
                stats.get("skipped_remeasure_command_history_quota_items", [])
                if isinstance(stats.get("skipped_remeasure_command_history_quota_items"), list)
                else []
            )
            skipped_items.append(
                {
                    "id": item_id,
                    "source": str(recommended.get("source", "summary") or "summary"),
                    "command": command,
                    "priority": str(recommended.get("priority", "medium") or "medium"),
                    "history_count": int(remeasure_history_command_counts.get(command, 0)),
                    "quota": int(remeasure_command_history_quota),
                    "hold_reason": "remeasure_command_history_quota",
                    "alternative_command": alternative_command,
                    "alternative_reason": alternative_reason,
                }
            )
            stats["skipped_remeasure_command_history_quota_items"] = skipped_items
            if alternative_command:
                _append_action(
                    command=alternative_command,
                    source="research_journal_alternative_probe",
                    priority=str(recommended.get("priority", "medium")),
                    reason=(
                        f"alternative_probe_for_quota_hold:{item_id}:"
                        f"{alternative_reason or 'targeted_fixture'}"
                    ),
                    affected_checks=["research_journal_summary", item_id, "remeasure_quota_hold"],
                )
            continue
        _append_action(
            command=command,
            source="research_journal_remeasure",
            priority=str(recommended.get("priority", "medium")),
            reason=(
                f"research_journal_{str(recommended.get('source', 'summary'))}:"
                f"{str(recommended.get('id', ''))}:count={int(recommended.get('count', 0) or 0)}"
            ),
            affected_checks=["research_journal_summary", str(recommended.get("id", ""))],
        )
    if not bool(operational_checklist.get("runbook_drop_rate_ok", True)):
        _append_action(
            command=(
                "python scripts/eval/operational_readiness.py "
                "--strict-production --runbook-max-actions 50 --runbook-max-per-source 0 "
                f"--runbook-drop-rate-threshold {runbook_drop_rate_threshold:.3f}"
            ),
            source="runbook_drop_rate_recovery",
            priority="medium",
            reason="runbook_drop_rate_exceeded",
            affected_checks=["runbook_drop_rate_ok"],
        )
    if not bool(operational_checklist.get("efficiency_shortcut_action_ok", True)):
        threshold = int(operational_checklist.get("efficiency_shortcut_action_threshold", 0) or 0)
        count = int(operational_checklist.get("efficiency_shortcut_action_count", 0) or 0)
        _append_action(
            command="python scripts/eval/operational_readiness.py --record-repair-command \"python scripts/eval/energy_efficiency_benchmark.py\" --record-repair-status success --record-repair-source efficiency_shortcut_cleanup",
            source="efficiency_shortcut_recovery",
            priority="medium",
            reason=f"efficiency_shortcut_action_count_exceeded:{count}>{threshold}",
            affected_checks=["efficiency_shortcut_action_ok"],
        )
    if not bool(operational_checklist.get("efficiency_shortcut_overuse_rate_ok", True)):
        rate = float(operational_checklist.get("efficiency_shortcut_overuse_rate", 0.0) or 0.0)
        rate_threshold = float(operational_checklist.get("efficiency_shortcut_overuse_rate_threshold", 0.0) or 0.0)
        _append_action(
            command="python scripts/eval/operational_readiness.py --strict-production --refresh-artifacts --include-accuracy",
            source="efficiency_shortcut_chronic_recovery",
            priority="high",
            reason=f"efficiency_shortcut_overuse_rate_exceeded:{rate:.3f}>{rate_threshold:.3f}",
            affected_checks=["efficiency_shortcut_overuse_rate_ok"],
        )
    metadata = {
        **stats,
        "max_actions": int(max_actions),
        "max_per_source": int(max_per_source),
        "remeasure_command_history_quota": int(remeasure_command_history_quota),
        "remeasure_history_command_counts": dict(sorted(remeasure_history_command_counts.items())),
        "skipped_source_cap_by_source": dict(
            sorted(
                (
                    str(name),
                    int(count or 0),
                )
                for name, count in (
                    stats.get("skipped_source_cap_by_source", {})
                    if isinstance(stats.get("skipped_source_cap_by_source"), dict)
                    else {}
                ).items()
                if str(name).strip() and int(count or 0) > 0
            )
        ),
        "skipped_duplicate_by_source": dict(
            sorted(
                (
                    str(name),
                    int(count or 0),
                )
                for name, count in (
                    stats.get("skipped_duplicate_by_source", {})
                    if isinstance(stats.get("skipped_duplicate_by_source"), dict)
                    else {}
                ).items()
                if str(name).strip() and int(count or 0) > 0
            )
        ),
        "skipped_empty_command_by_source": dict(
            sorted(
                (
                    str(name),
                    int(count or 0),
                )
                for name, count in (
                    stats.get("skipped_empty_command_by_source", {})
                    if isinstance(stats.get("skipped_empty_command_by_source"), dict)
                    else {}
                ).items()
                if str(name).strip() and int(count or 0) > 0
            )
        ),
        "skipped_max_actions_by_source": dict(
            sorted(
                (
                    str(name),
                    int(count or 0),
                )
                for name, count in (
                    stats.get("skipped_max_actions_by_source", {})
                    if isinstance(stats.get("skipped_max_actions_by_source"), dict)
                    else {}
                ).items()
                if str(name).strip() and int(count or 0) > 0
            )
        ),
        "skipped_remeasure_command_history_quota_by_command": dict(
            sorted(
                (
                    str(name),
                    int(count or 0),
                )
                for name, count in (
                    stats.get("skipped_remeasure_command_history_quota_by_command", {})
                    if isinstance(stats.get("skipped_remeasure_command_history_quota_by_command"), dict)
                    else {}
                ).items()
                if str(name).strip() and int(count or 0) > 0
            )
        ),
        "skipped_remeasure_command_history_quota_items": (
            stats.get("skipped_remeasure_command_history_quota_items", [])
            if isinstance(stats.get("skipped_remeasure_command_history_quota_items"), list)
            else []
        ),
    }
    if return_metadata:
        return actions, metadata
    return actions


def summarize_runbook_actions(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_counts: Dict[str, int] = {}
    priority_counts: Dict[str, int] = {}
    for item in actions:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip() or "unknown"
        priority = str(item.get("priority", "")).strip().lower() or "unknown"
        source_counts[source] = int(source_counts.get(source, 0)) + 1
        priority_counts[priority] = int(priority_counts.get(priority, 0)) + 1
    return {
        "total_actions": int(len([item for item in actions if isinstance(item, dict)])),
        "source_counts": dict(sorted(source_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "priority_counts": dict(sorted(priority_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def summarize_research_planner_task_status(
    research_review_compact: Dict[str, Any],
    research_journal_summary: Dict[str, Any],
    *,
    cleanup_threshold: int = 2,
) -> Dict[str, Any]:
    compact = research_review_compact if isinstance(research_review_compact, dict) else {}
    journal = research_journal_summary if isinstance(research_journal_summary, dict) else {}
    pending_cause_boundary = int(compact.get("cause_boundary_documentation_count", 0) or 0)
    pending_fixture_repair = int(compact.get("targeted_fixture_repair_count", 0) or 0)
    pending_count = int(pending_cause_boundary + pending_fixture_repair)
    completed_count = int(journal.get("completed_research_planner_task_count", 0) or 0)
    cleanup_pending_count = int(journal.get("research_planner_task_cleanup_pending_count", 0) or 0)
    cleanup_success_count = int(journal.get("research_planner_task_cleanup_success_count", 0) or 0)
    cleanup_skipped_count = int(journal.get("research_planner_task_cleanup_skipped_count", 0) or 0)
    cleanup_entries = (
        journal.get("research_planner_task_cleanup_entries", [])
        if isinstance(journal.get("research_planner_task_cleanup_entries", []), list)
        else []
    )
    total_count = int(pending_count + completed_count)
    completion_ratio = float(completed_count) / float(total_count) if total_count > 0 else 1.0
    threshold = int(max(cleanup_threshold, 1))
    pending_sources = {
        str(item.get("source", "") or "")
        for item in cleanup_entries
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "pending"
    }
    pending_commands = {
        str(item.get("command", "") or "")
        for item in cleanup_entries
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "pending"
    }
    stalled_reason = ""
    stalled_action_source = ""
    if cleanup_pending_count > 0 and pending_count >= threshold:
        if pending_fixture_repair > 0:
            stalled_reason = "fixture_implementation_wait"
            stalled_action_source = "research_planner_fixture_repair_followup"
        elif pending_cause_boundary > 0 and (
            any("manual" in source for source in pending_sources)
            or any("review" in command for command in pending_commands)
        ):
            stalled_reason = "manual_review_wait"
            stalled_action_source = "research_planner_manual_review_followup"
        elif pending_cause_boundary > 0:
            stalled_reason = "documentation_not_reflected"
            stalled_action_source = "research_planner_documentation_followup"
        else:
            stalled_reason = "cleanup_pending"
            stalled_action_source = "research_planner_task_cleanup_stalled"
    return {
        "pending_count": pending_count,
        "pending_cause_boundary_documentation_count": pending_cause_boundary,
        "pending_targeted_fixture_repair_count": pending_fixture_repair,
        "completed_count": completed_count,
        "total_count": total_count,
        "completion_ratio": float(completion_ratio),
        "cleanup_threshold": threshold,
        "cleanup_pending_count": cleanup_pending_count,
        "cleanup_success_count": cleanup_success_count,
        "cleanup_skipped_count": cleanup_skipped_count,
        "cleanup_stalled": bool(cleanup_pending_count > 0 and pending_count >= threshold),
        "cleanup_stalled_reason": stalled_reason,
        "cleanup_stalled_action_source": stalled_action_source,
        "cleanup_needed": bool(pending_count >= threshold and cleanup_pending_count <= 0),
    }


def summarize_roadmap_patch_refresh_policy(
    research_journal_summary: Dict[str, Any],
    *,
    low_ratio_threshold: float = 0.15,
    high_ratio_threshold: float = 0.85,
    min_rejected_items: int = 2,
) -> Dict[str, Any]:
    summary = research_journal_summary if isinstance(research_journal_summary, dict) else {}
    rejected_count = int(summary.get("roadmap_patch_rejected_item_count", 0) or 0)
    refreshed_count = int(summary.get("roadmap_patch_refreshed_item_count", 0) or 0)
    if rejected_count <= 0:
        rejected_items = summary.get("roadmap_patch_rejected_items", [])
        rejected_count = len(rejected_items) if isinstance(rejected_items, list) else 0
    if refreshed_count <= 0:
        refreshed_items = summary.get("roadmap_patch_refreshed_items", [])
        refreshed_count = len(refreshed_items) if isinstance(refreshed_items, list) else 0
    ratio = (
        float(summary.get("roadmap_patch_refresh_to_rejection_ratio", 0.0) or 0.0)
        if "roadmap_patch_refresh_to_rejection_ratio" in summary
        else (float(refreshed_count) / float(rejected_count) if rejected_count > 0 else 0.0)
    )
    low = float(max(low_ratio_threshold, 0.0))
    high = float(min(max(high_ratio_threshold, low), 1.0))
    minimum = int(max(min_rejected_items, 1))
    status = "insufficient_history"
    action_source = ""
    followup_pending_count = int(summary.get("roadmap_patch_refresh_policy_followup_pending_count", 0) or 0)
    followup_success_count = int(summary.get("roadmap_patch_refresh_policy_followup_success_count", 0) or 0)
    followup_skipped_count = int(summary.get("roadmap_patch_refresh_policy_followup_skipped_count", 0) or 0)
    followup_failed_count = int(summary.get("roadmap_patch_refresh_policy_followup_failed_count", 0) or 0)
    evidence_success_count = int(summary.get("roadmap_patch_evidence_collection_success_count", 0) or 0)
    if evidence_success_count > 0:
        return {
            "status": "evidence_collection_completed",
            "rejected_item_count": rejected_count,
            "refreshed_item_count": refreshed_count,
            "refresh_to_rejection_ratio": float(ratio),
            "low_ratio_threshold": low,
            "high_ratio_threshold": high,
            "min_rejected_items": minimum,
            "needs_followup": False,
            "action_source": "",
        }
    if followup_failed_count >= 2:
        return {
            "status": "followup_failed_evidence_collection_needed",
            "rejected_item_count": rejected_count,
            "refreshed_item_count": refreshed_count,
            "refresh_to_rejection_ratio": float(ratio),
            "low_ratio_threshold": low,
            "high_ratio_threshold": high,
            "min_rejected_items": minimum,
            "needs_followup": True,
            "action_source": "roadmap_patch_refresh_evidence_collection_fallback",
        }
    if followup_success_count > 0 or followup_skipped_count > 0:
        return {
            "status": "followup_completed",
            "rejected_item_count": rejected_count,
            "refreshed_item_count": refreshed_count,
            "refresh_to_rejection_ratio": float(ratio),
            "low_ratio_threshold": low,
            "high_ratio_threshold": high,
            "min_rejected_items": minimum,
            "needs_followup": False,
            "action_source": "",
        }
    if followup_pending_count > 0:
        return {
            "status": "followup_pending",
            "rejected_item_count": rejected_count,
            "refreshed_item_count": refreshed_count,
            "refresh_to_rejection_ratio": float(ratio),
            "low_ratio_threshold": low,
            "high_ratio_threshold": high,
            "min_rejected_items": minimum,
            "needs_followup": False,
            "action_source": "",
        }
    if rejected_count >= minimum:
        if ratio > high:
            status = "over_resurfacing"
            action_source = "roadmap_patch_refresh_over_resurfacing_followup"
        elif ratio < low:
            status = "over_suppression"
            action_source = "roadmap_patch_refresh_over_suppression_followup"
        else:
            status = "balanced"
    return {
        "status": status,
        "rejected_item_count": rejected_count,
        "refreshed_item_count": refreshed_count,
        "refresh_to_rejection_ratio": float(ratio),
        "low_ratio_threshold": low,
        "high_ratio_threshold": high,
        "min_rejected_items": minimum,
        "needs_followup": bool(status in {"over_resurfacing", "over_suppression"}),
        "action_source": action_source,
    }


def attach_roadmap_patch_refresh_policy_followups_to_research_journal_summary(
    research_journal_summary: Dict[str, Any],
    execution_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = dict(research_journal_summary) if isinstance(research_journal_summary, dict) else {}
    entries: List[Dict[str, Any]] = (
        [dict(item) for item in summary.get("roadmap_patch_refresh_policy_followup_entries", []) if isinstance(item, dict)]
        if isinstance(summary.get("roadmap_patch_refresh_policy_followup_entries", []), list)
        else []
    )
    evidence_entries: List[Dict[str, Any]] = (
        [dict(item) for item in summary.get("roadmap_patch_evidence_collection_entries", []) if isinstance(item, dict)]
        if isinstance(summary.get("roadmap_patch_evidence_collection_entries", []), list)
        else []
    )
    seen_keys = {json.dumps(item, sort_keys=True) for item in entries}
    seen_evidence_keys = {json.dumps(item, sort_keys=True) for item in evidence_entries}

    def _evidence_kind(command: str, source: str, checks: List[str]) -> str:
        haystack = " ".join([command, source, *checks]).lower()
        if "real_data" in haystack or "real-data" in haystack or "fixture" in haystack:
            return "real_data_fixture"
        if "release_soak" in haystack or "release-soak" in haystack or "soak_trend" in haystack:
            return "release_soak_trend"
        if "targeted_probe" in haystack or "targeted-probe" in haystack or "probe" in haystack:
            return "targeted_probe"
        return "targeted_probe"

    def _next_evidence_kind(latest_kind: str) -> str:
        if latest_kind == "targeted_probe":
            return "real_data_fixture"
        if latest_kind == "real_data_fixture":
            return "release_soak_trend"
        if latest_kind == "release_soak_trend":
            return "release_soak_trend"
        return "targeted_probe"

    for entry in execution_log if isinstance(execution_log, list) else []:
        if not isinstance(entry, dict):
            continue
        source = str(entry.get("source", "") or "")
        command = str(entry.get("command", "") or "")
        checks = (
            [str(item).strip() for item in entry.get("covered_checks", []) if str(item).strip()]
            if isinstance(entry.get("covered_checks"), list)
            else []
        )
        check_set = set(checks)
        if (
            "roadmap_patch_refresh_policy" not in source
            and "roadmap_patch_refresh_policy" not in command
            and "roadmap_patch_refresh_policy" not in check_set
        ):
            continue
        status = str(entry.get("status", "") or "").strip().lower()
        if status not in {"pending", "success", "skipped", "failed", "timeout", "error"}:
            continue
        normalized = {
            "status": status,
            "source": source,
            "command": command,
            "covered_checks": checks,
            "resolved_timestamp": _safe_float(
                entry.get("resolved_timestamp", entry.get("timestamp", 0.0)),
                0.0,
            ),
        }
        key = json.dumps(normalized, sort_keys=True)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        entries.append(normalized)
        if "evidence_collection" in source or "evidence_collection" in command or "evidence_collection" in check_set:
            evidence = {
                "status": status,
                "source": source,
                "command": command,
                "covered_checks": checks,
                "evidence_kind": _evidence_kind(command, source, checks),
                "resolved_timestamp": normalized["resolved_timestamp"],
            }
            evidence_key = json.dumps(evidence, sort_keys=True)
            if evidence_key not in seen_evidence_keys:
                seen_evidence_keys.add(evidence_key)
                evidence_entries.append(evidence)
    status_counts: Dict[str, int] = {}
    latest_status = ""
    latest_timestamp = 0.0
    for item in entries:
        status = str(item.get("status", "") or "")
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        timestamp = _safe_float(item.get("resolved_timestamp", 0.0), 0.0)
        if timestamp >= latest_timestamp:
            latest_status = status
            latest_timestamp = float(timestamp)
    summary["roadmap_patch_refresh_policy_followup_entries"] = entries
    summary["roadmap_patch_refresh_policy_followup_pending_count"] = int(status_counts.get("pending", 0))
    summary["roadmap_patch_refresh_policy_followup_success_count"] = int(status_counts.get("success", 0))
    summary["roadmap_patch_refresh_policy_followup_skipped_count"] = int(status_counts.get("skipped", 0))
    summary["roadmap_patch_refresh_policy_followup_failed_count"] = int(
        status_counts.get("failed", 0) + status_counts.get("timeout", 0) + status_counts.get("error", 0)
    )
    summary["roadmap_patch_refresh_policy_followup_latest_status"] = latest_status
    evidence_status_counts: Dict[str, int] = {}
    latest_evidence_status = ""
    latest_evidence_kind = ""
    latest_evidence_timestamp = 0.0
    evidence_kind_counts: Dict[str, int] = {}
    for item in evidence_entries:
        status = str(item.get("status", "") or "")
        evidence_status_counts[status] = int(evidence_status_counts.get(status, 0)) + 1
        kind = str(item.get("evidence_kind", "") or "targeted_probe")
        evidence_kind_counts[kind] = int(evidence_kind_counts.get(kind, 0)) + 1
        timestamp = _safe_float(item.get("resolved_timestamp", 0.0), 0.0)
        if timestamp >= latest_evidence_timestamp:
            latest_evidence_status = status
            latest_evidence_kind = kind
            latest_evidence_timestamp = float(timestamp)
    summary["roadmap_patch_evidence_collection_entries"] = evidence_entries
    summary["roadmap_patch_evidence_collection_success_count"] = int(evidence_status_counts.get("success", 0))
    summary["roadmap_patch_evidence_collection_pending_count"] = int(evidence_status_counts.get("pending", 0))
    summary["roadmap_patch_evidence_collection_failed_count"] = int(
        evidence_status_counts.get("failed", 0)
        + evidence_status_counts.get("timeout", 0)
        + evidence_status_counts.get("error", 0)
    )
    summary["roadmap_patch_evidence_collection_latest_status"] = latest_evidence_status
    summary["roadmap_patch_evidence_collection_kind_counts"] = dict(sorted(evidence_kind_counts.items()))
    summary["roadmap_patch_evidence_collection_latest_kind"] = latest_evidence_kind
    summary["roadmap_patch_evidence_collection_next_required_kind"] = (
        _next_evidence_kind(latest_evidence_kind) if latest_evidence_kind else ""
    )
    return summary


def attach_remeasure_quota_holds_to_research_journal_summary(
    research_journal_summary: Dict[str, Any],
    runbook_action_build_stats: Dict[str, Any],
) -> Dict[str, Any]:
    summary = dict(research_journal_summary) if isinstance(research_journal_summary, dict) else {}
    hold_items = (
        runbook_action_build_stats.get("skipped_remeasure_command_history_quota_items", [])
        if isinstance(runbook_action_build_stats, dict)
        and isinstance(runbook_action_build_stats.get("skipped_remeasure_command_history_quota_items"), list)
        else []
    )
    normalized: List[Dict[str, Any]] = []
    alternative_actions: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in hold_items:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command", "") or "").strip()
        item_id = str(item.get("id", "") or "").strip()
        if not command:
            continue
        key = f"{item_id}|{command}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        normalized.append(
            {
                "id": item_id,
                "source": str(item.get("source", "summary") or "summary"),
                "command": command,
                "priority": str(item.get("priority", "medium") or "medium"),
                "history_count": int(item.get("history_count", 0) or 0),
                "quota": int(item.get("quota", 0) or 0),
                "hold_reason": str(item.get("hold_reason", "remeasure_command_history_quota") or "remeasure_command_history_quota"),
                "alternative_command": str(item.get("alternative_command", "") or ""),
                "alternative_reason": str(item.get("alternative_reason", "") or ""),
            }
        )
        alternative_command = str(item.get("alternative_command", "") or "").strip()
        if alternative_command:
            alternative_actions.append(
                {
                    "id": item_id,
                    "source": "remeasure_quota_hold",
                    "command": alternative_command,
                    "priority": str(item.get("priority", "medium") or "medium"),
                    "reason": str(item.get("alternative_reason", "") or "targeted_fixture"),
                }
            )
    summary["remeasure_quota_hold_count"] = int(len(normalized))
    summary["remeasure_quota_holds"] = normalized
    summary["alternative_benchmark_actions"] = alternative_actions
    if any(
        str(item.get("id", "") or "") == STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID
        for item in alternative_actions
        if isinstance(item, dict)
    ):
        repair_loop = (
            dict(summary.get("stage_e_observed_acceptance_candidate_repair_loop", {}))
            if isinstance(
                summary.get("stage_e_observed_acceptance_candidate_repair_loop", {}),
                dict,
            )
            else {}
        )
        repair_loop.setdefault(
            "schema",
            "sara-stage-e-observed-acceptance-candidate-repair-loop-v1",
        )
        repair_loop["id"] = STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID
        repair_loop["alternative_probe_recommended"] = True
        repair_loop["needs_followup"] = True
        repair_loop["promotion_review_recommended"] = False
        repair_loop["next_review_action"] = ""
        summary["stage_e_observed_acceptance_candidate_repair_loop"] = repair_loop
    return summary


def attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_summary(
    research_journal_summary: Dict[str, Any],
    execution_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = dict(research_journal_summary) if isinstance(research_journal_summary, dict) else {}
    repair_loop = (
        dict(summary.get("stage_e_observed_acceptance_candidate_repair_loop", {}))
        if isinstance(summary.get("stage_e_observed_acceptance_candidate_repair_loop", {}), dict)
        else {}
    )
    existing_entries = (
        [
            dict(item)
            for item in summary.get("stage_e_observed_acceptance_candidate_recovery_review_entries", [])
            if isinstance(item, dict)
        ]
        if isinstance(summary.get("stage_e_observed_acceptance_candidate_recovery_review_entries", []), list)
        else []
    )
    seen_keys = {json.dumps(item, sort_keys=True) for item in existing_entries}
    for entry in execution_log if isinstance(execution_log, list) else []:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "") or "").strip().lower()
        if status not in {"pending", "success", "skipped", "failed", "timeout", "error"}:
            continue
        source = str(entry.get("source", "") or "").strip()
        command = str(entry.get("command", "") or "")
        checks = (
            [str(item).strip() for item in entry.get("covered_checks", []) if str(item).strip()]
            if isinstance(entry.get("covered_checks"), list)
            else []
        )
        check_set = set(checks)
        if not (
            "stage_e_observed_acceptance_candidate_recovery_review" in source
            or "stage_e_observed_acceptance_candidate_recovery_review" in command
            or "stage_e_observed_acceptance_candidate_repair_recovery" in check_set
        ):
            continue
        entry_type = (
            "targeted_probe_recheck"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck" in source
                or "recheck_stage_e_observed_acceptance_candidate_recovery_review_targeted_probe" in command
            )
            else
            "targeted_probe"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe" in source
                or "probe_stage_e_observed_acceptance_candidate_recovery_review_evidence" in command
            )
            else
            "evidence_recheck"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck" in source
                or "recheck_stage_e_observed_acceptance_candidate_recovery_review_evidence" in command
            )
            else
            "evidence_collection"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection" in source
                or "collect_stage_e_observed_acceptance_candidate_recovery_review_evidence" in command
            )
            else
            "escalation"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation" in source
                or "escalate_stage_e_observed_acceptance_candidate_recovery_review_followup_retry" in command
            )
            else "retry"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_followup_retry" in source
                or "retry_stage_e_observed_acceptance_candidate_recovery_review_followup" in command
            )
            else (
                "followup"
                if (
                    "stage_e_observed_acceptance_candidate_recovery_review_followup" in source
                    or "followup_stage_e_observed_acceptance_candidate_recovery_review" in command
                )
                else "review"
            )
        )
        review_entry = {
            "status": status,
            "source": source,
            "command": command,
            "entry_type": entry_type,
            "timestamp": _safe_float(
                entry.get("resolved_timestamp", entry.get("timestamp", 0.0)),
                0.0,
            ),
            "covered_checks": sorted(check_set),
        }
        key = json.dumps(review_entry, sort_keys=True)
        if key not in seen_keys:
            seen_keys.add(key)
            existing_entries.append(review_entry)

    status_counts: Dict[str, int] = {}
    latest_entry: Dict[str, Any] = {}
    for entry in existing_entries:
        status = str(entry.get("status", "") or "").strip().lower() or "unknown"
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        if _safe_float(entry.get("timestamp", 0.0), 0.0) >= _safe_float(
            latest_entry.get("timestamp", 0.0),
            0.0,
        ):
            latest_entry = entry
    latest_status = str(latest_entry.get("status", "") or "")
    completed = latest_status in {"success", "skipped"}
    in_progress = latest_status == "pending"
    followup_entries = [
        item
        for item in existing_entries
        if str(item.get("entry_type", "") or "")
        in {
            "followup",
            "retry",
            "escalation",
            "evidence_collection",
            "evidence_recheck",
            "targeted_probe",
            "targeted_probe_recheck",
        }
    ]
    retry_entries = [
        item
        for item in existing_entries
        if str(item.get("entry_type", "") or "") == "retry"
    ]
    escalation_entries = [
        item
        for item in existing_entries
        if str(item.get("entry_type", "") or "") == "escalation"
    ]
    evidence_collection_entries = [
        item
        for item in existing_entries
        if str(item.get("entry_type", "") or "") == "evidence_collection"
    ]
    evidence_recheck_entries = [
        item
        for item in existing_entries
        if str(item.get("entry_type", "") or "") == "evidence_recheck"
    ]
    targeted_probe_entries = [
        item
        for item in existing_entries
        if str(item.get("entry_type", "") or "") == "targeted_probe"
    ]
    targeted_probe_recheck_entries = [
        item
        for item in existing_entries
        if str(item.get("entry_type", "") or "") == "targeted_probe_recheck"
    ]
    latest_followup_entry: Dict[str, Any] = {}
    for entry in followup_entries:
        if _safe_float(entry.get("timestamp", 0.0), 0.0) >= _safe_float(
            latest_followup_entry.get("timestamp", 0.0),
            0.0,
        ):
            latest_followup_entry = entry
    latest_followup_status = str(latest_followup_entry.get("status", "") or "")
    followup_in_progress = latest_followup_status == "pending"
    followup_completed = latest_followup_status in {"success", "skipped"}
    followup_failed = latest_followup_status in {"failed", "timeout", "error"}
    latest_retry_entry: Dict[str, Any] = {}
    for entry in retry_entries:
        if _safe_float(entry.get("timestamp", 0.0), 0.0) >= _safe_float(
            latest_retry_entry.get("timestamp", 0.0),
            0.0,
        ):
            latest_retry_entry = entry
    latest_retry_status = str(latest_retry_entry.get("status", "") or "")
    retry_in_progress = latest_retry_status == "pending"
    retry_completed = latest_retry_status in {"success", "skipped"}
    retry_failed = latest_retry_status in {"failed", "timeout", "error"}
    latest_escalation_entry: Dict[str, Any] = {}
    for entry in escalation_entries:
        if _safe_float(entry.get("timestamp", 0.0), 0.0) >= _safe_float(
            latest_escalation_entry.get("timestamp", 0.0),
            0.0,
        ):
            latest_escalation_entry = entry
    latest_escalation_status = str(latest_escalation_entry.get("status", "") or "")
    escalation_in_progress = latest_escalation_status == "pending"
    escalation_completed = latest_escalation_status in {"success", "skipped"}
    escalation_failed = latest_escalation_status in {"failed", "timeout", "error"}
    latest_evidence_collection_entry: Dict[str, Any] = {}
    for entry in evidence_collection_entries:
        if _safe_float(entry.get("timestamp", 0.0), 0.0) >= _safe_float(
            latest_evidence_collection_entry.get("timestamp", 0.0),
            0.0,
        ):
            latest_evidence_collection_entry = entry
    latest_evidence_collection_status = str(
        latest_evidence_collection_entry.get("status", "") or ""
    )
    evidence_collection_in_progress = latest_evidence_collection_status == "pending"
    evidence_collection_completed = latest_evidence_collection_status in {"success", "skipped"}
    evidence_collection_failed = latest_evidence_collection_status in {
        "failed",
        "timeout",
        "error",
    }
    latest_evidence_recheck_entry: Dict[str, Any] = {}
    for entry in evidence_recheck_entries:
        if _safe_float(entry.get("timestamp", 0.0), 0.0) >= _safe_float(
            latest_evidence_recheck_entry.get("timestamp", 0.0),
            0.0,
        ):
            latest_evidence_recheck_entry = entry
    latest_evidence_recheck_status = str(latest_evidence_recheck_entry.get("status", "") or "")
    evidence_recheck_in_progress = latest_evidence_recheck_status == "pending"
    evidence_recheck_completed = latest_evidence_recheck_status in {"success", "skipped"}
    evidence_recheck_failed = latest_evidence_recheck_status in {"failed", "timeout", "error"}
    latest_targeted_probe_entry: Dict[str, Any] = {}
    for entry in targeted_probe_entries:
        if _safe_float(entry.get("timestamp", 0.0), 0.0) >= _safe_float(
            latest_targeted_probe_entry.get("timestamp", 0.0),
            0.0,
        ):
            latest_targeted_probe_entry = entry
    latest_targeted_probe_status = str(latest_targeted_probe_entry.get("status", "") or "")
    targeted_probe_in_progress = latest_targeted_probe_status == "pending"
    targeted_probe_completed = latest_targeted_probe_status in {"success", "skipped"}
    targeted_probe_failed = latest_targeted_probe_status in {"failed", "timeout", "error"}
    latest_targeted_probe_recheck_entry: Dict[str, Any] = {}
    for entry in targeted_probe_recheck_entries:
        if _safe_float(entry.get("timestamp", 0.0), 0.0) >= _safe_float(
            latest_targeted_probe_recheck_entry.get("timestamp", 0.0),
            0.0,
        ):
            latest_targeted_probe_recheck_entry = entry
    latest_targeted_probe_recheck_status = str(
        latest_targeted_probe_recheck_entry.get("status", "") or ""
    )
    targeted_probe_recheck_in_progress = latest_targeted_probe_recheck_status == "pending"
    targeted_probe_recheck_completed = latest_targeted_probe_recheck_status in {
        "success",
        "skipped",
    }
    targeted_probe_recheck_failed = latest_targeted_probe_recheck_status in {
        "failed",
        "timeout",
        "error",
    }
    latest_timestamp = _safe_float(latest_entry.get("timestamp", 0.0), 0.0)
    latest_age_seconds = max(time.time() - latest_timestamp, 0.0) if latest_timestamp > 0 else 0.0
    stale = bool(
        in_progress
        and latest_timestamp > 0
        and latest_age_seconds >= STAGE_E_RECOVERY_REVIEW_STALE_SECONDS
    )

    repair_loop.setdefault(
        "schema",
        "sara-stage-e-observed-acceptance-candidate-repair-loop-v1",
    )
    repair_loop.setdefault("id", STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID)
    repair_loop["promotion_review_completed"] = bool(completed)
    repair_loop["promotion_review_in_progress"] = bool(in_progress)
    repair_loop["promotion_review_stale"] = stale
    repair_loop["promotion_review_followup_in_progress"] = bool(followup_in_progress)
    repair_loop["promotion_review_followup_completed"] = bool(followup_completed)
    repair_loop["promotion_review_followup_failed"] = bool(followup_failed)
    repair_loop["promotion_review_followup_latest_status"] = latest_followup_status
    repair_loop["promotion_review_followup_retry_in_progress"] = bool(retry_in_progress)
    repair_loop["promotion_review_followup_retry_completed"] = bool(retry_completed)
    repair_loop["promotion_review_followup_retry_failed"] = bool(retry_failed)
    repair_loop["promotion_review_followup_retry_latest_status"] = latest_retry_status
    repair_loop["promotion_review_followup_retry_escalation_in_progress"] = bool(
        escalation_in_progress
    )
    repair_loop["promotion_review_followup_retry_escalation_completed"] = bool(
        escalation_completed
    )
    repair_loop["promotion_review_followup_retry_escalation_failed"] = bool(escalation_failed)
    repair_loop["promotion_review_followup_retry_escalation_latest_status"] = (
        latest_escalation_status
    )
    repair_loop["promotion_review_evidence_collection_in_progress"] = bool(
        evidence_collection_in_progress
    )
    repair_loop["promotion_review_evidence_collection_completed"] = bool(
        evidence_collection_completed
    )
    repair_loop["promotion_review_evidence_collection_failed"] = bool(evidence_collection_failed)
    repair_loop["promotion_review_evidence_collection_latest_status"] = (
        latest_evidence_collection_status
    )
    repair_loop["promotion_review_evidence_recheck_in_progress"] = bool(
        evidence_recheck_in_progress
    )
    repair_loop["promotion_review_evidence_recheck_completed"] = bool(
        evidence_recheck_completed
    )
    repair_loop["promotion_review_evidence_recheck_failed"] = bool(evidence_recheck_failed)
    repair_loop["promotion_review_evidence_recheck_latest_status"] = latest_evidence_recheck_status
    repair_loop["promotion_review_targeted_probe_in_progress"] = bool(
        targeted_probe_in_progress
    )
    repair_loop["promotion_review_targeted_probe_completed"] = bool(targeted_probe_completed)
    repair_loop["promotion_review_targeted_probe_failed"] = bool(targeted_probe_failed)
    repair_loop["promotion_review_targeted_probe_latest_status"] = latest_targeted_probe_status
    repair_loop["promotion_review_targeted_probe_recheck_in_progress"] = bool(
        targeted_probe_recheck_in_progress
    )
    repair_loop["promotion_review_targeted_probe_recheck_completed"] = bool(
        targeted_probe_recheck_completed
    )
    repair_loop["promotion_review_targeted_probe_recheck_failed"] = bool(
        targeted_probe_recheck_failed
    )
    repair_loop["promotion_review_targeted_probe_recheck_latest_status"] = (
        latest_targeted_probe_recheck_status
    )
    repair_loop["promotion_review_latest_status"] = latest_status
    repair_loop["promotion_review_latest_timestamp"] = float(latest_timestamp)
    repair_loop["promotion_review_latest_age_seconds"] = float(latest_age_seconds)
    repair_loop["promotion_review_stale_after_seconds"] = float(STAGE_E_RECOVERY_REVIEW_STALE_SECONDS)
    repair_loop["promotion_review_completed_count"] = int(
        status_counts.get("success", 0) + status_counts.get("skipped", 0)
    )
    if completed or in_progress:
        repair_loop["promotion_review_recommended"] = False
    if completed:
        repair_loop["needs_followup"] = False
        repair_loop["next_review_action"] = ""

    summary["stage_e_observed_acceptance_candidate_recovery_review_entries"] = existing_entries
    summary["stage_e_observed_acceptance_candidate_recovery_review_count"] = int(len(existing_entries))
    summary["stage_e_observed_acceptance_candidate_recovery_review_status_counts"] = dict(
        sorted(status_counts.items())
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_latest_status"] = latest_status
    summary["stage_e_observed_acceptance_candidate_recovery_review_latest_timestamp"] = float(latest_timestamp)
    summary["stage_e_observed_acceptance_candidate_recovery_review_latest_age_seconds"] = float(latest_age_seconds)
    summary["stage_e_observed_acceptance_candidate_recovery_review_stale_after_seconds"] = float(
        STAGE_E_RECOVERY_REVIEW_STALE_SECONDS
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_completed"] = bool(completed)
    summary["stage_e_observed_acceptance_candidate_recovery_review_in_progress"] = bool(in_progress)
    summary["stage_e_observed_acceptance_candidate_recovery_review_stale"] = stale
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_count"] = int(len(followup_entries))
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_in_progress"] = bool(
        followup_in_progress
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_completed"] = bool(
        followup_completed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_failed"] = bool(
        followup_failed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_latest_status"] = latest_followup_status
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_count"] = int(len(retry_entries))
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_in_progress"] = bool(
        retry_in_progress
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_completed"] = bool(
        retry_completed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_failed"] = bool(
        retry_failed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_latest_status"] = latest_retry_status
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_count"] = int(
        len(escalation_entries)
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_in_progress"] = bool(
        escalation_in_progress
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_completed"] = bool(
        escalation_completed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_failed"] = bool(
        escalation_failed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_latest_status"] = (
        latest_escalation_status
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_count"] = int(
        len(evidence_collection_entries)
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_in_progress"] = bool(
        evidence_collection_in_progress
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_completed"] = bool(
        evidence_collection_completed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_failed"] = bool(
        evidence_collection_failed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_latest_status"] = (
        latest_evidence_collection_status
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_count"] = int(
        len(evidence_recheck_entries)
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_in_progress"] = bool(
        evidence_recheck_in_progress
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_completed"] = bool(
        evidence_recheck_completed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_failed"] = bool(
        evidence_recheck_failed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_latest_status"] = (
        latest_evidence_recheck_status
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_count"] = int(
        len(targeted_probe_entries)
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_in_progress"] = bool(
        targeted_probe_in_progress
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_completed"] = bool(
        targeted_probe_completed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_failed"] = bool(
        targeted_probe_failed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_latest_status"] = (
        latest_targeted_probe_status
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_count"] = int(
        len(targeted_probe_recheck_entries)
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_in_progress"] = bool(
        targeted_probe_recheck_in_progress
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_completed"] = bool(
        targeted_probe_recheck_completed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_failed"] = bool(
        targeted_probe_recheck_failed
    )
    summary["stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_latest_status"] = (
        latest_targeted_probe_recheck_status
    )
    summary["stage_e_observed_acceptance_candidate_repair_loop"] = repair_loop
    return summary


def attach_research_planner_task_completions_to_research_journal_summary(
    research_journal_summary: Dict[str, Any],
    execution_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = dict(research_journal_summary) if isinstance(research_journal_summary, dict) else {}
    completed_boundary_ids: set[str] = {
        str(item).strip()
        for item in summary.get("completed_cause_boundary_documentation_ids", [])
        if str(item).strip()
    } if isinstance(summary.get("completed_cause_boundary_documentation_ids", []), list) else set()
    completed_fixture_ids: set[str] = {
        str(item).strip()
        for item in summary.get("completed_targeted_fixture_repair_ids", [])
        if str(item).strip()
    } if isinstance(summary.get("completed_targeted_fixture_repair_ids", []), list) else set()
    completion_entries: List[Dict[str, Any]] = (
        [dict(item) for item in summary.get("completed_research_planner_tasks", []) if isinstance(item, dict)]
        if isinstance(summary.get("completed_research_planner_tasks", []), list)
        else []
    )
    cleanup_entries: List[Dict[str, Any]] = (
        [dict(item) for item in summary.get("research_planner_task_cleanup_entries", []) if isinstance(item, dict)]
        if isinstance(summary.get("research_planner_task_cleanup_entries", []), list)
        else []
    )
    seen_keys = {json.dumps(item, sort_keys=True) for item in completion_entries}
    seen_cleanup_keys = {json.dumps(item, sort_keys=True) for item in cleanup_entries}

    for entry in execution_log if isinstance(execution_log, list) else []:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "") or "").strip().lower()
        source = str(entry.get("source", "") or "").strip()
        command = str(entry.get("command", "") or "")
        checks = (
            [str(item).strip() for item in entry.get("covered_checks", []) if str(item).strip()]
            if isinstance(entry.get("covered_checks"), list)
            else []
        )
        check_set = set(checks)
        if (
            "research_planner_task_cleanup" in source
            or "research_planner_task_cleanup" in command
            or "research_planner_task_cleanup" in check_set
        ):
            if status in {"pending", "success", "skipped", "failed", "timeout", "error"}:
                cleanup = {
                    "status": status,
                    "source": source,
                    "command": command,
                    "timestamp": _safe_float(
                        entry.get("resolved_timestamp", entry.get("timestamp", 0.0)),
                        0.0,
                    ),
                    "covered_checks": sorted(check_set),
                }
                key = json.dumps(cleanup, sort_keys=True)
                if key not in seen_cleanup_keys:
                    seen_cleanup_keys.add(key)
                    cleanup_entries.append(cleanup)
        if status not in {"success", "skipped"}:
            continue
        task_type = ""
        if "cause_boundary_documentation" in source or "cause_boundary_documentation" in check_set:
            task_type = "cause_boundary_documentation"
        elif "targeted_fixture_repair" in source or "targeted_fixture_repair" in check_set:
            task_type = "targeted_fixture_repair"
        if not task_type:
            continue
        target_ids = sorted(
            item
            for item in check_set
            if item
            and item
            not in {
                "research_review",
                "roadmap_patch_suggestion",
                "cause_boundary_documentation",
                "targeted_fixture_repair",
            }
        )
        if not target_ids:
            explicit_id = str(entry.get("task_id", "") or entry.get("id", "") or "").strip()
            if explicit_id:
                target_ids = [explicit_id]
        for target_id in target_ids:
            if task_type == "cause_boundary_documentation":
                completed_boundary_ids.add(target_id)
            else:
                completed_fixture_ids.add(target_id)
            completion = {
                "id": target_id,
                "task_type": task_type,
                "status": status,
                "source": source,
                "command": str(entry.get("command", "") or ""),
                "timestamp": _safe_float(
                    entry.get("resolved_timestamp", entry.get("timestamp", 0.0)),
                    0.0,
                ),
            }
            key = json.dumps(completion, sort_keys=True)
            if key not in seen_keys:
                seen_keys.add(key)
                completion_entries.append(completion)

    summary["completed_cause_boundary_documentation_ids"] = sorted(completed_boundary_ids)
    summary["completed_targeted_fixture_repair_ids"] = sorted(completed_fixture_ids)
    summary["completed_research_planner_task_count"] = int(len(completion_entries))
    summary["completed_research_planner_tasks"] = completion_entries
    summary["research_planner_task_cleanup_entries"] = cleanup_entries
    summary["research_planner_task_cleanup_pending_count"] = int(
        sum(1 for item in cleanup_entries if str(item.get("status", "")).lower() == "pending")
    )
    summary["research_planner_task_cleanup_success_count"] = int(
        sum(1 for item in cleanup_entries if str(item.get("status", "")).lower() == "success")
    )
    summary["research_planner_task_cleanup_skipped_count"] = int(
        sum(1 for item in cleanup_entries if str(item.get("status", "")).lower() == "skipped")
    )
    return summary


def summarize_runbook_action_build_stats(stats: Dict[str, Any]) -> Dict[str, float]:
    considered = int(stats.get("considered_count", 0) or 0)
    if considered <= 0:
        return {
            "drop_rate": 0.0,
            "duplicate_drop_rate": 0.0,
            "empty_drop_rate": 0.0,
            "source_cap_drop_rate": 0.0,
            "max_actions_drop_rate": 0.0,
        }
    duplicate = int(stats.get("skipped_duplicate_count", 0) or 0)
    empty = int(stats.get("skipped_empty_command_count", 0) or 0)
    source_cap = int(stats.get("skipped_source_cap_count", 0) or 0)
    max_actions = int(stats.get("skipped_max_actions_count", 0) or 0)
    remeasure_quota = int(stats.get("skipped_remeasure_command_history_quota_count", 0) or 0)
    dropped = duplicate + empty + source_cap + max_actions + remeasure_quota
    return {
        "drop_rate": float(dropped) / float(considered),
        "duplicate_drop_rate": float(duplicate) / float(considered),
        "empty_drop_rate": float(empty) / float(considered),
        "source_cap_drop_rate": float(source_cap) / float(considered),
        "max_actions_drop_rate": float(max_actions) / float(considered),
        "remeasure_command_history_quota_drop_rate": float(remeasure_quota) / float(considered),
    }


def _build_refresh_commands(
    soak_profile: str,
    include_accuracy: bool,
    *,
    phase3_regression_tolerance: float = 0.025,
) -> List[List[str]]:
    python_bin = sys.executable
    commands: List[List[str]] = [
        [
            python_bin,
            os.path.join("scripts", "eval", "phase3_accuracy_suite.py"),
            "--regression-tolerance",
            f"{max(float(phase3_regression_tolerance), 0.0):.6f}",
        ],
        [python_bin, os.path.join("scripts", "eval", "phase4_scale_continual_benchmark.py")],
        [python_bin, os.path.join("scripts", "eval", "phase5_predictive_coding_benchmark.py")],
        [python_bin, os.path.join("scripts", "eval", "phase5_entry_gate.py")],
        [python_bin, os.path.join("scripts", "eval", "sparse_diffusion_block_readiness.py")],
        [python_bin, os.path.join("scripts", "eval", "phase5_completion_gate.py")],
        [python_bin, os.path.join("scripts", "eval", "real_data_external_validity.py")],
        [python_bin, os.path.join("scripts", "eval", "real_data_external_validity_ladder.py")],
    ]
    soak_command = [
        python_bin,
        os.path.join("scripts", "eval", "release_soak.py"),
        "--profile",
        str(soak_profile),
    ]
    if include_accuracy:
        soak_command.append("--include-accuracy")
    commands.append(soak_command)
    commands.append([python_bin, os.path.join("scripts", "eval", "release_gate.py"), "--skip-accuracy"])
    return commands


def _is_within_workspace(path: str) -> bool:
    workspace_root = os.path.abspath(workspace_path(""))
    abs_path = os.path.abspath(path)
    return os.path.commonpath([abs_path, workspace_root]) == workspace_root


def collect_operational_checklist_status(
    report: Dict[str, Any],
    report_path: str,
    summary_path: str,
    repair_plan_path: str,
    runbook_path: str,
    runbook_actions_path: str,
    runbook_drop_rate_threshold: float = 0.9,
    efficiency_shortcut_action_threshold: int = 3,
    efficiency_shortcut_overuse_window: int = 10,
    efficiency_shortcut_overuse_rate_threshold: float = 0.5,
) -> Dict[str, Any]:
    report_path_resolved = os.path.abspath(report_path)
    summary_path_resolved = os.path.abspath(summary_path)
    repair_plan_path_resolved = os.path.abspath(repair_plan_path)
    runbook_path_resolved = os.path.abspath(runbook_path)
    runbook_actions_path_resolved = os.path.abspath(runbook_actions_path)
    managed_output_paths_ok = (
        _is_within_workspace(report_path_resolved)
        and _is_within_workspace(summary_path_resolved)
        and _is_within_workspace(repair_plan_path_resolved)
        and _is_within_workspace(runbook_path_resolved)
        and _is_within_workspace(runbook_actions_path_resolved)
    )
    report_summary_review_ready = bool(
        managed_output_paths_ok
        and isinstance(report.get("checks", {}), dict)
        and isinstance(report.get("repair_plan", {}), dict)
        and isinstance(report.get("iterative_repair_plan", {}), dict)
        and isinstance(report.get("runbook_actions", []), list)
    )
    runbook_action_build_stats = (
        report.get("runbook_action_build_stats", {})
        if isinstance(report.get("runbook_action_build_stats"), dict)
        else {}
    )
    runbook_action_build_rates = (
        report.get("runbook_action_build_rates", {})
        if isinstance(report.get("runbook_action_build_rates"), dict)
        else {}
    )
    runbook_manifest_hygiene_ok = bool(
        int(runbook_action_build_stats.get("skipped_empty_command_count", 0) or 0) == 0
    )
    runbook_drop_rate_ok = bool(
        float(runbook_action_build_rates.get("drop_rate", 0.0) or 0.0)
        <= float(max(runbook_drop_rate_threshold, 0.0))
    )
    runbook_action_summary = (
        report.get("runbook_action_summary", {})
        if isinstance(report.get("runbook_action_summary"), dict)
        else {}
    )
    source_counts = (
        runbook_action_summary.get("source_counts", {})
        if isinstance(runbook_action_summary.get("source_counts"), dict)
        else {}
    )
    efficiency_shortcut_action_count = int(source_counts.get("efficiency_incident_shortcut", 0) or 0)
    efficiency_shortcut_action_threshold = int(efficiency_shortcut_action_threshold)
    if efficiency_shortcut_action_threshold < 0:
        efficiency_shortcut_action_threshold = 0
    efficiency_shortcut_action_ok = bool(
        efficiency_shortcut_action_count <= int(efficiency_shortcut_action_threshold)
    )
    history = (
        report.get("efficiency_shortcut_overuse_timeline", [])
        if isinstance(report.get("efficiency_shortcut_overuse_timeline"), list)
        else []
    )
    overuse_window = max(int(efficiency_shortcut_overuse_window), 1)
    rate_threshold = float(max(efficiency_shortcut_overuse_rate_threshold, 0.0))
    historical_flags = [
        bool(item.get("overuse_active", False))
        for item in history
        if isinstance(item, dict)
    ]
    window_flags = (historical_flags + [not efficiency_shortcut_action_ok])[-overuse_window:]
    observed_window_size = len(window_flags)
    observed_overuse_count = int(sum(1 for item in window_flags if bool(item)))
    observed_overuse_rate = float(observed_overuse_count) / float(max(observed_window_size, 1))
    efficiency_shortcut_overuse_rate_ok = bool(observed_overuse_rate <= rate_threshold)
    checklist_passed = bool(
        report.get("passed", False)
        and managed_output_paths_ok
        and report_summary_review_ready
    )
    return {
        "passed": checklist_passed,
        "report_path": report_path_resolved,
        "summary_path": summary_path_resolved,
        "repair_plan_path": repair_plan_path_resolved,
        "runbook_path": runbook_path_resolved,
        "runbook_actions_path": runbook_actions_path_resolved,
        "managed_output_paths_ok": managed_output_paths_ok,
        "report_summary_review_ready": report_summary_review_ready,
        "runbook_manifest_hygiene_ok": runbook_manifest_hygiene_ok,
        "runbook_drop_rate_ok": runbook_drop_rate_ok,
        "runbook_drop_rate_threshold": float(max(runbook_drop_rate_threshold, 0.0)),
        "efficiency_shortcut_action_count": int(efficiency_shortcut_action_count),
        "efficiency_shortcut_action_threshold": int(efficiency_shortcut_action_threshold),
        "efficiency_shortcut_action_ok": efficiency_shortcut_action_ok,
        "efficiency_shortcut_overuse_window": int(overuse_window),
        "efficiency_shortcut_overuse_rate_threshold": float(rate_threshold),
        "efficiency_shortcut_overuse_rate": float(observed_overuse_rate),
        "efficiency_shortcut_overuse_count_in_window": int(observed_overuse_count),
        "efficiency_shortcut_overuse_observed_window_size": int(observed_window_size),
        "efficiency_shortcut_overuse_rate_ok": efficiency_shortcut_overuse_rate_ok,
    }


def _apply_operational_evaluation_to_output(
    output: Dict[str, Any],
    evaluation: Dict[str, Any],
    *,
    passed: bool,
    execution_log: List[Dict[str, Any]],
) -> None:
    output.update(
        {
            "passed": bool(passed),
            "error_count": int(evaluation.get("error_count", 0)),
            "readiness_score": float(evaluation.get("readiness_score", 0.0)),
            "checks": evaluation.get("checks", {}),
            "stage_b_promotion": evaluation.get("stage_b_promotion", {}),
            "stage_d_readiness": evaluation.get("stage_d_readiness", {}),
            "stage_e_readiness": evaluation.get("stage_e_readiness", {}),
            "phase5_entry_readiness": evaluation.get("phase5_entry_readiness", {}),
            "recovery_actions": evaluation.get("recovery_actions", []),
            "repair_plan": evaluation.get("repair_plan", {}),
            "error_details": evaluation.get("error_details", []),
            "error_details_summary": evaluation.get("error_details_summary", {}),
            "failure_focus": evaluation.get("failure_focus", {}),
            "iterative_repair_plan": evaluation.get("iterative_repair_plan", {}),
            "repair_retry_queue": evaluation.get("repair_retry_queue", []),
            "repair_retry_queue_count": int(evaluation.get("repair_retry_queue_count", 0) or 0),
            "repair_retry_cooldown_seconds": float(evaluation.get("repair_retry_cooldown_seconds", 0.0) or 0.0),
            "repair_retry_cooldown_blocked": evaluation.get("repair_retry_cooldown_blocked", []),
            "repair_retry_cooldown_blocked_count": int(evaluation.get("repair_retry_cooldown_blocked_count", 0) or 0),
            "repair_pending_count": int(evaluation.get("repair_pending_count", 0) or 0),
            "repair_timeout_count": int(evaluation.get("repair_timeout_count", 0) or 0),
            "execution_log": execution_log,
        }
    )


def _build_operational_repair_artifact(output: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "checks": output.get("checks", {}),
        "recovery_actions": output.get("recovery_actions", []),
        "repair_plan": output.get("repair_plan", {}),
        "error_details": output.get("error_details", []),
        "error_details_summary": output.get("error_details_summary", {}),
        "failure_focus": output.get("failure_focus", {}),
        "iterative_repair_plan": output.get("iterative_repair_plan", {}),
        "repair_retry_queue": output.get("repair_retry_queue", []),
        "repair_retry_queue_count": int(output.get("repair_retry_queue_count", 0) or 0),
        "repair_retry_cooldown_seconds": float(output.get("repair_retry_cooldown_seconds", 0.0) or 0.0),
        "repair_retry_cooldown_blocked": output.get("repair_retry_cooldown_blocked", []),
        "repair_retry_cooldown_blocked_count": int(output.get("repair_retry_cooldown_blocked_count", 0) or 0),
        "execution_log": output.get("execution_log", []),
        "repair_log_path": output.get("repair_log_path", ""),
        "operational_checklist": output.get("operational_checklist", {}),
        "runbook_actions": output.get("runbook_actions", []),
        "runbook_actions_path": output.get("runbook_actions_path", ""),
        "refresh_results": output.get("refresh_results", []),
        "failure_reason": output.get("failure_reason", ""),
        "generated_at": output.get("generated_at", time.time()),
    }


def _append_efficiency_shortcut_overuse_timeline(
    output: Dict[str, Any],
    *,
    previous_report: Optional[Dict[str, Any]] = None,
    max_entries: int = 128,
) -> None:
    checklist = (
        output.get("operational_checklist", {})
        if isinstance(output.get("operational_checklist"), dict)
        else {}
    )
    generated_at = float(output.get("generated_at", time.time()) or time.time())
    current = {
        "timestamp": generated_at,
        "overuse_active": not bool(checklist.get("efficiency_shortcut_action_ok", True)),
        "shortcut_action_count": int(checklist.get("efficiency_shortcut_action_count", 0) or 0),
        "shortcut_action_threshold": int(checklist.get("efficiency_shortcut_action_threshold", 0) or 0),
    }
    history: List[Dict[str, Any]] = []
    if isinstance(previous_report, dict):
        raw = previous_report.get("efficiency_shortcut_overuse_timeline", [])
        if isinstance(raw, list):
            history = [dict(item) for item in raw if isinstance(item, dict)]
    history.append(current)
    if max_entries > 0 and len(history) > max_entries:
        history = history[-max_entries:]
    total_overuse_events = int(
        sum(1 for item in history if isinstance(item, dict) and bool(item.get("overuse_active", False)))
    )
    output["efficiency_shortcut_overuse_timeline"] = history
    output["efficiency_shortcut_overuse_event_count"] = total_overuse_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and validate operational readiness gates for practical deployment.")
    parser.add_argument("--phase3-report-path", default=DEFAULT_PHASE3_REPORT_PATH, help="Managed path to Phase 3 report.")
    parser.add_argument("--phase4-report-path", default=DEFAULT_PHASE4_REPORT_PATH, help="Managed path to Phase 4 report.")
    parser.add_argument(
        "--phase5-entry-gate-report-path",
        default=DEFAULT_PHASE5_ENTRY_GATE_REPORT_PATH,
        help="Managed path to Phase 5 entry gate report.",
    )
    parser.add_argument(
        "--phase5-completion-gate-report-path",
        default=DEFAULT_PHASE5_COMPLETION_GATE_REPORT_PATH,
        help="Managed path to Phase 5 completion gate report.",
    )
    parser.add_argument(
        "--external-validity-report-path",
        default=DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH,
        help="Managed path to real-data external validity report.",
    )
    parser.add_argument(
        "--external-validity-ladder-report-path",
        default=DEFAULT_EXTERNAL_VALIDITY_LADDER_REPORT_PATH,
        help="Managed path to real-data external validity scale-ladder report.",
    )
    parser.add_argument(
        "--ann-efficiency-roadmap-report-path",
        default=DEFAULT_ANN_EFFICIENCY_ROADMAP_REPORT_PATH,
        help="Managed path to ANN-efficiency roadmap report. Optional; next evidence actions are merged when present.",
    )
    parser.add_argument(
        "--sara-ann-comparison-report-path",
        default=DEFAULT_SARA_ANN_COMPARISON_REPORT_PATH,
        help="Managed path to SARA-vs-ANN comparison report. Optional; comparison follow-up actions are merged when present.",
    )
    parser.add_argument(
        "--release-report-path",
        default=DEFAULT_RELEASE_SOAK_REPORT_PATH,
        help="Managed path to release soak report.",
    )
    parser.add_argument("--report-path", default=DEFAULT_OPERATIONAL_REPORT_PATH, help="Managed output path for operational report.")
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_OPERATIONAL_SUMMARY_PATH,
        help="Managed output path for operational summary.",
    )
    parser.add_argument(
        "--runbook-path",
        default=DEFAULT_OPERATIONAL_RUNBOOK_PATH,
        help="Managed output path for operational runbook Markdown.",
    )
    parser.add_argument(
        "--runbook-actions-path",
        default=DEFAULT_OPERATIONAL_RUNBOOK_ACTIONS_PATH,
        help="Managed output path for operational runbook action manifest JSON.",
    )
    parser.add_argument(
        "--runbook-max-actions",
        type=int,
        default=50,
        help="Maximum total runbook actions kept in manifest.",
    )
    parser.add_argument(
        "--runbook-max-per-source",
        type=int,
        default=0,
        help="Maximum runbook actions allowed per source (0 disables source cap).",
    )
    parser.add_argument(
        "--runbook-drop-rate-threshold",
        type=float,
        default=0.9,
        help="Checklist warning threshold for runbook action drop rate.",
    )
    parser.add_argument(
        "--efficiency-shortcut-action-threshold",
        type=int,
        default=3,
        help="Checklist warning threshold for efficiency incident shortcut action count.",
    )
    parser.add_argument(
        "--efficiency-shortcut-overuse-window",
        type=int,
        default=10,
        help="Sliding window size for efficiency shortcut overuse-rate checklist.",
    )
    parser.add_argument(
        "--efficiency-shortcut-overuse-rate-threshold",
        type=float,
        default=0.5,
        help="Maximum acceptable overuse rate in the sliding window.",
    )
    parser.add_argument(
        "--v1-actions-path",
        default=DEFAULT_V1_RELEASE_ACTIONS_PATH,
        help="Managed input path for v1 release gate action manifest JSON.",
    )
    parser.add_argument(
        "--v1-actions-max-age-seconds",
        type=float,
        default=86400.0,
        help="Maximum allowed age for v1 action entries (0 disables freshness filtering).",
    )
    parser.add_argument(
        "--refresh-artifacts",
        action="store_true",
        help="Run phase3/phase4/soak/release-gate commands before validation.",
    )
    parser.add_argument(
        "--soak-profile",
        default="release",
        choices=("quick", "release", "extended"),
        help="Profile used when --refresh-artifacts is enabled.",
    )
    parser.add_argument(
        "--include-accuracy",
        action="store_true",
        help="Pass --include-accuracy to release_soak when --refresh-artifacts is enabled.",
    )
    parser.add_argument(
        "--phase3-regression-tolerance",
        type=float,
        default=0.025,
        help="Regression tolerance passed to phase3_accuracy_suite during --refresh-artifacts.",
    )
    parser.add_argument(
        "--strict-production",
        action="store_true",
        help="Require extended-profile-equivalent soak thresholds for production promotion.",
    )
    parser.add_argument(
        "--repair-log-path",
        default=DEFAULT_OPERATIONAL_REPAIR_LOG_PATH,
        help="Managed path to operational repair execution log JSON/JSONL.",
    )
    parser.add_argument(
        "--repair-plan-path",
        default=DEFAULT_OPERATIONAL_REPAIR_PLAN_PATH,
        help="Managed output path for operational repair planning artifact JSON.",
    )
    parser.add_argument(
        "--research-journal-path",
        default=DEFAULT_OPERATIONAL_RESEARCH_JOURNAL_PATH,
        help="Managed input path for research journal JSONL summary.",
    )
    parser.add_argument(
        "--record-repair-command",
        default="",
        help="Append a repair execution entry command and exit.",
    )
    parser.add_argument(
        "--record-repair-status",
        default="pending",
        help="Status for --record-repair-command (pending/success/failed/skipped/timeout/error).",
    )
    parser.add_argument(
        "--record-repair-checks",
        default="",
        help="Comma-separated checks covered by --record-repair-command.",
    )
    parser.add_argument(
        "--record-repair-source",
        default="manual",
        help="Source label for --record-repair-command.",
    )
    parser.add_argument(
        "--tool-verification-trace-path",
        default=DEFAULT_TOOL_VERIFICATION_TRACE_PATH,
        help="Managed output path for tool verification trace artifacts.",
    )
    parser.add_argument(
        "--record-tool-verification-command",
        default="",
        help="Append a tool verification trace and matching repair-log entry, then exit.",
    )
    parser.add_argument(
        "--record-tool-verification-status",
        default="success",
        help="Status for --record-tool-verification-command.",
    )
    parser.add_argument(
        "--record-tool-verification-checks",
        default="",
        help="Comma-separated checks covered by the tool verification trace.",
    )
    parser.add_argument(
        "--record-tool-verification-source",
        default="tool_verification",
        help="Source label for the tool verification trace.",
    )
    parser.add_argument(
        "--record-tool-verification-summary",
        default="",
        help="Short human-readable summary stored in the tool verification trace.",
    )
    parser.add_argument(
        "--record-tool-verification-stdout",
        default="",
        help="Optional stdout excerpt stored in the managed trace artifact.",
    )
    parser.add_argument(
        "--record-tool-verification-stderr",
        default="",
        help="Optional stderr excerpt stored in the managed trace artifact.",
    )
    parser.add_argument(
        "--record-tool-verification-artifact",
        default="",
        help="Optional managed artifact path produced by the verified command.",
    )
    parser.add_argument(
        "--record-efficiency-incident-repair",
        action="store_true",
        help=(
            "Append the default 3-step efficiency-incident repair commands "
            "(energy_efficiency_benchmark -> phase3_accuracy_suite -> release_gate) and exit."
        ),
    )
    parser.add_argument(
        "--repair-complete-command",
        default="",
        help="Optional command to complete from pending repair entries.",
    )
    parser.add_argument(
        "--repair-complete-status",
        choices=["success", "failed", "skipped"],
        default="success",
        help="Completion status for --repair-complete-command.",
    )
    parser.add_argument(
        "--repair-complete-checks",
        default="",
        help="Comma-separated covered checks for --repair-complete-command.",
    )
    parser.add_argument(
        "--repair-complete-source",
        default="manual_cli_completion",
        help="Source label for --repair-complete-command.",
    )
    parser.add_argument(
        "--record-roadmap-patch-review",
        choices=["approved", "rejected"],
        default="",
        help="Record the human review decision for the current roadmap_patch_suggestion and exit.",
    )
    parser.add_argument(
        "--roadmap-patch-review-reason",
        default="",
        help="Short reason saved with --record-roadmap-patch-review.",
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=2,
        help="Maximum failed/timeout/error attempts before a command is excluded from retry queue.",
    )
    parser.add_argument(
        "--retry-cooldown-seconds",
        type=float,
        default=0.0,
        help="Cooldown window before failed/timeout/error commands re-enter retry queue (0 disables).",
    )
    parser.add_argument(
        "--auto-dispatch-retry",
        type=int,
        default=0,
        help="Automatically dispatch this many retry-queue commands to pending repair entries.",
    )
    parser.add_argument(
        "--auto-dispatch-min-priority",
        choices=["low", "medium", "high"],
        default="low",
        help="Minimum retry priority tier eligible for auto-dispatch.",
    )
    parser.add_argument(
        "--auto-dispatch-diversify-checks",
        action="store_true",
        help="Greedy-select retry commands to diversify covered checks within dispatch budget.",
    )
    parser.add_argument(
        "--auto-dispatch-max-per-check",
        type=int,
        default=0,
        help="Maximum auto-dispatch entries allowed per covered check (0 disables).",
    )
    parser.add_argument(
        "--append-iterative-next-actions",
        action="store_true",
        help="Append iterative next-step commands as pending entries to the repair log.",
    )
    parser.add_argument(
        "--append-runbook-actions",
        action="store_true",
        help="Append runbook action manifest commands as pending entries to the repair log.",
    )
    parser.add_argument(
        "--append-runbook-actions-max",
        type=int,
        default=0,
        help="Maximum runbook action entries to append (0 means no cap).",
    )
    parser.add_argument(
        "--append-runbook-actions-min-priority",
        choices=["low", "medium", "high"],
        default="low",
        help="Minimum runbook action priority tier to append to pending entries.",
    )
    parser.add_argument(
        "--pending-ttl-seconds",
        type=float,
        default=0.0,
        help="Automatically expire pending repair entries older than this TTL (0 disables).",
    )
    args = parser.parse_args()
    v1_actions: List[Dict[str, Any]] = []
    v1_actions_snapshot: Dict[str, Any] = {
        "path": os.path.abspath(args.v1_actions_path),
        "configured_max_age_seconds": float(max(args.v1_actions_max_age_seconds, 0.0)),
        "loaded_count": 0,
        "accepted_count": 0,
        "rejected_stale_count": 0,
        "rejected_missing_timestamp_count": 0,
        "age_filter_active": bool(args.v1_actions_max_age_seconds > 0),
        "load_error": "",
    }
    try:
        if os.path.exists(args.v1_actions_path):
            v1_actions, v1_actions_snapshot = _load_recent_v1_actions(
                args.v1_actions_path,
                max_age_seconds=float(args.v1_actions_max_age_seconds),
            )
    except (OSError, json.JSONDecodeError, ValueError):
        v1_actions = []
        v1_actions_snapshot["load_error"] = "failed_to_load_v1_actions"

    if str(args.record_repair_command).strip():
        repair_execution_log = load_operational_repair_execution_log(args.repair_log_path)
        updated = append_operational_repair_execution_entry(
            repair_execution_log,
            command=str(args.record_repair_command),
            status=str(args.record_repair_status),
            covered_checks=_parse_repair_checks_csv(str(args.record_repair_checks)),
            source=str(args.record_repair_source),
        )
        if not updated:
            print("Operational repair entry was not recorded (empty command or status).")
            return 1
        resolved_log_path = save_operational_repair_execution_log(args.repair_log_path, repair_execution_log)
        print("Operational repair entry recorded.")
        print(f"Saved repair log: {resolved_log_path}")
        return 0

    if str(args.record_tool_verification_command).strip():
        repair_execution_log = load_operational_repair_execution_log(args.repair_log_path)
        traces = load_tool_verification_traces(args.tool_verification_trace_path)
        result = append_tool_verification_trace(
            traces,
            repair_execution_log,
            command=str(args.record_tool_verification_command),
            status=str(args.record_tool_verification_status),
            covered_checks=_parse_repair_checks_csv(str(args.record_tool_verification_checks)),
            source=str(args.record_tool_verification_source),
            summary=str(args.record_tool_verification_summary),
            stdout_excerpt=str(args.record_tool_verification_stdout),
            stderr_excerpt=str(args.record_tool_verification_stderr),
            artifact_path=str(args.record_tool_verification_artifact),
        )
        if not bool(result.get("appended", False)):
            print("Tool verification trace was not recorded (empty command or status).")
            return 1
        resolved_trace_path = save_tool_verification_traces(args.tool_verification_trace_path, traces)
        resolved_log_path = save_operational_repair_execution_log(args.repair_log_path, repair_execution_log)
        print("Tool verification trace recorded.")
        print(f"Saved tool verification trace: {resolved_trace_path}")
        print(f"Saved repair log: {resolved_log_path}")
        return 0

    if bool(args.record_efficiency_incident_repair):
        repair_execution_log = load_operational_repair_execution_log(args.repair_log_path)
        appended_log = append_efficiency_incident_repair_shortcut(
            repair_execution_log,
            source="efficiency_incident_shortcut",
        )
        runbook_actions: List[Dict[str, Any]] = []
        try:
            if os.path.exists(args.runbook_actions_path):
                runbook_actions = _load_json_list(args.runbook_actions_path)
        except (OSError, json.JSONDecodeError, ValueError):
            runbook_actions = []
        appended_manifest = append_efficiency_incident_runbook_actions(
            runbook_actions,
            source="efficiency_incident_shortcut",
            priority="high",
        )
        if appended_log == 0 and appended_manifest == 0:
            print("Efficiency incident shortcut did not append new entries (already pending or empty).")
            return 1
        resolved_log_path = save_operational_repair_execution_log(args.repair_log_path, repair_execution_log)
        resolved_runbook_actions_path = ensure_parent_directory(args.runbook_actions_path)
        with open(resolved_runbook_actions_path, "w", encoding="utf-8") as handle:
            json.dump(runbook_actions, handle, indent=2, ensure_ascii=False)
        print("Efficiency incident repair shortcut recorded.")
        print(f"Appended repair-log commands: {appended_log}")
        print(f"Appended runbook actions: {appended_manifest}")
        print(f"Saved repair log: {resolved_log_path}")
        print(f"Saved runbook actions: {resolved_runbook_actions_path}")
        return 0

    if str(args.repair_complete_command).strip():
        repair_execution_log = load_operational_repair_execution_log(args.repair_log_path)
        completed = _finalize_pending_operational_repair_entries(
            repair_execution_log,
            command=str(args.repair_complete_command),
            status=str(args.repair_complete_status),
            covered_checks=_parse_repair_checks_csv(str(args.repair_complete_checks)),
            source=str(args.repair_complete_source),
        )
        if completed == 0:
            append_operational_repair_execution_entry(
                repair_execution_log,
                command=str(args.repair_complete_command),
                status=str(args.repair_complete_status),
                covered_checks=_parse_repair_checks_csv(str(args.repair_complete_checks)),
                source=str(args.repair_complete_source),
            )
        resolved_log_path = save_operational_repair_execution_log(args.repair_log_path, repair_execution_log)
        print("Operational repair completion recorded.")
        print(f"Saved repair log: {resolved_log_path}")
        return 0

    if str(args.record_roadmap_patch_review).strip():
        repair_execution_log = load_operational_repair_execution_log(args.repair_log_path)
        recorded = record_roadmap_patch_review_decision(
            repair_execution_log,
            decision=str(args.record_roadmap_patch_review),
            reason=str(args.roadmap_patch_review_reason),
        )
        if recorded <= 0:
            print("Roadmap patch review decision was not recorded.")
            return 1
        resolved_log_path = save_operational_repair_execution_log(args.repair_log_path, repair_execution_log)
        print("Roadmap patch review decision recorded.")
        print(f"Saved repair log: {resolved_log_path}")
        return 0

    refresh_results: List[Dict[str, Any]] = []
    if args.refresh_artifacts:
        for command in _build_refresh_commands(
            args.soak_profile,
            args.include_accuracy,
            phase3_regression_tolerance=float(max(args.phase3_regression_tolerance, 0.0)),
        ):
            result = _run_command(command)
            refresh_results.append(result)
            if not result.get("passed", False):
                break

    if refresh_results and not all(item.get("passed", False) for item in refresh_results):
        report_path = ensure_parent_directory(args.report_path)
        summary_path = ensure_parent_directory(args.summary_path)
        repair_plan_path = ensure_parent_directory(args.repair_plan_path)
        runbook_path = ensure_parent_directory(args.runbook_path)
        runbook_actions_path = ensure_parent_directory(args.runbook_actions_path)
        output = {
            "suite_name": "OperationalReadiness",
            "passed": False,
            "error_count": 1,
            "checks": {},
            "repair_plan": {},
            "iterative_repair_plan": {},
            "refresh_results": refresh_results,
            "failure_reason": "Artifact refresh command failed.",
            "v1_actions_snapshot": dict(v1_actions_snapshot),
            "runbook_max_actions": int(max(args.runbook_max_actions, 1)),
            "runbook_max_per_source": int(args.runbook_max_per_source),
            "runbook_drop_rate_threshold": float(max(args.runbook_drop_rate_threshold, 0.0)),
            "generated_at": time.time(),
        }
        runbook_actions_result = build_operational_runbook_actions(
            output,
            max_actions=int(max(args.runbook_max_actions, 1)),
            max_per_source=int(args.runbook_max_per_source),
            return_metadata=True,
            external_actions=v1_actions,
        )
        output["runbook_actions"], output["runbook_action_build_stats"] = runbook_actions_result
        output["runbook_action_build_rates"] = summarize_runbook_action_build_stats(
            output.get("runbook_action_build_stats", {})
            if isinstance(output.get("runbook_action_build_stats"), dict)
            else {}
        )
        output["runbook_action_summary"] = summarize_runbook_actions(
            output.get("runbook_actions", []) if isinstance(output.get("runbook_actions"), list) else []
        )
        output["runbook_actions_path"] = os.path.abspath(runbook_actions_path)
        output["operational_checklist"] = collect_operational_checklist_status(
            output,
            report_path=report_path,
            summary_path=summary_path,
            repair_plan_path=repair_plan_path,
            runbook_path=runbook_path,
            runbook_actions_path=runbook_actions_path,
            runbook_drop_rate_threshold=float(max(args.runbook_drop_rate_threshold, 0.0)),
            efficiency_shortcut_action_threshold=int(max(args.efficiency_shortcut_action_threshold, 0)),
            efficiency_shortcut_overuse_window=int(max(args.efficiency_shortcut_overuse_window, 1)),
            efficiency_shortcut_overuse_rate_threshold=float(max(args.efficiency_shortcut_overuse_rate_threshold, 0.0)),
        )
        _append_efficiency_shortcut_overuse_timeline(output, previous_report=None)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(format_operational_summary(output))
        with open(runbook_path, "w", encoding="utf-8") as handle:
            handle.write(build_operational_runbook(output))
        with open(runbook_actions_path, "w", encoding="utf-8") as handle:
            json.dump(output.get("runbook_actions", []), handle, indent=2, ensure_ascii=False)
        repair_artifact = _build_operational_repair_artifact(output)
        with open(repair_plan_path, "w", encoding="utf-8") as handle:
            json.dump(repair_artifact, handle, indent=2, ensure_ascii=False)
        print("Operational readiness failed during artifact refresh.")
        print(f"Saved report: {report_path}")
        print(f"Saved summary: {summary_path}")
        print(f"Saved runbook: {runbook_path}")
        print(f"Saved runbook actions: {runbook_actions_path}")
        print(f"Saved repair artifact: {repair_plan_path}")
        return 1

    try:
        phase3_report = _load_json_object(args.phase3_report_path)
        phase4_report = _load_json_object(args.phase4_report_path)
        phase5_entry_gate_report = _load_json_object(args.phase5_entry_gate_report_path)
        phase5_completion_gate_report = _load_json_object(args.phase5_completion_gate_report_path)
        external_validity_report = _load_json_object(args.external_validity_report_path)
        external_validity_ladder_report = _load_json_object(args.external_validity_ladder_report_path)
        release_report = _load_json_object(args.release_report_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Operational readiness failed: {exc}")
        return 1
    ann_efficiency_roadmap_report: Dict[str, Any] = {}
    if os.path.exists(args.ann_efficiency_roadmap_report_path):
        try:
            ann_efficiency_roadmap_report = _load_json_object(args.ann_efficiency_roadmap_report_path)
        except (OSError, json.JSONDecodeError, ValueError):
            ann_efficiency_roadmap_report = {}
    sara_ann_comparison_report: Dict[str, Any] = {}
    if os.path.exists(args.sara_ann_comparison_report_path):
        try:
            sara_ann_comparison_report = _load_json_object(args.sara_ann_comparison_report_path)
        except (OSError, json.JSONDecodeError, ValueError):
            sara_ann_comparison_report = {}
    execution_log = load_operational_repair_execution_log(args.repair_log_path)
    research_journal_entries = load_research_journal_entries(args.research_journal_path)
    research_journal_sync = attach_remeasure_results_to_research_journal_entries(
        research_journal_entries,
        execution_log,
    )
    research_journal_entries = research_journal_sync[0]
    research_journal_sync_summary = research_journal_sync[1]
    alternative_probe_sync = attach_alternative_probe_results_to_research_journal_entries(
        research_journal_entries,
        execution_log,
    )
    research_journal_entries = alternative_probe_sync[0]
    alternative_probe_sync_summary = alternative_probe_sync[1]
    planner_task_sync = attach_research_planner_task_completions_to_research_journal_entries(
        research_journal_entries,
        execution_log,
    )
    research_journal_entries = planner_task_sync[0]
    planner_task_sync_summary = planner_task_sync[1]
    evidence_collection_sync = attach_roadmap_patch_evidence_collection_completions_to_research_journal_entries(
        research_journal_entries,
        execution_log,
    )
    research_journal_entries = evidence_collection_sync[0]
    evidence_collection_sync_summary = evidence_collection_sync[1]
    stage_e_recovery_review_sync = attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_entries(
        research_journal_entries,
        execution_log,
    )
    research_journal_entries = stage_e_recovery_review_sync[0]
    stage_e_recovery_review_sync_summary = stage_e_recovery_review_sync[1]
    if (
        int(research_journal_sync_summary.get("linked_count", 0) or 0) > 0
        or int(alternative_probe_sync_summary.get("linked_count", 0) or 0) > 0
        or int(planner_task_sync_summary.get("linked_count", 0) or 0) > 0
        or int(evidence_collection_sync_summary.get("linked_count", 0) or 0) > 0
        or int(stage_e_recovery_review_sync_summary.get("linked_count", 0) or 0) > 0
    ):
        write_research_journal_entries(args.research_journal_path, research_journal_entries)
    research_journal_summary: Dict[str, Any] = summarize_research_journal_entries(
        research_journal_entries,
        now_timestamp=time.time(),
    )
    research_journal_summary["remeasure_sync"] = research_journal_sync_summary
    research_journal_summary["alternative_probe_sync"] = alternative_probe_sync_summary
    research_journal_summary["planner_task_sync"] = planner_task_sync_summary
    research_journal_summary["evidence_collection_sync"] = evidence_collection_sync_summary
    research_journal_summary["stage_e_recovery_review_sync"] = stage_e_recovery_review_sync_summary
    research_journal_summary = attach_research_planner_task_completions_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    research_journal_summary = attach_roadmap_patch_refresh_policy_followups_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    research_journal_summary = attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_summary(
        research_journal_summary,
        execution_log,
    )
    if float(args.pending_ttl_seconds) > 0:
        expire_pending_operational_repair_entries(
            execution_log,
            ttl_seconds=float(args.pending_ttl_seconds),
        )

    passed, evaluation = _evaluate_operational_readiness(
        phase3_report,
        phase4_report,
        release_report,
        phase5_entry_gate_report=phase5_entry_gate_report,
        phase5_completion_gate_report=phase5_completion_gate_report,
        external_validity_report=external_validity_report,
        external_validity_ladder_report=external_validity_ladder_report,
        execution_log=execution_log,
        strict_production=bool(args.strict_production),
        retry_max_attempts=int(args.retry_max_attempts),
        retry_cooldown_seconds=float(args.retry_cooldown_seconds),
    )
    output = {
        "suite_name": "OperationalReadiness",
        "strict_production": bool(evaluation.get("strict_production", False)),
        "repair_log_path": os.path.abspath(args.repair_log_path),
        "runbook_actions_path": os.path.abspath(args.runbook_actions_path),
        "runbook_max_actions": int(max(args.runbook_max_actions, 1)),
        "runbook_max_per_source": int(args.runbook_max_per_source),
        "runbook_drop_rate_threshold": float(max(args.runbook_drop_rate_threshold, 0.0)),
        "v1_actions_snapshot": dict(v1_actions_snapshot),
        "refresh_results": refresh_results,
        "source_reports": {
            "phase3": str(args.phase3_report_path),
            "phase4": str(args.phase4_report_path),
            "phase5_entry_gate": str(args.phase5_entry_gate_report_path),
            "phase5_completion_gate": str(args.phase5_completion_gate_report_path),
            "external_validity": str(args.external_validity_report_path),
            "external_validity_ladder": str(args.external_validity_ladder_report_path),
            "ann_efficiency_roadmap": str(args.ann_efficiency_roadmap_report_path),
            "sara_ann_comparison": str(args.sara_ann_comparison_report_path),
            "release": str(args.release_report_path),
            "research_journal": str(args.research_journal_path),
        },
        "ann_efficiency_roadmap": ann_efficiency_roadmap_report,
        "sara_ann_comparison": sara_ann_comparison_report,
        "research_journal_summary": research_journal_summary,
        "repair_auto_dispatch": {
            "requested": int(args.auto_dispatch_retry),
            "candidate_count": 0,
            "eligible_count": 0,
            "selected_count": 0,
            "selected_unique_check_count": 0,
            "min_priority_tier": str(args.auto_dispatch_min_priority).strip().lower(),
            "selection_mode": "priority_diversified" if bool(args.auto_dispatch_diversify_checks) else "priority",
            "max_per_check": int(args.auto_dispatch_max_per_check),
            "dispatched": 0,
            "dispatched_commands": [],
            "skipped_pending_commands": [],
            "skipped_limit_commands": [],
            "skipped_low_priority_commands": [],
            "skipped_low_priority_count": 0,
            "skipped_check_quota_commands": [],
            "skipped_check_quota_count": 0,
        },
        "generated_at": time.time(),
    }
    _apply_operational_evaluation_to_output(
        output,
        evaluation,
        passed=bool(passed),
        execution_log=execution_log,
    )
    if int(args.auto_dispatch_retry) > 0:
        retry_queue = (
            output.get("repair_retry_queue", [])
            if isinstance(output.get("repair_retry_queue"), list)
            else []
        )
        prioritized_queue = prioritize_operational_retry_queue(
            retry_queue,
            iterative_plan=output.get("iterative_repair_plan", {}) if isinstance(output.get("iterative_repair_plan"), dict) else {},
        )
        batch = select_operational_retry_dispatch_batch(
            prioritized_queue,
            max_dispatch=int(args.auto_dispatch_retry),
            min_priority_tier=str(args.auto_dispatch_min_priority).strip().lower(),
            diversify_checks=bool(args.auto_dispatch_diversify_checks),
            max_per_check=int(args.auto_dispatch_max_per_check),
        )
        dispatch_report = dispatch_operational_retry_queue_to_pending_with_report(
            execution_log,
            batch.get("selected", []) if isinstance(batch.get("selected"), list) else [],
            max_dispatch=int(args.auto_dispatch_retry),
        )
        output["repair_auto_dispatch"] = {
            **output["repair_auto_dispatch"],
            "candidate_count": int(len(prioritized_queue)),
            "eligible_count": int(batch.get("eligible_count", 0) or 0),
            "selected_count": int(batch.get("selected_count", 0) or 0),
            "selected_unique_check_count": int(batch.get("selected_unique_check_count", 0) or 0),
            "min_priority_tier": batch.get("min_priority_tier", "low"),
            "selection_mode": str(batch.get("selection_mode", "priority")).strip() or "priority",
            "max_per_check": int(batch.get("max_per_check", 0) or 0),
            "skipped_low_priority_commands": (
                batch.get("skipped_low_priority_commands", [])
                if isinstance(batch.get("skipped_low_priority_commands"), list)
                else []
            ),
            "skipped_low_priority_count": int(batch.get("skipped_low_priority_count", 0) or 0),
            "skipped_check_quota_commands": (
                batch.get("skipped_check_quota_commands", [])
                if isinstance(batch.get("skipped_check_quota_commands"), list)
                else []
            ),
            "skipped_check_quota_count": int(batch.get("skipped_check_quota_count", 0) or 0),
            **dispatch_report,
        }
        if int(dispatch_report.get("dispatched", 0) or 0) > 0:
            passed, evaluation = _evaluate_operational_readiness(
                phase3_report,
                phase4_report,
                release_report,
                phase5_entry_gate_report=phase5_entry_gate_report,
                phase5_completion_gate_report=phase5_completion_gate_report,
                external_validity_report=external_validity_report,
                external_validity_ladder_report=external_validity_ladder_report,
                execution_log=execution_log,
                strict_production=bool(args.strict_production),
                retry_max_attempts=int(args.retry_max_attempts),
                retry_cooldown_seconds=float(args.retry_cooldown_seconds),
            )
            _apply_operational_evaluation_to_output(
                output,
                evaluation,
                passed=bool(passed),
                execution_log=execution_log,
            )
    if bool(args.append_iterative_next_actions):
        appended = append_operational_iterative_next_actions_to_repair_log(
            execution_log,
            output.get("iterative_repair_plan", {})
            if isinstance(output.get("iterative_repair_plan"), dict)
            else {},
        )
        if appended > 0:
            passed, evaluation = _evaluate_operational_readiness(
                phase3_report,
                phase4_report,
                release_report,
                phase5_entry_gate_report=phase5_entry_gate_report,
                phase5_completion_gate_report=phase5_completion_gate_report,
                external_validity_report=external_validity_report,
                external_validity_ladder_report=external_validity_ladder_report,
                execution_log=execution_log,
                strict_production=bool(args.strict_production),
                retry_max_attempts=int(args.retry_max_attempts),
                retry_cooldown_seconds=float(args.retry_cooldown_seconds),
            )
            _apply_operational_evaluation_to_output(
                output,
                evaluation,
                passed=bool(passed),
                execution_log=execution_log,
            )
    runbook_actions_result = build_operational_runbook_actions(
        output,
        max_actions=int(max(args.runbook_max_actions, 1)),
        max_per_source=int(args.runbook_max_per_source),
        return_metadata=True,
        external_actions=v1_actions,
    )
    output["runbook_actions"], output["runbook_action_build_stats"] = runbook_actions_result
    output["research_journal_summary"] = attach_remeasure_quota_holds_to_research_journal_summary(
        output.get("research_journal_summary", {})
        if isinstance(output.get("research_journal_summary"), dict)
        else {},
        output.get("runbook_action_build_stats", {})
        if isinstance(output.get("runbook_action_build_stats"), dict)
        else {},
    )
    output["runbook_action_build_rates"] = summarize_runbook_action_build_stats(
        output.get("runbook_action_build_stats", {})
        if isinstance(output.get("runbook_action_build_stats"), dict)
        else {}
    )
    output["runbook_action_summary"] = summarize_runbook_actions(
        output.get("runbook_actions", []) if isinstance(output.get("runbook_actions"), list) else []
    )
    if bool(args.append_runbook_actions):
        appended = append_operational_runbook_actions_to_repair_log(
            execution_log,
            output.get("runbook_actions", [])
            if isinstance(output.get("runbook_actions"), list)
            else [],
            max_append=int(args.append_runbook_actions_max),
            min_priority=str(args.append_runbook_actions_min_priority).strip().lower(),
        )
        if appended > 0:
            passed, evaluation = _evaluate_operational_readiness(
                phase3_report,
                phase4_report,
                release_report,
                phase5_entry_gate_report=phase5_entry_gate_report,
                phase5_completion_gate_report=phase5_completion_gate_report,
                external_validity_report=external_validity_report,
                external_validity_ladder_report=external_validity_ladder_report,
                execution_log=execution_log,
                strict_production=bool(args.strict_production),
                retry_max_attempts=int(args.retry_max_attempts),
                retry_cooldown_seconds=float(args.retry_cooldown_seconds),
            )
            _apply_operational_evaluation_to_output(
                output,
                evaluation,
                passed=bool(passed),
                execution_log=execution_log,
            )

    report_path = ensure_parent_directory(args.report_path)
    previous_report: Optional[Dict[str, Any]] = None
    if os.path.exists(report_path):
        try:
            previous_report = _load_json_object(report_path)
        except (OSError, json.JSONDecodeError, ValueError):
            previous_report = None
    summary_path = ensure_parent_directory(args.summary_path)
    runbook_path = ensure_parent_directory(args.runbook_path)
    runbook_actions_path = ensure_parent_directory(args.runbook_actions_path)
    output["research_review"] = build_operational_research_review(
        phase3_report=phase3_report,
        release_report=release_report,
        operational_report=output,
        research_journal_summary=research_journal_summary,
    )
    runbook_actions_result = build_operational_runbook_actions(
        output,
        max_actions=int(max(args.runbook_max_actions, 1)),
        max_per_source=int(args.runbook_max_per_source),
        return_metadata=True,
        external_actions=v1_actions,
    )
    output["runbook_actions"], output["runbook_action_build_stats"] = runbook_actions_result
    output["research_journal_summary"] = attach_remeasure_quota_holds_to_research_journal_summary(
        output.get("research_journal_summary", {})
        if isinstance(output.get("research_journal_summary"), dict)
        else {},
        output.get("runbook_action_build_stats", {})
        if isinstance(output.get("runbook_action_build_stats"), dict)
        else {},
    )
    output["runbook_action_build_rates"] = summarize_runbook_action_build_stats(
        output.get("runbook_action_build_stats", {})
        if isinstance(output.get("runbook_action_build_stats"), dict)
        else {}
    )
    output["runbook_action_summary"] = summarize_runbook_actions(
        output.get("runbook_actions", []) if isinstance(output.get("runbook_actions"), list) else []
    )
    output["runbook_actions_path"] = os.path.abspath(runbook_actions_path)
    if isinstance(previous_report, dict):
        prior_timeline = previous_report.get("efficiency_shortcut_overuse_timeline", [])
        if isinstance(prior_timeline, list):
            output["efficiency_shortcut_overuse_timeline"] = [dict(item) for item in prior_timeline if isinstance(item, dict)]
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_operational_summary(output))
    repair_plan_path = ensure_parent_directory(args.repair_plan_path)
    output["operational_checklist"] = collect_operational_checklist_status(
        output,
        report_path=report_path,
        summary_path=summary_path,
        repair_plan_path=repair_plan_path,
        runbook_path=runbook_path,
        runbook_actions_path=runbook_actions_path,
        runbook_drop_rate_threshold=float(max(args.runbook_drop_rate_threshold, 0.0)),
        efficiency_shortcut_action_threshold=int(max(args.efficiency_shortcut_action_threshold, 0)),
        efficiency_shortcut_overuse_window=int(max(args.efficiency_shortcut_overuse_window, 1)),
        efficiency_shortcut_overuse_rate_threshold=float(max(args.efficiency_shortcut_overuse_rate_threshold, 0.0)),
    )
    _append_efficiency_shortcut_overuse_timeline(output, previous_report=previous_report)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_operational_summary(output))
    with open(runbook_path, "w", encoding="utf-8") as handle:
        handle.write(build_operational_runbook(output))
    with open(runbook_actions_path, "w", encoding="utf-8") as handle:
        json.dump(output.get("runbook_actions", []), handle, indent=2, ensure_ascii=False)
    repair_artifact = _build_operational_repair_artifact(output)
    with open(repair_plan_path, "w", encoding="utf-8") as handle:
        json.dump(repair_artifact, handle, indent=2, ensure_ascii=False)

    if not passed:
        print("Operational readiness failed:")
        checks = output.get("checks", {}) if isinstance(output.get("checks"), dict) else {}
        for section_name in (
            "phase3_accuracy",
            "phase3_completion",
            "phase4_completion",
            "phase5_entry_gate",
            "phase5_completion_gate",
            "external_validity",
            "release_gate",
            "production_profile",
        ):
            section = checks.get(section_name, {}) if isinstance(checks.get(section_name), dict) else {}
            errors = section.get("errors", []) if isinstance(section.get("errors"), list) else []
            for item in errors[:5]:
                print(f"- {section_name}: {str(item)}")
        failure_focus = output.get("failure_focus", {}) if isinstance(output.get("failure_focus"), dict) else {}
        print(
            "Failure focus: "
            f"primary={str(failure_focus.get('primary_category', ''))} "
            f"secondary={str(failure_focus.get('secondary_category', ''))} "
            f"confidence={float(failure_focus.get('confidence', 0.0) or 0.0):.3f}"
        )
        iterative = output.get("iterative_repair_plan", {}) if isinstance(output.get("iterative_repair_plan"), dict) else {}
        next_actions = iterative.get("next_actions", []) if isinstance(iterative.get("next_actions"), list) else []
        if next_actions:
            print("Iterative repair loop (next actions):")
            for action in next_actions[:5]:
                if not isinstance(action, dict):
                    continue
                checks_text = ", ".join(action.get("affected_checks", [])) if isinstance(action.get("affected_checks"), list) else ""
                print(
                    f"- step {int(action.get('step', 0) or 0)}: {action.get('title', '')} -> "
                    f"{action.get('command', '')} (covers={checks_text})"
                )
        retry_queue = output.get("repair_retry_queue", []) if isinstance(output.get("repair_retry_queue"), list) else []
        if retry_queue:
            print("Retry queue candidates:")
            for retry in retry_queue[:5]:
                if not isinstance(retry, dict):
                    continue
                print(
                    f"- {str(retry.get('command', ''))} "
                    f"(reason={str(retry.get('reason', ''))}, attempt={int(retry.get('next_attempt', 0) or 0)}/{int(retry.get('max_attempts', 0) or 0)})"
                )
        print(f"Saved report: {report_path}")
        print(f"Saved summary: {summary_path}")
        print(f"Saved runbook: {runbook_path}")
        print(f"Saved runbook actions: {runbook_actions_path}")
        print(f"Saved repair artifact: {repair_plan_path}")
        return 1

    print("Operational readiness evaluation completed.")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved runbook: {runbook_path}")
    print(f"Saved runbook actions: {runbook_actions_path}")
    print(f"Saved repair artifact: {repair_plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
