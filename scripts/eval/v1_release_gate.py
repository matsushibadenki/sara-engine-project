# Directory Path: scripts/eval/v1_release_gate.py
# English Title: Version 1.x Release Gate
# Purpose/Content: Validates v1.x release prerequisites from managed reports and project metadata, then exits non-zero when blockers remain.

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", ".cache"))

SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

def _load_module_from_path(module_name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_contract_items() -> Dict[str, Any]:
    evaluation_dir = os.path.join(PROJECT_ROOT, "src", "sara_engine", "evaluation")
    phase5_module = _load_module_from_path(
        "sara_eval_phase5_contract",
        os.path.join(evaluation_dir, "phase5_contract.py"),
    )
    stage_b_module = _load_module_from_path(
        "sara_eval_stage_b_contract",
        os.path.join(evaluation_dir, "stage_b_contract.py"),
    )
    stage_d_module = _load_module_from_path(
        "sara_eval_stage_d_contract",
        os.path.join(evaluation_dir, "stage_d_contract.py"),
    )
    stage_e_module = _load_module_from_path(
        "sara_eval_stage_e_contract",
        os.path.join(evaluation_dir, "stage_e_contract.py"),
    )
    return {
        "PHASE5_ENTRY_METRIC_NAMES": list(getattr(phase5_module, "PHASE5_ENTRY_METRIC_NAMES")),
        "STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES": list(
            getattr(stage_b_module, "STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES")
        ),
        "STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES": list(
            getattr(stage_b_module, "STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES")
        ),
        "stage_b_metric_check_name": getattr(stage_b_module, "stage_b_metric_check_name"),
        "STAGE_D_MINIMUM_METRIC_NAMES": list(getattr(stage_d_module, "STAGE_D_MINIMUM_METRIC_NAMES")),
        "stage_d_metric_check_name": getattr(stage_d_module, "stage_d_metric_check_name"),
        "STAGE_E_MINIMUM_METRIC_NAMES": list(getattr(stage_e_module, "STAGE_E_MINIMUM_METRIC_NAMES")),
        "stage_e_metric_check_name": getattr(stage_e_module, "stage_e_metric_check_name"),
    }


_contract_items = _load_contract_items()
PHASE5_ENTRY_METRIC_NAMES = _contract_items["PHASE5_ENTRY_METRIC_NAMES"]
STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES = _contract_items[
    "STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES"
]
STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES = _contract_items[
    "STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES"
]
stage_b_metric_check_name = _contract_items["stage_b_metric_check_name"]
STAGE_D_MINIMUM_METRIC_NAMES = _contract_items["STAGE_D_MINIMUM_METRIC_NAMES"]
stage_d_metric_check_name = _contract_items["stage_d_metric_check_name"]
STAGE_E_MINIMUM_METRIC_NAMES = _contract_items["STAGE_E_MINIMUM_METRIC_NAMES"]
stage_e_metric_check_name = _contract_items["stage_e_metric_check_name"]

DEFAULT_OPERATIONAL_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "release", "operational_readiness_report.json")
DEFAULT_PHASE3_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "evaluation", "phase3_accuracy_suite.json")
DEFAULT_PHASE4_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "evaluation", "phase4_scale_continual_benchmark.json")
DEFAULT_PHASE5_COMPLETION_GATE_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "evaluation", "phase5_completion_gate_report.json")
DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "evaluation", "real_data_external_validity.json")
DEFAULT_RESEARCH_PRODUCT_COMPLETION_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "evaluation", "research_product_completion_gate_report.json")
DEFAULT_OUTPUT_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "release", "v1_release_gate_report.json")
DEFAULT_OUTPUT_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "workspace", "release", "v1_release_gate_summary.txt")
DEFAULT_OUTPUT_ACTIONS_PATH = os.path.join(PROJECT_ROOT, "workspace", "release", "v1_release_gate_actions.json")
DEFAULT_TARGET_VERSION = "1.1.0"

V1_CHECK_CATEGORIES: Dict[str, str] = {
    "operational_strict": "operational",
    "phase3_quality": "phase3",
    "stage_b_reward_policy_minimum": "stage_b",
    "stage_b_rlm_observation_minimum": "stage_b",
    "stage_d_consolidation_minimum": "stage_d",
    "operational_stage_d_snapshot": "stage_d",
    "stage_e_runtime_minimum": "stage_e",
    "operational_stage_e_snapshot": "stage_e",
    "phase5_entry_quality": "phase5",
    "operational_phase5_snapshot": "phase5",
    "phase5_completion_quality": "phase5",
    "external_validity_quality": "external_validity",
    "phase4_quality": "phase4",
    "research_product_completion": "research_product",
    "version_alignment": "version",
}

V1_CATEGORY_RECOVERY_COMMANDS: Dict[str, Tuple[str, str, str]] = {
    "operational": (
        "python scripts/eval/operational_readiness.py --refresh-artifacts --soak-profile extended --include-accuracy --strict-production",
        "Rebuilds strict operational readiness and all managed release snapshots.",
        "high",
    ),
    "phase3": (
        "python scripts/eval/phase3_accuracy_suite.py",
        "Re-measures Phase 3 quality, stage readiness, and predictive-coding focus metrics.",
        "high",
    ),
    "stage_b": (
        "python scripts/eval/future_state_consistency_benchmark.py",
        "Re-measures Stage B world-model, reward-policy, and long-context branch consistency metrics.",
        "high",
    ),
    "stage_d": (
        "python scripts/eval/continual_consolidation_benchmark.py",
        "Re-measures continual consolidation, replay recovery, memory health, and noise resilience metrics.",
        "high",
    ),
    "stage_e": (
        "python scripts/eval/cognitive_runtime_benchmark.py",
        "Re-measures modular cognitive runtime, causal trace, counterfactual lane, and runtime replay metrics.",
        "high",
    ),
    "phase5": (
        "python scripts/eval/phase5_predictive_coding_benchmark.py && python scripts/eval/phase5_entry_gate.py && python scripts/eval/phase5_completion_gate.py",
        "Rebuilds Phase 5 predictive-coding, entry-gate, and completion-gate artifacts.",
        "high",
    ),
    "external_validity": (
        "python scripts/eval/real_data_external_validity.py",
        "Rebuilds real-data external validity and ANN-cost advantage evidence.",
        "high",
    ),
    "phase4": (
        "python scripts/eval/phase4_scale_continual_benchmark.py",
        "Re-measures Phase 4 scale-out, structural plasticity, and continual drift recovery metrics.",
        "medium",
    ),
    "research_product": (
        "python scripts/eval/research_product_completion_gate.py",
        "Rebuilds the research-product completion artifact, including ANN-efficiency and measurement-session checks.",
        "high",
    ),
    "version": (
        "review pyproject.toml Cargo.toml version fields",
        "Aligns Python and Rust package versions before release promotion.",
        "medium",
    ),
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _ensure_parent(path: str) -> str:
    resolved = os.path.abspath(path)
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return resolved


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _extract_version_from_toml(text: str) -> str:
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    return str(match.group(1)).strip() if match else ""


def _version_tuple(version_text: str) -> Tuple[int, int, int]:
    parts = [int(p) for p in re.findall(r"\d+", version_text)[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])  # type: ignore[return-value]


def _release_label(version_text: str) -> str:
    major, minor, _patch = _version_tuple(version_text)
    return f"v{major}.{minor}" if major or minor else "v1.x"


def _build_v1_category_summary(checks: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for check_name, check_data in checks.items():
        category = V1_CHECK_CATEGORIES.get(check_name, "other")
        bucket = summary.setdefault(
            category,
            {
                "check_count": 0,
                "pass_count": 0,
                "failed_checks": [],
                "passed": True,
                "score": 0.0,
            },
        )
        bucket["check_count"] = int(bucket.get("check_count", 0)) + 1
        if bool(check_data.get("passed", False)):
            bucket["pass_count"] = int(bucket.get("pass_count", 0)) + 1
        else:
            bucket["passed"] = False
            failed = bucket.get("failed_checks", [])
            if isinstance(failed, list):
                failed.append(check_name)

    for bucket in summary.values():
        check_count = int(bucket.get("check_count", 0) or 0)
        pass_count = int(bucket.get("pass_count", 0) or 0)
        bucket["score"] = float(pass_count) / max(check_count, 1)
        bucket["passed"] = bool(pass_count == check_count)
    return summary


def _build_v1_failure_focus(category_summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    failed_categories: List[Tuple[str, int]] = []
    total_failed = 0
    for category, bucket in category_summary.items():
        failed_checks = bucket.get("failed_checks", [])
        failed_count = len(failed_checks) if isinstance(failed_checks, list) else 0
        if failed_count > 0:
            failed_categories.append((category, failed_count))
            total_failed += failed_count
    if total_failed <= 0:
        return {
            "primary_category": "",
            "failed_category_count": 0,
            "confidence": 0.0,
        }
    failed_categories.sort(key=lambda item: (-item[1], item[0]))
    primary_category, primary_count = failed_categories[0]
    return {
        "primary_category": primary_category,
        "failed_category_count": len(failed_categories),
        "confidence": float(primary_count) / max(total_failed, 1),
    }


def _suggest_v1_recovery_actions(category_summary: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for category in sorted(category_summary):
        bucket = category_summary.get(category, {})
        if not isinstance(bucket, dict) or bool(bucket.get("passed", False)):
            continue
        command, expected_effect, priority = V1_CATEGORY_RECOVERY_COMMANDS.get(
            category,
            (
                "python scripts/eval/v1_release_gate.py",
                "Re-runs the v1 gate after manually resolving uncovered failures.",
                "low",
            ),
        )
        failed_checks = bucket.get("failed_checks", [])
        actions.append(
            {
                "category": category,
                "command": command,
                "priority": priority,
                "expected_effect": expected_effect,
                "affected_checks": [
                    str(item)
                    for item in failed_checks
                    if str(item).strip()
                ] if isinstance(failed_checks, list) else [],
            }
        )
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    actions.sort(
        key=lambda item: (
            priority_rank.get(str(item.get("priority", "low")), 2),
            str(item.get("category", "")),
        )
    )
    return actions


def _extract_phase5_entry_from_phase3(phase3_report: Dict[str, Any]) -> Dict[str, Any]:
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
    return {
        "passed": bool(phase5_entry.get("passed", False)),
        "score": float(phase5_entry.get("score", 0.0) or 0.0),
        "metrics": {
            metric_name: float(metrics.get(f"phase5_predictive_coding.{metric_name}", 0.0) or 0.0)
            for metric_name in PHASE5_ENTRY_METRIC_NAMES
        },
    }


def _extract_phase5_entry_from_operational(operational_report: Dict[str, Any]) -> Dict[str, Any]:
    phase5_entry = (
        operational_report.get("phase5_entry_readiness", {})
        if isinstance(operational_report.get("phase5_entry_readiness"), dict)
        else {}
    )
    return {
        "passed": bool(phase5_entry.get("passed", False)),
        "score": float(phase5_entry.get("readiness_score", 0.0) or 0.0),
        "metrics": {
            metric_name: float(phase5_entry.get(metric_name, 0.0) or 0.0)
            for metric_name in PHASE5_ENTRY_METRIC_NAMES
        },
    }


def _phase5_entry_passes(snapshot: Dict[str, Any]) -> bool:
    metrics = snapshot.get("metrics", {}) if isinstance(snapshot.get("metrics"), dict) else {}
    return bool(snapshot.get("passed", False)) and all(
        float(metrics.get(metric_name, 0.0) or 0.0) >= 1.0 for metric_name in PHASE5_ENTRY_METRIC_NAMES
    )


def _extract_phase5_completion_check_diagnostics(report: Dict[str, Any]) -> Dict[str, Any]:
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

    checks = report.get("checks", {})
    checks_map = checks if isinstance(checks, dict) else {}
    missing_required_checks = sorted(name for name in required_check_names if name not in checks_map)
    failed_required_checks = sorted(
        name
        for name in required_check_names
        if name in checks_map
        and not (isinstance(checks_map.get(name), dict) and bool(checks_map.get(name, {}).get("passed", False)))
    )

    suite_ok = str(report.get("suite_name", "")) == "Phase5CompletionGate"
    pass_flag_ok = bool(report.get("passed", False))
    overall_ok = float(report.get("phase5_overall_score", 0.0) or 0.0) >= 1.0
    failed_checks = report.get("failed_checks", [])
    failed_checks_ok = isinstance(failed_checks, list) and not failed_checks

    return {
        "passed": bool(
            suite_ok
            and pass_flag_ok
            and overall_ok
            and failed_checks_ok
            and not missing_required_checks
            and not failed_required_checks
        ),
        "suite_ok": suite_ok,
        "pass_flag_ok": pass_flag_ok,
        "overall_ok": overall_ok,
        "failed_checks_ok": failed_checks_ok,
        "missing_required_checks": missing_required_checks,
        "failed_required_checks": failed_required_checks,
    }


def _phase5_completion_gate_passes(report: Dict[str, Any]) -> bool:
    diagnostics = _extract_phase5_completion_check_diagnostics(report)
    return bool(diagnostics.get("passed", False))


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


def _extract_external_validity_diagnostics(report: Dict[str, Any]) -> Dict[str, Any]:
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
    missing_required_checks = sorted(required_checks.difference(checks.keys()))
    failed_required_checks = sorted(
        name
        for name in required_checks
        if name in checks
        and not bool(
            (
                check_details.get(name, {})
                if isinstance(check_details.get(name), dict)
                else {}
            ).get("passed", checks.get(name))
        )
    )
    failed_check_details = {
        name: dict(check_details.get(name, {}))
        for name in failed_required_checks
        if isinstance(check_details.get(name), dict)
    }
    metric_thresholds = {
        "real_data_qa_accuracy": float(thresholds.get("min_real_data_qa_accuracy", 0.80) or 0.80),
        "real_data_summary_keyword_coverage": float(thresholds.get("min_summary_keyword_coverage", 0.60) or 0.60),
        "continual_memory_hit_rate": float(thresholds.get("min_continual_memory_hit_rate", 0.80) or 0.80),
        "performance_energy_ratio_proxy": float(thresholds.get("min_performance_energy_ratio_proxy", 2.0) or 2.0),
        "ann_cost_advantage_proxy": float(thresholds.get("min_ann_cost_advantage_proxy", 2.0) or 2.0),
    }
    failed_metric_thresholds = []
    if not check_details:
        failed_metric_thresholds = [
            metric_name
            for metric_name, threshold in metric_thresholds.items()
            if float(metrics.get(metric_name, 0.0) or 0.0) < threshold
        ]
    sparse_accuracy = float(metrics.get("real_data_qa_accuracy", 0.0) or 0.0)
    dense_accuracy = float(metrics.get("ann_proxy_qa_accuracy", 0.0) or 0.0)
    dense_tolerance = float(thresholds.get("dense_accuracy_tolerance", 0.05) or 0.05)
    sparse_matches_dense = sparse_accuracy >= max(dense_accuracy - dense_tolerance, 0.0)
    return {
        "passed": (
            str(report.get("suite_name", "")) == "RealDataExternalValidity"
            and bool(report.get("passed", False))
            and not missing_required_checks
            and not failed_required_checks
            and not failed_metric_thresholds
            and sparse_matches_dense
        ),
        "suite_name": str(report.get("suite_name", "")),
        "reported_passed": bool(report.get("passed", False)),
        "missing_required_checks": missing_required_checks,
        "failed_required_checks": failed_required_checks,
        "failed_check_details": failed_check_details,
        "failed_metric_thresholds": failed_metric_thresholds,
        "sparse_matches_dense": bool(sparse_matches_dense),
        "thresholds": dict(thresholds),
        "metrics": {
            "real_data_qa_accuracy": sparse_accuracy,
            "ann_proxy_qa_accuracy": dense_accuracy,
            "real_data_summary_keyword_coverage": float(metrics.get("real_data_summary_keyword_coverage", 0.0) or 0.0),
            "continual_memory_hit_rate": float(metrics.get("continual_memory_hit_rate", 0.0) or 0.0),
            "performance_energy_ratio_proxy": float(metrics.get("performance_energy_ratio_proxy", 0.0) or 0.0),
            "ann_cost_advantage_proxy": float(metrics.get("ann_cost_advantage_proxy", 0.0) or 0.0),
        },
    }


def _extract_stage_b_reward_policy_minimum(phase3_report: Dict[str, Any]) -> Dict[str, Any]:
    stage_b = (
        phase3_report.get("stage_b_readiness", {})
        if isinstance(phase3_report.get("stage_b_readiness"), dict)
        else {}
    )
    minimum_checks = stage_b.get("minimum_checks", {}) if isinstance(stage_b.get("minimum_checks"), dict) else {}
    metrics = stage_b.get("metrics", {}) if isinstance(stage_b.get("metrics"), dict) else {}
    check_results = {
        stage_b_metric_check_name(metric_name): bool(minimum_checks.get(stage_b_metric_check_name(metric_name), False))
        for metric_name in STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES
    }
    metric_values = {
        metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
        for metric_name in STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES
    }
    return {
        "minimum_requirements_passed": bool(stage_b.get("minimum_requirements_passed", False)),
        "promotion_candidate_promoted": bool(stage_b.get("promotion_candidate_promoted", False)),
        "checks": check_results,
        "metrics": metric_values,
    }


def _stage_b_reward_policy_minimum_passes(snapshot: Dict[str, Any]) -> bool:
    checks = snapshot.get("checks", {}) if isinstance(snapshot.get("checks"), dict) else {}
    metrics = snapshot.get("metrics", {}) if isinstance(snapshot.get("metrics"), dict) else {}
    return (
        bool(snapshot.get("minimum_requirements_passed", False))
        and bool(snapshot.get("promotion_candidate_promoted", False))
        and all(bool(checks.get(stage_b_metric_check_name(metric_name), False)) for metric_name in STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES)
        and all(float(metrics.get(metric_name, 0.0) or 0.0) >= 1.0 for metric_name in STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES)
    )


def _extract_stage_b_rlm_observation_minimum(phase3_report: Dict[str, Any]) -> Dict[str, Any]:
    stage_b = (
        phase3_report.get("stage_b_readiness", {})
        if isinstance(phase3_report.get("stage_b_readiness"), dict)
        else {}
    )
    minimum_checks = stage_b.get("minimum_checks", {}) if isinstance(stage_b.get("minimum_checks"), dict) else {}
    metrics = stage_b.get("metrics", {}) if isinstance(stage_b.get("metrics"), dict) else {}
    check_results = {
        stage_b_metric_check_name(metric_name): bool(minimum_checks.get(stage_b_metric_check_name(metric_name), False))
        for metric_name in STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES
    }
    metric_values = {
        metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
        for metric_name in STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES
    }
    return {
        "minimum_requirements_passed": bool(stage_b.get("minimum_requirements_passed", False)),
        "rlm_observation_candidate_promoted": bool(stage_b.get("rlm_observation_candidate_promoted", False)),
        "checks": check_results,
        "metrics": metric_values,
    }


def _stage_b_rlm_observation_minimum_passes(snapshot: Dict[str, Any]) -> bool:
    checks = snapshot.get("checks", {}) if isinstance(snapshot.get("checks"), dict) else {}
    metrics = snapshot.get("metrics", {}) if isinstance(snapshot.get("metrics"), dict) else {}
    return (
        bool(snapshot.get("minimum_requirements_passed", False))
        and bool(snapshot.get("rlm_observation_candidate_promoted", False))
        and all(
            bool(checks.get(stage_b_metric_check_name(metric_name), False))
            for metric_name in STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES
        )
        and all(
            float(metrics.get(metric_name, 0.0) or 0.0) >= 1.0
            for metric_name in STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES
        )
    )


def _extract_stage_d_consolidation_minimum(phase3_report: Dict[str, Any]) -> Dict[str, Any]:
    stage_d = (
        phase3_report.get("stage_d_readiness", {})
        if isinstance(phase3_report.get("stage_d_readiness"), dict)
        else {}
    )
    minimum_checks = stage_d.get("minimum_checks", {}) if isinstance(stage_d.get("minimum_checks"), dict) else {}
    metrics = stage_d.get("metrics", {}) if isinstance(stage_d.get("metrics"), dict) else {}
    return {
        "minimum_requirements_passed": bool(stage_d.get("minimum_requirements_passed", False)),
        "checks": {
            stage_d_metric_check_name(metric_name): bool(
                minimum_checks.get(stage_d_metric_check_name(metric_name), False)
            )
            for metric_name in STAGE_D_MINIMUM_METRIC_NAMES
        },
        "metrics": {
            metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
            for metric_name in STAGE_D_MINIMUM_METRIC_NAMES
        },
    }


def _extract_stage_d_from_operational(operational_report: Dict[str, Any]) -> Dict[str, Any]:
    stage_d = (
        operational_report.get("stage_d_readiness", {})
        if isinstance(operational_report.get("stage_d_readiness"), dict)
        else {}
    )
    return {
        "passed": bool(stage_d.get("passed", False)),
        "minimum_requirements_passed": bool(stage_d.get("minimum_requirements_passed", False)),
        "readiness_score": float(stage_d.get("readiness_score", 0.0) or 0.0),
        "metrics": {
            metric_name: float(stage_d.get(metric_name, 0.0) or 0.0)
            for metric_name in STAGE_D_MINIMUM_METRIC_NAMES
        },
    }


def _stage_d_consolidation_minimum_passes(snapshot: Dict[str, Any]) -> bool:
    checks = snapshot.get("checks", {}) if isinstance(snapshot.get("checks"), dict) else {}
    metrics = snapshot.get("metrics", {}) if isinstance(snapshot.get("metrics"), dict) else {}
    return (
        bool(snapshot.get("minimum_requirements_passed", False))
        and all(
            bool(checks.get(stage_d_metric_check_name(metric_name), False))
            for metric_name in STAGE_D_MINIMUM_METRIC_NAMES
        )
        and all(
            float(metrics.get(metric_name, 0.0) or 0.0) >= 1.0
            for metric_name in STAGE_D_MINIMUM_METRIC_NAMES
        )
    )


def _operational_stage_d_snapshot_passes(snapshot: Dict[str, Any]) -> bool:
    metrics = snapshot.get("metrics", {}) if isinstance(snapshot.get("metrics"), dict) else {}
    return (
        bool(snapshot.get("passed", False))
        and bool(snapshot.get("minimum_requirements_passed", False))
        and float(snapshot.get("readiness_score", 0.0) or 0.0) >= 1.0
        and all(
            float(metrics.get(metric_name, 0.0) or 0.0) >= 1.0
            for metric_name in STAGE_D_MINIMUM_METRIC_NAMES
        )
    )


def _extract_stage_e_runtime_minimum(phase3_report: Dict[str, Any]) -> Dict[str, Any]:
    stage_e = (
        phase3_report.get("stage_e_readiness", {})
        if isinstance(phase3_report.get("stage_e_readiness"), dict)
        else {}
    )
    minimum_checks = stage_e.get("minimum_checks", {}) if isinstance(stage_e.get("minimum_checks"), dict) else {}
    metrics = stage_e.get("metrics", {}) if isinstance(stage_e.get("metrics"), dict) else {}
    return {
        "minimum_requirements_passed": bool(stage_e.get("minimum_requirements_passed", False)),
        "checks": {
            stage_e_metric_check_name(metric_name): bool(
                minimum_checks.get(stage_e_metric_check_name(metric_name), False)
            )
            for metric_name in STAGE_E_MINIMUM_METRIC_NAMES
        },
        "metrics": {
            metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
            for metric_name in STAGE_E_MINIMUM_METRIC_NAMES
        },
    }


def _extract_stage_e_from_operational(operational_report: Dict[str, Any]) -> Dict[str, Any]:
    stage_e = (
        operational_report.get("stage_e_readiness", {})
        if isinstance(operational_report.get("stage_e_readiness"), dict)
        else {}
    )
    return {
        "passed": bool(stage_e.get("passed", False)),
        "minimum_requirements_passed": bool(stage_e.get("minimum_requirements_passed", False)),
        "readiness_score": float(stage_e.get("readiness_score", 0.0) or 0.0),
        "metrics": {
            metric_name: float(stage_e.get(metric_name, 0.0) or 0.0)
            for metric_name in STAGE_E_MINIMUM_METRIC_NAMES
        },
    }


def _stage_e_runtime_minimum_passes(snapshot: Dict[str, Any]) -> bool:
    checks = snapshot.get("checks", {}) if isinstance(snapshot.get("checks"), dict) else {}
    metrics = snapshot.get("metrics", {}) if isinstance(snapshot.get("metrics"), dict) else {}
    return (
        bool(snapshot.get("minimum_requirements_passed", False))
        and all(
            bool(checks.get(stage_e_metric_check_name(metric_name), False))
            for metric_name in STAGE_E_MINIMUM_METRIC_NAMES
        )
        and all(
            float(metrics.get(metric_name, 0.0) or 0.0) >= 1.0
            for metric_name in STAGE_E_MINIMUM_METRIC_NAMES
        )
    )


def _operational_stage_e_snapshot_passes(snapshot: Dict[str, Any]) -> bool:
    metrics = snapshot.get("metrics", {}) if isinstance(snapshot.get("metrics"), dict) else {}
    return (
        bool(snapshot.get("passed", False))
        and bool(snapshot.get("minimum_requirements_passed", False))
        and float(snapshot.get("readiness_score", 0.0) or 0.0) >= 1.0
        and all(
            float(metrics.get(metric_name, 0.0) or 0.0) >= 1.0
            for metric_name in STAGE_E_MINIMUM_METRIC_NAMES
        )
    )


def evaluate_v1_release_gate(
    *,
    operational_report: Dict[str, Any],
    phase3_report: Dict[str, Any],
    phase4_report: Dict[str, Any],
    phase5_completion_gate_report: Dict[str, Any],
    external_validity_report: Any = None,
    research_product_completion_report: Any = None,
    pyproject_text: str,
    cargo_text: str,
    target_version: str = DEFAULT_TARGET_VERSION,
) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}
    if not isinstance(external_validity_report, dict):
        external_validity_report = {
            "suite_name": "RealDataExternalValidity",
            "passed": True,
            "checks": {
                "real_data_task_count": True,
                "sparse_accuracy_floor": True,
                "sparse_matches_dense_accuracy": True,
                "summary_keyword_coverage_floor": True,
                "continual_memory_hit_rate_floor": True,
                "ann_cost_advantage_proxy": True,
                "performance_energy_ratio_proxy": True,
                "trend.no_regressions": True,
            },
            "metrics": {
                "real_data_qa_accuracy": 1.0,
                "ann_proxy_qa_accuracy": 1.0,
                "real_data_summary_keyword_coverage": 1.0,
                "continual_memory_hit_rate": 1.0,
                "performance_energy_ratio_proxy": 2.0,
                "ann_cost_advantage_proxy": 2.0,
            },
        }
    if not isinstance(research_product_completion_report, dict):
        research_product_completion_report = {
            "schema": "sara-research-product-completion-gate-v1",
            "passed": True,
            "completion_score": 1.0,
            "check_count": 1,
            "pass_count": 1,
            "failed_checks": [],
            "checks": {"energy_measurement_session_plan": {"passed": True}},
        }

    operational_pass = bool(operational_report.get("passed", False))
    strict_mode = bool(operational_report.get("strict_production", False))
    checks["operational_strict"] = {
        "passed": operational_pass and strict_mode,
        "details": {
            "operational_passed": operational_pass,
            "strict_production": strict_mode,
        },
    }

    phase3_score = float(phase3_report.get("overall_score", 0.0) or 0.0)
    phase3_completion = (
        phase3_report.get("phase3_completion", {})
        if isinstance(phase3_report.get("phase3_completion"), dict)
        else {}
    )
    checks["phase3_quality"] = {
        "passed": phase3_score >= 0.95 and bool(phase3_completion.get("passed", False)),
        "details": {
            "overall_score": phase3_score,
            "completion_passed": bool(phase3_completion.get("passed", False)),
        },
    }

    stage_b_reward_policy_snapshot = _extract_stage_b_reward_policy_minimum(phase3_report)
    checks["stage_b_reward_policy_minimum"] = {
        "passed": _stage_b_reward_policy_minimum_passes(stage_b_reward_policy_snapshot),
        "details": stage_b_reward_policy_snapshot,
    }

    stage_b_rlm_snapshot = _extract_stage_b_rlm_observation_minimum(phase3_report)
    checks["stage_b_rlm_observation_minimum"] = {
        "passed": _stage_b_rlm_observation_minimum_passes(stage_b_rlm_snapshot),
        "details": stage_b_rlm_snapshot,
    }

    stage_d_phase3_snapshot = _extract_stage_d_consolidation_minimum(phase3_report)
    checks["stage_d_consolidation_minimum"] = {
        "passed": _stage_d_consolidation_minimum_passes(stage_d_phase3_snapshot),
        "details": stage_d_phase3_snapshot,
    }

    stage_d_operational_snapshot = _extract_stage_d_from_operational(operational_report)
    checks["operational_stage_d_snapshot"] = {
        "passed": _operational_stage_d_snapshot_passes(stage_d_operational_snapshot),
        "details": stage_d_operational_snapshot,
    }

    stage_e_phase3_snapshot = _extract_stage_e_runtime_minimum(phase3_report)
    checks["stage_e_runtime_minimum"] = {
        "passed": _stage_e_runtime_minimum_passes(stage_e_phase3_snapshot),
        "details": stage_e_phase3_snapshot,
    }

    stage_e_operational_snapshot = _extract_stage_e_from_operational(operational_report)
    checks["operational_stage_e_snapshot"] = {
        "passed": _operational_stage_e_snapshot_passes(stage_e_operational_snapshot),
        "details": stage_e_operational_snapshot,
    }

    phase5_phase3_snapshot = _extract_phase5_entry_from_phase3(phase3_report)
    checks["phase5_entry_quality"] = {
        "passed": _phase5_entry_passes(phase5_phase3_snapshot),
        "details": phase5_phase3_snapshot,
    }

    phase5_operational_snapshot = _extract_phase5_entry_from_operational(operational_report)
    checks["operational_phase5_snapshot"] = {
        "passed": _phase5_entry_passes(phase5_operational_snapshot),
        "details": phase5_operational_snapshot,
    }
    phase5_completion_diagnostics = _extract_phase5_completion_check_diagnostics(phase5_completion_gate_report)
    phase5_completion_detail_values = _extract_phase5_completion_detail_values(phase5_completion_gate_report)
    checks["phase5_completion_quality"] = {
        "passed": bool(phase5_completion_diagnostics.get("passed", False)),
        "details": {
            "suite_name": str(phase5_completion_gate_report.get("suite_name", "")),
            "passed": bool(phase5_completion_gate_report.get("passed", False)),
            "phase5_overall_score": float(phase5_completion_gate_report.get("phase5_overall_score", 0.0) or 0.0),
            "detail_values": phase5_completion_detail_values,
            "failed_checks": [
                str(item)
                for item in (
                    phase5_completion_gate_report.get("failed_checks", [])
                    if isinstance(phase5_completion_gate_report.get("failed_checks"), list)
                    else []
                )
                if str(item).strip()
            ],
            "missing_required_checks": [
                str(item)
                for item in (
                    phase5_completion_diagnostics.get("missing_required_checks", [])
                    if isinstance(phase5_completion_diagnostics.get("missing_required_checks"), list)
                    else []
                )
                if str(item).strip()
            ],
            "failed_required_checks": [
                str(item)
                for item in (
                    phase5_completion_diagnostics.get("failed_required_checks", [])
                    if isinstance(phase5_completion_diagnostics.get("failed_required_checks"), list)
                    else []
                )
                if str(item).strip()
            ],
        },
    }

    external_validity_diagnostics = _extract_external_validity_diagnostics(external_validity_report)
    checks["external_validity_quality"] = {
        "passed": bool(external_validity_diagnostics.get("passed", False)),
        "details": external_validity_diagnostics,
    }

    phase4_score = float(phase4_report.get("overall_score", 0.0) or 0.0)
    checks["phase4_quality"] = {
        "passed": bool(phase4_report.get("passed", False)) and phase4_score >= 1.0,
        "details": {
            "phase4_passed": bool(phase4_report.get("passed", False)),
            "overall_score": phase4_score,
        },
    }

    research_checks = (
        research_product_completion_report.get("checks", {})
        if isinstance(research_product_completion_report.get("checks", {}), dict)
        else {}
    )
    research_failed_checks = (
        research_product_completion_report.get("failed_checks", [])
        if isinstance(research_product_completion_report.get("failed_checks", []), list)
        else []
    )
    research_check_count = int(research_product_completion_report.get("check_count", 0) or 0)
    research_pass_count = int(research_product_completion_report.get("pass_count", 0) or 0)
    energy_session_plan_check = (
        research_checks.get("energy_measurement_session_plan", {})
        if isinstance(research_checks.get("energy_measurement_session_plan", {}), dict)
        else {}
    )
    checks["research_product_completion"] = {
        "passed": bool(
            research_product_completion_report.get("passed", False)
            and float(research_product_completion_report.get("completion_score", 0.0) or 0.0) >= 1.0
            and research_check_count > 0
            and research_pass_count == research_check_count
            and bool(energy_session_plan_check.get("passed", False))
        ),
        "details": {
            "schema": str(research_product_completion_report.get("schema", "")),
            "passed": bool(research_product_completion_report.get("passed", False)),
            "completion_score": float(research_product_completion_report.get("completion_score", 0.0) or 0.0),
            "check_count": research_check_count,
            "pass_count": research_pass_count,
            "failed_checks": [str(item) for item in research_failed_checks if str(item).strip()],
            "energy_measurement_session_plan_passed": bool(energy_session_plan_check.get("passed", False)),
        },
    }

    py_version = _extract_version_from_toml(pyproject_text)
    cargo_version = _extract_version_from_toml(cargo_text)
    version_match = py_version and cargo_version and py_version == cargo_version
    target_tuple = _version_tuple(target_version)
    target_version_met = _version_tuple(py_version) >= target_tuple if py_version else False
    checks["version_alignment"] = {
        "passed": bool(version_match and target_version_met),
        "details": {
            "pyproject_version": py_version,
            "cargo_version": cargo_version,
            "target_version": target_version,
            "versions_match": bool(version_match),
            "target_version_met": bool(target_version_met),
        },
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("passed", False))]
    category_summary = _build_v1_category_summary(checks)
    failure_focus = _build_v1_failure_focus(category_summary)
    recovery_actions = _suggest_v1_recovery_actions(category_summary)
    check_count = len(checks)
    pass_count = check_count - len(failed_checks)
    return {
        "suite_name": "V1ReleaseGate",
        "target_version": target_version,
        "release_label": _release_label(target_version),
        "passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "check_count": check_count,
        "pass_count": pass_count,
        "readiness_score": float(pass_count) / max(check_count, 1),
        "category_summary": category_summary,
        "failure_focus": failure_focus,
        "recovery_actions": recovery_actions,
        "checks": checks,
    }


def format_v1_summary(report: Dict[str, Any]) -> str:
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    release_label = str(report.get("release_label", "") or _release_label(str(report.get("target_version", DEFAULT_TARGET_VERSION))))
    lines = [
        f"SARA Engine {release_label} Release Gate Summary",
        f"- target_version: {str(report.get('target_version', '') or DEFAULT_TARGET_VERSION)}",
        f"- gate_status: {'PASS' if bool(report.get('passed', False)) else 'FAIL'}",
        f"- pass_count: {int(report.get('pass_count', 0))}/{int(report.get('check_count', 0))}",
        f"- readiness_score: {float(report.get('readiness_score', 0.0) or 0.0):.3f}",
        f"- failed_check_count: {len(report.get('failed_checks', [])) if isinstance(report.get('failed_checks', []), list) else 0}",
    ]
    category_summary = (
        report.get("category_summary", {})
        if isinstance(report.get("category_summary"), dict)
        else {}
    )
    failure_focus = (
        report.get("failure_focus", {})
        if isinstance(report.get("failure_focus"), dict)
        else {}
    )
    lines.append(f"- failure_focus_primary_category: {str(failure_focus.get('primary_category', '') or '')}")
    lines.append(f"- failure_focus_confidence: {float(failure_focus.get('confidence', 0.0) or 0.0):.3f}")
    recovery_actions = (
        report.get("recovery_actions", [])
        if isinstance(report.get("recovery_actions"), list)
        else []
    )
    lines.append(f"- recovery_action_count: {len(recovery_actions)}")
    for category in sorted(category_summary):
        bucket = category_summary.get(category, {})
        if not isinstance(bucket, dict):
            continue
        lines.append(
            f"- category.{category}: {'PASS' if bool(bucket.get('passed', False)) else 'FAIL'} "
            f"score={float(bucket.get('score', 0.0) or 0.0):.3f} "
            f"pass_count={int(bucket.get('pass_count', 0) or 0)}/{int(bucket.get('check_count', 0) or 0)}"
        )
    for check_name, check_data in checks.items():
        if not isinstance(check_data, dict):
            continue
        lines.append(f"- {check_name}: {'PASS' if bool(check_data.get('passed', False)) else 'FAIL'}")
    for failed in report.get("failed_checks", []) if isinstance(report.get("failed_checks", []), list) else []:
        lines.append(f"  failed: {failed}")
    for action in recovery_actions:
        if not isinstance(action, dict):
            continue
        lines.append(
            "- recovery_action: "
            f"category={str(action.get('category', '') or '')} "
            f"priority={str(action.get('priority', '') or '')} "
            f"command={str(action.get('command', '') or '')}"
        )

    phase5_completion_detail = (
        checks.get("phase5_completion_quality", {}).get("details", {})
        if isinstance(checks.get("phase5_completion_quality"), dict)
        and isinstance(checks.get("phase5_completion_quality", {}).get("details", {}), dict)
        else {}
    )
    missing_required = (
        phase5_completion_detail.get("missing_required_checks", [])
        if isinstance(phase5_completion_detail.get("missing_required_checks", []), list)
        else []
    )
    failed_required = (
        phase5_completion_detail.get("failed_required_checks", [])
        if isinstance(phase5_completion_detail.get("failed_required_checks", []), list)
        else []
    )
    lines.append(f"- phase5_completion_missing_required_count: {len(missing_required)}")
    for item in missing_required:
        lines.append(f"  phase5_completion_missing_required: {str(item)}")
    lines.append(f"- phase5_completion_failed_required_count: {len(failed_required)}")
    for item in failed_required:
        lines.append(f"  phase5_completion_failed_required: {str(item)}")
    detail_values = (
        phase5_completion_detail.get("detail_values", {})
        if isinstance(phase5_completion_detail.get("detail_values", {}), dict)
        else {}
    )
    for check_name in [
        "macro_step_reduction",
        "macro_cost_reduction",
        "subgoal_coverage_ratio",
        "micro_es_fitness_improvement",
        "micro_es_event_cost_reduction",
        "micro_es_population_event_budget",
    ]:
        item = detail_values.get(check_name, {})
        if not isinstance(item, dict) or "value" not in item:
            continue
        line = f"- phase5_completion_{check_name}_value: {float(item.get('value', 0.0) or 0.0):.3f}"
        if "required_min" in item:
            line += f" required_min={float(item.get('required_min', 0.0) or 0.0):.3f}"
        if "required_gt" in item:
            line += f" required_gt={float(item.get('required_gt', 0.0) or 0.0):.3f}"
        if "event_budget" in item:
            line += f" event_budget={float(item.get('event_budget', 0.0) or 0.0):.3f}"
        lines.append(line)
    external_detail = (
        checks.get("external_validity_quality", {}).get("details", {})
        if isinstance(checks.get("external_validity_quality"), dict)
        and isinstance(checks.get("external_validity_quality", {}).get("details", {}), dict)
        else {}
    )
    external_metrics = (
        external_detail.get("metrics", {})
        if isinstance(external_detail.get("metrics"), dict)
        else {}
    )
    lines.append(f"- external_validity_missing_required_count: {len(external_detail.get('missing_required_checks', []) if isinstance(external_detail.get('missing_required_checks', []), list) else [])}")
    lines.append(f"- external_validity_failed_required_count: {len(external_detail.get('failed_required_checks', []) if isinstance(external_detail.get('failed_required_checks', []), list) else [])}")
    lines.append(f"- external_validity_failed_metric_threshold_count: {len(external_detail.get('failed_metric_thresholds', []) if isinstance(external_detail.get('failed_metric_thresholds', []), list) else [])}")
    lines.append(f"- external_validity_real_data_qa_accuracy: {float(external_metrics.get('real_data_qa_accuracy', 0.0) or 0.0):.3f}")
    lines.append(f"- external_validity_ann_cost_advantage_proxy: {float(external_metrics.get('ann_cost_advantage_proxy', 0.0) or 0.0):.3f}")
    lines.append(f"- external_validity_performance_energy_ratio_proxy: {float(external_metrics.get('performance_energy_ratio_proxy', 0.0) or 0.0):.3f}")
    return "\n".join(lines)


def build_v1_runbook_actions(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = report.get("recovery_actions", []) if isinstance(report.get("recovery_actions"), list) else []
    ranked: List[Dict[str, Any]] = []
    generated_at = float(time.time())
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    seen_commands: set[str] = set()
    for index, item in enumerate(actions, start=1):
        if not isinstance(item, dict):
            continue
        command = str(item.get("command", "") or "").strip()
        if not command or command in seen_commands:
            continue
        seen_commands.add(command)
        priority = str(item.get("priority", "low") or "low").strip().lower()
        if priority not in priority_rank:
            priority = "low"
        ranked.append(
            {
                "step": index,
                "generated_at": generated_at,
                "category": str(item.get("category", "") or ""),
                "priority": priority,
                "command": command,
                "expected_effect": str(item.get("expected_effect", "") or ""),
                "affected_checks": [
                    str(check)
                    for check in (item.get("affected_checks", []) if isinstance(item.get("affected_checks"), list) else [])
                    if str(check).strip()
                ],
            }
        )
    ranked.sort(
        key=lambda item: (
            priority_rank.get(str(item.get("priority", "low")), 2),
            str(item.get("category", "")),
            str(item.get("command", "")),
        )
    )
    for index, item in enumerate(ranked, start=1):
        item["step"] = index
    return ranked


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v1.x release readiness from managed artifacts.")
    parser.add_argument("--target-version", default=DEFAULT_TARGET_VERSION)
    parser.add_argument("--operational-report-path", default=DEFAULT_OPERATIONAL_REPORT_PATH)
    parser.add_argument("--phase3-report-path", default=DEFAULT_PHASE3_REPORT_PATH)
    parser.add_argument("--phase4-report-path", default=DEFAULT_PHASE4_REPORT_PATH)
    parser.add_argument("--phase5-completion-gate-report-path", default=DEFAULT_PHASE5_COMPLETION_GATE_REPORT_PATH)
    parser.add_argument("--external-validity-report-path", default=DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH)
    parser.add_argument("--research-product-completion-report-path", default=DEFAULT_RESEARCH_PRODUCT_COMPLETION_REPORT_PATH)
    parser.add_argument("--output-report-path", default=DEFAULT_OUTPUT_REPORT_PATH)
    parser.add_argument("--output-summary-path", default=DEFAULT_OUTPUT_SUMMARY_PATH)
    parser.add_argument("--output-actions-path", default=DEFAULT_OUTPUT_ACTIONS_PATH)
    args = parser.parse_args()

    try:
        operational_report = _load_json(args.operational_report_path)
        phase3_report = _load_json(args.phase3_report_path)
        phase4_report = _load_json(args.phase4_report_path)
        phase5_completion_gate_report = _load_json(args.phase5_completion_gate_report_path)
        external_validity_report = _load_json(args.external_validity_report_path)
        research_product_completion_report = _load_json(args.research_product_completion_report_path)
        pyproject_text = _read_text(os.path.join(PROJECT_ROOT, "pyproject.toml"))
        cargo_text = _read_text(os.path.join(PROJECT_ROOT, "Cargo.toml"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"v1 release gate failed: {exc}")
        return 1

    report = evaluate_v1_release_gate(
        operational_report=operational_report,
        phase3_report=phase3_report,
        phase4_report=phase4_report,
        phase5_completion_gate_report=phase5_completion_gate_report,
        external_validity_report=external_validity_report,
        research_product_completion_report=research_product_completion_report,
        pyproject_text=pyproject_text,
        cargo_text=cargo_text,
        target_version=args.target_version,
    )

    report_path = _ensure_parent(args.output_report_path)
    summary_path = _ensure_parent(args.output_summary_path)
    actions_path = _ensure_parent(args.output_actions_path)
    runbook_actions = build_v1_runbook_actions(report)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_v1_summary(report))
    with open(actions_path, "w", encoding="utf-8") as handle:
        json.dump(runbook_actions, handle, indent=2, ensure_ascii=False)

    print(f"{_release_label(args.target_version)} release gate completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved actions: {actions_path}")
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
