#!/usr/bin/env python3
"""Gate SARA's roadmap toward ANN accuracy-per-energy advantage."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.report_artifacts import artifact_state, format_artifact_state_line  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_ENERGY_REPORT_PATH = workspace_path("evaluation", "energy_efficiency_benchmark.json")
DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH = workspace_path("evaluation", "real_data_external_validity.json")
DEFAULT_EXTERNAL_LADDER_REPORT_PATH = workspace_path("evaluation", "real_data_external_validity_ladder.json")
DEFAULT_ENERGY_MEASUREMENT_REPORT_PATH = workspace_path("evaluation", "energy_measurement_readiness.json")
DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH = workspace_path(
    "evaluation", "internal_maintenance_efficiency_benchmark.json"
)
DEFAULT_EVENT_MEMORY_MAINTENANCE_COUPLING_REPORT_PATH = workspace_path(
    "evaluation", "event_memory_maintenance_coupling_benchmark.json"
)
DEFAULT_OPERATIONAL_REPORT_PATH = workspace_path("release", "operational_readiness_report.json")
DEFAULT_OUTPUT_REPORT_PATH = workspace_path("evaluation", "ann_efficiency_roadmap_gate.json")
DEFAULT_OUTPUT_SUMMARY_PATH = workspace_path("evaluation", "ann_efficiency_roadmap_gate_summary.txt")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = report.get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _checks(report: Mapping[str, Any]) -> Mapping[str, Any]:
    checks = report.get("checks", {})
    return checks if isinstance(checks, Mapping) else {}


def _float(mapping: Mapping[str, Any], name: str) -> float:
    value = mapping.get(name, 0.0)
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _measurement_plan(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    plan = (report or {}).get("measurement_plan", {}) if isinstance(report or {}, Mapping) else {}
    return plan if isinstance(plan, Mapping) else {}


def _measurement_session_plan(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    plan = (report or {}).get("measurement_session_plan", {}) if isinstance(report or {}, Mapping) else {}
    return plan if isinstance(plan, Mapping) else {}


def _measurement_session_progress(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    progress = (report or {}).get("measurement_session_progress", {}) if isinstance(report or {}, Mapping) else {}
    return progress if isinstance(progress, Mapping) else {}


def _reference_readiness(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    readiness = (report or {}).get("reference_readiness", {}) if isinstance(report or {}, Mapping) else {}
    return readiness if isinstance(readiness, Mapping) else {}


def _internal_maintenance_counts(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    counts = (report or {}).get("counts", {}) if isinstance(report or {}, Mapping) else {}
    return counts if isinstance(counts, Mapping) else {}


def _internal_maintenance_normalized(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    metrics = (report or {}).get("normalized_metrics", {}) if isinstance(report or {}, Mapping) else {}
    return metrics if isinstance(metrics, Mapping) else {}


def _maintenance_alignment(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    payload = (report or {}).get("maintenance_alignment", {}) if isinstance(report or {}, Mapping) else {}
    return payload if isinstance(payload, Mapping) else {}


def _event_memory_maintenance_coupling_reference(
    report: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    payload = (
        (report or {}).get("event_memory_maintenance_coupling_reference", {})
        if isinstance(report or {}, Mapping)
        else {}
    )
    return payload if isinstance(payload, Mapping) else {}


def _phase6_internal_maintenance_actions(
    internal_maintenance_report: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    if not internal_maintenance_report:
        return [
            {
                "source": "internal_maintenance_reference",
                "category": "missing_internal_maintenance_reference",
                "priority": "medium",
                "task": "phase6_maintenance_efficiency",
                "return_phase": "phase6",
                "command": "python scripts/sara_cli.py eval-internal-maintenance-efficiency",
            }
        ]
    metrics = _metrics(internal_maintenance_report)
    actions: List[Dict[str, Any]] = []
    if _float(metrics, "maintenance_event_cost_efficiency_observed") < 1.0:
        actions.append(
            {
                "source": "internal_maintenance_reference",
                "category": "weak_internal_maintenance_efficiency",
                "priority": "medium",
                "task": "phase6_maintenance_efficiency",
                "return_phase": "phase6",
                "command": "python scripts/sara_cli.py eval-internal-maintenance-efficiency",
            }
        )
    if _float(metrics, "maintenance_self_state_continuity_observed") < 1.0:
        actions.append(
            {
                "source": "internal_maintenance_reference",
                "category": "weak_internal_self_state_continuity",
                "priority": "medium",
                "task": "phase6_maintenance_efficiency",
                "return_phase": "phase6",
                "command": "python scripts/sara_cli.py eval-persistent-self-state",
            }
        )
    return actions


def _phase6_maintenance_alignment_actions(
    energy_measurement_report: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    alignment = _maintenance_alignment(energy_measurement_report)
    if not bool(alignment.get("available", False)):
        return []
    ratio = _float(alignment, "maintenance_event_cost_per_selected_ratio")
    if ratio <= 1.5:
        return []
    priority = "high" if ratio >= 2.0 else "medium"
    severity = "critical" if ratio >= 3.0 else ("high" if ratio >= 2.0 else "moderate")
    return [
        {
            "source": "physical_internal_maintenance_alignment",
            "category": "maintenance_alignment_drift",
            "priority": priority,
            "task": "phase6_maintenance_alignment",
            "return_phase": "phase6",
            "ratio": ratio,
            "severity": severity,
            "command": "Inspect physical pair maintenance traces and rerun python scripts/sara_cli.py eval-internal-maintenance-efficiency before claiming stable self-state efficiency.",
        }
    ]


def _phase6_event_memory_maintenance_coupling_actions(
    energy_measurement_report: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    coupling = _event_memory_maintenance_coupling_reference(energy_measurement_report)
    if not bool(coupling.get("available", False)):
        return [
            {
                "source": "event_memory_maintenance_coupling_reference",
                "category": "missing_event_memory_maintenance_coupling_reference",
                "priority": "medium",
                "task": "phase6_compression_maintenance_coupling",
                "return_phase": "phase6",
                "command": "python scripts/sara_cli.py eval-event-memory-maintenance-coupling",
            }
        ]
    if (
        _float(coupling, "best_profile_compression_efficiency_per_maintenance") <= 0.0
        or _float(coupling, "best_profile_self_state_continuity") < 0.5
    ):
        return [
            {
                "source": "event_memory_maintenance_coupling_reference",
                "category": "weak_event_memory_maintenance_coupling_reference",
                "priority": "medium",
                "task": "phase6_compression_maintenance_coupling",
                "return_phase": "phase6",
                "command": "python scripts/sara_cli.py eval-event-memory-maintenance-coupling",
            }
        ]
    bundle_contribution = _float(
        coupling,
        "best_profile_multimodal_bundle_compression_contribution",
    )
    if bundle_contribution < 0.5:
        return [
            {
                "source": "event_memory_maintenance_coupling_reference",
                "category": "weak_bundle_compression_contribution",
                "priority": "medium",
                "task": "phase6_compression_maintenance_coupling",
                "return_phase": "phase7",
                "return_lane": "phase7_source_aware_bundle_fixtures",
                "bundle_contribution": bundle_contribution,
                "command": (
                    "Expand multimodal bundle-bearing Event Memory coupling fixtures and rerun "
                    "python scripts/sara_cli.py eval-event-memory-maintenance-coupling so compression "
                    "wins remain attributable to verified cross-modal binding rather than untyped collapse."
                ),
            }
        ]
    return []


def _phase8_reference_actions(external_validity_report: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    readiness = _reference_readiness(external_validity_report)
    references = readiness.get("references", []) if isinstance(readiness.get("references"), list) else []
    unresolved = [
        item
        for item in references
        if isinstance(item, Mapping) and not bool(item.get("available", False))
    ]
    reason_rank = {
        "missing_directory": 3,
        "RuntimeError": 2,
        "ImportError": 2,
        "ModuleNotFoundError": 2,
        "not_configured": 1,
        "": 0,
    }
    family_rank = {
        "ann_cross_encoder_reference": 3,
        "ann_pretrained_embedding_faiss_reference": 2,
        "ann_pretrained_embedding_reference": 1,
    }
    unresolved = sorted(
        unresolved,
        key=lambda item: (
            -reason_rank.get(str(item.get("reason", "") or ""), 0),
            -family_rank.get(str(item.get("reference_id", "") or ""), 0),
        ),
    )
    actions: List[Dict[str, Any]] = []
    for item in unresolved:
        reference_id = str(item.get("reference_id", "") or "")
        reason = str(item.get("reason", "") or "")
        label = str(item.get("label", "") or reference_id)
        configured_path = str(item.get("configured_path", "") or "")
        if reason == "missing_directory":
            command = (
                f"Provide a valid local directory for {label}"
                + (f" at {configured_path}" if configured_path else "")
                + " and rerun python scripts/sara_cli.py eval-external-validity."
            )
            priority = "high"
            category = "missing_reference_directory"
        elif reason in {"RuntimeError", "ImportError", "ModuleNotFoundError"}:
            command = (
                f"Install the optional CPU-only dependencies required by {label}"
                " and rerun python scripts/sara_cli.py eval-external-validity."
            )
            priority = "medium"
            category = "missing_reference_dependency"
        else:
            command = (
                f"Configure {label} for eval-external-validity"
                " with --pretrained-embedding-model or --cross-encoder-model."
            )
            priority = "medium"
            category = "configure_reference"
        actions.append(
            {
                "source": "external_reference_readiness",
                "category": category,
                "priority": priority,
                "task": "phase8_reference_strength",
                "return_phase": "phase8",
                "baseline_id": reference_id,
                "reason": reason,
                "command": command,
            }
        )
    return actions


def _build_next_evidence_actions(
    energy_measurement_report: Mapping[str, Any] | None,
    external_validity_report: Mapping[str, Any] | None,
    internal_maintenance_report: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    plan = _measurement_plan(energy_measurement_report)
    session_plan = _measurement_session_plan(energy_measurement_report)
    session_progress = _measurement_session_progress(energy_measurement_report)
    actions: List[Dict[str, Any]] = []
    pair_statuses = (
        session_progress.get("pair_statuses", [])
        if isinstance(session_progress.get("pair_statuses"), list)
        else []
    )
    orphan_pairs = (
        session_progress.get("orphan_pairs", [])
        if isinstance(session_progress.get("orphan_pairs"), list)
        else []
    )
    invalid_measurement_row_count = int(
        session_progress.get("invalid_measurement_row_count", 0) or 0
    )
    actionable_statuses = {"partial_pair", "invalid_pair", "missing_pair"}
    if invalid_measurement_row_count > 0:
        actions.append(
            {
                "source": "energy_measurement_session_progress",
                "category": "invalid_measurement_rows",
                "priority": "high",
                "task": "phase6_measurement_data_hygiene",
                "return_phase": "phase6",
                "invalid_measurement_row_count": invalid_measurement_row_count,
                "command": (
                    "Inspect data/raw/energy_measurements.jsonl for malformed or incomplete rows "
                    "and rerun python scripts/sara_cli.py eval-physical-energy-session-progress."
                ),
            }
        )
    if pair_statuses:
        for item in pair_statuses:
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status", "") or "")
            if status not in actionable_statuses:
                continue
            action_category = status
            action_priority = str(item.get("priority", "high") or "high")
            action_command = str(item.get("pair_command", "") or "")
            if status == "invalid_pair":
                invalid_reason_category = str(item.get("invalid_reason_category", "") or "")
                invalid_reason_priority = str(item.get("invalid_reason_priority", "") or "")
                invalid_reason_fields = (
                    [str(value).strip() for value in item.get("invalid_reason_fields", []) if str(value).strip()]
                    if isinstance(item.get("invalid_reason_fields"), list)
                    else []
                )
                if invalid_reason_category == "run_order_conflict":
                    action_category = "invalid_pair_run_order_conflict"
                    action_priority = invalid_reason_priority or "medium"
                    action_command = action_command or (
                        "Rerun the physical pair with alternating or explicitly differentiated run order, "
                        "then rerun python scripts/sara_cli.py eval-physical-energy-session-progress."
                    )
                elif invalid_reason_category in {
                    "fairness_field_mismatch",
                    "fairness_and_run_order_conflict",
                }:
                    action_category = "invalid_pair_fairness_mismatch"
                    action_priority = invalid_reason_priority or "high"
                    field_text = ", ".join(invalid_reason_fields) if invalid_reason_fields else "fairness fields"
                    action_command = (
                        f"Repair mismatched pair conditions ({field_text}) before rerunning this physical pair, "
                        "then rerun python scripts/sara_cli.py eval-physical-energy-session-progress."
                    )
            actions.append(
                {
                    "source": "energy_measurement_session_progress",
                    "category": action_category,
                    "priority": action_priority,
                    "task": str(item.get("task", "") or ""),
                    "return_phase": "phase6",
                    "pair_id": str(item.get("pair_id", "") or ""),
                    "replicate_index": int(item.get("replicate_index", 0) or 0),
                    "invalid_reason_category": str(item.get("invalid_reason_category", "") or ""),
                    "invalid_reason_fields": (
                        [str(value).strip() for value in item.get("invalid_reason_fields", []) if str(value).strip()]
                        if isinstance(item.get("invalid_reason_fields"), list)
                        else []
                    ),
                    "command": action_command,
                }
            )
    if orphan_pairs:
        for item in orphan_pairs:
            if not isinstance(item, Mapping):
                continue
            actions.append(
                {
                    "source": "energy_measurement_session_progress",
                    "category": "orphan_pair",
                    "priority": "medium",
                    "task": str(item.get("task", "") or ""),
                    "return_phase": "phase6",
                    "pair_id": str(item.get("pair_id", "") or ""),
                    "replicate_index": int(item.get("replicate_index", 0) or 0),
                    "command": (
                        "Inspect data/raw/energy_measurements.jsonl for rows that do not belong "
                        "to the active physical session batch, then rerun "
                        "python scripts/sara_cli.py eval-physical-energy-session-progress."
                    ),
                }
            )
    if actions:
        actions.extend(_phase8_reference_actions(external_validity_report))
        actions.extend(_phase6_internal_maintenance_actions(internal_maintenance_report))
        actions.extend(_phase6_maintenance_alignment_actions(energy_measurement_report))
        actions.extend(
            _phase6_event_memory_maintenance_coupling_actions(
                energy_measurement_report
            )
        )
        return actions
    planned_runs = (
        session_plan.get("planned_runs", [])
        if isinstance(session_plan.get("planned_runs"), list)
        else []
    )
    if planned_runs:
        for item in planned_runs:
            if not isinstance(item, Mapping):
                continue
            actions.append(
                {
                    "source": "energy_measurement_session_plan",
                    "category": str(item.get("category", "pending_joule_pair") or "pending_joule_pair"),
                    "priority": str(item.get("priority", "high") or "high"),
                    "task": str(item.get("task", "") or ""),
                    "return_phase": "phase6",
                    "system": str(item.get("system", "") or ""),
                    "run_id_template": str(item.get("run_id_template", "") or ""),
                    "command": str(item.get("command_template", "") or ""),
                }
            )
        actions.extend(_phase8_reference_actions(external_validity_report))
        actions.extend(_phase6_internal_maintenance_actions(internal_maintenance_report))
        actions.extend(_phase6_maintenance_alignment_actions(energy_measurement_report))
        actions.extend(
            _phase6_event_memory_maintenance_coupling_actions(
                energy_measurement_report
            )
        )
        return actions

    pending_pairs = plan.get("pending_pairs", []) if isinstance(plan.get("pending_pairs"), list) else []
    for item in pending_pairs:
        if not isinstance(item, Mapping):
            continue
        actions.append(
            {
                "source": "energy_measurement_plan",
                "category": "pending_joule_pair",
                "priority": str(item.get("priority", "high") or "high"),
                "task": str(item.get("task", "") or ""),
                "return_phase": "phase6",
                "system": str(item.get("missing_system", "") or ""),
                "command": str(item.get("command_template", "") or ""),
            }
        )
    weak_pairs = plan.get("weak_pairs", []) if isinstance(plan.get("weak_pairs"), list) else []
    for item in weak_pairs:
        if not isinstance(item, Mapping):
            continue
        actions.append(
            {
                "source": "energy_measurement_plan",
                "category": "weak_joule_pair",
                "priority": str(item.get("priority", "medium") or "medium"),
                "task": str(item.get("task", "") or ""),
                "return_phase": "phase6",
                "ratio": _float(item, "ann_to_sara_joule_efficiency_ratio"),
                "required_min": _float(item, "required_min"),
                "relative_ratio": _float(item, "relative_ratio"),
                "ratio_gap": _float(item, "ratio_gap"),
                "severity": str(item.get("severity", "") or ""),
                "command": str(item.get("next_action", "") or ""),
            }
        )
    actions.extend(_phase8_reference_actions(external_validity_report))
    actions.extend(_phase6_internal_maintenance_actions(internal_maintenance_report))
    actions.extend(_phase6_maintenance_alignment_actions(energy_measurement_report))
    actions.extend(
        _phase6_event_memory_maintenance_coupling_actions(energy_measurement_report)
    )
    return actions


def _stage(
    *,
    name: str,
    objective: str,
    checks: Mapping[str, bool],
    metrics: Mapping[str, Any],
    next_actions: Sequence[str],
) -> Dict[str, Any]:
    failed = [check_name for check_name, passed in checks.items() if not bool(passed)]
    return {
        "name": name,
        "objective": objective,
        "passed": not failed,
        "checks": {str(check_name): bool(passed) for check_name, passed in checks.items()},
        "failed_checks": failed,
        "metrics": dict(metrics),
        "next_actions": list(next_actions if failed else []),
    }


def build_ann_efficiency_roadmap_report(
    *,
    energy_report: Mapping[str, Any],
    external_validity_report: Mapping[str, Any],
    external_ladder_report: Mapping[str, Any],
    energy_measurement_report: Mapping[str, Any] | None = None,
    internal_maintenance_report: Mapping[str, Any] | None = None,
    operational_report: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    energy_metrics = _metrics(energy_report)
    external_metrics = _metrics(external_validity_report)
    external_checks = _checks(external_validity_report)
    ladder_metrics = _metrics(external_ladder_report)
    ladder_checks = _checks(external_ladder_report)
    measurement_metrics = _metrics(energy_measurement_report or {})
    measurement_checks = _checks(energy_measurement_report or {})
    measurement_plan = _measurement_plan(energy_measurement_report)
    maintenance_alignment = _maintenance_alignment(energy_measurement_report)
    event_memory_maintenance_coupling_reference = _event_memory_maintenance_coupling_reference(
        energy_measurement_report
    )
    internal_maintenance_metrics = _metrics(internal_maintenance_report or {})
    internal_maintenance_counts = _internal_maintenance_counts(internal_maintenance_report)
    internal_maintenance_normalized = _internal_maintenance_normalized(internal_maintenance_report)
    next_evidence_actions = _build_next_evidence_actions(
        energy_measurement_report,
        external_validity_report,
        internal_maintenance_report,
    )
    operational_checks = _checks(operational_report or {})
    real_joule_present = bool(
        (energy_measurement_report or {}).get("real_joule_measurements_present", False)
    )

    neuromorphic_metric_names = [
        "neuromorphic_ir_schema_integrity_observed",
        "neuromorphic_capability_manifest_integrity_observed",
        "neuromorphic_backend_profile_compatibility_observed",
        "neuromorphic_sparse_event_budget_observed",
        "neuromorphic_profile_report_integrity_observed",
        "neuromorphic_stage_e_state_trace_ir_observed",
        "neuromorphic_stage_e_routing_hint_coverage_observed",
        "neuromorphic_stage_e_online_update_policy_observed",
        "neuromorphic_stage_e_event_budget_observed",
        "neuromorphic_profile_history_regression_observed",
    ]

    stages = [
        _stage(
            name="stage_1_instrumented_sparse_proxy",
            objective="Keep CPU-first sparse-event performance-per-energy proxies measurable and above the current ANN-style dense baseline.",
            checks={
                "energy_benchmark_passed": bool(energy_report.get("passed", False)),
                "performance_energy_ratio_proxy": _float(energy_metrics, "performance_energy_ratio_proxy") >= 0.20,
                "ann_cost_advantage_proxy": _float(energy_metrics, "ann_cost_advantage_proxy") >= 8.0,
                "sparse_event_cost_score": _float(energy_metrics, "sparse_event_cost_score") >= 1.0,
                "brain_efficiency_alignment_proxy": _float(energy_metrics, "brain_efficiency_alignment_proxy") >= 0.85,
            },
            metrics={
                "performance_energy_ratio_proxy": _float(energy_metrics, "performance_energy_ratio_proxy"),
                "ann_cost_advantage_proxy": _float(energy_metrics, "ann_cost_advantage_proxy"),
                "sparse_event_cost_score": _float(energy_metrics, "sparse_event_cost_score"),
                "brain_efficiency_alignment_proxy": _float(energy_metrics, "brain_efficiency_alignment_proxy"),
            },
            next_actions=[
                "Reduce event/state units before increasing model scale.",
                "Add a negative control case where ANN dense scan should win, then optimize the sparse route only if the trace explains the gap.",
            ],
        ),
        _stage(
            name="stage_2_limited_real_data_advantage",
            objective="Show that sparse retrieval preserves real-data task quality while beating an ANN-style dense-scan cost proxy.",
            checks={
                "external_validity_passed": bool(external_validity_report.get("passed", False)),
                "real_data_qa_accuracy": _float(external_metrics, "real_data_qa_accuracy") >= 0.80,
                "summary_keyword_coverage": _float(external_metrics, "real_data_summary_keyword_coverage") >= 0.60,
                "continual_memory_hit_rate": _float(external_metrics, "continual_memory_hit_rate") >= 0.80,
                "ann_cost_advantage_proxy": _float(external_metrics, "ann_cost_advantage_proxy") >= 2.0,
                "performance_energy_ratio_proxy": _float(external_metrics, "performance_energy_ratio_proxy") >= 2.0,
                "negative_control_abstention_integrity": (
                    _float(external_metrics, "negative_control_abstention_integrity") >= 1.0
                ),
                "negative_control_cost_advantage_proxy": (
                    _float(external_metrics, "negative_control_cost_advantage_proxy") >= 2.0
                ),
                "partial_evidence_abstention_integrity": (
                    _float(external_metrics, "partial_evidence_abstention_integrity") >= 1.0
                ),
                "partial_evidence_cost_advantage_proxy": (
                    _float(external_metrics, "partial_evidence_cost_advantage_proxy") >= 2.0
                ),
                "contrastive_control_accuracy": (
                    _float(external_metrics, "contrastive_control_accuracy") >= 1.0
                ),
                "contrastive_control_cost_advantage_proxy": (
                    _float(external_metrics, "contrastive_control_cost_advantage_proxy") >= 2.0
                ),
                "dense_embedding_ann_cost_advantage_proxy": (
                    _float(external_metrics, "dense_embedding_ann_cost_advantage_proxy") >= 2.0
                ),
                "sparse_diffusion_real_data_denoise_accuracy": (
                    _float(external_metrics, "sparse_diffusion_real_data_denoise_accuracy") >= 1.0
                ),
                "sparse_diffusion_real_data_event_cost_advantage": (
                    _float(external_metrics, "sparse_diffusion_real_data_event_cost_advantage") >= 2.0
                ),
                "sparse_diffusion_real_data_partition_integrity": (
                    _float(external_metrics, "sparse_diffusion_real_data_partition_integrity") >= 1.0
                ),
                "sparse_diffusion_real_data_single_pass_integrity": (
                    _float(external_metrics, "sparse_diffusion_real_data_single_pass_integrity") >= 1.0
                ),
                "trend_no_regressions": bool(external_checks.get("trend.no_regressions", False)),
            },
            metrics={
                "real_data_qa_accuracy": _float(external_metrics, "real_data_qa_accuracy"),
                "real_data_summary_keyword_coverage": _float(external_metrics, "real_data_summary_keyword_coverage"),
                "continual_memory_hit_rate": _float(external_metrics, "continual_memory_hit_rate"),
                "ann_cost_advantage_proxy": _float(external_metrics, "ann_cost_advantage_proxy"),
                "performance_energy_ratio_proxy": _float(external_metrics, "performance_energy_ratio_proxy"),
                "negative_control_abstention_integrity": _float(
                    external_metrics, "negative_control_abstention_integrity"
                ),
                "negative_control_cost_advantage_proxy": _float(
                    external_metrics, "negative_control_cost_advantage_proxy"
                ),
                "partial_evidence_abstention_integrity": _float(
                    external_metrics, "partial_evidence_abstention_integrity"
                ),
                "partial_evidence_cost_advantage_proxy": _float(
                    external_metrics, "partial_evidence_cost_advantage_proxy"
                ),
                "contrastive_control_accuracy": _float(
                    external_metrics, "contrastive_control_accuracy"
                ),
                "contrastive_control_cost_advantage_proxy": _float(
                    external_metrics, "contrastive_control_cost_advantage_proxy"
                ),
                "dense_embedding_ann_cost_advantage_proxy": _float(
                    external_metrics, "dense_embedding_ann_cost_advantage_proxy"
                ),
                "sparse_diffusion_real_data_denoise_accuracy": _float(
                    external_metrics, "sparse_diffusion_real_data_denoise_accuracy"
                ),
                "sparse_diffusion_real_data_event_cost_advantage": _float(
                    external_metrics, "sparse_diffusion_real_data_event_cost_advantage"
                ),
                "sparse_diffusion_real_data_partition_integrity": _float(
                    external_metrics, "sparse_diffusion_real_data_partition_integrity"
                ),
                "sparse_diffusion_real_data_single_pass_integrity": _float(
                    external_metrics, "sparse_diffusion_real_data_single_pass_integrity"
                ),
            },
            next_actions=[
                "Improve rare-token sparse routing or verified fallback until QA quality recovers without dense scanning.",
                "Record a fresh external-validity history entry after the fix.",
            ],
        ),
        _stage(
            name="stage_3_scale_ladder_advantage",
            objective="Keep the ANN-style cost advantage intact from small to large real-data profiles.",
            checks={
                "external_ladder_passed": bool(external_ladder_report.get("passed", False)),
                "all_profiles_passed": bool(ladder_checks.get("all_profiles_passed", False)),
                "large_profile_present": bool(ladder_checks.get("large_profile_present", False)),
                "scale_doc_counts_monotonic": bool(ladder_checks.get("scale_doc_counts_monotonic", False)),
                "min_ann_cost_advantage_proxy": _float(ladder_metrics, "min_ann_cost_advantage_proxy") >= 2.0,
                "min_performance_energy_ratio_proxy": _float(ladder_metrics, "min_performance_energy_ratio_proxy") >= 2.0,
                "negative_control_abstention_all_profiles": bool(
                    ladder_checks.get("negative_control_abstention_all_profiles", False)
                ),
                "negative_control_cost_advantage_all_profiles": bool(
                    ladder_checks.get("negative_control_cost_advantage_all_profiles", False)
                ),
                "partial_evidence_abstention_all_profiles": bool(
                    ladder_checks.get("partial_evidence_abstention_all_profiles", False)
                ),
                "partial_evidence_cost_advantage_all_profiles": bool(
                    ladder_checks.get("partial_evidence_cost_advantage_all_profiles", False)
                ),
                "contrastive_control_accuracy_all_profiles": bool(
                    ladder_checks.get("contrastive_control_accuracy_all_profiles", False)
                ),
                "contrastive_control_cost_advantage_all_profiles": bool(
                    ladder_checks.get("contrastive_control_cost_advantage_all_profiles", False)
                ),
                "dense_embedding_cost_advantage_all_profiles": bool(
                    ladder_checks.get("dense_embedding_cost_advantage_all_profiles", False)
                ),
                "sparse_diffusion_real_data_denoise_all_profiles": bool(
                    ladder_checks.get("sparse_diffusion_real_data_denoise_all_profiles", False)
                ),
                "sparse_diffusion_real_data_cost_advantage_all_profiles": bool(
                    ladder_checks.get("sparse_diffusion_real_data_cost_advantage_all_profiles", False)
                ),
                "sparse_diffusion_real_data_partition_all_profiles": bool(
                    ladder_checks.get("sparse_diffusion_real_data_partition_all_profiles", False)
                ),
                "sparse_diffusion_real_data_single_pass_all_profiles": bool(
                    ladder_checks.get("sparse_diffusion_real_data_single_pass_all_profiles", False)
                ),
                "no_trend_regressions_all_profiles": bool(
                    ladder_checks.get("no_trend_regressions_all_profiles", False)
                ),
            },
            metrics={
                "profile_count": _float(ladder_metrics, "profile_count"),
                "min_real_data_qa_accuracy": _float(ladder_metrics, "min_real_data_qa_accuracy"),
                "min_ann_cost_advantage_proxy": _float(ladder_metrics, "min_ann_cost_advantage_proxy"),
                "min_performance_energy_ratio_proxy": _float(ladder_metrics, "min_performance_energy_ratio_proxy"),
                "min_negative_control_abstention_integrity": _float(
                    ladder_metrics, "min_negative_control_abstention_integrity"
                ),
                "min_negative_control_cost_advantage_proxy": _float(
                    ladder_metrics, "min_negative_control_cost_advantage_proxy"
                ),
                "min_partial_evidence_abstention_integrity": _float(
                    ladder_metrics, "min_partial_evidence_abstention_integrity"
                ),
                "min_partial_evidence_cost_advantage_proxy": _float(
                    ladder_metrics, "min_partial_evidence_cost_advantage_proxy"
                ),
                "min_contrastive_control_accuracy": _float(
                    ladder_metrics, "min_contrastive_control_accuracy"
                ),
                "min_contrastive_control_cost_advantage_proxy": _float(
                    ladder_metrics, "min_contrastive_control_cost_advantage_proxy"
                ),
                "min_dense_embedding_ann_cost_advantage_proxy": _float(
                    ladder_metrics, "min_dense_embedding_ann_cost_advantage_proxy"
                ),
                "min_sparse_diffusion_real_data_denoise_accuracy": _float(
                    ladder_metrics, "min_sparse_diffusion_real_data_denoise_accuracy"
                ),
                "min_sparse_diffusion_real_data_event_cost_advantage": _float(
                    ladder_metrics, "min_sparse_diffusion_real_data_event_cost_advantage"
                ),
                "min_sparse_diffusion_real_data_partition_integrity": _float(
                    ladder_metrics, "min_sparse_diffusion_real_data_partition_integrity"
                ),
                "min_sparse_diffusion_real_data_single_pass_integrity": _float(
                    ladder_metrics, "min_sparse_diffusion_real_data_single_pass_integrity"
                ),
            },
            next_actions=[
                "Keep scale profiles separate in history and fix the lowest-ratio profile first.",
                "Promote only mechanisms that improve the minimum ladder ratio, not just the average.",
            ],
        ),
        _stage(
            name="stage_4_production_regression_guard",
            objective="Make ANN-efficiency regressions block strict production promotion.",
            checks={
                "operational_report_present": operational_report is not None,
                "operational_passed": bool((operational_report or {}).get("passed", False)),
                "strict_production": bool((operational_report or {}).get("strict_production", False)),
                "external_validity_check": bool(
                    isinstance(operational_checks.get("external_validity"), Mapping)
                    and operational_checks.get("external_validity", {}).get("passed", False)
                ),
                "external_validity_ladder_check": bool(
                    isinstance(operational_checks.get("external_validity_ladder"), Mapping)
                    and operational_checks.get("external_validity_ladder", {}).get("passed", False)
                ),
            },
            metrics={
                "readiness_score": _float(operational_report or {}, "readiness_score"),
            },
            next_actions=[
                "Run operational_readiness.py with --refresh-artifacts --soak-profile extended --strict-production.",
                "Treat external-validity and ladder failures as release blockers.",
            ],
        ),
        _stage(
            name="stage_5_neuromorphic_transfer_readiness",
            objective="Keep the sparse-event implementation portable to neuromorphic-style backends before real joule measurements.",
            checks={
                metric_name: _float(energy_metrics, metric_name) >= 1.0
                for metric_name in neuromorphic_metric_names
            },
            metrics={metric_name: _float(energy_metrics, metric_name) for metric_name in neuromorphic_metric_names},
            next_actions=[
                "Repair the neuromorphic export manifest before adding new model features.",
                "Keep low-precision state traces and online-update policy visible in benchmark history.",
            ],
        ),
        _stage(
            name="stage_6_real_joule_measurement_readiness",
            objective="Accept real joule-per-success evidence when available and keep proxy-only claims labeled until measurements exist.",
            checks={
                "measurement_report_present": energy_measurement_report is not None,
                "measurement_schema_ready": bool(measurement_checks.get("schema_ready", False)),
                "measurement_rows_valid": bool(measurement_checks.get("rows_valid", False)),
                "real_joule_claim_guard": (
                    (not real_joule_present)
                    or (
                        bool(measurement_checks.get("joule_efficiency_ratio_passed", False))
                        and bool(measurement_checks.get("paired_task_measurements_present", False))
                        and bool(measurement_checks.get("paired_task_rows_balanced", False))
                        and bool(measurement_checks.get("paired_task_efficiency_ratio_passed", False))
                    )
                ),
            },
            metrics={
                "real_joule_measurements_present": 1.0 if real_joule_present else 0.0,
                "maintenance_trace_rows_present": 1.0
                if bool(measurement_metrics.get("maintenance_trace_rows_present", False))
                else 0.0,
                "sara_joule_per_success": _float(measurement_metrics, "sara_joule_per_success"),
                "ann_joule_per_success": _float(measurement_metrics, "ann_joule_per_success"),
                "ann_to_sara_joule_efficiency_ratio": _float(
                    measurement_metrics, "ann_to_sara_joule_efficiency_ratio"
                ),
                "paired_task_count": _float(measurement_metrics, "paired_task_count"),
                "min_paired_task_ann_to_sara_ratio": _float(
                    measurement_metrics, "min_paired_task_ann_to_sara_ratio"
                ),
                "sara_maintenance_event_cost_per_success": _float(
                    measurement_metrics, "sara_maintenance_event_cost_per_success"
                ),
                "ann_maintenance_event_cost_per_success": _float(
                    measurement_metrics, "ann_maintenance_event_cost_per_success"
                ),
                "measurement_session_invalid_measurement_row_count": _float(
                    _measurement_session_progress(energy_measurement_report),
                    "invalid_measurement_row_count",
                ),
                "measurement_session_orphan_pair_count": _float(
                    _measurement_session_progress(energy_measurement_report),
                    "orphan_pair_count",
                ),
                "measurement_pending_pair_count": _float(measurement_plan, "pending_pair_count"),
                "measurement_weak_pair_count": _float(measurement_plan, "weak_pair_count"),
                "internal_maintenance_event_cost_per_selected": _float(
                    internal_maintenance_normalized, "maintenance_event_cost_per_selected"
                ),
                "internal_maintenance_selected_count": _float(
                    internal_maintenance_counts, "maintenance_selected_count"
                ),
                "internal_maintenance_refresh_count": _float(
                    internal_maintenance_counts, "maintenance_refresh_count"
                ),
                "internal_maintenance_self_state_continuity_observed": _float(
                    internal_maintenance_metrics, "maintenance_self_state_continuity_observed"
                ),
                "internal_maintenance_event_cost_efficiency_observed": _float(
                    internal_maintenance_metrics, "maintenance_event_cost_efficiency_observed"
                ),
                "physical_internal_maintenance_alignment_available": 1.0
                if bool(maintenance_alignment.get("available", False))
                else 0.0,
                "physical_internal_maintenance_event_cost_per_selected": _float(
                    maintenance_alignment, "sara_physical_maintenance_event_cost_per_selected"
                ),
                "physical_internal_maintenance_alignment_ratio": _float(
                    maintenance_alignment, "maintenance_event_cost_per_selected_ratio"
                ),
                "physical_internal_maintenance_alignment_delta": _float(
                    maintenance_alignment, "maintenance_event_cost_per_selected_delta"
                ),
                "event_memory_maintenance_coupling_available": 1.0
                if bool(event_memory_maintenance_coupling_reference.get("available", False))
                else 0.0,
                "event_memory_maintenance_best_efficiency": _float(
                    event_memory_maintenance_coupling_reference,
                    "best_profile_compression_efficiency_per_maintenance",
                ),
                "event_memory_maintenance_best_bundle_compression_contribution": _float(
                    event_memory_maintenance_coupling_reference,
                    "best_profile_multimodal_bundle_compression_contribution",
                ),
                "event_memory_maintenance_best_continuity": _float(
                    event_memory_maintenance_coupling_reference,
                    "best_profile_self_state_continuity",
                ),
            },
            next_actions=[
                "Collect paired SARA/ANN joule measurements into data/raw/energy_measurements.jsonl.",
                "Record spontaneous maintenance counts and maintenance_event_cost whenever persistent self-state or idle replay stays enabled during the run.",
                "Keep the internal maintenance reference benchmark green so maintenance-side event cost stays interpretable before physical joule capture.",
                "Keep the Event Memory compression-versus-maintenance coupling surface green so compression gains do not hide self-state carry costs.",
                "Keep public claims labeled as proxy-only until real_joule_measurements_present=true.",
            ],
        ),
    ]

    passed_stage_count = sum(1 for stage in stages if bool(stage["passed"]))
    failed_stages = [str(stage["name"]) for stage in stages if not bool(stage["passed"])]
    completion_score = passed_stage_count / max(len(stages), 1)
    return {
        "schema": "sara-ann-efficiency-roadmap-gate-v1",
        "objective": "Beat ANN-style AI on accuracy or task success per energy/event cost, starting with bounded tasks and scaling only when sparse evidence holds.",
        "passed": not failed_stages,
        "status": "ready_for_next_evidence_loop" if not failed_stages else "needs_targeted_repair",
        "completion_score": completion_score,
        "stage_count": len(stages),
        "passed_stage_count": passed_stage_count,
        "failed_stages": failed_stages,
        "next_evidence_action_count": len(next_evidence_actions),
        "next_evidence_actions": next_evidence_actions,
        "artifact_state": {
            "energy_efficiency_benchmark": artifact_state(energy_report),
            "real_data_external_validity": artifact_state(external_validity_report),
            "real_data_external_validity_ladder": artifact_state(external_ladder_report),
            "energy_measurement_readiness": artifact_state(
                energy_measurement_report, pass_field=None
            ),
            "internal_maintenance_efficiency": artifact_state(
                internal_maintenance_report, pass_field=None
            ),
            "event_memory_maintenance_coupling": artifact_state(
                event_memory_maintenance_coupling_reference, pass_field=None
            ),
            "operational_readiness": artifact_state(operational_report),
        },
        "stages": stages,
        "roadmap": [
            "1. Preserve policy constraints: CPU-first, no backprop runtime dependency, no dense-matrix-first runtime, no GPU dependency.",
            "2. Keep sparse proxy wins measurable with energy_efficiency_benchmark before increasing model complexity.",
            "3. Prove limited real-data quality with external-validity QA, summary, and continual-memory tasks.",
            "4. Expand through the small/medium/large ladder and optimize the weakest profile by minimum ratio.",
            "5. Attach strict operational and neuromorphic-portability gates before treating the result as research-product evidence.",
            "6. Ingest paired SARA/ANN joule_per_success measurements and keep proxy-only claims labeled until real measurements pass.",
        ],
    }


def format_ann_efficiency_roadmap_summary(report: Mapping[str, Any]) -> str:
    artifact_state = report.get("artifact_state", {})
    artifact_state = artifact_state if isinstance(artifact_state, Mapping) else {}
    lines = [
        "# SARA ANN Efficiency Roadmap Gate",
        f"- passed: {bool(report.get('passed', False))}",
        f"- status: {str(report.get('status', ''))}",
        f"- completion_score: {float(report.get('completion_score', 0.0) or 0.0):.3f}",
        f"- passed_stage_count: {int(report.get('passed_stage_count', 0) or 0)}",
        f"- stage_count: {int(report.get('stage_count', 0) or 0)}",
        format_artifact_state_line(
            "- artifact_state",
            [
                ("proxy", artifact_state.get("energy_efficiency_benchmark")),
                ("phase8_single", artifact_state.get("real_data_external_validity")),
                ("phase8_ladder", artifact_state.get("real_data_external_validity_ladder")),
                ("phase6", artifact_state.get("energy_measurement_readiness")),
                ("maintenance", artifact_state.get("internal_maintenance_efficiency")),
                ("coupling", artifact_state.get("event_memory_maintenance_coupling")),
                ("operational", artifact_state.get("operational_readiness")),
            ],
        ),
    ]
    failed_stages = report.get("failed_stages", [])
    lines.append("- failed_stages: " + (", ".join(str(item) for item in failed_stages) if failed_stages else "none"))
    stages = report.get("stages", [])
    if isinstance(stages, Sequence):
        lines.append("Stages:")
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            lines.append(f"- {stage.get('name', '')}: {'PASS' if bool(stage.get('passed', False)) else 'FAIL'}")
            metrics = stage.get("metrics", {})
            if isinstance(metrics, Mapping):
                compact_metrics = ", ".join(
                    f"{name}={float(value):.3f}"
                    for name, value in sorted(metrics.items())
                    if isinstance(value, (int, float))
                )
                if compact_metrics:
                    lines.append(f"  - metrics: {compact_metrics}")
            next_actions = stage.get("next_actions", [])
            if isinstance(next_actions, Sequence) and next_actions:
                lines.append(f"  - next: {next_actions[0]}")
    evidence_actions = report.get("next_evidence_actions", [])
    lines.append(f"Next Evidence Actions: {int(report.get('next_evidence_action_count', 0) or 0)}")
    if isinstance(evidence_actions, Sequence):
        for action in evidence_actions[:8]:
            if not isinstance(action, Mapping):
                continue
            command = str(action.get("command", "") or "")
            detail = command if command else str(action.get("task", "") or "")
            lines.append(
                "- "
                f"{action.get('category', '')}: "
                f"priority={action.get('priority', '')}, "
                f"task={action.get('task', '')}, "
                f"return_phase={action.get('return_phase', '')}, "
                f"{detail}"
            )
    return "\n".join(lines) + "\n"


def _refresh_artifacts() -> None:
    commands = [
        [sys.executable, "scripts/eval/energy_efficiency_benchmark.py"],
        [sys.executable, "scripts/eval/real_data_external_validity.py"],
        [sys.executable, "scripts/eval/real_data_external_validity_ladder.py"],
        [sys.executable, "scripts/eval/energy_measurement_readiness.py"],
        [sys.executable, "scripts/eval/internal_maintenance_efficiency_benchmark.py"],
        [sys.executable, "scripts/eval/event_memory_maintenance_coupling_benchmark.py"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Refresh command failed: {' '.join(command)}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the ANN-efficiency research roadmap.")
    parser.add_argument("--energy-report-path", default=DEFAULT_ENERGY_REPORT_PATH)
    parser.add_argument("--external-validity-report-path", default=DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH)
    parser.add_argument("--external-ladder-report-path", default=DEFAULT_EXTERNAL_LADDER_REPORT_PATH)
    parser.add_argument("--energy-measurement-report-path", default=DEFAULT_ENERGY_MEASUREMENT_REPORT_PATH)
    parser.add_argument("--internal-maintenance-report-path", default=DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH)
    parser.add_argument(
        "--event-memory-maintenance-coupling-report-path",
        default=DEFAULT_EVENT_MEMORY_MAINTENANCE_COUPLING_REPORT_PATH,
    )
    parser.add_argument("--operational-report-path", default=DEFAULT_OPERATIONAL_REPORT_PATH)
    parser.add_argument("--output-report-path", default=DEFAULT_OUTPUT_REPORT_PATH)
    parser.add_argument("--output-summary-path", default=DEFAULT_OUTPUT_SUMMARY_PATH)
    parser.add_argument("--refresh-artifacts", action="store_true")
    parser.add_argument("--allow-missing-operational", action="store_true")
    args = parser.parse_args(argv)

    if args.refresh_artifacts:
        _refresh_artifacts()

    required_paths = [
        args.energy_report_path,
        args.external_validity_report_path,
        args.external_ladder_report_path,
        args.energy_measurement_report_path,
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        print("ANN efficiency roadmap gate failed: missing artifacts")
        for path in missing_paths:
            print(f"- {path}")
        return 1

    operational_report = None
    if os.path.exists(args.operational_report_path):
        operational_report = _load_json(args.operational_report_path)
    elif not args.allow_missing_operational:
        print(f"ANN efficiency roadmap gate failed: missing operational report: {args.operational_report_path}")
        return 1

    report = build_ann_efficiency_roadmap_report(
        energy_report=_load_json(args.energy_report_path),
        external_validity_report=_load_json(args.external_validity_report_path),
        external_ladder_report=_load_json(args.external_ladder_report_path),
        energy_measurement_report=(
            lambda payload: (
                dict(payload, event_memory_maintenance_coupling_reference=(
                    _load_json(args.event_memory_maintenance_coupling_report_path)
                    if os.path.exists(args.event_memory_maintenance_coupling_report_path)
                    else {}
                ))
            )
        )(_load_json(args.energy_measurement_report_path)),
        internal_maintenance_report=(
            _load_json(args.internal_maintenance_report_path)
            if os.path.exists(args.internal_maintenance_report_path)
            else None
        ),
        operational_report=operational_report,
    )
    output_report_path = ensure_parent_directory(args.output_report_path)
    output_summary_path = ensure_parent_directory(args.output_summary_path)
    with open(output_report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with open(output_summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_ann_efficiency_roadmap_summary(report))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
