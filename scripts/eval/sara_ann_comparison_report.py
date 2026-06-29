#!/usr/bin/env python3
"""Build a research-facing SARA versus ANN comparison surface."""

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


DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH = workspace_path("evaluation", "real_data_external_validity.json")
DEFAULT_EXTERNAL_LADDER_REPORT_PATH = workspace_path("evaluation", "real_data_external_validity_ladder.json")
DEFAULT_ENERGY_MEASUREMENT_REPORT_PATH = workspace_path("evaluation", "energy_measurement_readiness.json")
DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH = workspace_path(
    "evaluation", "internal_maintenance_efficiency_benchmark.json"
)
DEFAULT_EVENT_MEMORY_REPORT_PATH = workspace_path(
    "evaluation", "event_memory_ingest_pipeline.json"
)
DEFAULT_EVENT_MEMORY_MAINTENANCE_COUPLING_REPORT_PATH = workspace_path(
    "evaluation", "event_memory_maintenance_coupling_benchmark.json"
)
DEFAULT_REPORT_PATH = workspace_path("evaluation", "sara_ann_comparison_report.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "sara_ann_comparison_report.txt")


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


def _reference_readiness(report: Mapping[str, Any]) -> Mapping[str, Any]:
    readiness = report.get("reference_readiness", {})
    return readiness if isinstance(readiness, Mapping) else {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _card(
    *,
    baseline_id: str,
    label: str,
    family: str,
    evidence_tier: str,
    available: bool,
    source_artifact: str,
    quality_metric_name: str = "",
    quality_score: float = 0.0,
    cost_metric_name: str = "",
    cost_score: float = 0.0,
    latency_ms: float = 0.0,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "baseline_id": baseline_id,
        "label": label,
        "family": family,
        "evidence_tier": evidence_tier,
        "available": bool(available),
        "source_artifact": source_artifact,
        "quality_metric_name": quality_metric_name,
        "quality_score": float(quality_score),
        "cost_metric_name": cost_metric_name,
        "cost_score": float(cost_score),
        "latency_ms": float(latency_ms),
        "notes": list(notes or []),
    }


def _build_baseline_cards(
    *,
    external_validity_report: Mapping[str, Any],
    energy_measurement_report: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    external_metrics = _metrics(external_validity_report)
    energy_metrics = _metrics(energy_measurement_report)
    energy_checks = _checks(energy_measurement_report)
    pretrained = (
        external_validity_report.get("ann_pretrained_embedding_reference", {})
        if isinstance(external_validity_report.get("ann_pretrained_embedding_reference", {}), Mapping)
        else {}
    )
    pretrained_faiss = (
        external_validity_report.get("ann_pretrained_embedding_faiss_reference", {})
        if isinstance(external_validity_report.get("ann_pretrained_embedding_faiss_reference", {}), Mapping)
        else {}
    )
    cross_encoder = (
        external_validity_report.get("ann_cross_encoder_reference", {})
        if isinstance(external_validity_report.get("ann_cross_encoder_reference", {}), Mapping)
        else {}
    )
    bm25 = (
        external_validity_report.get("bm25_offline_proxy", {})
        if isinstance(external_validity_report.get("bm25_offline_proxy", {}), Mapping)
        else {}
    )
    cards = [
        _card(
            baseline_id="ann_dense_proxy",
            label="ANN Dense Scan Proxy",
            family="ann_proxy",
            evidence_tier="proxy",
            available=True,
            source_artifact="real_data_external_validity",
            quality_metric_name="qa_accuracy",
            quality_score=_safe_float(external_metrics.get("ann_proxy_qa_accuracy")),
            cost_metric_name="cost_advantage_proxy",
            cost_score=_safe_float(external_metrics.get("ann_cost_advantage_proxy")),
            latency_ms=_safe_float(external_metrics.get("ann_proxy_avg_latency_ms")),
            notes=["Dense-scan style proxy. Not a standalone offline reference implementation."],
        ),
        _card(
            baseline_id="ann_dense_embedding_proxy",
            label="ANN Dense Embedding Proxy",
            family="ann_proxy",
            evidence_tier="proxy",
            available=True,
            source_artifact="real_data_external_validity",
            quality_metric_name="qa_accuracy",
            quality_score=_safe_float(external_metrics.get("dense_embedding_ann_proxy_qa_accuracy")),
            cost_metric_name="cost_advantage_proxy",
            cost_score=_safe_float(external_metrics.get("dense_embedding_ann_cost_advantage_proxy")),
            latency_ms=_safe_float(external_metrics.get("dense_embedding_ann_proxy_avg_latency_ms")),
            notes=["Hashed dense-vector proxy. Kept outside the SARA runtime path."],
        ),
        _card(
            baseline_id="bm25_offline_proxy",
            label="BM25 Offline Baseline",
            family="offline_lexical_reference",
            evidence_tier="offline_reference",
            available=True,
            source_artifact="real_data_external_validity",
            quality_metric_name="qa_accuracy",
            quality_score=_safe_float(external_metrics.get("bm25_offline_proxy_qa_accuracy") or bm25.get("accuracy")),
            cost_metric_name="cost_advantage_proxy",
            cost_score=_safe_float(external_metrics.get("bm25_offline_cost_advantage_proxy")),
            latency_ms=_safe_float(bm25.get("avg_latency_ms")),
            notes=["Offline lexical reference baseline."],
        ),
        _card(
            baseline_id="ann_pretrained_embedding_reference",
            label="Local Pretrained Embedding Reference",
            family="offline_dense_reference",
            evidence_tier="offline_reference",
            available=bool(pretrained.get("available", False)),
            source_artifact="real_data_external_validity",
            quality_metric_name="qa_accuracy",
            quality_score=_safe_float(external_metrics.get("real_pretrained_embedding_reference_qa_accuracy")),
            cost_metric_name="cost_advantage_proxy",
            cost_score=_safe_float(external_metrics.get("real_pretrained_embedding_reference_cost_advantage_proxy")),
            latency_ms=_safe_float(external_metrics.get("real_pretrained_embedding_reference_avg_latency_ms")),
            notes=[str(pretrained.get("reason", "") or "")] if not bool(pretrained.get("available", False)) else [],
        ),
        _card(
            baseline_id="ann_pretrained_embedding_faiss_reference",
            label="Local Pretrained Embedding FAISS Reference",
            family="offline_dense_reference",
            evidence_tier="offline_reference",
            available=bool(pretrained_faiss.get("available", False)),
            source_artifact="real_data_external_validity",
            quality_metric_name="qa_accuracy",
            quality_score=_safe_float(external_metrics.get("real_pretrained_embedding_faiss_reference_qa_accuracy")),
            cost_metric_name="cost_advantage_proxy",
            cost_score=_safe_float(external_metrics.get("real_pretrained_embedding_faiss_reference_cost_advantage_proxy")),
            latency_ms=_safe_float(external_metrics.get("real_pretrained_embedding_faiss_reference_avg_latency_ms")),
            notes=[str(pretrained_faiss.get("reason", "") or "")] if not bool(pretrained_faiss.get("available", False)) else [],
        ),
        _card(
            baseline_id="ann_cross_encoder_reference",
            label="Local Cross-Encoder Reference",
            family="offline_dense_reference",
            evidence_tier="offline_reference",
            available=bool(cross_encoder.get("available", False)),
            source_artifact="real_data_external_validity",
            quality_metric_name="qa_accuracy",
            quality_score=_safe_float(external_metrics.get("real_cross_encoder_reference_qa_accuracy")),
            cost_metric_name="cost_advantage_proxy",
            cost_score=_safe_float(external_metrics.get("real_cross_encoder_reference_cost_advantage_proxy")),
            latency_ms=_safe_float(external_metrics.get("real_cross_encoder_reference_avg_latency_ms")),
            notes=[str(cross_encoder.get("reason", "") or "")] if not bool(cross_encoder.get("available", False)) else [],
        ),
        _card(
            baseline_id="physical_ann_measurement",
            label="Physical ANN Measurement",
            family="physical_energy_evidence",
            evidence_tier="physical",
            available=bool(energy_measurement_report.get("real_joule_measurements_present", False)),
            source_artifact="energy_measurement_readiness",
            quality_metric_name="paired_task_count",
            quality_score=_safe_float(energy_metrics.get("paired_task_count")),
            cost_metric_name="ann_to_sara_joule_efficiency_ratio",
            cost_score=_safe_float(energy_metrics.get("ann_to_sara_joule_efficiency_ratio")),
            latency_ms=0.0,
            notes=[
                "quality_parity_passed=" + str(bool(energy_checks.get("quality_parity_passed", False))),
                "paired_task_rows_balanced=" + str(bool(energy_checks.get("paired_task_rows_balanced", False))),
            ],
        ),
    ]
    return cards


def _best_available_reference(cards: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    candidates = [
        card for card in cards
        if bool(card.get("available", False))
        and str(card.get("evidence_tier", "")) == "offline_reference"
    ]
    if not candidates:
        return {
            "baseline_id": "",
            "label": "",
            "available": False,
        }
    family_rank = {
        "offline_dense_reference": 2,
        "offline_lexical_reference": 1,
    }
    best = max(
        candidates,
        key=lambda item: (
            family_rank.get(str(item.get("family", "") or ""), 0),
            _safe_float(item.get("quality_score")),
            _safe_float(item.get("cost_score")),
            -_safe_float(item.get("latency_ms")),
        ),
    )
    return {
        "baseline_id": str(best.get("baseline_id", "") or ""),
        "label": str(best.get("label", "") or ""),
        "available": True,
        "quality_score": _safe_float(best.get("quality_score")),
        "cost_score": _safe_float(best.get("cost_score")),
    }


def _build_reference_next_actions(external_validity_report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    readiness = _reference_readiness(external_validity_report)
    references = (
        readiness.get("references", [])
        if isinstance(readiness.get("references"), list)
        else []
    )
    family_rank = {
        "ann_cross_encoder_reference": 3,
        "ann_pretrained_embedding_faiss_reference": 2,
        "ann_pretrained_embedding_reference": 1,
    }
    reason_rank = {
        "missing_directory": 3,
        "RuntimeError": 2,
        "ImportError": 2,
        "ModuleNotFoundError": 2,
        "not_configured": 1,
        "": 0,
    }
    unresolved = [
        item
        for item in references
        if isinstance(item, Mapping) and not bool(item.get("available", False))
    ]
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
                "priority": priority,
                "category": category,
                "baseline_id": reference_id,
                "reason": reason,
                "command": command,
            }
        )
    return actions


def _maintenance_surface(
    *,
    internal_maintenance_report: Mapping[str, Any] | None,
    energy_measurement_report: Mapping[str, Any],
) -> Dict[str, Any]:
    internal_report = internal_maintenance_report or {}
    internal_metrics = (
        internal_report.get("metrics", {})
        if isinstance(internal_report.get("metrics", {}), Mapping)
        else {}
    )
    internal_counts = (
        internal_report.get("counts", {})
        if isinstance(internal_report.get("counts", {}), Mapping)
        else {}
    )
    internal_normalized = (
        internal_report.get("normalized_metrics", {})
        if isinstance(internal_report.get("normalized_metrics", {}), Mapping)
        else {}
    )
    energy_metrics = _metrics(energy_measurement_report)
    alignment = (
        energy_measurement_report.get("maintenance_alignment", {})
        if isinstance(energy_measurement_report.get("maintenance_alignment", {}), Mapping)
        else {}
    )
    return {
        "available": bool(internal_maintenance_report),
        "observed_only": bool(internal_report.get("observed_only", False)),
        "maintenance_selected_count": _safe_int(internal_counts.get("maintenance_selected_count")),
        "maintenance_refresh_count": _safe_int(internal_counts.get("maintenance_refresh_count")),
        "maintenance_idle_self_state_ok_count": _safe_int(
            internal_counts.get("maintenance_idle_self_state_ok_count")
        ),
        "maintenance_event_cost": _safe_float(internal_normalized.get("maintenance_event_cost")),
        "maintenance_event_cost_per_selected": _safe_float(
            internal_normalized.get("maintenance_event_cost_per_selected")
        ),
        "self_state_continuity_observed": _safe_float(
            internal_metrics.get("maintenance_self_state_continuity_observed")
        ),
        "cache_refresh_observed": _safe_float(
            internal_metrics.get("maintenance_cache_refresh_observed")
        ),
        "physical_maintenance_trace_rows_present": bool(
            energy_metrics.get("maintenance_trace_rows_present", False)
        ),
        "sara_maintenance_event_cost_per_success": _safe_float(
            energy_metrics.get("sara_maintenance_event_cost_per_success")
        ),
        "ann_maintenance_event_cost_per_success": _safe_float(
            energy_metrics.get("ann_maintenance_event_cost_per_success")
        ),
        "physical_alignment_available": bool(alignment.get("available", False)),
        "physical_maintenance_event_cost_per_selected": _safe_float(
            alignment.get("sara_physical_maintenance_event_cost_per_selected")
        ),
        "reference_maintenance_event_cost_per_selected": _safe_float(
            alignment.get("reference_maintenance_event_cost_per_selected")
        ),
        "maintenance_event_cost_per_selected_alignment_ratio": _safe_float(
            alignment.get("maintenance_event_cost_per_selected_ratio")
        ),
        "maintenance_event_cost_per_selected_alignment_delta": _safe_float(
            alignment.get("maintenance_event_cost_per_selected_delta")
        ),
    }


def _compression_surface(
    *,
    event_memory_report: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    payload = event_memory_report or {}
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), Mapping) else {}
    counts = payload.get("counts", {}) if isinstance(payload.get("counts", {}), Mapping) else {}
    return {
        "available": bool(event_memory_report),
        "passed": bool(payload.get("passed", False)),
        "observed_event_count": _safe_int(counts.get("observed_events")),
        "episode_count": _safe_int(counts.get("episodes")),
        "verified_relation_count": _safe_int(counts.get("verified_relations")),
        "eventization_emission_ratio": _safe_float(metrics.get("eventization_emission_ratio")),
        "candidate_event_acceptance_rate": _safe_float(
            metrics.get("candidate_event_acceptance_rate")
        ),
        "episode_compression_ratio": _safe_float(metrics.get("episode_compression_ratio")),
        "relation_verification_yield": _safe_float(metrics.get("relation_verification_yield")),
        "lineage_coverage_ratio": _safe_float(metrics.get("lineage_coverage_ratio")),
        "self_state_continuity": _safe_float(metrics.get("self_state_continuity")),
        "self_state_external_event_ratio": _safe_float(
            metrics.get("self_state_external_event_ratio")
        ),
    }


def _compression_maintenance_coupling_surface(
    *,
    event_memory_maintenance_coupling_report: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    payload = event_memory_maintenance_coupling_report or {}
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), Mapping) else {}
    best_profile = (
        payload.get("best_profile", {})
        if isinstance(payload.get("best_profile", {}), Mapping)
        else {}
    )
    return {
        "available": bool(event_memory_maintenance_coupling_report),
        "passed": bool(payload.get("passed", False)),
        "profile_count": _safe_int(payload.get("profile_count")),
        "best_profile_id": str(best_profile.get("profile_id", "") or ""),
        "compression_to_maintenance_correlation": _safe_float(
            metrics.get("compression_to_maintenance_correlation")
        ),
        "best_profile_compression_efficiency_per_maintenance": _safe_float(
            metrics.get("best_profile_compression_efficiency_per_maintenance")
        ),
        "best_profile_self_state_continuity": _safe_float(
            metrics.get("best_profile_self_state_continuity")
        ),
        "best_profile_episode_compression_ratio": _safe_float(
            metrics.get("best_profile_episode_compression_ratio")
        ),
    }


def build_sara_ann_comparison_report(
    *,
    external_validity_report: Mapping[str, Any],
    external_ladder_report: Mapping[str, Any],
    energy_measurement_report: Mapping[str, Any],
    internal_maintenance_report: Mapping[str, Any] | None = None,
    event_memory_report: Mapping[str, Any] | None = None,
    event_memory_maintenance_coupling_report: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    external_metrics = _metrics(external_validity_report)
    external_checks = _checks(external_validity_report)
    external_reference_readiness = _reference_readiness(external_validity_report)
    ladder_metrics = _metrics(external_ladder_report)
    ladder_checks = _checks(external_ladder_report)
    energy_checks = _checks(energy_measurement_report)
    baseline_cards = _build_baseline_cards(
        external_validity_report=external_validity_report,
        energy_measurement_report=energy_measurement_report,
    )
    maintenance_surface = _maintenance_surface(
        internal_maintenance_report=internal_maintenance_report,
        energy_measurement_report=energy_measurement_report,
    )
    compression_surface = _compression_surface(
        event_memory_report=event_memory_report,
    )
    compression_maintenance_coupling_surface = _compression_maintenance_coupling_surface(
        event_memory_maintenance_coupling_report=event_memory_maintenance_coupling_report,
    )
    stronger_real_reference_present = any(
        str(card.get("baseline_id", "")) in {
            "ann_pretrained_embedding_reference",
            "ann_pretrained_embedding_faiss_reference",
            "ann_cross_encoder_reference",
        }
        and bool(card.get("available", False))
        for card in baseline_cards
    )
    bm25_present = any(
        str(card.get("baseline_id", "")) == "bm25_offline_proxy" and bool(card.get("available", False))
        for card in baseline_cards
    )
    physical_present = any(
        str(card.get("baseline_id", "")) == "physical_ann_measurement" and bool(card.get("available", False))
        for card in baseline_cards
    )
    best_reference = _best_available_reference(baseline_cards)
    checks = {
        "external_validity_present": bool(external_validity_report),
        "external_validity_passed": bool(external_validity_report.get("passed", False)),
        "ladder_present": bool(external_ladder_report),
        "ladder_passed": bool(external_ladder_report.get("passed", False)),
        "bm25_reference_present": bm25_present,
        "stronger_real_reference_present": stronger_real_reference_present,
        "reference_readiness_visible": bool(external_reference_readiness),
        "per_task_summary_present": _safe_float(external_metrics.get("per_task_external_validity_summary_available")) >= 1.0,
        "quality_and_cost_reported_together": (
            _safe_float(external_metrics.get("real_data_qa_accuracy")) > 0.0
            and _safe_float(external_metrics.get("ann_cost_advantage_proxy")) > 0.0
        ),
        "offline_references_labeled": all(
            str(card.get("evidence_tier", "")) in {"proxy", "offline_reference", "physical"}
            for card in baseline_cards
        ),
        "physical_evidence_separated": physical_present == bool(energy_measurement_report.get("real_joule_measurements_present", False)),
        "physical_quality_guard_passed": (
            (not physical_present)
            or (
                bool(energy_checks.get("quality_parity_passed", False))
                and bool(energy_checks.get("paired_task_rows_balanced", False))
            )
        ),
        "maintenance_alignment_visible": (
            (not physical_present)
            or bool(maintenance_surface.get("physical_alignment_available", False))
        ),
        "event_memory_compression_visible": bool(compression_surface.get("available", False)),
        "event_memory_maintenance_coupling_visible": bool(
            compression_maintenance_coupling_surface.get("available", False)
        ),
    }
    completion_score = sum(1 for passed in checks.values() if bool(passed)) / max(len(checks), 1)
    if physical_present and stronger_real_reference_present and bm25_present:
        status = "phase6_and_phase8_evidence_surface_ready"
    elif stronger_real_reference_present and bm25_present:
        status = "phase8_reference_surface_ready_phase6_pending"
    else:
        status = "proxy_only_or_partial_reference_surface"
    next_actions: List[Dict[str, Any]] = []
    if not bm25_present:
        next_actions.append(
            {
                "priority": "high",
                "category": "missing_bm25_reference",
                "command": "python scripts/sara_cli.py eval-external-validity",
            }
        )
    if not stronger_real_reference_present:
        next_actions.extend(_build_reference_next_actions(external_validity_report))
        if not next_actions:
            next_actions.append(
                {
                    "priority": "high",
                    "category": "missing_real_reference",
                    "command": "Configure --pretrained-embedding-model or --cross-encoder-model for eval-external-validity.",
                }
            )
    if not physical_present:
        next_actions.append(
            {
                "priority": "high",
                "category": "missing_physical_measurement",
                "command": "python scripts/eval/energy_measurement_readiness.py",
            }
        )
    if not bool(maintenance_surface.get("available", False)):
        next_actions.append(
            {
                "priority": "medium",
                "category": "missing_internal_maintenance_reference",
                "command": "python scripts/sara_cli.py eval-internal-maintenance-efficiency",
            }
        )
    elif physical_present and not bool(maintenance_surface.get("physical_alignment_available", False)):
        next_actions.append(
            {
                "priority": "medium",
                "category": "missing_physical_maintenance_alignment",
                "command": "python scripts/eval/energy_measurement_readiness.py",
            }
        )
    elif (
        bool(maintenance_surface.get("physical_alignment_available", False))
        and _safe_float(
            maintenance_surface.get("maintenance_event_cost_per_selected_alignment_ratio")
        )
        > 1.5
    ):
        next_actions.append(
            {
                "priority": "medium",
                "category": "maintenance_alignment_drift",
                "command": "Inspect physical maintenance traces and rerun python scripts/sara_cli.py eval-internal-maintenance-efficiency before promoting the current Phase 6 result.",
            }
        )
    if not bool(compression_surface.get("available", False)):
        next_actions.append(
            {
                "priority": "medium",
                "category": "missing_event_memory_compression_surface",
                "command": "python scripts/sara_cli.py eval-event-memory-ingest-pipeline",
            }
        )
    elif (
        _safe_float(compression_surface.get("episode_compression_ratio")) < 1.0
        or _safe_float(compression_surface.get("relation_verification_yield")) < 0.5
    ):
        next_actions.append(
            {
                "priority": "medium",
                "category": "weak_event_memory_compression_surface",
                "command": "Review Event Memory compression, proposal verification, and episode segmentation before promoting the current comparison surface.",
            }
        )
    if not bool(compression_maintenance_coupling_surface.get("available", False)):
        next_actions.append(
            {
                "priority": "medium",
                "category": "missing_event_memory_maintenance_coupling_surface",
                "command": "python scripts/sara_cli.py eval-event-memory-maintenance-coupling",
            }
        )
    elif (
        _safe_float(
            compression_maintenance_coupling_surface.get(
                "best_profile_compression_efficiency_per_maintenance"
            )
        )
        <= 0.0
        or _safe_float(
            compression_maintenance_coupling_surface.get(
                "best_profile_self_state_continuity"
            )
        )
        < 0.5
    ):
        next_actions.append(
            {
                "priority": "medium",
                "category": "weak_event_memory_maintenance_coupling_surface",
                "command": "Review Event Memory profile width, episode segmentation, and self-state carry-over before promoting the current comparison surface.",
            }
        )
    report = {
        "schema": "sara-ann-comparison-report-v1",
        "passed": bool(all(checks.values())),
        "status": status,
        "completion_score": float(completion_score),
        "checks": checks,
        "metrics": {
            "real_data_qa_accuracy": _safe_float(external_metrics.get("real_data_qa_accuracy")),
            "ann_cost_advantage_proxy": _safe_float(external_metrics.get("ann_cost_advantage_proxy")),
            "bm25_offline_proxy_qa_accuracy": _safe_float(external_metrics.get("bm25_offline_proxy_qa_accuracy")),
            "real_pretrained_embedding_reference_available": _safe_float(external_metrics.get("real_pretrained_embedding_reference_available")),
            "real_pretrained_embedding_faiss_reference_available": _safe_float(external_metrics.get("real_pretrained_embedding_faiss_reference_available")),
            "real_cross_encoder_reference_available": _safe_float(external_metrics.get("real_cross_encoder_reference_available")),
            "paired_task_count": _safe_float(_metrics(energy_measurement_report).get("paired_task_count")),
            "ann_to_sara_joule_efficiency_ratio": _safe_float(_metrics(energy_measurement_report).get("ann_to_sara_joule_efficiency_ratio")),
            "ladder_profile_count": _safe_float(ladder_metrics.get("profile_count")),
            "reference_ready_count": _safe_float(external_metrics.get("reference_ready_count")),
            "reference_configured_count": _safe_float(external_metrics.get("reference_configured_count")),
            "reference_dependency_error_count": _safe_float(external_metrics.get("reference_dependency_error_count")),
            "maintenance_event_cost_per_selected": _safe_float(
                maintenance_surface.get("maintenance_event_cost_per_selected")
            ),
            "sara_maintenance_event_cost_per_success": _safe_float(
                maintenance_surface.get("sara_maintenance_event_cost_per_success")
            ),
            "ann_maintenance_event_cost_per_success": _safe_float(
                maintenance_surface.get("ann_maintenance_event_cost_per_success")
            ),
            "physical_maintenance_event_cost_per_selected": _safe_float(
                maintenance_surface.get("physical_maintenance_event_cost_per_selected")
            ),
            "maintenance_event_cost_per_selected_alignment_ratio": _safe_float(
                maintenance_surface.get("maintenance_event_cost_per_selected_alignment_ratio")
            ),
            "event_memory_episode_compression_ratio": _safe_float(
                compression_surface.get("episode_compression_ratio")
            ),
            "event_memory_relation_verification_yield": _safe_float(
                compression_surface.get("relation_verification_yield")
            ),
            "event_memory_self_state_continuity": _safe_float(
                compression_surface.get("self_state_continuity")
            ),
            "event_memory_maintenance_best_profile": str(
                compression_maintenance_coupling_surface.get("best_profile_id", "") or ""
            ),
            "event_memory_maintenance_correlation": _safe_float(
                compression_maintenance_coupling_surface.get(
                    "compression_to_maintenance_correlation"
                )
            ),
            "event_memory_maintenance_best_efficiency": _safe_float(
                compression_maintenance_coupling_surface.get(
                    "best_profile_compression_efficiency_per_maintenance"
                )
            ),
            "event_memory_maintenance_best_continuity": _safe_float(
                compression_maintenance_coupling_surface.get(
                    "best_profile_self_state_continuity"
                )
            ),
        },
        "best_available_offline_reference": best_reference,
        "baseline_cards": baseline_cards,
        "reference_readiness": dict(external_reference_readiness),
        "maintenance_surface": maintenance_surface,
        "compression_surface": compression_surface,
        "compression_maintenance_coupling_surface": compression_maintenance_coupling_surface,
        "artifact_state": {
            "external_validity_passed": bool(external_validity_report.get("passed", False)),
            "ladder_passed": bool(external_ladder_report.get("passed", False)),
            "physical_measurement_present": physical_present,
            "internal_maintenance_reference_present": bool(maintenance_surface.get("available", False)),
            "physical_maintenance_alignment_present": bool(
                maintenance_surface.get("physical_alignment_available", False)
            ),
            "event_memory_compression_present": bool(compression_surface.get("available", False)),
            "event_memory_maintenance_coupling_present": bool(
                compression_maintenance_coupling_surface.get("available", False)
            ),
            "trend_no_regressions": bool(external_checks.get("trend.no_regressions", False)),
            "ladder_all_profiles_passed": bool(ladder_checks.get("all_profiles_passed", False)),
        },
        "next_action_count": len(next_actions),
        "next_actions": next_actions,
    }
    return report


def format_sara_ann_comparison_summary(report: Mapping[str, Any]) -> str:
    checks = report.get("checks", {}) if isinstance(report.get("checks"), Mapping) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), Mapping) else {}
    best_reference = (
        report.get("best_available_offline_reference", {})
        if isinstance(report.get("best_available_offline_reference"), Mapping)
        else {}
    )
    baseline_cards = report.get("baseline_cards", []) if isinstance(report.get("baseline_cards"), list) else []
    next_actions = report.get("next_actions", []) if isinstance(report.get("next_actions"), list) else []
    maintenance_surface = (
        report.get("maintenance_surface", {})
        if isinstance(report.get("maintenance_surface"), Mapping)
        else {}
    )
    compression_surface = (
        report.get("compression_surface", {})
        if isinstance(report.get("compression_surface"), Mapping)
        else {}
    )
    compression_maintenance_coupling_surface = (
        report.get("compression_maintenance_coupling_surface", {})
        if isinstance(report.get("compression_maintenance_coupling_surface"), Mapping)
        else {}
    )
    lines = [
        "# SARA ANN Comparison Report",
        f"- passed: {bool(report.get('passed', False))}",
        f"- status: {report.get('status', '')}",
        f"- completion_score: {_safe_float(report.get('completion_score')):.3f}",
        f"- real_data_qa_accuracy: {_safe_float(metrics.get('real_data_qa_accuracy')):.3f}",
        f"- ann_cost_advantage_proxy: {_safe_float(metrics.get('ann_cost_advantage_proxy')):.3f}",
        f"- bm25_offline_proxy_qa_accuracy: {_safe_float(metrics.get('bm25_offline_proxy_qa_accuracy')):.3f}",
        f"- paired_task_count: {_safe_float(metrics.get('paired_task_count')):.3f}",
        f"- ann_to_sara_joule_efficiency_ratio: {_safe_float(metrics.get('ann_to_sara_joule_efficiency_ratio')):.3f}",
        f"- reference_ready_count: {_safe_float(metrics.get('reference_ready_count')):.3f}",
        f"- reference_configured_count: {_safe_float(metrics.get('reference_configured_count')):.3f}",
        f"- best_available_offline_reference: {best_reference.get('label', '')}",
        f"- maintenance_event_cost_per_selected: {_safe_float(metrics.get('maintenance_event_cost_per_selected')):.3f}",
        f"- sara_maintenance_event_cost_per_success: {_safe_float(metrics.get('sara_maintenance_event_cost_per_success')):.3f}",
        f"- ann_maintenance_event_cost_per_success: {_safe_float(metrics.get('ann_maintenance_event_cost_per_success')):.3f}",
        f"- event_memory_episode_compression_ratio: {_safe_float(metrics.get('event_memory_episode_compression_ratio')):.3f}",
        f"- event_memory_relation_verification_yield: {_safe_float(metrics.get('event_memory_relation_verification_yield')):.3f}",
        f"- event_memory_self_state_continuity: {_safe_float(metrics.get('event_memory_self_state_continuity')):.3f}",
        f"- event_memory_maintenance_best_profile: {metrics.get('event_memory_maintenance_best_profile', '')}",
        f"- event_memory_maintenance_correlation: {_safe_float(metrics.get('event_memory_maintenance_correlation')):.3f}",
        f"- event_memory_maintenance_best_efficiency: {_safe_float(metrics.get('event_memory_maintenance_best_efficiency')):.3f}",
        f"- event_memory_maintenance_best_continuity: {_safe_float(metrics.get('event_memory_maintenance_best_continuity')):.3f}",
        "Checks:",
    ]
    for name in sorted(checks):
        lines.append(f"- {name}: {'PASS' if bool(checks[name]) else 'FAIL'}")
    lines.append("Baselines:")
    for card in baseline_cards:
        if not isinstance(card, Mapping):
            continue
        lines.append(
            "- "
            f"id={card.get('baseline_id', '')}, "
            f"tier={card.get('evidence_tier', '')}, "
            f"available={bool(card.get('available', False))}, "
            f"quality={_safe_float(card.get('quality_score')):.3f}, "
            f"cost={_safe_float(card.get('cost_score')):.3f}"
        )
    readiness = (
        report.get("reference_readiness", {})
        if isinstance(report.get("reference_readiness"), Mapping)
        else {}
    )
    references = (
        readiness.get("references", [])
        if isinstance(readiness.get("references"), list)
        else []
    )
    if references:
        lines.append("Reference Readiness:")
        for item in references:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"id={item.get('reference_id', '')}, "
                f"available={bool(item.get('available', False))}, "
                f"reason={item.get('reason', '')}, "
                f"path={item.get('configured_path', '')}"
            )
    if maintenance_surface:
        lines.append("Maintenance Surface:")
        lines.append(
            "- "
            f"available={bool(maintenance_surface.get('available', False))}, "
            f"observed_only={bool(maintenance_surface.get('observed_only', False))}, "
            f"selected={_safe_int(maintenance_surface.get('maintenance_selected_count'))}, "
            f"refresh={_safe_int(maintenance_surface.get('maintenance_refresh_count'))}, "
            f"event_cost_per_selected={_safe_float(maintenance_surface.get('maintenance_event_cost_per_selected')):.3f}, "
            f"physical_trace_rows={bool(maintenance_surface.get('physical_maintenance_trace_rows_present', False))}, "
            f"physical_alignment={bool(maintenance_surface.get('physical_alignment_available', False))}, "
            f"physical_event_cost_per_selected={_safe_float(maintenance_surface.get('physical_maintenance_event_cost_per_selected')):.3f}, "
            f"alignment_ratio={_safe_float(maintenance_surface.get('maintenance_event_cost_per_selected_alignment_ratio')):.3f}"
        )
    if compression_surface:
        lines.append("Compression Surface:")
        lines.append(
            "- "
            f"available={bool(compression_surface.get('available', False))}, "
            f"passed={bool(compression_surface.get('passed', False))}, "
            f"episode_compression_ratio={_safe_float(compression_surface.get('episode_compression_ratio')):.3f}, "
            f"relation_verification_yield={_safe_float(compression_surface.get('relation_verification_yield')):.3f}, "
            f"self_state_continuity={_safe_float(compression_surface.get('self_state_continuity')):.3f}"
        )
    if compression_maintenance_coupling_surface:
        lines.append("Compression Maintenance Coupling Surface:")
        lines.append(
            "- "
            f"available={bool(compression_maintenance_coupling_surface.get('available', False))}, "
            f"passed={bool(compression_maintenance_coupling_surface.get('passed', False))}, "
            f"best_profile={compression_maintenance_coupling_surface.get('best_profile_id', '')}, "
            f"correlation={_safe_float(compression_maintenance_coupling_surface.get('compression_to_maintenance_correlation')):.3f}, "
            f"best_efficiency={_safe_float(compression_maintenance_coupling_surface.get('best_profile_compression_efficiency_per_maintenance')):.3f}, "
            f"best_continuity={_safe_float(compression_maintenance_coupling_surface.get('best_profile_self_state_continuity')):.3f}"
        )
    lines.append(f"Next Actions: {len(next_actions)}")
    for action in next_actions:
        if not isinstance(action, Mapping):
            continue
        lines.append(
            "- "
            f"priority={action.get('priority', '')}, "
            f"category={action.get('category', '')}, "
            f"command={action.get('command', '')}"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-facing SARA versus ANN comparison report.")
    parser.add_argument("--external-validity-report-path", default=DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH)
    parser.add_argument("--external-ladder-report-path", default=DEFAULT_EXTERNAL_LADDER_REPORT_PATH)
    parser.add_argument("--energy-measurement-report-path", default=DEFAULT_ENERGY_MEASUREMENT_REPORT_PATH)
    parser.add_argument("--internal-maintenance-report-path", default=DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH)
    parser.add_argument("--event-memory-report-path", default=DEFAULT_EVENT_MEMORY_REPORT_PATH)
    parser.add_argument(
        "--event-memory-maintenance-coupling-report-path",
        default=DEFAULT_EVENT_MEMORY_MAINTENANCE_COUPLING_REPORT_PATH,
    )
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    external_validity_report = _load_json(args.external_validity_report_path)
    external_ladder_report = _load_json(args.external_ladder_report_path)
    energy_measurement_report = _load_json(args.energy_measurement_report_path)
    internal_maintenance_report = (
        _load_json(args.internal_maintenance_report_path)
        if os.path.exists(args.internal_maintenance_report_path)
        else None
    )
    event_memory_report = (
        _load_json(args.event_memory_report_path)
        if os.path.exists(args.event_memory_report_path)
        else None
    )
    event_memory_maintenance_coupling_report = (
        _load_json(args.event_memory_maintenance_coupling_report_path)
        if os.path.exists(args.event_memory_maintenance_coupling_report_path)
        else None
    )
    report = build_sara_ann_comparison_report(
        external_validity_report=external_validity_report,
        external_ladder_report=external_ladder_report,
        energy_measurement_report=energy_measurement_report,
        internal_maintenance_report=internal_maintenance_report,
        event_memory_report=event_memory_report,
        event_memory_maintenance_coupling_report=event_memory_maintenance_coupling_report,
    )
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_sara_ann_comparison_summary(report))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
