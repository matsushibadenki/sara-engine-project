#!/usr/bin/env python3
"""Run the compact SARA research benchmark suite and write a manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    interim_data_path,
    processed_data_path,
    workspace_path,
)
from sara_engine.evaluation.report_artifacts import artifact_state, display_artifact_value, format_artifact_state_line  # noqa: E402


DEFAULT_MANIFEST_PATH = workspace_path("evaluation", "research_benchmark_manifest.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "research_benchmark_summary.txt")


@dataclass(frozen=True)
class BenchmarkCommand:
    command_id: str
    purpose: str
    command: List[str]
    managed_outputs: List[str]


def build_recommended_commands(*, rust_iterations: int) -> List[BenchmarkCommand]:
    return [
        BenchmarkCommand(
            command_id="research_fixture_readiness",
            purpose="Validate repository-safe benchmark fixtures for QA, abstention, noisy, adversarial, and delayed-recall cases.",
            command=[
                sys.executable,
                "scripts/eval/research_fixture_readiness.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "research_fixture_readiness.json"),
                workspace_path("evaluation", "research_fixture_readiness_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="rust_core_readiness",
            purpose="Validate Rust sparse-runtime source readiness, Cargo tests, and PyO3 export metadata.",
            command=[
                sys.executable,
                "scripts/eval/rust_core_readiness.py",
                "--run-cargo-test",
            ],
            managed_outputs=[
                workspace_path("evaluation", "rust_core_readiness.json"),
                workspace_path("evaluation", "rust_core_readiness_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="rust_core_benchmark",
            purpose="Compare Rust sparse-runtime exports with Python reference paths when the extension is available.",
            command=[
                sys.executable,
                "scripts/eval/rust_core_benchmark.py",
                "--iterations",
                str(rust_iterations),
            ],
            managed_outputs=[
                workspace_path("evaluation", "rust_core_benchmark.json"),
                workspace_path("evaluation", "rust_core_benchmark_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="neuromorphic_capability_matrix",
            purpose="Record hardware-portability profile coverage for chip-neutral sparse event IR.",
            command=[
                sys.executable,
                "scripts/eval/neuromorphic_capability_matrix.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "neuromorphic_capability_matrix.json"),
                workspace_path("evaluation", "neuromorphic_capability_matrix_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="own_latent_learning",
            purpose="Record observed-only sparse own-latent sample-efficiency evidence.",
            command=[
                sys.executable,
                "scripts/eval/own_latent_learning_benchmark.py",
                "--no-history-update",
            ],
            managed_outputs=[
                workspace_path("evaluation", "own_latent_learning_benchmark.json"),
                workspace_path("evaluation", "own_latent_learning_benchmark_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="own_latent_manifest",
            purpose="Build source-backed sparse latent manifests from autobot learning materials.",
            command=[
                sys.executable,
                "scripts/eval/own_latent_manifest_builder.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "own_latent_manifest_builder.json"),
                workspace_path("evaluation", "own_latent_manifest_builder_summary.txt"),
                processed_data_path("autobot", "latent_manifest.jsonl"),
            ],
        ),
        BenchmarkCommand(
            command_id="gap_materials_closed_loop",
            purpose="Measure whether deterministic gap materials reduce own-latent fixture coverage gaps.",
            command=[
                sys.executable,
                "scripts/eval/gap_materials_closed_loop_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "gap_materials_closed_loop_benchmark.json"),
                workspace_path("evaluation", "gap_materials_closed_loop_benchmark_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="autobot_gap_loop_readiness",
            purpose="Validate that managed gap requests become deterministic repair or replay curriculum and reach the training queue.",
            command=[
                sys.executable,
                "scripts/eval/autobot_gap_loop_readiness.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "autobot_gap_loop_readiness.json"),
                workspace_path("evaluation", "autobot_gap_loop_readiness_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="dendritic_feedback_gate",
            purpose="Record observed-only sparse dendritic feedback robustness evidence.",
            command=[
                sys.executable,
                "scripts/eval/dendritic_feedback_gate_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "dendritic_feedback_gate_benchmark.json"),
                workspace_path("evaluation", "dendritic_feedback_gate_benchmark_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="sparse_plan_trace_verifier",
            purpose="Record observed-only sparse plan-trace verification and repair-material evidence.",
            command=[
                sys.executable,
                "scripts/eval/sparse_plan_trace_verifier.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "sparse_plan_trace_verifier.json"),
                workspace_path("evaluation", "sparse_plan_trace_verifier_summary.txt"),
                processed_data_path("benchmark_fixtures", "sparse_plan_trace_cases.jsonl"),
                processed_data_path("autobot", "plan_trace_repair_materials.jsonl"),
            ],
        ),
        BenchmarkCommand(
            command_id="sparse_reasoning_prior",
            purpose="Record observed-only sparse future-state reasoning consistency, relevance, and external-context abstention evidence.",
            command=[
                sys.executable,
                "scripts/eval/sparse_reasoning_prior_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "sparse_reasoning_prior_benchmark.json"),
                workspace_path("evaluation", "sparse_reasoning_prior_benchmark_summary.txt"),
                workspace_path("evaluation", "sparse_reasoning_prior_traces.jsonl"),
                processed_data_path(
                    "benchmark_fixtures", "sparse_reasoning_prior_cases.jsonl"
                ),
            ],
        ),
        BenchmarkCommand(
            command_id="resonance_credit",
            purpose="Record observed-only SARA-specific verified multi-signal plasticity gating evidence.",
            command=[
                sys.executable,
                "scripts/eval/resonance_credit_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "resonance_credit_benchmark.json"),
                workspace_path("evaluation", "resonance_credit_benchmark_summary.txt"),
                workspace_path("evaluation", "resonance_credit_traces.jsonl"),
                workspace_path("evaluation", "resonance_credit_state.json"),
                processed_data_path("benchmark_fixtures", "resonance_credit_cases.jsonl"),
            ],
        ),
        BenchmarkCommand(
            command_id="synesthetic_multimodal_binding",
            purpose="Record observed-only equal-modality sparse binding, routing, plug-swap, and sensory-substitution evidence.",
            command=[
                sys.executable,
                "scripts/eval/synesthetic_multimodal_binding_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "synesthetic_multimodal_binding_benchmark.json"),
                workspace_path(
                    "evaluation", "synesthetic_multimodal_binding_benchmark_summary.txt"
                ),
                workspace_path("evaluation", "synesthetic_multimodal_binding_traces.jsonl"),
                workspace_path("evaluation", "sparse_cortical_column_plug_swap_report.json"),
                processed_data_path("benchmark_fixtures", "synesthetic_multimodal_cases.jsonl"),
                interim_data_path("autobot", "synesthetic_cross_links.jsonl"),
                processed_data_path("autobot", "synesthetic_binding_manifest.jsonl"),
            ],
        ),
        BenchmarkCommand(
            command_id="resonance_credit_integration",
            purpose="Bridge managed reasoning, planning, multimodal, dendritic, own-latent, and metabolic evidence into verified plasticity decisions.",
            command=[
                sys.executable,
                "scripts/eval/resonance_credit_integration_benchmark.py",
            ],
            managed_outputs=[
                workspace_path(
                    "evaluation", "resonance_credit_integration_benchmark.json"
                ),
                workspace_path(
                    "evaluation", "resonance_credit_integration_benchmark_summary.txt"
                ),
                workspace_path(
                    "evaluation", "resonance_credit_integration_traces.jsonl"
                ),
            ],
        ),
        BenchmarkCommand(
            command_id="adaptive_credit_field",
            purpose="Record observed-only event-driven local credit assignment, sparse route updates, and quantized-credit behavior.",
            command=[
                sys.executable,
                "scripts/eval/adaptive_credit_field_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "adaptive_credit_field_benchmark.json"),
                workspace_path(
                    "evaluation",
                    "adaptive_credit_field_benchmark_summary.txt",
                ),
                workspace_path("evaluation", "adaptive_credit_field_traces.jsonl"),
                workspace_path("evaluation", "adaptive_credit_field_state.json"),
                processed_data_path(
                    "benchmark_fixtures",
                    "adaptive_credit_field_cases.jsonl",
                ),
            ],
        ),
        BenchmarkCommand(
            command_id="adaptive_credit_event_memory",
            purpose="Record observed-only Adaptive Credit Field support for Event Memory admission, weak-entry eviction, and contradiction-preserving memory guards.",
            command=[
                sys.executable,
                "scripts/eval/adaptive_credit_event_memory_benchmark.py",
            ],
            managed_outputs=[
                workspace_path(
                    "evaluation",
                    "adaptive_credit_event_memory_benchmark.json",
                ),
                workspace_path(
                    "evaluation",
                    "adaptive_credit_event_memory_benchmark_summary.txt",
                ),
                workspace_path(
                    "evaluation",
                    "adaptive_credit_event_memory_traces.jsonl",
                ),
                processed_data_path(
                    "benchmark_fixtures",
                    "adaptive_credit_event_memory_cases.jsonl",
                ),
            ],
        ),
        BenchmarkCommand(
            command_id="event_state_cache",
            purpose="Compare bounded verified event-state retention across fixed, linear, and logarithmic cache profiles.",
            command=[
                sys.executable,
                "scripts/eval/event_state_cache_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "event_state_cache_benchmark.json"),
                workspace_path(
                    "evaluation",
                    "event_state_cache_benchmark_summary.txt",
                ),
                workspace_path("evaluation", "event_state_cache_traces.jsonl"),
                workspace_path("evaluation", "event_state_cache_state.json"),
                processed_data_path(
                    "benchmark_fixtures",
                    "event_state_cache_cases.jsonl",
                ),
                interim_data_path("event_state_cache", "candidates.jsonl"),
                processed_data_path("event_state_cache", "manifest.jsonl"),
            ],
        ),
        BenchmarkCommand(
            command_id="event_state_cache_integration",
            purpose="Bridge source-aware autobot latent materials and managed resonance evidence into verified hierarchical caching.",
            command=[
                sys.executable,
                "scripts/eval/event_state_cache_integration_benchmark.py",
            ],
            managed_outputs=[
                workspace_path(
                    "evaluation",
                    "event_state_cache_integration_benchmark.json",
                ),
                workspace_path(
                    "evaluation",
                    "event_state_cache_integration_benchmark_summary.txt",
                ),
                workspace_path(
                    "evaluation",
                    "event_state_cache_integration_traces.jsonl",
                ),
                workspace_path(
                    "evaluation",
                    "event_state_cache_round_trip_state.json",
                ),
                workspace_path(
                    "evaluation",
                    "event_state_cache_concept_revalidation_queue.json",
                ),
                workspace_path(
                    "evaluation",
                    "event_state_cache_concept_review_report.json",
                ),
            ],
        ),
        BenchmarkCommand(
            command_id="concept_revalidation_fixture_builder",
            purpose="Generate harder source-aware concept revalidation cases for follow-up Event Memory evaluation.",
            command=[
                sys.executable,
                "scripts/eval/build_concept_revalidation_fixture.py",
            ],
            managed_outputs=[
                processed_data_path(
                    "benchmark_fixtures",
                    "concept_revalidation_cases.jsonl",
                ),
                workspace_path(
                    "evaluation",
                    "concept_revalidation_fixture_builder.json",
                ),
                workspace_path(
                    "evaluation",
                    "concept_revalidation_fixture_builder_summary.txt",
                ),
            ],
        ),
        BenchmarkCommand(
            command_id="persistent_self_state",
            purpose="Record observed-only bounded spontaneous activity, internal prediction, and Event Memory reactivation evidence.",
            command=[
                sys.executable,
                "scripts/eval/persistent_self_state_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "persistent_self_state_benchmark.json"),
                workspace_path("evaluation", "persistent_self_state_benchmark_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="idle_replay",
            purpose="Record observed-only bounded idle replay selection, self-state continuity, and astro-style replay modulation evidence.",
            command=[
                sys.executable,
                "scripts/eval/idle_replay_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "idle_replay_benchmark.json"),
                workspace_path("evaluation", "idle_replay_benchmark_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="internal_maintenance_efficiency",
            purpose="Record observed-only bounded maintenance cost, self-state continuity, and replay-refresh efficiency on one fixed internal loop.",
            command=[
                sys.executable,
                "scripts/eval/internal_maintenance_efficiency_benchmark.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "internal_maintenance_efficiency_benchmark.json"),
                workspace_path("evaluation", "internal_maintenance_efficiency_benchmark_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="event_memory_ingest_pipeline",
            purpose="Record Event Memory compression, relation-verification yield, lineage coverage, and self-state continuity on the bounded ingest path.",
            command=[
                sys.executable,
                "scripts/eval/event_memory_ingest_pipeline.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "event_memory_ingest_pipeline.json"),
                workspace_path("evaluation", "event_memory_ingest_pipeline_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="event_memory_maintenance_coupling",
            purpose="Record how Event Memory compression profiles trade off against bounded self-state maintenance load and continuity.",
            command=[
                sys.executable,
                "scripts/eval/event_memory_maintenance_coupling_benchmark.py",
            ],
            managed_outputs=[
                workspace_path(
                    "evaluation",
                    "event_memory_maintenance_coupling_benchmark.json",
                ),
                workspace_path(
                    "evaluation",
                    "event_memory_maintenance_coupling_benchmark_summary.txt",
                ),
            ],
        ),
        BenchmarkCommand(
            command_id="sara_ann_comparison",
            purpose="Summarize proxy, offline-reference, and physical SARA-versus-ANN evidence in one managed comparison surface.",
            command=[
                sys.executable,
                "scripts/eval/sara_ann_comparison_report.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "sara_ann_comparison_report.json"),
                workspace_path("evaluation", "sara_ann_comparison_report.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="research_product_completion",
            purpose="Validate policy, ROADMAP closure, release evidence, managed outputs, and Rust readiness.",
            command=[
                sys.executable,
                "scripts/eval/research_product_completion_gate.py",
            ],
            managed_outputs=[
                workspace_path("evaluation", "research_product_completion_gate_report.json"),
                workspace_path("evaluation", "research_product_completion_gate_summary.txt"),
            ],
        ),
        BenchmarkCommand(
            command_id="v1_release_gate",
            purpose="Run the integrated v1.1 release-gate surface for reproducibility.",
            command=[
                sys.executable,
                "scripts/eval/v1_release_gate.py",
            ],
            managed_outputs=[
                workspace_path("release", "v1_release_gate_report.json"),
                workspace_path("release", "v1_release_gate_summary.txt"),
                workspace_path("release", "v1_release_gate_actions.json"),
            ],
        ),
    ]


def _bundle_support_gap(
    *,
    event_memory_ingest_report: Optional[Dict[str, Any]],
    event_memory_maintenance_coupling_report: Optional[Dict[str, Any]],
    sara_ann_comparison_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ingest_metrics = (
        event_memory_ingest_report.get("metrics", {})
        if isinstance(event_memory_ingest_report, dict)
        and isinstance(event_memory_ingest_report.get("metrics"), dict)
        else {}
    )
    coupling_metrics = (
        event_memory_maintenance_coupling_report.get("metrics", {})
        if isinstance(event_memory_maintenance_coupling_report, dict)
        and isinstance(event_memory_maintenance_coupling_report.get("metrics"), dict)
        else {}
    )
    comparison_next_actions = (
        sara_ann_comparison_report.get("next_actions", [])
        if isinstance(sara_ann_comparison_report, dict)
        and isinstance(sara_ann_comparison_report.get("next_actions"), list)
        else []
    )
    comparison_categories = [
        str(item.get("category", "") or "").strip()
        for item in comparison_next_actions
        if isinstance(item, dict)
    ]
    ingest_bundle_contribution = ingest_metrics.get("multimodal_bundle_compression_contribution")
    coupling_bundle_contribution = coupling_metrics.get(
        "best_profile_multimodal_bundle_compression_contribution"
    )
    ingest_weak = bool(
        isinstance(ingest_bundle_contribution, (int, float))
        and float(ingest_bundle_contribution) < 0.5
    )
    coupling_weak = bool(
        isinstance(coupling_bundle_contribution, (int, float))
        and float(coupling_bundle_contribution) < 0.5
    )
    comparison_weak = any(
        category in {
            "weak_event_memory_bundle_compression_surface",
            "weak_event_memory_bundle_coupling_surface",
        }
        for category in comparison_categories
    )
    gap_present = ingest_weak or coupling_weak or comparison_weak
    trigger = ""
    if coupling_weak:
        trigger = "maintenance_coupling"
    elif ingest_weak:
        trigger = "ingest_pipeline"
    elif comparison_weak:
        trigger = "comparison_surface"
    return {
        "present": gap_present,
        "trigger": trigger,
        "repair_target": "phase7_source_aware_bundle_fixtures" if gap_present else None,
        "ingest_weak": ingest_weak,
        "coupling_weak": coupling_weak,
        "comparison_weak": comparison_weak,
    }


def _bundle_support_fixture_repairs(
    *,
    bundle_support_gap: Mapping[str, Any],
    autobot_gap_loop_readiness_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not bool(bundle_support_gap.get("present")):
        return {
            "action_count": 0,
            "request_ids": [],
            "coverage_ready": True,
        }
    if not isinstance(autobot_gap_loop_readiness_report, dict):
        return {
            "action_count": 0,
            "request_ids": [],
            "coverage_ready": False,
        }
    fixture_repair_actions = (
        autobot_gap_loop_readiness_report.get("fixture_repair_actions", [])
        if isinstance(autobot_gap_loop_readiness_report.get("fixture_repair_actions"), list)
        else []
    )
    preferred_tokens = (
        "bundle",
        "source_diversity",
        "counterexample",
        "repair_support",
        "revision_conflict",
    )
    preferred_actions = [
        item
        for item in fixture_repair_actions
        if isinstance(item, dict)
        and any(token in str(item.get("request_id", "") or "") for token in preferred_tokens)
    ]
    selected_actions = preferred_actions or [
        item for item in fixture_repair_actions if isinstance(item, dict)
    ]
    request_ids = sorted(
        {
            str(item.get("request_id", "") or "")
            for item in selected_actions
            if str(item.get("request_id", "") or "")
        }
    )
    return {
        "action_count": len(selected_actions),
        "request_ids": request_ids,
        "coverage_ready": bool(selected_actions),
    }


def _bundle_support_closed_loop_effect(
    *,
    bundle_support_gap: Mapping[str, Any],
    bundle_support_fixture_repairs: Mapping[str, Any],
    gap_materials_closed_loop_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not bool(bundle_support_gap.get("present")):
        return {
            "request_overlap_count": 0,
            "request_overlap_ids": [],
            "gap_reduction": 0,
            "coverage_ready": True,
        }
    if not isinstance(gap_materials_closed_loop_report, dict):
        return {
            "request_overlap_count": 0,
            "request_overlap_ids": [],
            "gap_reduction": 0,
            "coverage_ready": False,
        }
    bundle_request_ids = {
        str(item)
        for item in (
            bundle_support_fixture_repairs.get("request_ids", [])
            if isinstance(bundle_support_fixture_repairs.get("request_ids", []), list)
            else []
        )
        if str(item)
    }
    closed_loop_request_ids = {
        str(item)
        for item in (
            gap_materials_closed_loop_report.get("bundle_relevant_built_request_ids", [])
            if isinstance(gap_materials_closed_loop_report.get("bundle_relevant_built_request_ids", []), list)
            else []
        )
        if str(item)
    }
    overlap_ids = sorted(bundle_request_ids & closed_loop_request_ids)
    return {
        "request_overlap_count": len(overlap_ids),
        "request_overlap_ids": overlap_ids,
        "gap_reduction": int(gap_materials_closed_loop_report.get("coverage_gap_reduction", 0) or 0),
        "coverage_ready": bool(overlap_ids)
        and int(gap_materials_closed_loop_report.get("coverage_gap_reduction", 0) or 0) > 0,
    }


def _select_request_audit_subset(
    request_audit: Mapping[str, Any],
    request_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(request_audit, Mapping):
        return {}
    selected: Dict[str, Dict[str, Any]] = {}
    for request_id in request_ids:
        key = str(request_id or "")
        value = request_audit.get(key)
        if key and isinstance(value, dict):
            selected[key] = dict(value)
    return dict(sorted(selected.items()))


def _bundle_overlap_isolation_risk_summary(
    request_audit: Mapping[str, Any],
    request_ids: Sequence[str],
) -> Dict[str, Any]:
    selected = _select_request_audit_subset(request_audit, request_ids)
    axis_weight = {
        "source_lineage": 3,
        "source_domain": 3,
        "collection_time": 1,
    }
    all_missing_axes = sorted(
        {
            str(axis)
            for payload in selected.values()
            if isinstance(payload, dict)
            for axis in (
                payload.get("missing_axes", [])
                if isinstance(payload.get("missing_axes", []), list)
                else []
            )
            if str(axis)
        }
    )
    risk_count = sum(
        1
        for payload in selected.values()
        if isinstance(payload, dict)
        and bool(
            payload.get("missing_axes", [])
            if isinstance(payload.get("missing_axes", []), list)
            else []
        )
    )
    highest_risk_axis = ""
    highest_risk_weight = 0
    for axis in all_missing_axes:
        weight = int(axis_weight.get(axis, 1))
        if weight > highest_risk_weight:
            highest_risk_axis = axis
            highest_risk_weight = weight
    risk_priority = "none"
    if highest_risk_weight >= 3:
        risk_priority = "high"
    elif highest_risk_weight > 0:
        risk_priority = "medium"
    return {
        "request_audit": selected,
        "missing_axes": all_missing_axes,
        "risk_count": risk_count,
        "highest_risk_axis": highest_risk_axis,
        "risk_priority": risk_priority,
    }


def _load_json_if_present(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_list_if_present(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
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


def _run_command(item: BenchmarkCommand, *, dry_run: bool) -> Dict[str, Any]:
    started_at = time.time()
    if dry_run:
        return {
            "command_id": item.command_id,
            "purpose": item.purpose,
            "command": item.command,
            "managed_outputs": item.managed_outputs,
            "returncode": None,
            "duration_seconds": 0.0,
            "status": "planned",
        }

    result = subprocess.run(item.command)
    duration_seconds = round(time.time() - started_at, 6)
    output_status = {
        path: os.path.exists(path)
        for path in item.managed_outputs
    }
    return {
        "command_id": item.command_id,
        "purpose": item.purpose,
        "command": item.command,
        "managed_outputs": item.managed_outputs,
        "managed_outputs_present": output_status,
        "returncode": int(result.returncode),
        "duration_seconds": duration_seconds,
        "status": "passed" if result.returncode == 0 and all(output_status.values()) else "failed",
    }


def build_manifest(
    *,
    command_results: Sequence[Dict[str, Any]],
    dry_run: bool,
    rust_iterations: int,
) -> Dict[str, Any]:
    passed_count = sum(1 for item in command_results if item.get("status") in {"passed", "planned"})
    v1_report = _load_json_if_present(workspace_path("release", "v1_release_gate_report.json"))
    operational_report = _load_json_if_present(workspace_path("release", "operational_readiness_report.json"))
    operational_repair_log = _load_json_list_if_present(
        workspace_path("release", "operational_repair_execution_log.json")
    )
    research_report = _load_json_if_present(workspace_path("evaluation", "research_product_completion_gate_report.json"))
    rust_report = _load_json_if_present(workspace_path("evaluation", "rust_core_readiness.json"))
    fixture_report = _load_json_if_present(workspace_path("evaluation", "research_fixture_readiness.json"))
    neuromorphic_report = _load_json_if_present(workspace_path("evaluation", "neuromorphic_capability_matrix.json"))
    own_latent_report = _load_json_if_present(workspace_path("evaluation", "own_latent_learning_benchmark.json"))
    own_latent_manifest_report = _load_json_if_present(workspace_path("evaluation", "own_latent_manifest_builder.json"))
    gap_materials_closed_loop_report = _load_json_if_present(
        workspace_path("evaluation", "gap_materials_closed_loop_benchmark.json")
    )
    autobot_gap_loop_readiness_report = _load_json_if_present(
        workspace_path("evaluation", "autobot_gap_loop_readiness.json")
    )
    dendritic_report = _load_json_if_present(workspace_path("evaluation", "dendritic_feedback_gate_benchmark.json"))
    sparse_plan_report = _load_json_if_present(workspace_path("evaluation", "sparse_plan_trace_verifier.json"))
    reasoning_prior_report = _load_json_if_present(
        workspace_path("evaluation", "sparse_reasoning_prior_benchmark.json")
    )
    resonance_credit_report = _load_json_if_present(
        workspace_path("evaluation", "resonance_credit_benchmark.json")
    )
    resonance_integration_report = _load_json_if_present(
        workspace_path("evaluation", "resonance_credit_integration_benchmark.json")
    )
    adaptive_credit_field_report = _load_json_if_present(
        workspace_path("evaluation", "adaptive_credit_field_benchmark.json")
    )
    adaptive_credit_event_memory_report = _load_json_if_present(
        workspace_path("evaluation", "adaptive_credit_event_memory_benchmark.json")
    )
    synesthetic_report = _load_json_if_present(
        workspace_path("evaluation", "synesthetic_multimodal_binding_benchmark.json")
    )
    event_state_cache_report = _load_json_if_present(
        workspace_path("evaluation", "event_state_cache_benchmark.json")
    )
    event_state_cache_integration_report = _load_json_if_present(
        workspace_path(
            "evaluation",
            "event_state_cache_integration_benchmark.json",
        )
    )
    concept_revalidation_fixture_report = _load_json_if_present(
        workspace_path(
            "evaluation",
            "concept_revalidation_fixture_builder.json",
        )
    )
    persistent_self_state_report = _load_json_if_present(
        workspace_path("evaluation", "persistent_self_state_benchmark.json")
    )
    idle_replay_report = _load_json_if_present(
        workspace_path("evaluation", "idle_replay_benchmark.json")
    )
    internal_maintenance_report = _load_json_if_present(
        workspace_path("evaluation", "internal_maintenance_efficiency_benchmark.json")
    )
    event_memory_ingest_report = _load_json_if_present(
        workspace_path("evaluation", "event_memory_ingest_pipeline.json")
    )
    event_memory_maintenance_coupling_report = _load_json_if_present(
        workspace_path("evaluation", "event_memory_maintenance_coupling_benchmark.json")
    )
    sara_ann_comparison_report = _load_json_if_present(
        workspace_path("evaluation", "sara_ann_comparison_report.json")
    )
    bundle_support_gap = _bundle_support_gap(
        event_memory_ingest_report=event_memory_ingest_report,
        event_memory_maintenance_coupling_report=event_memory_maintenance_coupling_report,
        sara_ann_comparison_report=sara_ann_comparison_report,
    )
    bundle_support_fixture_repairs = _bundle_support_fixture_repairs(
        bundle_support_gap=bundle_support_gap,
        autobot_gap_loop_readiness_report=autobot_gap_loop_readiness_report,
    )
    bundle_support_closed_loop_effect = _bundle_support_closed_loop_effect(
        bundle_support_gap=bundle_support_gap,
        bundle_support_fixture_repairs=bundle_support_fixture_repairs,
        gap_materials_closed_loop_report=gap_materials_closed_loop_report,
    )
    bundle_overlap_isolation_risk = _bundle_overlap_isolation_risk_summary(
        autobot_gap_loop_readiness_report.get("fixture_request_isolation_audit", {})
        if isinstance(autobot_gap_loop_readiness_report, dict)
        else {},
        bundle_support_closed_loop_effect.get("request_overlap_ids", [])
        if isinstance(bundle_support_closed_loop_effect.get("request_overlap_ids", []), list)
        else [],
    )

    what_is_proven = [
        "The managed v1.1 release gate can be reproduced from repository commands.",
        "Policy-compatible sparse-runtime source readiness and Cargo tests are recorded.",
        "Chip-neutral neuromorphic backend capability coverage is recorded as a profile matrix.",
        "Sparse own-latent learning evidence is recorded as observed-only sample-efficiency data.",
        "Source-backed sparse own-latent manifests are generated from autobot learning materials.",
        "Phase 7 autobot gap-loop readiness records whether managed gap requests become repair or replay curriculum and reach the training queue.",
        "Sparse dendritic feedback gate evidence is recorded as observed-only robustness data.",
        "Sparse plan-trace verification evidence is recorded as observed-only repair data.",
        "Sparse future-state reasoning priors record deterministic logic consistency and external-context abstention.",
        "SARA resonance credit permits local plasticity only when verified sparse evidence channels agree.",
        "Managed SARA evaluator reports are bridged into resonance credit with explicit failure isolation.",
        "Adaptive Credit Field records event-driven sparse credit assignment without turning dense backpropagation into the runtime learning rule.",
        "Adaptive Credit Field pressure can be traced into Event Memory admission without bypassing contradiction or source verification guards.",
        "Equal-modality sparse binding and non-language sensory-substitution routes are recorded as observed-only evidence.",
        "Verified multimodal event bundles can be traced into Event Memory promotion checks without collapsing modality-local payloads.",
        "Verified hierarchical event-state caching records bounded delayed-recall, abstention, and state-growth evidence.",
        "Source-aware cache promotion, read-only reactivation, persistence validation, and corruption rejection are recorded.",
        "Benchmark and gate outputs stay inside managed workspace or release paths.",
        "The current SARA-versus-ANN evidence surface separates proxy, offline-reference, and physical claims.",
    ]
    what_is_not_proven = [
        "Physical joule-per-success measurements remain optional unless real meter rows are supplied.",
        "Rust extension speedup is not proven when sara_engine.sara_rust_core is not built in the active Python environment.",
        "External generalization beyond the included real-data fixtures requires additional source-aware datasets.",
        "Event-state cache gains remain observed-only until larger source-aware delayed-recall evaluations pass.",
    ]
    if persistent_self_state_report is not None and bool(persistent_self_state_report.get("passed")):
        what_is_proven.append(
            "Bounded persistent self-state can maintain sparse spontaneous activity, Event Memory reactivation, and internal prediction without external input."
        )
    else:
        what_is_not_proven.append(
            "Bounded persistent self-state maintenance is not yet recorded in the current managed benchmark surface."
        )
    if idle_replay_report is not None and bool(idle_replay_report.get("passed")):
        what_is_proven.append(
            "Bounded idle replay can prioritize self-state-aligned verified memories under an explicit event budget and modulation signal."
        )
    else:
        what_is_not_proven.append(
            "Idle replay maintenance quality is not yet recorded in the current managed benchmark surface."
        )
    if internal_maintenance_report is not None and bool(internal_maintenance_report.get("passed")):
        what_is_proven.append(
            "One fixed internal maintenance loop can preserve self-state continuity and cache refresh with bounded maintenance event cost."
        )
    else:
        what_is_not_proven.append(
            "Internal maintenance event-cost efficiency is not yet recorded in the current managed benchmark surface."
        )
    if event_memory_ingest_report is not None and bool(event_memory_ingest_report.get("passed")):
        what_is_proven.append(
            "Event Memory ingest compression now records bounded eventization, episode compression, verified-relation yield, lineage coverage, and self-state continuity in one managed surface."
        )
    else:
        what_is_not_proven.append(
            "Event Memory compression quality is not yet recorded in the current managed benchmark surface."
        )
    if event_memory_maintenance_coupling_report is not None and bool(
        event_memory_maintenance_coupling_report.get("passed")
    ):
        what_is_proven.append(
            "Event Memory compression profiles can now be compared against bounded self-state maintenance load, continuity, and compression efficiency in one managed coupling surface."
        )
    else:
        what_is_not_proven.append(
            "Compression-to-maintenance coupling is not yet recorded in the current managed benchmark surface."
        )
    if bool(bundle_support_gap.get("present")):
        what_is_not_proven.append(
            "Verified multimodal bundle support remains too weak for a clean compression claim, so the repair loop should return to Phase 7 source-aware bundle-fixture strengthening before promoting this as a SARA-native efficiency win."
        )
        overlap_missing_axes = sorted(
            {
                str(axis)
                for payload in _select_request_audit_subset(
                    autobot_gap_loop_readiness_report.get("fixture_request_isolation_audit", {})
                    if isinstance(autobot_gap_loop_readiness_report, dict)
                    else {},
                    bundle_support_closed_loop_effect.get("request_overlap_ids", [])
                    if isinstance(bundle_support_closed_loop_effect.get("request_overlap_ids", []), list)
                    else [],
                ).values()
                if isinstance(payload, dict)
                for axis in (
                    payload.get("missing_axes", [])
                    if isinstance(payload.get("missing_axes", []), list)
                    else []
                )
                if str(axis)
            }
        )
        if overlap_missing_axes:
            what_is_not_proven.append(
                "Closed-loop bundle repairs currently touch requests with incomplete Phase 7 isolation audit axes "
                f"({', '.join(overlap_missing_axes)}), so the repair win should remain under isolation review before promotion."
            )
            overlap_request_ids = [
                str(item)
                for item in (
                    bundle_support_closed_loop_effect.get("request_overlap_ids", [])
                    if isinstance(bundle_support_closed_loop_effect.get("request_overlap_ids", []), list)
                    else []
                )
                if str(item)
            ]
            if overlap_request_ids:
                what_is_not_proven.append(
                    "The current blocked overlap repair requests are "
                    f"{', '.join(overlap_request_ids)}."
                )
            if str(bundle_overlap_isolation_risk.get("highest_risk_axis", "") or ""):
                what_is_not_proven.append(
                    "Current overlap isolation risk is prioritized as "
                    f"{bundle_overlap_isolation_risk.get('risk_priority')} due to "
                    f"{bundle_overlap_isolation_risk.get('highest_risk_axis')} gaps."
                )
    else:
        what_is_proven.append(
            "The current managed compression surfaces do not show a standalone bundle-support gap that would force a return from Phase 6/8 evidence work back to Phase 7 fixture strengthening."
        )
    if gap_materials_closed_loop_report is not None and bool(gap_materials_closed_loop_report.get("passed")):
        what_is_proven.insert(
            5,
            "Deterministic gap materials can be shown to reduce own-latent fixture coverage gaps in a closed loop.",
        )
    else:
        what_is_not_proven.append(
            "Closed-loop gap-material improvement is not yet proven for the current default autobot materials and targets."
        )
    if not (
        autobot_gap_loop_readiness_report is not None
        and bool(autobot_gap_loop_readiness_report.get("passed"))
    ):
        what_is_not_proven.append(
            "Phase 7 autonomous gap-loop readiness is not yet proven for the current managed materials, targets, and queue handoff."
        )
    adaptive_credit_operational_checks = (
        operational_report.get("checks", {})
        if isinstance(operational_report, dict) and isinstance(operational_report.get("checks"), dict)
        else {}
    )
    adaptive_credit_operational_has_visibility = bool(
        "adaptive_credit_field" in adaptive_credit_operational_checks
        or "adaptive_credit_event_memory" in adaptive_credit_operational_checks
    )
    if adaptive_credit_operational_has_visibility:
        what_is_proven.append(
            "Operational readiness now classifies adaptive-credit failures explicitly and binds them to benchmark rerun repair actions."
        )
    else:
        what_is_not_proven.append(
            "Adaptive-credit failure classification and repair binding are not yet surfaced in the current operational readiness artifact."
        )
    adaptive_credit_repair_log_entries = [
        dict(item)
        for item in operational_repair_log
        if (
            isinstance(item, dict)
            and (
                str(item.get("source", "")).strip() == "adaptive_credit_repair"
                or any(
                    str(check).startswith("adaptive_credit_")
                    for check in (
                        item.get("covered_checks", [])
                        if isinstance(item.get("covered_checks"), list)
                        else []
                    )
                )
            )
        )
    ]
    adaptive_credit_repair_log_success_count = sum(
        1
        for item in adaptive_credit_repair_log_entries
        if str(item.get("status", "")).strip().lower() == "success"
    )
    adaptive_credit_repair_log_pending_count = sum(
        1
        for item in adaptive_credit_repair_log_entries
        if str(item.get("status", "")).strip().lower() == "pending"
    )
    adaptive_credit_repair_log_failure_count = sum(
        1
        for item in adaptive_credit_repair_log_entries
        if str(item.get("status", "")).strip().lower() in {"failed", "error", "timeout"}
    )
    adaptive_credit_repair_log_recovered = bool(
        adaptive_credit_repair_log_success_count > 0
        and adaptive_credit_repair_log_pending_count == 0
        and adaptive_credit_repair_log_failure_count == 0
    )
    adaptive_credit_repair_log_chronic = bool(
        adaptive_credit_repair_log_failure_count >= 2
    )
    if adaptive_credit_repair_log_recovered:
        what_is_proven.append(
            "Adaptive-credit repair history shows at least one clean recovery without pending or repeated failed repair entries."
        )
    elif adaptive_credit_repair_log_chronic:
        what_is_not_proven.append(
            "Adaptive-credit repair history still shows repeated failed repair attempts, so recovery stability is not yet proven."
        )
    elif adaptive_credit_repair_log_pending_count > 0:
        what_is_not_proven.append(
            "Adaptive-credit repair history still has pending repair work, so recovery stability is not yet proven."
        )

    return {
        "schema": "sara-research-benchmark-manifest-v1",
        "suite_name": "SARAResearchBenchmarkSuite",
        "dry_run": bool(dry_run),
        "rust_iterations": int(rust_iterations),
        "command_count": len(command_results),
        "passed_count": passed_count,
        "passed": passed_count == len(command_results),
        "commands": list(command_results),
        "artifact_state": {
            "v1_release_gate": artifact_state(v1_report),
            "operational_readiness": artifact_state(operational_report),
            "research_product_completion_gate": artifact_state(research_report),
            "rust_core_readiness": artifact_state(
                rust_report, pass_field="source_readiness_passed"
            ),
            "research_fixture_readiness": artifact_state(fixture_report),
            "neuromorphic_capability_matrix": artifact_state(neuromorphic_report),
            "own_latent_learning": artifact_state(own_latent_report),
            "own_latent_manifest_builder": artifact_state(own_latent_manifest_report),
            "gap_materials_closed_loop": artifact_state(gap_materials_closed_loop_report),
            "autobot_gap_loop_readiness": artifact_state(
                autobot_gap_loop_readiness_report
            ),
            "dendritic_feedback_gate": artifact_state(dendritic_report),
            "sparse_plan_trace_verifier": artifact_state(sparse_plan_report),
            "sparse_reasoning_prior": artifact_state(reasoning_prior_report),
            "resonance_credit": artifact_state(resonance_credit_report),
            "resonance_credit_integration": artifact_state(
                resonance_integration_report
            ),
            "adaptive_credit_field": artifact_state(adaptive_credit_field_report),
            "adaptive_credit_event_memory": artifact_state(
                adaptive_credit_event_memory_report
            ),
            "synesthetic_multimodal_binding": artifact_state(synesthetic_report),
            "event_state_cache": artifact_state(event_state_cache_report),
            "event_state_cache_integration": artifact_state(
                event_state_cache_integration_report
            ),
            "concept_revalidation_fixture_builder": artifact_state(
                concept_revalidation_fixture_report
            ),
            "persistent_self_state": artifact_state(persistent_self_state_report),
            "idle_replay": artifact_state(idle_replay_report),
            "internal_maintenance_efficiency": artifact_state(internal_maintenance_report),
            "event_memory_ingest_pipeline": artifact_state(event_memory_ingest_report),
            "event_memory_maintenance_coupling": artifact_state(
                event_memory_maintenance_coupling_report
            ),
            "sara_ann_comparison": artifact_state(sara_ann_comparison_report),
        },
        "evidence": {
            "v1_release_passed": None if v1_report is None else bool(v1_report.get("passed")),
            "operational_readiness_passed": None
            if operational_report is None
            else bool(operational_report.get("passed")),
            "research_product_passed": None if research_report is None else bool(research_report.get("passed")),
            "rust_core_status": None if rust_report is None else rust_report.get("status"),
            "rust_extension_available": None
            if rust_report is None
            else rust_report.get("python_extension_available"),
            "research_fixture_passed": None if fixture_report is None else bool(fixture_report.get("passed")),
            "research_fixture_case_count": None if fixture_report is None else fixture_report.get("case_count"),
            "neuromorphic_capability_matrix_passed": None
            if neuromorphic_report is None
            else bool(neuromorphic_report.get("passed")),
            "neuromorphic_profile_count": None
            if neuromorphic_report is None
            else len(neuromorphic_report.get("profiles", [])),
            "own_latent_learning_passed": None
            if own_latent_report is None
            else bool(own_latent_report.get("passed")),
            "own_latent_observed_only": None
            if own_latent_report is None
            else bool(own_latent_report.get("observed_only")),
            "own_latent_manifest_passed": None
            if own_latent_manifest_report is None
            else bool(own_latent_manifest_report.get("passed")),
            "own_latent_manifest_count": None
            if own_latent_manifest_report is None
            else own_latent_manifest_report.get("manifest_count"),
            "own_latent_fixture_feedback_loaded": None
            if own_latent_manifest_report is None
            else own_latent_manifest_report.get("fixture_feedback_loaded"),
            "own_latent_fixture_material_coverage_gap_count": None
            if own_latent_manifest_report is None
            else own_latent_manifest_report.get("fixture_material_coverage_gap_count"),
            "own_latent_fixture_material_request_count": None
            if own_latent_manifest_report is None
            else own_latent_manifest_report.get("fixture_material_request_count"),
            "own_latent_fixture_expansion_plan": []
            if own_latent_manifest_report is None
            else own_latent_manifest_report.get("fixture_expansion_plan", []),
            "gap_materials_closed_loop_passed": None
            if gap_materials_closed_loop_report is None
            else bool(gap_materials_closed_loop_report.get("passed")),
            "gap_materials_closed_loop_baseline_gap_count": None
            if gap_materials_closed_loop_report is None
            else gap_materials_closed_loop_report.get("baseline_fixture_material_coverage_gap_count"),
            "gap_materials_closed_loop_augmented_gap_count": None
            if gap_materials_closed_loop_report is None
            else gap_materials_closed_loop_report.get("augmented_fixture_material_coverage_gap_count"),
            "gap_materials_closed_loop_gap_reduction": None
            if gap_materials_closed_loop_report is None
            else gap_materials_closed_loop_report.get("coverage_gap_reduction"),
            "gap_materials_closed_loop_bundle_relevant_request_coverage": None
            if gap_materials_closed_loop_report is None
            else gap_materials_closed_loop_report.get("bundle_relevant_request_coverage"),
            "gap_materials_closed_loop_bundle_relevant_built_request_ids": []
            if gap_materials_closed_loop_report is None
            else gap_materials_closed_loop_report.get("bundle_relevant_built_request_ids", []),
            "autobot_gap_loop_readiness_passed": None
            if autobot_gap_loop_readiness_report is None
            else bool(autobot_gap_loop_readiness_report.get("passed")),
            "autobot_gap_loop_requested_slot_count": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("requested_slot_count"),
            "autobot_gap_loop_build_coverage": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("gap_build_coverage"),
            "autobot_gap_loop_fixture_request_count": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("fixture_request_count"),
            "autobot_gap_loop_fixture_requested_slot_count": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("fixture_requested_slot_count"),
            "autobot_gap_loop_fixture_gap_material_built_count": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("fixture_gap_material_built_count"),
            "autobot_gap_loop_fixture_build_coverage": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("fixture_gap_build_coverage"),
            "autobot_gap_loop_fixture_source_domain_count": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("fixture_source_domain_count"),
            "autobot_gap_loop_fixture_candidate_source_domain_count": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("fixture_candidate_source_domain_count"),
            "autobot_gap_loop_fixture_accepted_source_domain_count": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("fixture_accepted_source_domain_count"),
            "autobot_gap_loop_fixture_source_lineage_coverage": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("fixture_source_lineage_coverage"),
            "autobot_gap_loop_fixture_collection_time_coverage": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("fixture_collection_time_coverage"),
            "autobot_gap_loop_fixture_source_isolation_ready": None
            if autobot_gap_loop_readiness_report is None
            else bool(
                autobot_gap_loop_readiness_report.get("checks", {})
                .get("fixture_source_isolation_ready", {})
                .get("passed")
            ),
            "autobot_gap_loop_fixture_source_lineage_ready": None
            if autobot_gap_loop_readiness_report is None
            else bool(
                autobot_gap_loop_readiness_report.get("checks", {})
                .get("fixture_source_lineage_ready", {})
                .get("passed")
            ),
            "autobot_gap_loop_fixture_collection_time_ready": None
            if autobot_gap_loop_readiness_report is None
            else bool(
                autobot_gap_loop_readiness_report.get("checks", {})
                .get("fixture_collection_time_ready", {})
                .get("passed")
            ),
            "autobot_gap_loop_fixture_missing_isolation_axes": []
            if autobot_gap_loop_readiness_report is None
            else (
                autobot_gap_loop_readiness_report.get("fixture_isolation_audit", {}).get("missing_axes", [])
                if isinstance(autobot_gap_loop_readiness_report.get("fixture_isolation_audit"), dict)
                else []
            ),
            "autobot_gap_loop_fixture_request_isolation_audit": {}
            if autobot_gap_loop_readiness_report is None
            else (
                autobot_gap_loop_readiness_report.get("fixture_request_isolation_audit", {})
                if isinstance(autobot_gap_loop_readiness_report.get("fixture_request_isolation_audit"), dict)
                else {}
            ),
            "autobot_gap_loop_fixture_requested_slots_by_request": {}
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("fixture_lane", {}).get(
                "requested_slots_by_request", {}
            ),
            "autobot_gap_loop_fixture_built_by_request": {}
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("fixture_lane", {}).get(
                "built_by_request", {}
            ),
            "autobot_gap_loop_fixture_skipped_by_request": {}
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("fixture_lane", {}).get(
                "skipped_by_request", {}
            ),
            "autobot_gap_loop_fixture_repair_action_count": 0
            if autobot_gap_loop_readiness_report is None
            else len(
                autobot_gap_loop_readiness_report.get("fixture_repair_actions", [])
                if isinstance(autobot_gap_loop_readiness_report.get("fixture_repair_actions"), list)
                else []
            ),
            "autobot_gap_loop_fixture_repair_request_ids": []
            if autobot_gap_loop_readiness_report is None
            else [
                str(item.get("request_id", "") or "")
                for item in autobot_gap_loop_readiness_report.get("fixture_repair_actions", [])
                if isinstance(item, dict) and str(item.get("request_id", "") or "")
            ],
            "autobot_gap_loop_enqueue_coverage": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("gap_enqueue_coverage"),
            "autobot_gap_loop_skip_ratio": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("gap_skip_ratio"),
            "autobot_gap_loop_repair_curriculum_share": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("repair_curriculum_share"),
            "autobot_gap_loop_replay_curriculum_share": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("replay_curriculum_share"),
            "dendritic_feedback_gate_passed": None
            if dendritic_report is None
            else bool(dendritic_report.get("passed")),
            "dendritic_feedback_observed_only": None
            if dendritic_report is None
            else bool(dendritic_report.get("observed_only")),
            "dendritic_feedback_robustness_delta": None
            if dendritic_report is None
            else dendritic_report.get("robustness_delta"),
            "sparse_plan_trace_verifier_passed": None
            if sparse_plan_report is None
            else bool(sparse_plan_report.get("passed")),
            "sparse_plan_trace_repair_material_count": None
            if sparse_plan_report is None
            else sparse_plan_report.get("repair_material_count"),
            "sparse_reasoning_prior_passed": None
            if reasoning_prior_report is None
            else bool(reasoning_prior_report.get("passed")),
            "sparse_reasoning_prior_logic_consistency": None
            if reasoning_prior_report is None
            else reasoning_prior_report.get("metrics", {}).get("logic_to_state_consistency"),
            "sparse_reasoning_prior_external_abstention": None
            if reasoning_prior_report is None
            else reasoning_prior_report.get("metrics", {}).get(
                "external_event_missing_abstention"
            ),
            "resonance_credit_passed": None
            if resonance_credit_report is None
            else bool(resonance_credit_report.get("passed")),
            "resonance_harmful_update_suppression": None
            if resonance_credit_report is None
            else resonance_credit_report.get("metrics", {}).get(
                "harmful_update_suppression"
            ),
            "resonance_integration_passed": None
            if resonance_integration_report is None
            else bool(resonance_integration_report.get("passed")),
            "resonance_integration_decision_integrity": None
            if resonance_integration_report is None
            else resonance_integration_report.get("metrics", {}).get(
                "decision_integrity"
            ),
            "adaptive_credit_field_passed": None
            if adaptive_credit_field_report is None
            else bool(adaptive_credit_field_report.get("passed")),
            "adaptive_credit_field_sparse_active_fraction": None
            if adaptive_credit_field_report is None
            else adaptive_credit_field_report.get("metrics", {}).get(
                "sparse_active_fraction_vs_naive"
            ),
            "adaptive_credit_field_quantized_match": None
            if adaptive_credit_field_report is None
            else adaptive_credit_field_report.get("metrics", {}).get(
                "quantized_behavior_match"
            ),
            "adaptive_credit_event_memory_passed": None
            if adaptive_credit_event_memory_report is None
            else bool(adaptive_credit_event_memory_report.get("passed")),
            "adaptive_credit_event_memory_strong_entry_present": None
            if adaptive_credit_event_memory_report is None
            else adaptive_credit_event_memory_report.get("metrics", {}).get(
                "credit_strong_entry_present"
            ),
            "adaptive_credit_event_memory_weak_entry_evicted": None
            if adaptive_credit_event_memory_report is None
            else adaptive_credit_event_memory_report.get("metrics", {}).get(
                "credit_weak_entry_evicted"
            ),
            "adaptive_credit_event_memory_harmful_block_preserved_count": None
            if adaptive_credit_event_memory_report is None
            else adaptive_credit_event_memory_report.get("metrics", {}).get(
                "harmful_block_preserved_count"
            ),
            "adaptive_credit_operational_visibility": adaptive_credit_operational_has_visibility,
            "adaptive_credit_operational_error_count": 0
            if operational_report is None
            else sum(
                1
                for item in (
                    operational_report.get("error_details", [])
                    if isinstance(operational_report.get("error_details"), list)
                    else []
                )
                if isinstance(item, dict)
                and str(item.get("category", "")).startswith("adaptive_credit_")
            ),
            "adaptive_credit_operational_repair_action_count": 0
            if operational_report is None
            else sum(
                1
                for item in (
                    operational_report.get("recovery_actions", [])
                    if isinstance(operational_report.get("recovery_actions"), list)
                    else []
                )
                if isinstance(item, dict)
                and any(
                    str(check).startswith("adaptive_credit_")
                    for check in (
                        item.get("affected_checks", [])
                        if isinstance(item.get("affected_checks"), list)
                        else []
                    )
                )
            ),
            "adaptive_credit_operational_primary_focus": None
            if operational_report is None
            else (
                operational_report.get("failure_focus", {}).get("primary_category")
                if isinstance(operational_report.get("failure_focus"), dict)
                else None
            ),
            "adaptive_credit_repair_log_entry_count": len(adaptive_credit_repair_log_entries),
            "adaptive_credit_repair_log_success_count": adaptive_credit_repair_log_success_count,
            "adaptive_credit_repair_log_pending_count": adaptive_credit_repair_log_pending_count,
            "adaptive_credit_repair_log_failure_count": adaptive_credit_repair_log_failure_count,
            "adaptive_credit_repair_log_recovered": adaptive_credit_repair_log_recovered,
            "adaptive_credit_repair_log_chronic": adaptive_credit_repair_log_chronic,
            "operational_bundle_repair_log_entry_count": None
            if operational_report is None
            else (
                operational_report.get("bundle_repair_log_summary", {}).get("entry_count")
                if isinstance(operational_report.get("bundle_repair_log_summary"), dict)
                else None
            ),
            "operational_bundle_repair_log_recovered_count": None
            if operational_report is None
            else (
                operational_report.get("bundle_repair_log_summary", {}).get("recovered_count")
                if isinstance(operational_report.get("bundle_repair_log_summary"), dict)
                else None
            ),
            "operational_bundle_repair_log_max_gap_reduction": None
            if operational_report is None
            else (
                operational_report.get("bundle_repair_log_summary", {}).get("max_gap_reduction")
                if isinstance(operational_report.get("bundle_repair_log_summary"), dict)
                else None
            ),
            "operational_bundle_isolation_clear_release_success_count": None
            if operational_report is None
            else (
                operational_report.get("bundle_repair_log_summary", {}).get(
                    "isolation_clear_release_success_count"
                )
                if isinstance(operational_report.get("bundle_repair_log_summary"), dict)
                else None
            ),
            "operational_bundle_isolation_clear_release_request_ids": []
            if operational_report is None
            else (
                [
                    str(item)
                    for item in (
                        operational_report.get("bundle_repair_log_summary", {}).get(
                            "isolation_clear_release_request_ids", []
                        )
                        if isinstance(operational_report.get("bundle_repair_log_summary"), dict)
                        and isinstance(
                            operational_report.get("bundle_repair_log_summary", {}).get(
                                "isolation_clear_release_request_ids", []
                            ),
                            list,
                        )
                        else []
                    )
                    if str(item)
                ]
            ),
            "operational_bundle_retry_queue_fresh_count": None
            if operational_report is None
            else sum(
                1
                for item in (
                    operational_report.get("repair_retry_queue", [])
                    if isinstance(operational_report.get("repair_retry_queue"), list)
                    else []
                )
                if isinstance(item, dict)
                and (
                    "autobot_bundle_fixture_repair" in str(item.get("source", "")).strip().lower()
                    or bool(item.get("bundle_closed_loop_overlap", False))
                )
                and not bool(item.get("bundle_recovered_before", False))
            ),
            "operational_bundle_retry_queue_recovered_before_count": None
            if operational_report is None
            else sum(
                1
                for item in (
                    operational_report.get("repair_retry_queue", [])
                    if isinstance(operational_report.get("repair_retry_queue"), list)
                    else []
                )
                if isinstance(item, dict)
                and (
                    "autobot_bundle_fixture_repair" in str(item.get("source", "")).strip().lower()
                    or bool(item.get("bundle_closed_loop_overlap", False))
                )
                and bool(item.get("bundle_recovered_before", False))
            ),
            "operational_bundle_retry_queue_isolation_review_churn_count": None
            if operational_report is None
            else (
                operational_report.get("bundle_retry_queue_summary", {}).get(
                    "isolation_review_churn_count"
                )
                if isinstance(operational_report.get("bundle_retry_queue_summary"), dict)
                else None
            ),
            "operational_bundle_retry_queue_isolation_review_churn_request_ids": []
            if operational_report is None
            else (
                [
                    str(item)
                    for item in (
                        operational_report.get("bundle_retry_queue_summary", {}).get(
                            "isolation_review_churn_request_ids", []
                        )
                        if isinstance(operational_report.get("bundle_retry_queue_summary"), dict)
                        and isinstance(
                            operational_report.get("bundle_retry_queue_summary", {}).get(
                                "isolation_review_churn_request_ids", []
                            ),
                            list,
                        )
                        else []
                    )
                    if str(item)
                ]
            ),
            "operational_bundle_retry_queue_isolation_reblocked_count": None
            if operational_report is None
            else (
                operational_report.get("bundle_retry_queue_summary", {}).get(
                    "isolation_reblocked_count"
                )
                if isinstance(operational_report.get("bundle_retry_queue_summary"), dict)
                else None
            ),
            "operational_bundle_retry_queue_isolation_reblocked_request_ids": []
            if operational_report is None
            else (
                [
                    str(item)
                    for item in (
                        operational_report.get("bundle_retry_queue_summary", {}).get(
                            "isolation_reblocked_request_ids", []
                        )
                        if isinstance(operational_report.get("bundle_retry_queue_summary"), dict)
                        and isinstance(
                            operational_report.get("bundle_retry_queue_summary", {}).get(
                                "isolation_reblocked_request_ids", []
                            ),
                            list,
                        )
                        else []
                    )
                    if str(item)
                ]
            ),
            "operational_bundle_phase7_routed_action_count": None
            if operational_report is None
            else sum(
                1
                for item in (
                    operational_report.get("runbook_actions", [])
                    if isinstance(operational_report.get("runbook_actions"), list)
                    else []
                )
                if isinstance(item, dict)
                and str(item.get("source", "")).strip().lower() == "autobot_bundle_fixture_repair"
                and (
                    str(item.get("return_phase", "")).strip().lower() == "phase7"
                    or "return_phase=phase7" in str(item.get("reason", "")).strip().lower()
                )
            ),
            "operational_bundle_phase7_routed_retry_count": None
            if operational_report is None
            else sum(
                1
                for item in (
                    operational_report.get("repair_retry_queue", [])
                    if isinstance(operational_report.get("repair_retry_queue"), list)
                    else []
                )
                if isinstance(item, dict)
                and (
                    "autobot_bundle_fixture_repair" in str(item.get("source", "")).strip().lower()
                    or bool(item.get("bundle_closed_loop_overlap", False))
                )
                and (
                    str(item.get("return_phase", "")).strip().lower() == "phase7"
                    or "return_phase=phase7" in str(item.get("reason", "")).strip().lower()
                )
            ),
            "operational_bundle_phase7_isolation_ready": None
            if operational_report is None
            else (
                bool(
                    autobot_gap_loop_readiness_report.get("checks", {})
                    .get("fixture_source_isolation_ready", {})
                    .get("passed")
                )
                if autobot_gap_loop_readiness_report is not None
                else None
            ),
            "operational_bundle_phase7_lineage_ready": None
            if operational_report is None
            else (
                bool(
                    autobot_gap_loop_readiness_report.get("checks", {})
                    .get("fixture_source_lineage_ready", {})
                    .get("passed")
                )
                if autobot_gap_loop_readiness_report is not None
                else None
            ),
            "operational_bundle_phase7_collection_time_ready": None
            if operational_report is None
            else (
                bool(
                    autobot_gap_loop_readiness_report.get("checks", {})
                    .get("fixture_collection_time_ready", {})
                    .get("passed")
                )
                if autobot_gap_loop_readiness_report is not None
                else None
            ),
            "operational_bundle_phase7_missing_isolation_axes": []
            if operational_report is None
            else (
                autobot_gap_loop_readiness_report.get("fixture_isolation_audit", {}).get("missing_axes", [])
                if autobot_gap_loop_readiness_report is not None
                and isinstance(autobot_gap_loop_readiness_report.get("fixture_isolation_audit"), dict)
                else []
            ),
            "operational_bundle_phase7_request_isolation_audit": {}
            if operational_report is None
            else (
                autobot_gap_loop_readiness_report.get("fixture_request_isolation_audit", {})
                if autobot_gap_loop_readiness_report is not None
                and isinstance(autobot_gap_loop_readiness_report.get("fixture_request_isolation_audit"), dict)
                else {}
            ),
            "operational_bundle_isolation_blocked_request_count": None
            if operational_report is None
            else operational_report.get("bundle_isolation_blocked_request_count"),
            "operational_bundle_isolation_blocked_request_ids": []
            if operational_report is None
            else (
                [
                    str(item)
                    for item in (
                        operational_report.get("bundle_isolation_blocked_request_ids", [])
                        if isinstance(operational_report.get("bundle_isolation_blocked_request_ids"), list)
                        else []
                    )
                    if str(item)
                ]
            ),
            "operational_bundle_isolation_blocked_missing_axes": []
            if operational_report is None
            else (
                [
                    str(item)
                    for item in (
                        operational_report.get("bundle_isolation_blocked_missing_axes", [])
                        if isinstance(operational_report.get("bundle_isolation_blocked_missing_axes"), list)
                        else []
                    )
                    if str(item)
                ]
            ),
            "operational_bundle_overlap_blocked_request_ids": []
            if operational_report is None
            else (
                sorted(
                    str(item)
                    for item in (
                        operational_report.get("bundle_isolation_blocked_request_ids", [])
                        if isinstance(operational_report.get("bundle_isolation_blocked_request_ids"), list)
                        else []
                    )
                    if str(item)
                    and str(item)
                    in {
                        str(request_id)
                        for request_id in (
                            bundle_support_closed_loop_effect.get("request_overlap_ids", [])
                            if isinstance(bundle_support_closed_loop_effect.get("request_overlap_ids", []), list)
                            else []
                        )
                        if str(request_id)
                    }
                )
            ),
            "operational_bundle_isolation_resolved_request_ids": []
            if operational_report is None
            else (
                sorted(
                    request_id
                    for request_id in (
                        [
                            str(item)
                            for item in (
                                operational_report.get("bundle_repair_log_summary", {}).get(
                                    "isolation_clear_release_request_ids", []
                                )
                                if isinstance(operational_report.get("bundle_repair_log_summary"), dict)
                                and isinstance(
                                    operational_report.get("bundle_repair_log_summary", {}).get(
                                        "isolation_clear_release_request_ids", []
                                    ),
                                    list,
                                )
                                else []
                            )
                            if str(item)
                        ]
                    )
                    if request_id
                    and request_id
                    not in {
                        str(item)
                        for item in (
                            operational_report.get("bundle_isolation_blocked_request_ids", [])
                            if isinstance(operational_report.get("bundle_isolation_blocked_request_ids"), list)
                            else []
                        )
                        if str(item)
                    }
                )
            ),
            "synesthetic_multimodal_binding_passed": None
            if synesthetic_report is None
            else bool(synesthetic_report.get("passed")),
            "synesthetic_multimodal_observed_only": None
            if synesthetic_report is None
            else bool(synesthetic_report.get("observed_only")),
            "synesthetic_non_language_route_usefulness": None
            if synesthetic_report is None
            else synesthetic_report.get("metrics", {}).get("non_language_route_usefulness"),
            "event_state_cache_passed": None
            if event_state_cache_report is None
            else bool(event_state_cache_report.get("passed")),
            "event_state_cache_delayed_recall": None
            if event_state_cache_report is None
            else event_state_cache_report.get("metrics", {}).get(
                "logarithmic_delayed_recall"
            ),
            "event_state_cache_state_ratio": None
            if event_state_cache_report is None
            else event_state_cache_report.get("metrics", {}).get(
                "logarithmic_to_linear_state_ratio"
            ),
            "event_state_cache_integration_passed": None
            if event_state_cache_integration_report is None
            else bool(event_state_cache_integration_report.get("passed")),
            "event_state_cache_source_aware_recall": None
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("metrics", {}).get(
                "source_aware_logarithmic_delayed_recall"
            ),
            "event_state_cache_round_trip_integrity": None
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("metrics", {}).get(
                "round_trip_integrity"
            ),
            "event_state_cache_concept_revalidation_case_count": None
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("metrics", {}).get(
                "concept_revalidation_case_count"
            ),
            "event_state_cache_concept_revalidation_recovery_rate": None
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("metrics", {}).get(
                "concept_revalidation_recovery_rate"
            ),
            "event_state_cache_concept_revalidation_blocked_count": None
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("metrics", {}).get(
                "concept_revalidation_blocked_count"
            ),
            "event_state_cache_concept_source_diversity_blocked_count": None
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("metrics", {}).get(
                "concept_revalidation_source_diversity_blocked_count"
            ),
            "event_state_cache_concept_revision_conflict_blocked_count": None
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("metrics", {}).get(
                "concept_revalidation_revision_conflict_blocked_count"
            ),
            "event_state_cache_concept_counterexample_blocked_count": None
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("metrics", {}).get(
                "concept_revalidation_counterexample_blocked_count"
            ),
            "event_state_cache_concept_attempt_budget_blocked_count": None
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("metrics", {}).get(
                "concept_revalidation_attempt_budget_blocked_count"
            ),
            "event_state_cache_concept_next_actions": []
            if event_state_cache_integration_report is None
            else event_state_cache_integration_report.get("next_actions", []),
            "concept_revalidation_fixture_builder_passed": None
            if concept_revalidation_fixture_report is None
            else bool(concept_revalidation_fixture_report.get("passed")),
            "concept_revalidation_fixture_case_count": None
            if concept_revalidation_fixture_report is None
            else concept_revalidation_fixture_report.get("case_count"),
            "concept_revalidation_fixture_case_type_counts": {}
            if concept_revalidation_fixture_report is None
            else concept_revalidation_fixture_report.get("case_type_counts", {}),
            "concept_revalidation_fixture_manifest_material_type_counts": {}
            if concept_revalidation_fixture_report is None
            else concept_revalidation_fixture_report.get("manifest_material_type_counts", {}),
            "concept_revalidation_fixture_next_actions": []
            if concept_revalidation_fixture_report is None
            else concept_revalidation_fixture_report.get("next_actions", []),
            "concept_revalidation_fixture_expansion_plan": []
            if concept_revalidation_fixture_report is None
            else concept_revalidation_fixture_report.get("expansion_plan", []),
            "persistent_self_state_passed": None
            if persistent_self_state_report is None
            else bool(persistent_self_state_report.get("passed")),
            "persistent_self_state_observed_only": None
            if persistent_self_state_report is None
            else bool(persistent_self_state_report.get("observed_only")),
            "persistent_self_state_idle_activity": None
            if persistent_self_state_report is None
            else persistent_self_state_report.get("metrics", {}).get(
                "persistent_self_state_idle_activity"
            ),
            "persistent_self_state_continuity": None
            if persistent_self_state_report is None
            else persistent_self_state_report.get("metrics", {}).get(
                "persistent_self_state_continuity"
            ),
            "persistent_self_state_memory_reactivation": None
            if persistent_self_state_report is None
            else persistent_self_state_report.get("metrics", {}).get(
                "persistent_self_state_memory_reactivation"
            ),
            "persistent_self_state_internal_prediction": None
            if persistent_self_state_report is None
            else persistent_self_state_report.get("metrics", {}).get(
                "persistent_self_state_internal_prediction"
            ),
            "idle_replay_passed": None
            if idle_replay_report is None
            else bool(idle_replay_report.get("passed")),
            "idle_replay_observed_only": None
            if idle_replay_report is None
            else bool(idle_replay_report.get("observed_only")),
            "idle_replay_candidate_selection": None
            if idle_replay_report is None
            else idle_replay_report.get("metrics", {}).get(
                "idle_replay_candidate_selection_observed"
            ),
            "idle_replay_budget": None
            if idle_replay_report is None
            else idle_replay_report.get("metrics", {}).get(
                "idle_replay_budget_observed"
            ),
            "idle_replay_self_state_alignment": None
            if idle_replay_report is None
            else idle_replay_report.get("metrics", {}).get(
                "idle_replay_self_state_alignment_observed"
            ),
            "idle_replay_memory_reactivation": None
            if idle_replay_report is None
            else idle_replay_report.get("metrics", {}).get(
                "idle_replay_memory_reactivation_observed"
            ),
            "idle_replay_state_continuity": None
            if idle_replay_report is None
            else idle_replay_report.get("metrics", {}).get(
                "idle_replay_state_continuity_observed"
            ),
            "internal_maintenance_efficiency_passed": None
            if internal_maintenance_report is None
            else bool(internal_maintenance_report.get("passed")),
            "internal_maintenance_observed_only": None
            if internal_maintenance_report is None
            else bool(internal_maintenance_report.get("observed_only")),
            "internal_maintenance_selected_count": None
            if internal_maintenance_report is None
            else internal_maintenance_report.get("counts", {}).get(
                "maintenance_selected_count"
            ),
            "internal_maintenance_refresh_count": None
            if internal_maintenance_report is None
            else internal_maintenance_report.get("counts", {}).get(
                "maintenance_refresh_count"
            ),
            "internal_maintenance_event_cost": None
            if internal_maintenance_report is None
            else internal_maintenance_report.get("normalized_metrics", {}).get(
                "maintenance_event_cost"
            ),
            "internal_maintenance_event_cost_per_selected": None
            if internal_maintenance_report is None
            else internal_maintenance_report.get("normalized_metrics", {}).get(
                "maintenance_event_cost_per_selected"
            ),
            "event_memory_ingest_pipeline_passed": None
            if event_memory_ingest_report is None
            else bool(event_memory_ingest_report.get("passed")),
            "event_memory_episode_compression_ratio": None
            if event_memory_ingest_report is None
            else event_memory_ingest_report.get("metrics", {}).get(
                "episode_compression_ratio"
            ),
            "event_memory_relation_verification_yield": None
            if event_memory_ingest_report is None
            else event_memory_ingest_report.get("metrics", {}).get(
                "relation_verification_yield"
            ),
            "event_memory_self_state_continuity": None
            if event_memory_ingest_report is None
            else event_memory_ingest_report.get("metrics", {}).get(
                "self_state_continuity"
            ),
            "event_memory_multimodal_bundle_promotion_rate": None
            if event_memory_ingest_report is None
            else event_memory_ingest_report.get("metrics", {}).get(
                "multimodal_bundle_promotion_rate"
            ),
            "event_memory_multimodal_bundle_relation_verification_yield": None
            if event_memory_ingest_report is None
            else event_memory_ingest_report.get("metrics", {}).get(
                "multimodal_bundle_relation_verification_yield"
            ),
            "event_memory_multimodal_bundle_compression_contribution": None
            if event_memory_ingest_report is None
            else event_memory_ingest_report.get("metrics", {}).get(
                "multimodal_bundle_compression_contribution"
            ),
            "event_memory_multimodal_bundle_promotion_count": None
            if event_memory_ingest_report is None
            else event_memory_ingest_report.get("traces", {}).get(
                "multimodal_bundle_admission", {}
            ).get("promotion_allowed_count"),
            "event_memory_maintenance_coupling_passed": None
            if event_memory_maintenance_coupling_report is None
            else bool(event_memory_maintenance_coupling_report.get("passed")),
            "event_memory_maintenance_best_profile": None
            if event_memory_maintenance_coupling_report is None
            else event_memory_maintenance_coupling_report.get("best_profile", {}).get(
                "profile_id"
            ),
            "event_memory_maintenance_correlation": None
            if event_memory_maintenance_coupling_report is None
            else event_memory_maintenance_coupling_report.get("metrics", {}).get(
                "compression_to_maintenance_correlation"
            ),
            "event_memory_maintenance_best_efficiency": None
            if event_memory_maintenance_coupling_report is None
            else event_memory_maintenance_coupling_report.get("metrics", {}).get(
                "best_profile_compression_efficiency_per_maintenance"
            ),
            "event_memory_maintenance_best_bundle_compression_contribution": None
            if event_memory_maintenance_coupling_report is None
            else event_memory_maintenance_coupling_report.get("metrics", {}).get(
                "best_profile_multimodal_bundle_compression_contribution"
            ),
            "event_memory_bundle_support_gap_present": bool(
                bundle_support_gap.get("present")
            ),
            "event_memory_bundle_support_gap_trigger": bundle_support_gap.get("trigger"),
            "event_memory_bundle_support_repair_target": bundle_support_gap.get(
                "repair_target"
            ),
            "event_memory_bundle_support_fixture_repair_action_count": bundle_support_fixture_repairs.get(
                "action_count"
            ),
            "event_memory_bundle_support_fixture_request_ids": bundle_support_fixture_repairs.get(
                "request_ids"
            ),
            "event_memory_bundle_support_fixture_coverage_ready": bundle_support_fixture_repairs.get(
                "coverage_ready"
            ),
            "event_memory_bundle_support_closed_loop_overlap_count": bundle_support_closed_loop_effect.get(
                "request_overlap_count"
            ),
            "event_memory_bundle_support_closed_loop_overlap_ids": bundle_support_closed_loop_effect.get(
                "request_overlap_ids"
            ),
            "event_memory_bundle_support_closed_loop_gap_reduction": bundle_support_closed_loop_effect.get(
                "gap_reduction"
            ),
            "event_memory_bundle_support_closed_loop_coverage_ready": bundle_support_closed_loop_effect.get(
                "coverage_ready"
            ),
            "event_memory_bundle_support_overlap_request_isolation_audit": bundle_overlap_isolation_risk.get(
                "request_audit"
            ),
            "event_memory_bundle_support_overlap_missing_isolation_axes": bundle_overlap_isolation_risk.get(
                "missing_axes"
            ),
            "event_memory_bundle_support_overlap_isolation_risk_count": bundle_overlap_isolation_risk.get(
                "risk_count"
            ),
            "event_memory_bundle_support_overlap_highest_risk_axis": bundle_overlap_isolation_risk.get(
                "highest_risk_axis"
            ),
            "event_memory_bundle_support_overlap_risk_priority": bundle_overlap_isolation_risk.get(
                "risk_priority"
            ),
            "event_memory_maintenance_best_continuity": None
            if event_memory_maintenance_coupling_report is None
            else event_memory_maintenance_coupling_report.get("metrics", {}).get(
                "best_profile_self_state_continuity"
            ),
            "sara_ann_comparison_passed": None
            if sara_ann_comparison_report is None
            else bool(sara_ann_comparison_report.get("passed")),
            "sara_ann_comparison_status": None
            if sara_ann_comparison_report is None
            else sara_ann_comparison_report.get("status"),
            "sara_ann_comparison_completion_score": None
            if sara_ann_comparison_report is None
            else sara_ann_comparison_report.get("completion_score"),
            "sara_ann_best_offline_reference": None
            if sara_ann_comparison_report is None
            else sara_ann_comparison_report.get("best_available_offline_reference", {}).get("label"),
            "sara_ann_physical_ratio": None
            if sara_ann_comparison_report is None
            else sara_ann_comparison_report.get("metrics", {}).get("ann_to_sara_joule_efficiency_ratio"),
            "sara_ann_next_action_count": None
            if sara_ann_comparison_report is None
            else sara_ann_comparison_report.get("next_action_count"),
        },
        "what_is_proven": what_is_proven,
        "what_is_not_proven": what_is_not_proven,
    }


def write_outputs(manifest: Dict[str, Any], manifest_path: str, summary_path: str) -> None:
    resolved_manifest_path = ensure_parent_directory(manifest_path)
    with open(resolved_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    resolved_summary_path = ensure_parent_directory(summary_path)
    evidence = manifest.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    artifact_state = manifest.get("artifact_state", {})
    artifact_state = artifact_state if isinstance(artifact_state, dict) else {}
    fixture_requested_by_request = (
        evidence.get("autobot_gap_loop_fixture_requested_slots_by_request", {})
        if isinstance(evidence.get("autobot_gap_loop_fixture_requested_slots_by_request", {}), dict)
        else {}
    )
    fixture_built_by_request = (
        evidence.get("autobot_gap_loop_fixture_built_by_request", {})
        if isinstance(evidence.get("autobot_gap_loop_fixture_built_by_request", {}), dict)
        else {}
    )
    fixture_skipped_by_request = (
        evidence.get("autobot_gap_loop_fixture_skipped_by_request", {})
        if isinstance(evidence.get("autobot_gap_loop_fixture_skipped_by_request", {}), dict)
        else {}
    )
    fixture_breakdown = ", ".join(
        (
            f"{request_id}:"
            f"{int(fixture_built_by_request.get(request_id, 0) or 0)}"
            f"/{int(fixture_requested_by_request.get(request_id, 0) or 0)}"
            f"/{int(fixture_skipped_by_request.get(request_id, 0) or 0)}"
        )
        for request_id in sorted(fixture_requested_by_request)
    )

    lines = [
        f"Research benchmark suite: {'PASS' if manifest.get('passed') else 'FAIL'}",
        f"Dry run: {manifest.get('dry_run')}",
        f"Commands: {manifest.get('passed_count')}/{manifest.get('command_count')}",
        f"Rust core status: {display_artifact_value(evidence.get('rust_core_status'))}",
        format_artifact_state_line(
            "Artifact state",
            [
                ("phase6", artifact_state.get("research_product_completion_gate")),
                ("phase8", artifact_state.get("research_fixture_readiness")),
                ("phase7", artifact_state.get("autobot_gap_loop_readiness")),
            ],
        ),
        (
            "Phase 6 energy metrics: "
            f"research_product_passed={display_artifact_value(evidence.get('research_product_passed'))}, "
            f"v1_release_passed={display_artifact_value(evidence.get('v1_release_passed'))}, "
            f"rust_ready={display_artifact_value(evidence.get('rust_core_status'))}"
        ),
        (
            "Phase 8 baseline metrics: "
            f"fixture_cases={display_artifact_value(evidence.get('research_fixture_case_count'))}, "
            f"profile_count={display_artifact_value(evidence.get('neuromorphic_profile_count'))}, "
            f"own_latent_observed_only={display_artifact_value(evidence.get('own_latent_observed_only'))}, "
            f"comparison_state={display_artifact_value(evidence.get('sara_ann_comparison_status'))}, "
            f"best_offline_reference={display_artifact_value(evidence.get('sara_ann_best_offline_reference'))}"
        ),
        (
            "Phase 7 loop metrics: "
            f"requested_slots={display_artifact_value(evidence.get('autobot_gap_loop_requested_slot_count'))}, "
            f"build_coverage={display_artifact_value(evidence.get('autobot_gap_loop_build_coverage'))}, "
            f"fixture_requests={display_artifact_value(evidence.get('autobot_gap_loop_fixture_request_count'))}, "
            f"fixture_slots={display_artifact_value(evidence.get('autobot_gap_loop_fixture_requested_slot_count'))}, "
            f"fixture_build_coverage={display_artifact_value(evidence.get('autobot_gap_loop_fixture_build_coverage'))}, "
            f"fixture_lineage_coverage={display_artifact_value(evidence.get('autobot_gap_loop_fixture_source_lineage_coverage'))}, "
            f"fixture_breakdown={fixture_breakdown or 'missing_artifact'}, "
            f"fixture_repair_actions={display_artifact_value(evidence.get('autobot_gap_loop_fixture_repair_action_count'))}, "
            f"enqueue_coverage={display_artifact_value(evidence.get('autobot_gap_loop_enqueue_coverage'))}, "
            f"skip_ratio={display_artifact_value(evidence.get('autobot_gap_loop_skip_ratio'))}"
        ),
        (
            "Self-state maintenance: "
            f"persistent_state={display_artifact_value(artifact_state.get('persistent_self_state'))}, "
            f"idle_activity={display_artifact_value(evidence.get('persistent_self_state_idle_activity'))}, "
            f"continuity={display_artifact_value(evidence.get('persistent_self_state_continuity'))}, "
            f"idle_replay={display_artifact_value(artifact_state.get('idle_replay'))}, "
            f"replay_alignment={display_artifact_value(evidence.get('idle_replay_self_state_alignment'))}, "
            f"maintenance_efficiency={display_artifact_value(artifact_state.get('internal_maintenance_efficiency'))}, "
            f"event_cost_per_selected={display_artifact_value(evidence.get('internal_maintenance_event_cost_per_selected'))}"
        ),
        (
            "Event Memory coupling: "
            f"compression_state={display_artifact_value(artifact_state.get('event_memory_ingest_pipeline'))}, "
            f"maintenance_coupling={display_artifact_value(artifact_state.get('event_memory_maintenance_coupling'))}, "
            f"compression_ratio={display_artifact_value(evidence.get('event_memory_episode_compression_ratio'))}, "
            f"relation_yield={display_artifact_value(evidence.get('event_memory_relation_verification_yield'))}, "
            f"bundle_promotion_rate={display_artifact_value(evidence.get('event_memory_multimodal_bundle_promotion_rate'))}, "
            f"bundle_relation_yield={display_artifact_value(evidence.get('event_memory_multimodal_bundle_relation_verification_yield'))}, "
            f"bundle_compression_contribution={display_artifact_value(evidence.get('event_memory_multimodal_bundle_compression_contribution'))}, "
            f"bundle_promotion_count={display_artifact_value(evidence.get('event_memory_multimodal_bundle_promotion_count'))}, "
            f"coupling_best_profile={display_artifact_value(evidence.get('event_memory_maintenance_best_profile'))}, "
            f"coupling_efficiency={display_artifact_value(evidence.get('event_memory_maintenance_best_efficiency'))}, "
            f"coupling_bundle_contribution={display_artifact_value(evidence.get('event_memory_maintenance_best_bundle_compression_contribution'))}, "
            f"coupling_continuity={display_artifact_value(evidence.get('event_memory_maintenance_best_continuity'))}, "
            f"bundle_gap={display_artifact_value(evidence.get('event_memory_bundle_support_gap_present'))}, "
            f"bundle_gap_trigger={display_artifact_value(evidence.get('event_memory_bundle_support_gap_trigger'))}, "
            f"bundle_repair_target={display_artifact_value(evidence.get('event_memory_bundle_support_repair_target'))}, "
            f"bundle_fixture_repairs={display_artifact_value(evidence.get('event_memory_bundle_support_fixture_repair_action_count'))}, "
            f"bundle_fixture_coverage_ready={display_artifact_value(evidence.get('event_memory_bundle_support_fixture_coverage_ready'))}, "
            f"bundle_closed_loop_overlap={display_artifact_value(evidence.get('event_memory_bundle_support_closed_loop_overlap_count'))}, "
            f"bundle_closed_loop_gap_reduction={display_artifact_value(evidence.get('event_memory_bundle_support_closed_loop_gap_reduction'))}, "
            f"bundle_closed_loop_ready={display_artifact_value(evidence.get('event_memory_bundle_support_closed_loop_coverage_ready'))}, "
            f"bundle_overlap_isolation_risk={display_artifact_value(evidence.get('event_memory_bundle_support_overlap_isolation_risk_count'))}, "
            f"bundle_overlap_missing_axes={','.join(str(item) for item in evidence.get('event_memory_bundle_support_overlap_missing_isolation_axes', []) if str(item)) or 'none'}, "
            f"bundle_overlap_highest_risk_axis={display_artifact_value(evidence.get('event_memory_bundle_support_overlap_highest_risk_axis'))}, "
            f"bundle_overlap_risk_priority={display_artifact_value(evidence.get('event_memory_bundle_support_overlap_risk_priority'))}"
        ),
        (
            "Own-latent fixture alignment: "
            f"feedback_loaded={display_artifact_value(evidence.get('own_latent_fixture_feedback_loaded'))}, "
            f"coverage_gaps={display_artifact_value(evidence.get('own_latent_fixture_material_coverage_gap_count'))}, "
            f"requests={display_artifact_value(evidence.get('own_latent_fixture_material_request_count'))}"
        ),
        (
            "Gap closed loop: "
            f"state={display_artifact_value(artifact_state.get('gap_materials_closed_loop'))}, "
            f"passed={display_artifact_value(evidence.get('gap_materials_closed_loop_passed'))}, "
            f"baseline_gaps={display_artifact_value(evidence.get('gap_materials_closed_loop_baseline_gap_count'))}, "
            f"augmented_gaps={display_artifact_value(evidence.get('gap_materials_closed_loop_augmented_gap_count'))}, "
            f"reduction={display_artifact_value(evidence.get('gap_materials_closed_loop_gap_reduction'))}, "
            f"bundle_request_coverage={display_artifact_value(evidence.get('gap_materials_closed_loop_bundle_relevant_request_coverage'))}"
        ),
        (
            "Gap loop readiness: "
            f"state={display_artifact_value(artifact_state.get('autobot_gap_loop_readiness'))}, "
            f"passed={display_artifact_value(evidence.get('autobot_gap_loop_readiness_passed'))}, "
            f"requested_slots={display_artifact_value(evidence.get('autobot_gap_loop_requested_slot_count'))}, "
            f"build_coverage={display_artifact_value(evidence.get('autobot_gap_loop_build_coverage'))}, "
            f"fixture_requests={display_artifact_value(evidence.get('autobot_gap_loop_fixture_request_count'))}, "
            f"fixture_build_coverage={display_artifact_value(evidence.get('autobot_gap_loop_fixture_build_coverage'))}, "
            f"fixture_lineage_coverage={display_artifact_value(evidence.get('autobot_gap_loop_fixture_source_lineage_coverage'))}, "
            f"enqueue_coverage={display_artifact_value(evidence.get('autobot_gap_loop_enqueue_coverage'))}, "
            f"skip_ratio={display_artifact_value(evidence.get('autobot_gap_loop_skip_ratio'))}"
        ),
        (
            "Operational bundle repair: "
            f"log_entries={display_artifact_value(evidence.get('operational_bundle_repair_log_entry_count'))}, "
            f"recovered={display_artifact_value(evidence.get('operational_bundle_repair_log_recovered_count'))}, "
            f"max_gap_reduction={display_artifact_value(evidence.get('operational_bundle_repair_log_max_gap_reduction'))}, "
            f"clear_release_success_count={display_artifact_value(evidence.get('operational_bundle_isolation_clear_release_success_count'))}, "
            f"clear_release_request_ids={','.join(str(item) for item in evidence.get('operational_bundle_isolation_clear_release_request_ids', []) if str(item)) or 'none'}, "
            f"resolved_request_ids={','.join(str(item) for item in evidence.get('operational_bundle_isolation_resolved_request_ids', []) if str(item)) or 'none'}, "
            f"fresh_retry={display_artifact_value(evidence.get('operational_bundle_retry_queue_fresh_count'))}, "
            f"recovered_retry={display_artifact_value(evidence.get('operational_bundle_retry_queue_recovered_before_count'))}, "
            f"churn_retry={display_artifact_value(evidence.get('operational_bundle_retry_queue_isolation_review_churn_count'))}, "
            f"churn_request_ids={','.join(str(item) for item in evidence.get('operational_bundle_retry_queue_isolation_review_churn_request_ids', []) if str(item)) or 'none'}, "
            f"reblocked_retry={display_artifact_value(evidence.get('operational_bundle_retry_queue_isolation_reblocked_count'))}, "
            f"reblocked_request_ids={','.join(str(item) for item in evidence.get('operational_bundle_retry_queue_isolation_reblocked_request_ids', []) if str(item)) or 'none'}, "
            f"phase7_routed_actions={display_artifact_value(evidence.get('operational_bundle_phase7_routed_action_count'))}, "
            f"phase7_routed_retry={display_artifact_value(evidence.get('operational_bundle_phase7_routed_retry_count'))}, "
            f"phase7_isolation_ready={display_artifact_value(evidence.get('operational_bundle_phase7_isolation_ready'))}, "
            f"phase7_lineage_ready={display_artifact_value(evidence.get('operational_bundle_phase7_lineage_ready'))}, "
            f"phase7_collection_time_ready={display_artifact_value(evidence.get('operational_bundle_phase7_collection_time_ready'))}, "
            f"phase7_missing_axes={','.join(str(item) for item in evidence.get('operational_bundle_phase7_missing_isolation_axes', []) if str(item)) or 'none'}, "
            f"blocked_request_count={display_artifact_value(evidence.get('operational_bundle_isolation_blocked_request_count'))}, "
            f"blocked_request_ids={','.join(str(item) for item in evidence.get('operational_bundle_isolation_blocked_request_ids', []) if str(item)) or 'none'}, "
            f"blocked_missing_axes={','.join(str(item) for item in evidence.get('operational_bundle_isolation_blocked_missing_axes', []) if str(item)) or 'none'}, "
            f"overlap_blocked_request_ids={','.join(str(item) for item in evidence.get('operational_bundle_overlap_blocked_request_ids', []) if str(item)) or 'none'}"
        ),
        (
            "Adaptive credit: "
            f"field_state={display_artifact_value(artifact_state.get('adaptive_credit_field'))}, "
            f"field_passed={display_artifact_value(evidence.get('adaptive_credit_field_passed'))}, "
            f"sparse_fraction={display_artifact_value(evidence.get('adaptive_credit_field_sparse_active_fraction'))}, "
            f"quantized_match={display_artifact_value(evidence.get('adaptive_credit_field_quantized_match'))}, "
            f"memory_state={display_artifact_value(artifact_state.get('adaptive_credit_event_memory'))}, "
            f"memory_passed={display_artifact_value(evidence.get('adaptive_credit_event_memory_passed'))}, "
            f"strong_entry={display_artifact_value(evidence.get('adaptive_credit_event_memory_strong_entry_present'))}, "
            f"weak_evicted={display_artifact_value(evidence.get('adaptive_credit_event_memory_weak_entry_evicted'))}, "
            f"operational_visibility={display_artifact_value(evidence.get('adaptive_credit_operational_visibility'))}, "
            f"repair_actions={display_artifact_value(evidence.get('adaptive_credit_operational_repair_action_count'))}, "
            f"primary_focus={display_artifact_value(evidence.get('adaptive_credit_operational_primary_focus'))}, "
            f"repair_log_entries={display_artifact_value(evidence.get('adaptive_credit_repair_log_entry_count'))}, "
            f"repair_log_success={display_artifact_value(evidence.get('adaptive_credit_repair_log_success_count'))}, "
            f"repair_log_pending={display_artifact_value(evidence.get('adaptive_credit_repair_log_pending_count'))}, "
            f"repair_log_failures={display_artifact_value(evidence.get('adaptive_credit_repair_log_failure_count'))}, "
            f"recovered={display_artifact_value(evidence.get('adaptive_credit_repair_log_recovered'))}, "
            f"chronic={display_artifact_value(evidence.get('adaptive_credit_repair_log_chronic'))}"
        ),
        (
            "Concept revalidation: "
            f"cases={display_artifact_value(evidence.get('event_state_cache_concept_revalidation_case_count'))}, "
            f"recovery_rate={display_artifact_value(evidence.get('event_state_cache_concept_revalidation_recovery_rate'))}, "
            f"blocked={display_artifact_value(evidence.get('event_state_cache_concept_revalidation_blocked_count'))}, "
            f"source_diversity_blocked={display_artifact_value(evidence.get('event_state_cache_concept_source_diversity_blocked_count'))}, "
            f"revision_conflict_blocked={display_artifact_value(evidence.get('event_state_cache_concept_revision_conflict_blocked_count'))}, "
            f"counterexample_blocked={display_artifact_value(evidence.get('event_state_cache_concept_counterexample_blocked_count'))}, "
            f"attempt_budget_blocked={display_artifact_value(evidence.get('event_state_cache_concept_attempt_budget_blocked_count'))}"
        ),
        (
            "Concept revalidation fixture builder: "
            f"passed={display_artifact_value(evidence.get('concept_revalidation_fixture_builder_passed'))}, "
            f"cases={display_artifact_value(evidence.get('concept_revalidation_fixture_case_count'))}"
        ),
        "",
        "What is proven:",
    ]
    fixture_case_types = manifest.get("evidence", {}).get(
        "concept_revalidation_fixture_case_type_counts", {}
    )
    if isinstance(fixture_case_types, dict) and fixture_case_types:
        lines.append("Concept revalidation fixture case types:")
        for key, value in sorted(fixture_case_types.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")
    fixture_material_types = manifest.get("evidence", {}).get(
        "concept_revalidation_fixture_manifest_material_type_counts", {}
    )
    if isinstance(fixture_material_types, dict) and fixture_material_types:
        lines.append("Concept revalidation fixture material types:")
        for key, value in sorted(fixture_material_types.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(f"- {item}" for item in manifest.get("what_is_proven", []))
    lines.append("")
    lines.append("What is not proven:")
    lines.extend(f"- {item}" for item in manifest.get("what_is_not_proven", []))

    with open(resolved_summary_path, "w", encoding="utf-8") as handle:
        own_latent_expansion_plan = manifest.get("evidence", {}).get(
            "own_latent_fixture_expansion_plan", []
        )
        if isinstance(own_latent_expansion_plan, list) and own_latent_expansion_plan:
            lines.append("")
            lines.append("Own-latent fixture alignment:")
            for item in own_latent_expansion_plan:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "- "
                    f"{item.get('action', '')} "
                    f"(missing_now={','.join(item.get('missing_material_types_now', []))}, "
                    f"preferred={','.join(item.get('preferred_material_types', []))})"
                )
        integration_next_actions = manifest.get("evidence", {}).get(
            "event_state_cache_concept_next_actions", []
        )
        if isinstance(integration_next_actions, list) and integration_next_actions:
            lines.append("")
            lines.append("Concept revalidation follow-up:")
            for item in integration_next_actions:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "- "
                    f"{item.get('action', '')} "
                    f"(reason={item.get('reason', '')}, priority={item.get('priority', '')})"
                )
        fixture_next_actions = manifest.get("evidence", {}).get(
            "concept_revalidation_fixture_next_actions", []
        )
        if isinstance(fixture_next_actions, list) and fixture_next_actions:
            lines.append("")
            lines.append("Fixture expansion priorities:")
            for item in fixture_next_actions:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "- "
                    f"{item.get('action', '')} "
                    f"(reason={item.get('reason', '')}, priority={item.get('priority', '')}, "
                    f"case_type={item.get('case_type', '')}, case_count={item.get('case_count', '')})"
                )
        fixture_expansion_plan = manifest.get("evidence", {}).get(
            "concept_revalidation_fixture_expansion_plan", []
        )
        if isinstance(fixture_expansion_plan, list) and fixture_expansion_plan:
            lines.append("")
            lines.append("Fixture expansion plan:")
            for item in fixture_expansion_plan:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "- "
                    f"{item.get('action', '')} "
                    f"(preferred={','.join(item.get('preferred_material_types', []))}, "
                    f"missing={','.join(item.get('missing_material_types', []))})"
                )
        handle.write("\n".join(lines) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the compact SARA research benchmark suite.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rust-iterations", type=int, default=50)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.rust_iterations <= 0:
        raise ValueError("rust_iterations must be positive")

    command_results = [
        _run_command(item, dry_run=args.dry_run)
        for item in build_recommended_commands(rust_iterations=args.rust_iterations)
    ]
    manifest = build_manifest(
        command_results=command_results,
        dry_run=args.dry_run,
        rust_iterations=args.rust_iterations,
    )
    write_outputs(manifest, args.manifest_path, args.summary_path)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "manifest_path": os.path.abspath(args.manifest_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
