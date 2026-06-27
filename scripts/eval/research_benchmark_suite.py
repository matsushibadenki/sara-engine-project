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
from typing import Any, Dict, List, Optional, Sequence


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


def _load_json_if_present(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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
        "Equal-modality sparse binding and non-language sensory-substitution routes are recorded as observed-only evidence.",
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
            "autobot_gap_loop_readiness_passed": None
            if autobot_gap_loop_readiness_report is None
            else bool(autobot_gap_loop_readiness_report.get("passed")),
            "autobot_gap_loop_requested_slot_count": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("requested_slot_count"),
            "autobot_gap_loop_build_coverage": None
            if autobot_gap_loop_readiness_report is None
            else autobot_gap_loop_readiness_report.get("metrics", {}).get("gap_build_coverage"),
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
            f"coupling_best_profile={display_artifact_value(evidence.get('event_memory_maintenance_best_profile'))}, "
            f"coupling_efficiency={display_artifact_value(evidence.get('event_memory_maintenance_best_efficiency'))}, "
            f"coupling_continuity={display_artifact_value(evidence.get('event_memory_maintenance_best_continuity'))}"
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
            f"reduction={display_artifact_value(evidence.get('gap_materials_closed_loop_gap_reduction'))}"
        ),
        (
            "Gap loop readiness: "
            f"state={display_artifact_value(artifact_state.get('autobot_gap_loop_readiness'))}, "
            f"passed={display_artifact_value(evidence.get('autobot_gap_loop_readiness_passed'))}, "
            f"requested_slots={display_artifact_value(evidence.get('autobot_gap_loop_requested_slot_count'))}, "
            f"build_coverage={display_artifact_value(evidence.get('autobot_gap_loop_build_coverage'))}, "
            f"enqueue_coverage={display_artifact_value(evidence.get('autobot_gap_loop_enqueue_coverage'))}, "
            f"skip_ratio={display_artifact_value(evidence.get('autobot_gap_loop_skip_ratio'))}"
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
