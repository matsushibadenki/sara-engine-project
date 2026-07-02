# Directory Path: scripts/eval/continual_consolidation_benchmark.py
# English Title: Continual Consolidation Benchmark
# Purpose/Content: Runs a lightweight Stage D benchmark for replay recovery and consolidation retention under CPU-only constraints.

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from sara_engine.inference import SaraInference
from sara_engine.dynamics import PersistentSelfStateController, stable_self_state_id
from sara_engine.ingest import FrequentSequence, make_candidate_relation
from sara_engine.learning.astro_modulator import AstroReplayModulator
from sara_engine.learning.astro_structural_gate import AstroStructuralGateConfig, evaluate_astro_structural_gate
from sara_engine.learning.delta_retention_policy import (
    DeltaRetentionPolicyConfig,
    evaluate_delta_erase_write_decoupling,
    evaluate_delta_retention_policy,
    evaluate_delta_retention_policy_stress,
)
from sara_engine.learning.metabolic_budget import MetabolicBudgetConfig, evaluate_structural_metabolic_budget
from sara_engine.learning.memory_phase import MemoryPhaseConfig, evaluate_memory_phase_transitions
from sara_engine.learning.sleep_consolidation import SleepConsolidationConfig, evaluate_sleep_consolidation
from sara_engine.learning.synaptic_tag import SynapticTagConfig, evaluate_synaptic_tags
from sara_engine.memory.concept_admission import ConceptRevalidationEntry
from sara_engine.memory.event_state_cache import EventStateCandidate, VerifiedHierarchicalEventStateCache
from sara_engine.memory.idle_consolidation_loop import IdleConsolidationLoop
from sara_engine.nn.delta_associative_memory import evaluate_delta_associative_spike_memory
from sara_engine.nn.local_manifold_memory import LocalManifoldTransitionMemory
from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path

from scripts.utils.memory_health import inspect_inference_memory
from scripts.utils.upgrade_memory import upgrade_inference_memory


def _build_engine() -> SaraInference:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {}
    engine.context_index = {}
    engine.refractory_buffer = []
    engine.lif_network = None
    return engine


def _predict_next_token(engine: SaraInference, context_tokens: list[int]) -> Optional[int]:
    key = engine._find_best_matching_key(context_tokens)
    if key is None or key not in engine.direct_map:
        return None
    predicted = engine._sample_next_token(
        key,
        top_k=1,
        temperature=0.0,
        refractory_penalty=1.0,
    )
    return int(predicted) if predicted is not None else None


def _run_replay_recovery_case() -> Dict[str, Any]:
    engine = _build_engine()
    engine.learn_sequence([1, 2, 3])
    distractor_sequences = [[offset, offset + 1, offset + 2] for offset in range(20, 28)]
    for sequence in distractor_sequences:
        engine.learn_sequence(sequence)

    predicted_before = _predict_next_token(engine, [1, 2])

    # Simulate heavy drift by rebuilding memory from unrelated traces only.
    engine.direct_map = {}
    engine.context_index = {}
    for sequence in distractor_sequences:
        engine.learn_sequence(sequence)
    predicted_after_drift = _predict_next_token(engine, [1, 2])

    engine.learn_sequence([1, 2, 3])
    predicted_after_replay = _predict_next_token(engine, [1, 2])

    success = (
        predicted_before == 3
        and predicted_after_drift != 3
        and predicted_after_replay == 3
    )
    return {
        "success": success,
        "predicted_before": predicted_before,
        "predicted_after_drift": predicted_after_drift,
        "predicted_after_replay": predicted_after_replay,
        "description": "Replay should recover a degraded memory path after simulated drift.",
    }


def _run_long_horizon_consolidation_case() -> Dict[str, Any]:
    engine = _build_engine()
    engine.learn_sequence([1, 2, 3])
    engine.learn_sequence([1, 2, 3])
    for offset in range(40, 68):
        engine.learn_sequence([offset, offset + 1, offset + 2])
    engine.learn_sequence([1, 2, 3])
    predicted = _predict_next_token(engine, [1, 2])

    success = predicted == 3
    return {
        "success": success,
        "predicted_token": predicted,
        "expected_token": 3,
        "description": "Replay reinforcement should preserve long-horizon retention under continual updates.",
    }


def _run_counterfactual_replay_case() -> Dict[str, Any]:
    engine = _build_engine()
    engine.learn_sequence([7, 8, 9])
    engine.learn_sequence([7, 8, 10])
    engine.learn_sequence([7, 8, 9])
    engine.learn_sequence([7, 8, 9])
    predicted = _predict_next_token(engine, [7, 8])

    success = predicted == 9
    return {
        "success": success,
        "predicted_token": predicted,
        "expected_token": 9,
        "description": "Counterfactual replay should reinforce the selected branch consistently.",
    }


def _run_upgrade_health_pipeline_case() -> Dict[str, Any]:
    input_path = model_path("tests", "stage_d_pipeline_seed.msgpack")
    upgraded_path = model_path("tests", "stage_d_pipeline_upgraded.msgpack")
    replay_path = workspace_path("evaluation", "stage_d_pipeline_replay.jsonl")
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    os.makedirs(os.path.dirname(replay_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = input_path
    writer.direct_map = {}
    writer.context_index = {}
    writer.retrieval_diagnostics = []
    writer.refractory_buffer = []
    writer.lif_network = None
    writer.learn_sequence([10, 20, 30, 40])
    writer.context_index = {}
    writer.save_pretrained(input_path)

    with open(replay_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"tokens": [10, 20, 30, 40]}, ensure_ascii=False) + "\n")

    upgrade_report = upgrade_inference_memory(
        input_path,
        upgraded_path,
        replay_data_path=replay_path,
    )
    health_report = inspect_inference_memory(upgraded_path)
    health_checks = health_report.get("health_checks", {})
    if not isinstance(health_checks, dict):
        health_checks = {}

    replay_reindex_ok = (
        int(upgrade_report.get("context_count_after", 0)) > int(upgrade_report.get("context_count_before", 0))
        and int(upgrade_report.get("reindex_summary", {}).get("matched_contexts", 0)) >= 1
    )
    memory_health_index_ok = (
        bool(health_checks.get("supports_fuzzy_retrieval", False))
        and bool(health_checks.get("contexts_cover_patterns", False))
        and str(health_report.get("context_encoding", "")) == "stable_v1"
    )

    success = replay_reindex_ok and memory_health_index_ok
    return {
        "success": success,
        "replay_reindex_ok": bool(replay_reindex_ok),
        "memory_health_index_ok": bool(memory_health_index_ok),
        "upgrade_report": {
            "context_count_before": int(upgrade_report.get("context_count_before", 0)),
            "context_count_after": int(upgrade_report.get("context_count_after", 0)),
            "matched_contexts": int(upgrade_report.get("reindex_summary", {}).get("matched_contexts", 0)),
            "unresolved_pattern_count": int(upgrade_report.get("unresolved_pattern_count", 0)),
        },
        "health_report": {
            "context_encoding": str(health_report.get("context_encoding", "")),
            "supports_fuzzy_retrieval": bool(health_checks.get("supports_fuzzy_retrieval", False)),
            "contexts_cover_patterns": bool(health_checks.get("contexts_cover_patterns", False)),
            "has_patterns": bool(health_checks.get("has_patterns", False)),
        },
        "description": "Replay→upgrade-memory→memory-health pipeline should restore indexed retrieval coverage for continual consolidation diagnostics.",
    }


def _run_noisy_replay_resilience_case() -> Dict[str, Any]:
    engine = _build_engine()
    anchor_sequence = [15, 16, 17]
    engine.learn_sequence(anchor_sequence)
    for offset in range(70, 82):
        engine.learn_sequence([offset, offset + 1, offset + 2])
    for _ in range(3):
        # Inject noisy/corrupted replay traces that should not overwrite the anchor path.
        engine.learn_sequence([15, 18, 19])
    engine.learn_sequence(anchor_sequence)
    predicted = _predict_next_token(engine, [15, 16])

    success = predicted == 17
    return {
        "success": success,
        "predicted_token": predicted,
        "expected_token": 17,
        "description": "Noisy replay traces should not collapse the anchor memory path after replay reinforcement.",
    }


def _run_astro_modulation_case() -> Dict[str, Any]:
    modulator = AstroReplayModulator(
        stress_decay=0.85,
        support_recovery=0.25,
        stress_gain=0.25,
    )
    base_weight = 1.0
    replay_trace = [
        {"interference": 0.9, "recovery": 0.1},
        {"interference": 0.8, "recovery": 0.2},
        {"interference": 0.5, "recovery": 0.6},
        {"interference": 0.3, "recovery": 0.9},
    ]

    baseline_weight = base_weight
    baseline_series = []
    modulated_weight = base_weight
    modulated_series = []
    for step in replay_trace:
        interference = float(step["interference"])
        recovery = float(step["recovery"])

        baseline_weight *= max(0.1, 1.0 - 0.35 * interference)
        baseline_series.append(float(baseline_weight))

        modulator.update(
            interference_ratio=interference,
            replay_recovery_signal=recovery,
        )
        base_decay = modulated_weight * max(0.1, 1.0 - 0.35 * interference)
        astro_weight = modulator.modulate_replay_weight(base_decay)
        modulated_weight = 0.6 * base_decay + 0.4 * astro_weight
        modulated_series.append(float(modulated_weight))

    baseline_avg = sum(baseline_series) / max(len(baseline_series), 1)
    modulated_avg = sum(modulated_series) / max(len(modulated_series), 1)
    astro_gain = (modulated_avg - baseline_avg) / max(abs(baseline_avg), 1e-9)
    state = modulator.snapshot()
    success = (
        modulated_weight >= baseline_weight
        and state["stability_level"] > 0.35
        and state["support_level"] >= 0.10
    )
    return {
        "success": success,
        "baseline_weight": float(baseline_weight),
        "modulated_weight": float(modulated_weight),
        "baseline_average_weight": float(baseline_avg),
        "modulated_average_weight": float(modulated_avg),
        "astro_gain_ratio": float(astro_gain),
        "astro_state": state,
        "description": "Astrocyte-inspired slow modulation should improve replay retention stability under interference.",
    }


def _run_manifold_continual_retention_case() -> Dict[str, Any]:
    memory = LocalManifoldTransitionMemory(capacity=3)
    memory.add_trajectory(
        "anchor-release-path",
        source_events=[1, 2, 3],
        next_events=[10, 11],
        correction_events=[11],
        causal_edges=[
            {"from": "goal=release", "to": "status=ready", "support": 0.93},
            {"from": "status=ready", "to": "audit=complete", "support": 0.91},
        ],
        event_cost_proxy=0.20,
    )
    memory.add_trajectory(
        "handoff-path",
        source_events=[10, 11],
        next_events=[20, 21],
        causal_edges=[
            {"from": "audit=complete", "to": "handoff=documented", "support": 0.89},
        ],
        event_cost_proxy=0.18,
    )
    memory.add_trajectory(
        "risk-path",
        source_events=[1, 2],
        next_events=[30],
        correction_events=[30],
        causal_edges=[
            {"from": "goal=release", "to": "risk=pending", "support": 0.86},
        ],
        event_cost_proxy=0.15,
    )
    report = memory.evaluate(
        query_events=[1, 2, 3],
        withheld_expected_events=[10, 11],
        case_specs=[
            {
                "case_id": "anchor",
                "query_events": [1, 2, 3],
                "expected_trajectory_id": "anchor-release-path",
                "expected_next_events": [10, 11],
            },
            {
                "case_id": "handoff",
                "query_events": [10, 11],
                "expected_trajectory_id": "handoff-path",
                "expected_next_events": [20, 21],
            },
            {
                "case_id": "risk",
                "query_events": [1, 2],
                "expected_trajectory_id": "risk-path",
                "expected_next_events": [30],
            },
        ],
    )
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    success = bool(
        metrics.get("manifold_trajectory_case_coverage", 0.0) >= 1.0
        and metrics.get("manifold_average_case_recall", 0.0) >= 1.0
        and metrics.get("causal_route_sparsity", 0.0) >= 1.0
    )
    return {
        "success": success,
        "manifold_report": report,
        "trajectory_count": len(memory.trajectory_graph()),
        "description": "Bounded local manifold memory should preserve sparse trajectory recall across continual updates.",
    }


def _run_delta_associative_memory_case() -> Dict[str, Any]:
    report = evaluate_delta_associative_spike_memory()
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    success = (
        metrics.get("delta_memory_residual_write_integrity", 0.0) >= 1.0
        and metrics.get("delta_memory_retention_gate_stability", 0.0) >= 1.0
        and metrics.get("delta_memory_context_recall_without_text_reinjection", 0.0) >= 1.0
        and metrics.get("delta_memory_state_budget_integrity", 0.0) >= 1.0
        and metrics.get("delta_memory_interference_guard", 0.0) >= 1.0
    )
    return {
        "success": bool(success),
        "delta_memory_report": report,
        "description": "Delta associative spike memory should write only residual events while preserving bounded online state.",
    }


def _run_manifold_capacity_pressure_case() -> Dict[str, Any]:
    memory = LocalManifoldTransitionMemory(capacity=10)
    for index in range(8):
        base = 100 + index * 10
        memory.add_trajectory(
            f"distractor-{index}",
            source_events=[base, base + 1],
            next_events=[base + 2],
            causal_edges=[{"from": f"distractor={index}", "to": "route=unused", "support": 0.88}],
            event_cost_proxy=0.24,
        )
    memory.add_trajectory(
        "capacity-critical-path",
        source_events=[7, 8, 9],
        next_events=[70, 71],
        correction_events=[71],
        causal_edges=[{"from": "capacity=pressured", "to": "trajectory=retained", "support": 0.94}],
        event_cost_proxy=0.12,
    )
    report = memory.evaluate(
        query_events=[7, 8, 9],
        withheld_expected_events=[70, 71],
        scan_budget=2,
        case_specs=[
            {
                "case_id": "capacity-critical",
                "query_events": [7, 8, 9],
                "expected_trajectory_id": "capacity-critical-path",
                "expected_next_events": [70, 71],
            },
        ],
    )
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    success = bool(
        metrics.get("manifold_trajectory_case_coverage", 0.0) >= 1.0
        and metrics.get("manifold_average_case_recall", 0.0) >= 1.0
        and metrics.get("manifold_index_scan_reduction", 0.0) >= 1.0
        and float(report.get("indexed_scan_reduction_ratio", 0.0)) > 0.80
    )
    return {
        "success": success,
        "manifold_report": report,
        "trajectory_count": len(memory.trajectory_graph()),
        "description": "Indexed local manifold memory should preserve recall under distractor-heavy capacity pressure.",
    }


def _run_manifold_replay_refresh_case() -> Dict[str, Any]:
    memory = LocalManifoldTransitionMemory(capacity=3)
    memory.add_trajectory(
        "refresh-anchor-path",
        source_events=[3, 4, 5],
        next_events=[35],
        causal_edges=[{"from": "anchor=old", "to": "anchor=refreshed", "support": 0.93}],
        event_cost_proxy=0.14,
    )
    memory.add_trajectory("refresh-distractor-a", source_events=[100, 101], next_events=[102])
    memory.add_trajectory("refresh-distractor-b", source_events=[110, 111], next_events=[112])

    memory.add_trajectory(
        "refresh-anchor-path",
        source_events=[3, 4, 5],
        next_events=[35],
        causal_edges=[{"from": "anchor=old", "to": "anchor=refreshed", "support": 0.93}],
        event_cost_proxy=0.14,
    )
    memory.add_trajectory("refresh-distractor-c", source_events=[120, 121], next_events=[122])
    memory.add_trajectory("refresh-distractor-d", source_events=[130, 131], next_events=[132])

    trajectory_graph = memory.trajectory_graph()
    graph_ids = [str(item["trajectory_id"]) for item in trajectory_graph]
    anchor_trajectory = next(
        (item for item in trajectory_graph if str(item.get("trajectory_id", "")) == "refresh-anchor-path"),
        {},
    )
    report = memory.evaluate(
        query_events=[3, 4, 5],
        withheld_expected_events=[35],
        scan_budget=2,
        case_specs=[
            {
                "case_id": "replay-refreshed-anchor",
                "query_events": [3, 4, 5],
                "expected_trajectory_id": "refresh-anchor-path",
                "expected_next_events": [35],
            },
        ],
    )
    anchor_retained = "refresh-anchor-path" in graph_ids
    stale_distractors_evicted = "refresh-distractor-a" not in graph_ids and "refresh-distractor-b" not in graph_ids
    anchor_refresh_count = int(anchor_trajectory.get("refresh_count", 0) or 0)
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    success = bool(
        anchor_retained
        and stale_distractors_evicted
        and anchor_refresh_count >= 2
        and metrics.get("manifold_trajectory_case_coverage", 0.0) >= 1.0
        and metrics.get("manifold_average_case_recall", 0.0) >= 1.0
    )
    return {
        "success": success,
        "anchor_retained": bool(anchor_retained),
        "anchor_refresh_count": int(anchor_refresh_count),
        "stale_distractors_evicted": bool(stale_distractors_evicted),
        "trajectory_ids": graph_ids,
        "manifold_report": report,
        "trajectory_count": len(graph_ids),
        "description": "Replay refresh should keep a critical local manifold trajectory inside bounded memory while stale distractors age out.",
    }


def _run_synaptic_tag_case() -> Dict[str, Any]:
    trace = [
        {"step": 1, "pre_id": "goal", "post_id": "release", "pre_step": 1, "post_step": 2, "weight": 0.72, "replayed": True, "replay_useful": True},
        {"step": 3, "pre_id": "goal", "post_id": "release", "pre_step": 3, "post_step": 4, "weight": 0.78, "replayed": True, "replay_useful": True},
        {"step": 5, "pre_id": "goal", "post_id": "release", "pre_step": 5, "post_step": 6, "weight": 0.83, "replayed": True, "replay_useful": True},
        {"step": 7, "pre_id": "goal", "post_id": "release", "pre_step": 7, "post_step": 8, "weight": 0.88, "replayed": True, "replay_useful": True},
        {"step": 2, "pre_id": "noise", "post_id": "branch", "pre_step": 2, "post_step": 8, "weight": 0.10, "replayed": True, "replay_useful": False},
        {"step": 9, "pre_id": "noise", "post_id": "branch", "pre_step": 9, "post_step": 14, "weight": 0.08, "replayed": True, "replay_useful": False},
    ]
    report = evaluate_synaptic_tags(trace, SynapticTagConfig(state_budget=4))
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    tags = report.get("tags", []) if isinstance(report.get("tags"), list) else []
    top_tag = tags[0] if tags else {}
    prune_tags = [tag for tag in tags if isinstance(tag, dict) and bool(tag.get("pruning_candidate", False))]
    success = bool(
        metrics.get("synaptic_tag_integrity", 0.0) >= 1.0
        and str(top_tag.get("tag", "")) == "consolidate"
        and len(prune_tags) >= 1
    )
    return {
        "success": success,
        "synaptic_tag_report": report,
        "top_synaptic_tag": top_tag,
        "pruning_candidate_count": len(prune_tags),
        "description": "Gradient-free synaptic tags should separate replay-worthy connections from pruning candidates using local spike statistics.",
    }


def _run_memory_phase_case() -> Dict[str, Any]:
    observations = [
        {"step": 1, "memory_id": "release-anchor", "stability": 0.20, "replay_success": 0.20, "interference": 0.20},
        {"step": 2, "memory_id": "release-anchor", "stability": 0.55, "replay_success": 0.62, "interference": 0.16},
        {"step": 3, "memory_id": "release-anchor", "stability": 0.86, "replay_success": 0.91, "interference": 0.08},
        {"step": 1, "memory_id": "fresh-context", "stability": 0.16, "replay_success": 0.10, "interference": 0.32},
        {"step": 2, "memory_id": "fresh-context", "stability": 0.22, "replay_success": 0.18, "interference": 0.30},
        {"step": 1, "memory_id": "noisy-distractor", "stability": 0.34, "replay_success": 0.15, "interference": 0.70},
        {"step": 2, "memory_id": "noisy-distractor", "stability": 0.52, "replay_success": 0.20, "interference": 0.68},
    ]
    report = evaluate_memory_phase_transitions(observations, MemoryPhaseConfig(state_budget=4))
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    tracks = report.get("phase_tracks", []) if isinstance(report.get("phase_tracks"), list) else []
    anchor = next(
        (track for track in tracks if isinstance(track, dict) and track.get("memory_id") == "release-anchor"),
        {},
    )
    success = bool(
        metrics.get("memory_phase_transition_integrity", 0.0) >= 1.0
        and metrics.get("memory_phase_overfixation_guard_observed", 0.0) >= 1.0
        and anchor.get("phase_path") == ["liquid", "glass", "crystal"]
    )
    return {
        "success": success,
        "memory_phase_report": report,
        "anchor_phase_path": anchor.get("phase_path", []),
        "description": "Memory phase tracking should protect stable replay memories without overfixing noisy or fresh contexts.",
    }


def _run_metabolic_budget_case() -> Dict[str, Any]:
    operations = [
        {"kind": "grow", "synapse_delta": 2, "event_cost": 0.70, "reserve_cost": 0.18, "importance": 0.82},
        {"kind": "rewire", "synapse_delta": 1, "event_cost": 0.55, "reserve_cost": 0.14, "importance": 0.76},
        {"kind": "grow", "synapse_delta": 2, "event_cost": 0.80, "reserve_cost": 0.20, "importance": 0.68},
        {"kind": "grow", "synapse_delta": 1, "event_cost": 0.70, "reserve_cost": 0.18, "importance": 0.24},
        {"kind": "grow", "synapse_delta": 3, "event_cost": 0.95, "reserve_cost": 0.24, "importance": 0.30},
        {"kind": "prune", "synapse_delta": -2, "event_cost": 0.20, "reserve_cost": 0.02, "importance": 0.12, "reason": "low_importance_under_pressure"},
    ]
    report = evaluate_structural_metabolic_budget(
        operations,
        MetabolicBudgetConfig(max_synapses=6, event_budget=3.40, plasticity_reserve=0.78),
    )
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    rejected = report.get("rejected_operations", []) if isinstance(report.get("rejected_operations"), list) else []
    success = bool(
        metrics.get("metabolic_budget_integrity", 0.0) >= 1.0
        and metrics.get("plasticity_reserve_integrity", 0.0) >= 1.0
        and metrics.get("structural_growth_bounded_observed", 0.0) >= 1.0
        and any(item.get("reason") == "low_importance_under_resource_pressure" for item in rejected if isinstance(item, dict))
    )
    return {
        "success": success,
        "metabolic_budget_report": report,
        "rejected_reason_count": len(rejected),
        "description": "Metabolic budget should bound structural growth while preserving pruning reasons and resource pressure.",
    }


def _run_sleep_consolidation_case() -> Dict[str, Any]:
    replay_events = [
        {
            "memory_id": "release-anchor",
            "baseline_retention": 0.78,
            "post_retention": 0.88,
            "baseline_noise": 0.26,
            "post_noise": 0.18,
            "health_before": 0.76,
            "health_after": 0.86,
            "event_cost": 0.42,
            "latent_branch_count": 3,
            "selected_branch": "stable-release-plan",
        },
        {
            "memory_id": "handoff-branch",
            "baseline_retention": 0.72,
            "post_retention": 0.81,
            "baseline_noise": 0.22,
            "post_noise": 0.17,
            "health_before": 0.74,
            "health_after": 0.80,
            "event_cost": 0.36,
            "latent_branch_count": 2,
            "selected_branch": "documented-handoff",
        },
    ]
    report = evaluate_sleep_consolidation(
        replay_events,
        SleepConsolidationConfig(event_budget=1.0, min_retention=0.70, max_noise=0.30, min_health=0.70),
    )
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    success = bool(
        metrics.get("sleep_consolidation_retention_observed", 0.0) >= 1.0
        and metrics.get("latent_replay_noise_resilience_observed", 0.0) >= 1.0
        and metrics.get("sleep_consolidation_energy_budget_observed", 0.0) >= 1.0
    )
    return {
        "success": success,
        "sleep_consolidation_report": report,
        "description": "Sleep consolidation should improve retention and noise resilience while staying inside an offline replay event budget.",
    }


def _run_astro_structural_gate_case() -> Dict[str, Any]:
    replay_steps = [
        {"world_model_event": "baseline_replay", "prediction_error": 0.22, "replay_recovery": 0.20},
        {"world_model_event": "surprise_transition", "prediction_error": 0.82, "replay_recovery": 0.10},
        {"world_model_event": "counterfactual_replay", "prediction_error": 0.52, "replay_recovery": 0.64},
        {"world_model_event": "stabilized_replay", "prediction_error": 0.16, "replay_recovery": 0.95},
    ]
    report = evaluate_astro_structural_gate(
        replay_steps,
        AstroStructuralGateConfig(max_policy_events=6),
    )
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    success = bool(
        metrics.get("astro_structural_unlock_observed", 0.0) >= 1.0
        and metrics.get("astro_structural_lock_observed", 0.0) >= 1.0
        and metrics.get("world_model_replay_policy_trace_observed", 0.0) >= 1.0
    )
    return {
        "success": success,
        "astro_structural_gate_report": report,
        "description": "Astro modulation should unlock structural plasticity under high prediction error and lock back to bounded STDP after world-model replay stabilizes.",
    }


def _run_delta_retention_policy_case() -> Dict[str, Any]:
    memory_events = [
        {
            "phase": "crystal",
            "astro_stability": 0.96,
            "context_events": [701],
            "predicted_events": [],
            "observed_events": [901],
        },
        {
            "phase": "glass",
            "astro_stability": 0.78,
            "context_events": [711],
            "predicted_events": [],
            "observed_events": [911],
        },
        {
            "phase": "liquid",
            "astro_stability": 0.18,
            "context_events": [721],
            "predicted_events": [],
            "observed_events": [921],
        },
        {
            "phase": "crystal",
            "astro_stability": 0.94,
            "context_events": [701],
            "predicted_events": [901],
            "observed_events": [901],
            "write_gate": 0.0,
        },
    ]
    report = evaluate_delta_retention_policy(
        memory_events,
        DeltaRetentionPolicyConfig(capacity=6),
    )
    stress_histories = [
        {
            "branch_id": "release-anchor",
            "phase": "crystal",
            "astro_stability": 0.98,
            "context_events": [801],
            "predicted_events": [],
            "observed_events": [951],
            "expected_recall_ids": [951],
        },
        {
            "branch_id": "handoff-anchor",
            "phase": "crystal",
            "astro_stability": 0.95,
            "context_events": [802],
            "predicted_events": [],
            "observed_events": [952],
            "expected_recall_ids": [952],
        },
        {
            "branch_id": "volatile-topic",
            "phase": "liquid",
            "astro_stability": 0.12,
            "context_events": [821],
            "predicted_events": [],
            "observed_events": [971],
        },
        {
            "branch_id": "bridge-topic",
            "phase": "glass",
            "astro_stability": 0.80,
            "context_events": [811],
            "predicted_events": [],
            "observed_events": [961],
            "expected_recall_ids": [961],
        },
    ]
    stress_report = evaluate_delta_retention_policy_stress(
        stress_histories,
        DeltaRetentionPolicyConfig(capacity=8),
    )
    erase_write_report = evaluate_delta_erase_write_decoupling(
        [
            {
                "phase": "crystal",
                "astro_stability": 0.98,
                "residual_magnitude": 1.0,
                "context_events": [1001],
                "predicted_events": [],
                "observed_events": [2001],
                "expected_write_ids": [2001],
            },
            {
                "phase": "crystal",
                "astro_stability": 0.98,
                "residual_magnitude": 0.05,
                "context_events": [1001],
                "predicted_events": [2001],
                "observed_events": [2001],
                "expected_stable_ids": [2001],
            },
            {
                "phase": "glass",
                "astro_stability": 0.82,
                "residual_magnitude": 0.95,
                "context_events": [1002],
                "predicted_events": [2002],
                "observed_events": [2002, 2003],
                "expected_write_ids": [2003],
            },
        ],
        DeltaRetentionPolicyConfig(capacity=8),
    )
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    stress_metrics = stress_report.get("metrics", {}) if isinstance(stress_report.get("metrics"), dict) else {}
    erase_write_metrics = (
        erase_write_report.get("metrics", {})
        if isinstance(erase_write_report.get("metrics"), dict)
        else {}
    )
    success = bool(
        metrics.get("delta_memory_phase_retention_policy_observed", 0.0) >= 1.0
        and metrics.get("delta_memory_crystal_retention_observed", 0.0) >= 1.0
        and metrics.get("delta_memory_liquid_forget_observed", 0.0) >= 1.0
        and stress_metrics.get("delta_memory_multi_history_recall_observed", 0.0) >= 1.0
        and stress_metrics.get("delta_memory_multi_history_noise_resilience_observed", 0.0) >= 1.0
        and stress_metrics.get("delta_memory_multi_history_health_observed", 0.0) >= 1.0
        and stress_metrics.get("delta_memory_multi_history_manifold_guard_observed", 0.0) >= 1.0
        and erase_write_metrics.get("delta_memory_erase_write_decoupling_observed", 0.0) >= 1.0
        and erase_write_metrics.get("delta_memory_erase_preserves_stable_memory_observed", 0.0) >= 1.0
        and erase_write_metrics.get("delta_memory_write_commits_residual_observed", 0.0) >= 1.0
    )
    return {
        "success": success,
        "delta_retention_policy_report": report,
        "delta_retention_policy_stress_report": stress_report,
        "delta_erase_write_decoupling_report": erase_write_report,
        "description": "Delta retention policy should preserve crystal memory while decaying volatile liquid context through astro- and phase-modulated gates.",
    }


def _run_idle_maintenance_trace_case() -> Dict[str, Any]:
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic")
    cache.admit(
        EventStateCandidate(
            entry_id="concept-memory",
            signature=(21, 23, 27),
            source_ref="concept:aligned",
            time_segment=1,
            own_latent_id="predicts:vision:visual_cluster_018->audio:audio_cluster_044",
            confidence=0.9,
            uncertainty=0.1,
            source_reliability=0.9,
            resonance_score=0.82,
            sequence_support_score=0.45,
            sequence_support_count=2,
            metabolic_headroom=0.8,
            observed=True,
            source_backed=True,
            verified=True,
        )
    )
    queue = [
        ConceptRevalidationEntry(
            concept_key="predicts:vision:visual_cluster_018->audio:audio_cluster_044",
            decision="quarantine_source_revision_conflict",
            supporting_relation_ids=("predicts:vision:visual_cluster_018->audio:audio_cluster_044",),
            source_refs=("episode-1",),
            source_hashes=("hash-a",),
            revision_conflict_count=1,
            contradiction_score=0.2,
            next_action="wait",
            attempt_count=0,
            blocked_at_segment=3,
            last_review_segment=3,
            retry_after_segment=4,
        )
    ]
    relations = [
        make_candidate_relation(
            {
                "record_id": "rel-1",
                "relation": "predicts",
                "source_event_id": "vision:visual_cluster_018",
                "target_event_id": "audio:audio_cluster_044",
                "delay_lower_ms": 60,
                "delay_upper_ms": 140,
                "confidence": 0.88,
                "source_ref": "episode-1",
                "source_hash": "hash-a",
                "extractor_name": "prediction_gain",
                "extractor_version": "v1",
                "evidence_count": 5,
                "counterexample_count": 0,
                "prediction_gain": 0.18,
            }
        ),
        make_candidate_relation(
            {
                "record_id": "rel-2",
                "relation": "predicts",
                "source_event_id": "vision:visual_cluster_018",
                "target_event_id": "audio:audio_cluster_044",
                "delay_lower_ms": 62,
                "delay_upper_ms": 138,
                "confidence": 0.89,
                "source_ref": "episode-2",
                "source_hash": "hash-b",
                "extractor_name": "prediction_gain",
                "extractor_version": "v1",
                "evidence_count": 5,
                "counterexample_count": 0,
                "prediction_gain": 0.19,
            }
        ),
    ]
    sequences = [
        FrequentSequence(
            sequence_key="visual_cluster_018 -> audio_cluster_044",
            labels=("visual_cluster_018", "audio_cluster_044"),
            support_episode_count=2,
            occurrence_count=2,
            source_count=2,
            mean_span_ms=50.0,
            parent_episode_ids=("episode-1", "episode-2"),
        )
    ]
    controller = PersistentSelfStateController(
        core_event_ids=(
            stable_self_state_id("vision:visual_cluster_018"),
            stable_self_state_id("audio:audio_cluster_044"),
        )
    )
    modulator = AstroReplayModulator()
    loop_result = IdleConsolidationLoop().run(
        cache,
        queue,
        relations,
        current_segment=6,
        frequent_sequences=sequences,
        persistent_self_state=controller,
        astro_modulator=modulator,
        sleep_config=SleepConsolidationConfig(event_budget=12.0),
        memory_phase_config=MemoryPhaseConfig(state_budget=4),
        delta_retention_config=DeltaRetentionPolicyConfig(capacity=6),
    ).to_dict()

    selected = loop_result.get("idle_replay_report", {}).get("selected", [])
    phase_tracks = loop_result.get("memory_phase_report", {}).get("phase_tracks", [])
    refresh = loop_result.get("cache_refresh", [])
    delta_metrics = loop_result.get("delta_retention_policy_report", {}).get("metrics", {})
    success = bool(
        selected
        and phase_tracks
        and refresh
        and loop_result.get("sleep_consolidation_report", {}).get("event_budget_ok", False)
        and float(delta_metrics.get("delta_memory_policy_state_budget_observed", 0.0)) >= 1.0
    )
    return {
        "success": success,
        "idle_consolidation_loop_report": loop_result,
        "selected_count": len(selected),
        "phase_count": len(phase_tracks),
        "refresh_count": len(refresh),
        "bundle_candidate_count": int(
            loop_result.get("multimodal_bundle_summary", {}).get("bundle_candidate_count", 0) or 0
        ),
        "bundle_candidate_ratio": float(
            loop_result.get("multimodal_bundle_summary", {}).get("bundle_candidate_ratio", 0.0) or 0.0
        ),
        "description": "Integrated idle maintenance should expose one auditable trace from replay selection through phase-aware retention and cache refresh.",
    }


def run_continual_consolidation_benchmark() -> Dict[str, Any]:
    replay_recovery = _run_replay_recovery_case()
    long_horizon = _run_long_horizon_consolidation_case()
    counterfactual = _run_counterfactual_replay_case()
    upgrade_health_pipeline = _run_upgrade_health_pipeline_case()
    noisy_replay_resilience = _run_noisy_replay_resilience_case()
    astro_modulation = _run_astro_modulation_case()
    delta_associative_memory = _run_delta_associative_memory_case()
    manifold_continual_retention = _run_manifold_continual_retention_case()
    manifold_capacity_pressure = _run_manifold_capacity_pressure_case()
    manifold_replay_refresh = _run_manifold_replay_refresh_case()
    synaptic_tag = _run_synaptic_tag_case()
    memory_phase = _run_memory_phase_case()
    metabolic_budget = _run_metabolic_budget_case()
    sleep_consolidation = _run_sleep_consolidation_case()
    astro_structural_gate = _run_astro_structural_gate_case()
    delta_retention_policy = _run_delta_retention_policy_case()
    idle_maintenance_trace = _run_idle_maintenance_trace_case()
    cases = [
        replay_recovery,
        long_horizon,
        counterfactual,
        upgrade_health_pipeline,
        noisy_replay_resilience,
        astro_modulation,
        delta_associative_memory,
        manifold_continual_retention,
        manifold_capacity_pressure,
        manifold_replay_refresh,
        synaptic_tag,
        memory_phase,
        metabolic_budget,
        sleep_consolidation,
        astro_structural_gate,
        delta_retention_policy,
        idle_maintenance_trace,
    ]

    metrics = {
        "replay_recovery_integrity": 1.0 if replay_recovery["success"] else 0.0,
        "long_horizon_consolidation_retention": 1.0 if long_horizon["success"] else 0.0,
        "counterfactual_replay_selection_integrity": 1.0 if counterfactual["success"] else 0.0,
        "replay_upgrade_reindex_integrity": 1.0 if upgrade_health_pipeline["replay_reindex_ok"] else 0.0,
        "memory_health_index_integrity": 1.0 if upgrade_health_pipeline["memory_health_index_ok"] else 0.0,
        "replay_noise_resilience_integrity": 1.0 if noisy_replay_resilience["success"] else 0.0,
        "astro_modulation_stability": 1.0 if astro_modulation["success"] else 0.0,
        "delta_memory_residual_write_integrity_observed": float(
            delta_associative_memory["delta_memory_report"]["metrics"].get(
                "delta_memory_residual_write_integrity",
                0.0,
            )
        ),
        "delta_memory_retention_gate_stability_observed": float(
            delta_associative_memory["delta_memory_report"]["metrics"].get(
                "delta_memory_retention_gate_stability",
                0.0,
            )
        ),
        "delta_memory_context_recall_without_text_reinjection_observed": float(
            delta_associative_memory["delta_memory_report"]["metrics"].get(
                "delta_memory_context_recall_without_text_reinjection",
                0.0,
            )
        ),
        "delta_memory_state_budget_integrity_observed": float(
            delta_associative_memory["delta_memory_report"]["metrics"].get(
                "delta_memory_state_budget_integrity",
                0.0,
            )
        ),
        "delta_memory_interference_guard_observed": float(
            delta_associative_memory["delta_memory_report"]["metrics"].get(
                "delta_memory_interference_guard",
                0.0,
            )
        ),
        "manifold_continual_retention_observed": 1.0 if manifold_continual_retention["success"] else 0.0,
        "manifold_trajectory_case_coverage_observed": float(
            manifold_continual_retention["manifold_report"]["metrics"].get("manifold_trajectory_case_coverage", 0.0)
        ),
        "manifold_average_case_recall_observed": float(
            manifold_continual_retention["manifold_report"]["metrics"].get("manifold_average_case_recall", 0.0)
        ),
        "manifold_scan_budget_integrity_observed": float(
            manifold_continual_retention["manifold_report"]["metrics"].get("manifold_scan_budget_integrity", 0.0)
        ),
        "manifold_indexed_candidate_integrity_observed": float(
            manifold_continual_retention["manifold_report"]["metrics"].get(
                "manifold_indexed_candidate_integrity",
                0.0,
            )
        ),
        "manifold_index_scan_reduction_observed": float(
            manifold_continual_retention["manifold_report"]["metrics"].get(
                "manifold_index_scan_reduction",
                0.0,
            )
        ),
        "manifold_capacity_pressure_recall_observed": 1.0 if manifold_capacity_pressure["success"] else 0.0,
        "manifold_capacity_pressure_scan_reduction_observed": float(
            manifold_capacity_pressure["manifold_report"].get("indexed_scan_reduction_ratio", 0.0)
        ),
        "manifold_replay_refresh_retention_observed": 1.0 if manifold_replay_refresh["success"] else 0.0,
        "manifold_replay_refresh_eviction_integrity_observed": (
            1.0 if manifold_replay_refresh["stale_distractors_evicted"] else 0.0
        ),
        "synaptic_tag_integrity_observed": float(
            synaptic_tag["synaptic_tag_report"]["metrics"].get("synaptic_tag_integrity", 0.0)
        ),
        "synaptic_tag_importance_score_observed": float(
            synaptic_tag["synaptic_tag_report"]["metrics"].get(
                "synaptic_tag_importance_score_observed",
                0.0,
            )
        ),
        "synaptic_tag_replay_priority_observed": float(
            synaptic_tag["synaptic_tag_report"]["metrics"].get(
                "synaptic_tag_replay_priority_observed",
                0.0,
            )
        ),
        "synaptic_tag_pruning_candidate_observed": float(
            synaptic_tag["synaptic_tag_report"]["metrics"].get(
                "synaptic_tag_pruning_candidate_observed",
                0.0,
            )
        ),
        "synaptic_tag_state_budget_observed": float(
            synaptic_tag["synaptic_tag_report"]["metrics"].get(
                "synaptic_tag_state_budget_observed",
                0.0,
            )
        ),
        "memory_phase_transition_integrity_observed": float(
            memory_phase["memory_phase_report"]["metrics"].get("memory_phase_transition_integrity", 0.0)
        ),
        "memory_phase_retention_protection_observed": float(
            memory_phase["memory_phase_report"]["metrics"].get(
                "memory_phase_retention_protection_observed",
                0.0,
            )
        ),
        "memory_phase_plasticity_guard_observed": float(
            memory_phase["memory_phase_report"]["metrics"].get(
                "memory_phase_plasticity_guard_observed",
                0.0,
            )
        ),
        "memory_phase_overfixation_guard_observed": float(
            memory_phase["memory_phase_report"]["metrics"].get(
                "memory_phase_overfixation_guard_observed",
                0.0,
            )
        ),
        "memory_phase_state_budget_observed": float(
            memory_phase["memory_phase_report"]["metrics"].get(
                "memory_phase_state_budget_observed",
                0.0,
            )
        ),
        "metabolic_budget_integrity_observed": float(
            metabolic_budget["metabolic_budget_report"]["metrics"].get("metabolic_budget_integrity", 0.0)
        ),
        "plasticity_reserve_integrity_observed": float(
            metabolic_budget["metabolic_budget_report"]["metrics"].get("plasticity_reserve_integrity", 0.0)
        ),
        "structural_growth_bounded_observed": float(
            metabolic_budget["metabolic_budget_report"]["metrics"].get("structural_growth_bounded_observed", 0.0)
        ),
        "pruning_reason_trace_observed": float(
            metabolic_budget["metabolic_budget_report"]["metrics"].get("pruning_reason_trace_observed", 0.0)
        ),
        "resource_pressure_observed": float(
            metabolic_budget["metabolic_budget_report"]["metrics"].get("resource_pressure_observed", 0.0)
        ),
        "sleep_consolidation_retention_observed": float(
            sleep_consolidation["sleep_consolidation_report"]["metrics"].get(
                "sleep_consolidation_retention_observed",
                0.0,
            )
        ),
        "latent_replay_noise_resilience_observed": float(
            sleep_consolidation["sleep_consolidation_report"]["metrics"].get(
                "latent_replay_noise_resilience_observed",
                0.0,
            )
        ),
        "sleep_consolidation_memory_health_observed": float(
            sleep_consolidation["sleep_consolidation_report"]["metrics"].get(
                "sleep_consolidation_memory_health_observed",
                0.0,
            )
        ),
        "latent_replay_counterfactual_branch_observed": float(
            sleep_consolidation["sleep_consolidation_report"]["metrics"].get(
                "latent_replay_counterfactual_branch_observed",
                0.0,
            )
        ),
        "sleep_consolidation_energy_budget_observed": float(
            sleep_consolidation["sleep_consolidation_report"]["metrics"].get(
                "sleep_consolidation_energy_budget_observed",
                0.0,
            )
        ),
        "astro_structural_unlock_observed": float(
            astro_structural_gate["astro_structural_gate_report"]["metrics"].get(
                "astro_structural_unlock_observed",
                0.0,
            )
        ),
        "astro_structural_lock_observed": float(
            astro_structural_gate["astro_structural_gate_report"]["metrics"].get(
                "astro_structural_lock_observed",
                0.0,
            )
        ),
        "astro_bounded_stdp_fallback_observed": float(
            astro_structural_gate["astro_structural_gate_report"]["metrics"].get(
                "astro_bounded_stdp_fallback_observed",
                0.0,
            )
        ),
        "world_model_replay_policy_trace_observed": float(
            astro_structural_gate["astro_structural_gate_report"]["metrics"].get(
                "world_model_replay_policy_trace_observed",
                0.0,
            )
        ),
        "astro_policy_state_budget_observed": float(
            astro_structural_gate["astro_structural_gate_report"]["metrics"].get(
                "astro_policy_state_budget_observed",
                0.0,
            )
        ),
        "delta_memory_phase_retention_policy_observed": float(
            delta_retention_policy["delta_retention_policy_report"]["metrics"].get(
                "delta_memory_phase_retention_policy_observed",
                0.0,
            )
        ),
        "delta_memory_crystal_retention_observed": float(
            delta_retention_policy["delta_retention_policy_report"]["metrics"].get(
                "delta_memory_crystal_retention_observed",
                0.0,
            )
        ),
        "delta_memory_liquid_forget_observed": float(
            delta_retention_policy["delta_retention_policy_report"]["metrics"].get(
                "delta_memory_liquid_forget_observed",
                0.0,
            )
        ),
        "delta_memory_astro_gate_alignment_observed": float(
            delta_retention_policy["delta_retention_policy_report"]["metrics"].get(
                "delta_memory_astro_gate_alignment_observed",
                0.0,
            )
        ),
        "delta_memory_policy_state_budget_observed": float(
            delta_retention_policy["delta_retention_policy_report"]["metrics"].get(
                "delta_memory_policy_state_budget_observed",
                0.0,
            )
        ),
        "delta_memory_multi_history_recall_observed": float(
            delta_retention_policy["delta_retention_policy_stress_report"]["metrics"].get(
                "delta_memory_multi_history_recall_observed",
                0.0,
            )
        ),
        "delta_memory_multi_history_noise_resilience_observed": float(
            delta_retention_policy["delta_retention_policy_stress_report"]["metrics"].get(
                "delta_memory_multi_history_noise_resilience_observed",
                0.0,
            )
        ),
        "delta_memory_multi_history_health_observed": float(
            delta_retention_policy["delta_retention_policy_stress_report"]["metrics"].get(
                "delta_memory_multi_history_health_observed",
                0.0,
            )
        ),
        "delta_memory_multi_history_manifold_guard_observed": float(
            delta_retention_policy["delta_retention_policy_stress_report"]["metrics"].get(
                "delta_memory_multi_history_manifold_guard_observed",
                0.0,
            )
        ),
        "delta_memory_erase_write_decoupling_observed": float(
            delta_retention_policy["delta_erase_write_decoupling_report"]["metrics"].get(
                "delta_memory_erase_write_decoupling_observed",
                0.0,
            )
        ),
        "delta_memory_erase_preserves_stable_memory_observed": float(
            delta_retention_policy["delta_erase_write_decoupling_report"]["metrics"].get(
                "delta_memory_erase_preserves_stable_memory_observed",
                0.0,
            )
        ),
        "delta_memory_write_commits_residual_observed": float(
            delta_retention_policy["delta_erase_write_decoupling_report"]["metrics"].get(
                "delta_memory_write_commits_residual_observed",
                0.0,
            )
        ),
        "idle_maintenance_trace_integrity_observed": 1.0 if idle_maintenance_trace["success"] else 0.0,
        "idle_maintenance_phase_alignment_observed": float(
            bool(idle_maintenance_trace["phase_count"] >= idle_maintenance_trace["selected_count"] >= 1)
        ),
        "idle_maintenance_cache_refresh_observed": float(
            bool(idle_maintenance_trace["refresh_count"] >= 1)
        ),
        "idle_maintenance_multimodal_bundle_visibility_observed": float(
            "multimodal_bundle_summary" in idle_maintenance_trace["idle_consolidation_loop_report"]
        ),
    }
    thresholds = {
        "replay_recovery_integrity": 1.0,
        "long_horizon_consolidation_retention": 1.0,
        "counterfactual_replay_selection_integrity": 1.0,
        "replay_upgrade_reindex_integrity": 1.0,
        "memory_health_index_integrity": 1.0,
        "replay_noise_resilience_integrity": 1.0,
        "astro_modulation_stability": 1.0,
    }
    threshold_results = {
        name: metrics.get(name, 0.0) >= threshold
        for name, threshold in thresholds.items()
    }

    required_metric_values = [float(metrics.get(name, 0.0)) for name in thresholds]

    return {
        "evaluator_name": "ContinualConsolidationBenchmark",
        "overall_score": sum(required_metric_values) / max(len(required_metric_values), 1),
        "metrics": metrics,
        "details": {"test_results": cases},
        "thresholds": thresholds,
        "threshold_results": threshold_results,
        "passed": all(threshold_results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight continual-consolidation benchmark.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "continual_consolidation_benchmark.json"),
        help="Managed output path for the benchmark report.",
    )
    args = parser.parse_args()

    report = run_continual_consolidation_benchmark()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("Continual-consolidation benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
