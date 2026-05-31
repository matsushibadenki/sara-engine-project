# Directory Path: scripts/eval/cognitive_runtime_benchmark.py
# English Title: Cognitive Runtime Benchmark
# Purpose/Content: Evaluates Stage E common-spike-space and high-order reasoning primitives with CPU-only sparse events.

import argparse
import json
import os
import sys
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

MPL_CACHE_PATH = os.path.join(PROJECT_ROOT, "workspace", "matplotlib")
os.makedirs(MPL_CACHE_PATH, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_PATH)
XDG_CACHE_PATH = os.path.join(PROJECT_ROOT, "workspace", "cache")
os.makedirs(XDG_CACHE_PATH, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", XDG_CACHE_PATH)


from sara_engine.nn.common_spike_space import (
    CommonSpikeSpaceEncoder,
    DendriticContextGate,
    ModalityTemporalBudget,
    ModularCognitiveRuntime,
    TemporalCompressionPolicy,
    build_causal_candidate_trace,
    build_event_relation_trace,
    build_runtime_trace_digest,
    build_spiking_hjepa_transition_trace,
    compare_spiking_hjepa_transition_branches,
    compare_runtime_trace_digests,
    evaluate_lejepa_sparse_latent_health_trace,
    evaluate_micro_turn_interaction_trace,
    evaluate_phase_assigned_submodel_block_trace,
    build_reverse_reasoning_trace,
)
from sara_engine.nn.delta_associative_memory import evaluate_delta_memory_steering_trace
from sara_engine.nn.forward_only_local_update import evaluate_forward_only_local_update_trace
from sara_engine.nn.local_manifold_memory import LocalManifoldTransitionMemory
from sara_engine.nn.multi_timescale_leak_state import MultiTimescaleLeakState
from sara_engine.nn.phase_synchronized_binding_trace import (
    evaluate_phase_synchronized_binding_trace,
)
from sara_engine.nn.plastic_submodel_registry import (
    evaluate_plastic_submodel_credit_assignment_trace,
    evaluate_plastic_submodel_intervention_trace,
    evaluate_plastic_submodel_open_ended_hypothesis_bank_trace,
    evaluate_plastic_submodel_registry_trace,
    evaluate_plastic_submodel_scientific_model_trace,
    evaluate_plastic_submodel_structural_adaptation_trace,
)
from sara_engine.nn.sparse_verifier import (
    evaluate_bounded_tree_search_trace,
    evaluate_hierarchical_reasoning_trace,
    evaluate_reasoning_forest_lane_trace,
    evaluate_self_correction_trace,
    evaluate_sparse_best_of_n_trace,
    evaluate_sparse_verifier_trace,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _spike_ids(events: List[Any]) -> List[int]:
    return sorted(
        {
            int(event.spike_id)
            for event in events
            if hasattr(event, "spike_id")
        }
    )


def _predictive_error_gated_spike_case(
    *,
    predicted_events: List[Any],
    observed_events: List[Any],
    surprise_events: List[Any],
) -> Dict[str, Any]:
    predicted_ids = set(_spike_ids(predicted_events))
    observed_ids = set(_spike_ids(observed_events))
    surprise_ids = set(_spike_ids(surprise_events))
    expected_residual_ids = sorted(observed_ids.difference(predicted_ids))
    surprise_residual_ids = sorted(surprise_ids.difference(predicted_ids))

    correction_state = MultiTimescaleLeakState(max_state_units=12)
    expected_trace = correction_state.update(input_events=expected_residual_ids)
    surprise_trace = correction_state.update(input_events=surprise_residual_ids)
    expected_event_cost = len(expected_residual_ids)
    surprise_event_cost = len(surprise_residual_ids)
    entropy_reduction = (
        1.0 - (float(expected_event_cost) / float(max(surprise_event_cost, 1)))
        if surprise_event_cost > 0
        else 0.0
    )
    return {
        "observed_only": True,
        "predicted_ids": sorted(predicted_ids),
        "observed_ids": sorted(observed_ids),
        "surprise_ids": sorted(surprise_ids),
        "expected_residual_ids": expected_residual_ids,
        "surprise_residual_ids": surprise_residual_ids,
        "expected_correction_spikes": expected_event_cost,
        "surprise_correction_spikes": surprise_event_cost,
        "entropy_reduction": float(entropy_reduction),
        "state_budget_ok": bool(
            expected_trace.get("state_budget_ok", False)
            and surprise_trace.get("state_budget_ok", False)
        ),
        "expected_trace": expected_trace,
        "surprise_trace": surprise_trace,
        "state_snapshot": correction_state.snapshot(),
    }


def run_cognitive_runtime_benchmark() -> Dict[str, Any]:
    encoder = CommonSpikeSpaceEncoder(dimension=2048, active_bits=3)
    text_events = encoder.encode_text(
        "ship release after pytest gate passes",
        timestep=0,
        confidence=0.92,
    )
    state_events = encoder.encode_structured_state(
        {
            "goal": "release",
            "risk": "pytest",
            "status": "needs_gate",
        },
        timestep=1,
        confidence=0.88,
    )
    image_adapter_events = encoder.encode_adapter_features(
        "image",
        ["release_panel", "green_check"],
        timestep=2,
        confidence=0.60,
    )
    all_events = text_events + state_events + image_adapter_events

    schema_ok = all(
        event.modality
        and 0 <= event.spike_id < encoder.dimension
        and event.timestep >= 0
        and event.channel
        for event in all_events
    )
    modality_coverage = {event.modality for event in all_events}
    common_spike_space_integrity = 1.0 if schema_ok and {"text", "state", "image"}.issubset(modality_coverage) else 0.0

    compressor = TemporalCompressionPolicy(max_window=4, max_events_per_modality=10)
    compressed_events, compression_report = compressor.compress(all_events)
    compressed_modalities = {event.modality for event in compressed_events}
    temporal_compression_efficiency = 1.0 if (
        compressed_events
        and len(compressed_events) <= len(all_events)
        and compression_report["max_timestep"] <= 3.0
        and modality_coverage.issubset(compressed_modalities)
    ) else 0.0

    budgeter = ModalityTemporalBudget(max_budget=6)
    text_budget = budgeter.allocate("text", confidence=0.90, surprise=0.10)
    image_budget = budgeter.allocate("image", confidence=0.40, surprise=0.80)
    state_budget = budgeter.allocate("state", confidence=0.95, surprise=0.20)
    budget_reports = [text_budget, image_budget, state_budget]
    modality_temporal_budget_integrity = 1.0 if (
        all(bool(report["bounded"]) for report in budget_reports)
        and image_budget["budget"] >= state_budget["budget"]
        and len({report["budget"] for report in budget_reports}) >= 2
    ) else 0.0

    gate = DendriticContextGate(short_term_limit=12, long_term_limit=18, error_limit=8)
    first_gate_report = gate.update(compressed_events, consolidate=True)
    predicted_events = compressed_events[: max(1, len(compressed_events) - 2)]
    second_gate_report = gate.update(compressed_events, prediction_events=predicted_events, consolidate=False)
    dendritic_context_gate_stability = 1.0 if (
        first_gate_report["long_term_count"] > 0
        and second_gate_report["prediction_error_count"] <= 8
        and second_gate_report["context_stability"] >= 1.0
    ) else 0.0

    latent_source_events = encoder.encode_structured_state(
        {"goal": "release", "status": "needs_gate"},
        timestep=0,
        confidence=0.92,
    )
    latent_predicted_events = encoder.encode_structured_state(
        {"status": "release_ready"},
        timestep=1,
        confidence=0.88,
    )
    latent_observed_events = encoder.encode_structured_state(
        {"status": "release_ready", "audit": "complete"},
        timestep=2,
        confidence=0.90,
    )
    latent_correction_events = encoder.encode_structured_state(
        {"audit": "complete"},
        timestep=3,
        confidence=0.94,
    )
    counterfactual_predicted_events = encoder.encode_structured_state(
        {"status": "release_deferred"},
        timestep=1,
        confidence=0.70,
    )
    counterfactual_observed_events = encoder.encode_structured_state(
        {"status": "release_deferred", "risk": "pytest_pending"},
        timestep=2,
        confidence=0.72,
    )
    counterfactual_correction_events = encoder.encode_structured_state(
        {"risk": "pytest_pending"},
        timestep=3,
        confidence=0.75,
    )
    latent_transition = build_spiking_hjepa_transition_trace(
        source_events=latent_source_events,
        predicted_events=latent_predicted_events,
        observed_events=latent_observed_events,
        correction_events=latent_correction_events,
        operator="release_gate.latent_transition",
        branch_id="primary",
    )
    counterfactual_transition = build_spiking_hjepa_transition_trace(
        source_events=latent_source_events,
        predicted_events=counterfactual_predicted_events,
        observed_events=counterfactual_observed_events,
        correction_events=counterfactual_correction_events,
        operator="release_gate.counterfactual_transition",
        branch_id="counterfactual-1",
    )
    latent_branch_comparison = compare_spiking_hjepa_transition_branches(latent_transition, counterfactual_transition)
    lejepa_latent_health = evaluate_lejepa_sparse_latent_health_trace(
        latent_transition,
        counterfactual_transition,
        latent_branch_comparison,
    )
    lejepa_latent_health_metrics = (
        lejepa_latent_health.get("metrics", {})
        if isinstance(lejepa_latent_health.get("metrics"), dict)
        else {}
    )
    spiking_hjepa_latent_transition = 1.0 if (
        latent_transition["trace_complete"] and latent_branch_comparison["separable"]
    ) else 0.0
    manifold_trace_memory = LocalManifoldTransitionMemory(capacity=4)
    manifold_trace_memory.add_trajectory(
        "stage-e-primary-latent-trajectory",
        source_events=latent_source_events + latent_predicted_events,
        next_events=latent_observed_events,
        correction_events=latent_correction_events,
        causal_edges=[
            {"from": "status=needs_gate", "to": "status=release_ready", "support": 0.93},
            {"from": "status=release_ready", "to": "audit=complete", "support": 0.91},
        ],
        event_cost_proxy=0.18,
    )
    manifold_trace_memory.add_trajectory(
        "stage-e-counterfactual-latent-trajectory",
        source_events=latent_source_events + counterfactual_predicted_events,
        next_events=counterfactual_observed_events,
        correction_events=counterfactual_correction_events,
        causal_edges=[
            {"from": "status=needs_gate", "to": "status=release_deferred", "support": 0.87},
            {"from": "status=release_deferred", "to": "risk=pytest_pending", "support": 0.85},
        ],
        event_cost_proxy=0.22,
    )
    manifold_trace_memory.add_trajectory(
        "stage-e-unrelated-latent-trajectory",
        source_events=encoder.encode_structured_state(
            {"status": "unrelated_probe"},
            timestep=1,
            confidence=0.40,
        ),
        next_events=encoder.encode_structured_state(
            {"status": "unrelated_observed"},
            timestep=2,
            confidence=0.42,
        ),
        correction_events=(),
        causal_edges=[
            {"from": "status=unrelated_probe", "to": "status=unrelated_observed", "support": 0.84},
        ],
        event_cost_proxy=0.20,
    )
    manifold_trace_support = manifold_trace_memory.evaluate(
        query_events=latent_source_events + latent_predicted_events,
        withheld_expected_events=latent_observed_events,
        scan_budget=2,
        case_specs=[
            {
                "case_id": "stage-e-primary",
                "query_events": latent_source_events + latent_predicted_events,
                "expected_trajectory_id": "stage-e-primary-latent-trajectory",
                "expected_next_events": latent_observed_events,
            },
            {
                "case_id": "stage-e-counterfactual",
                "query_events": latent_source_events + counterfactual_predicted_events,
                "expected_trajectory_id": "stage-e-counterfactual-latent-trajectory",
                "expected_next_events": counterfactual_observed_events,
            },
        ],
    )
    manifold_trace_metrics = (
        manifold_trace_support.get("metrics", {})
        if isinstance(manifold_trace_support.get("metrics"), dict)
        else {}
    )
    delta_memory_steering = evaluate_delta_memory_steering_trace()
    delta_memory_metrics = (
        delta_memory_steering.get("metrics", {})
        if isinstance(delta_memory_steering.get("metrics"), dict)
        else {}
    )
    predictive_gated_spike = _predictive_error_gated_spike_case(
        predicted_events=latent_observed_events,
        observed_events=latent_observed_events,
        surprise_events=counterfactual_observed_events,
    )
    predictive_spike_entropy_reduction = 1.0 if (
        int(predictive_gated_spike.get("expected_correction_spikes", 0)) == 0
        and int(predictive_gated_spike.get("surprise_correction_spikes", 0) or 0) > 0
        and float(predictive_gated_spike.get("entropy_reduction", 0.0) or 0.0) >= 1.0
        and bool(predictive_gated_spike.get("state_budget_ok", False))
    ) else 0.0
    phase_binding_trace = evaluate_phase_synchronized_binding_trace()
    phase_binding_metrics = (
        phase_binding_trace.get("metrics", {})
        if isinstance(phase_binding_trace.get("metrics"), dict)
        else {}
    )
    forward_only_update = evaluate_forward_only_local_update_trace()
    forward_only_metrics = (
        forward_only_update.get("metrics", {})
        if isinstance(forward_only_update.get("metrics"), dict)
        else {}
    )
    plastic_submodel_registry = evaluate_plastic_submodel_registry_trace()
    plastic_submodel_metrics = (
        plastic_submodel_registry.get("metrics", {})
        if isinstance(plastic_submodel_registry.get("metrics"), dict)
        else {}
    )
    plastic_submodel_intervention = evaluate_plastic_submodel_intervention_trace()
    plastic_submodel_intervention_metrics = (
        plastic_submodel_intervention.get("metrics", {})
        if isinstance(plastic_submodel_intervention.get("metrics"), dict)
        else {}
    )
    plastic_submodel_credit_assignment = evaluate_plastic_submodel_credit_assignment_trace()
    plastic_submodel_credit_metrics = (
        plastic_submodel_credit_assignment.get("metrics", {})
        if isinstance(plastic_submodel_credit_assignment.get("metrics"), dict)
        else {}
    )
    plastic_submodel_structural_adaptation = evaluate_plastic_submodel_structural_adaptation_trace()
    plastic_submodel_structural_metrics = (
        plastic_submodel_structural_adaptation.get("metrics", {})
        if isinstance(plastic_submodel_structural_adaptation.get("metrics"), dict)
        else {}
    )
    plastic_submodel_scientific_model = evaluate_plastic_submodel_scientific_model_trace()
    plastic_submodel_scientific_metrics = (
        plastic_submodel_scientific_model.get("metrics", {})
        if isinstance(plastic_submodel_scientific_model.get("metrics"), dict)
        else {}
    )
    plastic_submodel_hypothesis_bank = evaluate_plastic_submodel_open_ended_hypothesis_bank_trace()
    plastic_submodel_hypothesis_bank_metrics = (
        plastic_submodel_hypothesis_bank.get("metrics", {})
        if isinstance(plastic_submodel_hypothesis_bank.get("metrics"), dict)
        else {}
    )

    relation_trace = build_event_relation_trace(
        cause="pytest gate passes",
        relation="enables",
        effect="release ready",
        branch_id="primary",
    )
    reverse_trace = build_reverse_reasoning_trace(
        outcome="release blocked",
        candidate_causes=["pytest gate failed", "version mismatch"],
        selected_cause="pytest gate failed",
        branch_id="counterfactual-a",
    )
    causal_trace = build_causal_candidate_trace(
        relation_trace=relation_trace,
        reverse_trace=build_reverse_reasoning_trace(
            outcome="release blocked",
            candidate_causes=["pytest gate passes:missing", "needs_gate:unchanged"],
            selected_cause="needs_gate:unchanged",
            branch_id="primary",
        ),
        selected_action="pytest gate passes",
        branch_id="primary",
    )
    reverse_reasoning_trace_integrity = 1.0 if (
        relation_trace["trace_complete"]
        and reverse_trace["trace_complete"]
        and relation_trace["branch_id"] != reverse_trace["branch_id"]
    ) else 0.0
    causal_candidate_trace_integrity = 1.0 if causal_trace["trace_complete"] else 0.0

    runtime = ModularCognitiveRuntime(
        encoder=CommonSpikeSpaceEncoder(dimension=2048, active_bits=3),
        compressor=TemporalCompressionPolicy(max_window=4, max_events_per_modality=10),
        budgeter=ModalityTemporalBudget(max_budget=6),
        gate=DendriticContextGate(short_term_limit=12, long_term_limit=18, error_limit=8),
    )
    runtime_report = runtime.run(
        text="ship release after pytest gate passes",
        state={
            "goal": "release",
            "status": "needs_gate",
            "risk": "pytest",
        },
        candidate_actions=[
            "run release gate",
            "defer release",
        ],
    )
    module_orchestration_integrity = 1.0 if runtime_report["module_orchestration_complete"] else 0.0
    counterfactual_lane_integrity = 1.0 if runtime_report["counterfactual_lane_complete"] else 0.0
    action_trace_observability = 1.0 if runtime_report["action_trace_complete"] else 0.0
    runtime_replay_report = runtime.run(
        text="ship release after pytest gate passes",
        state={
            "goal": "release",
            "status": "needs_gate",
            "risk": "pytest",
        },
        candidate_actions=[
            "run release gate",
            "defer release",
        ],
    )
    feedback_runtime_report = runtime.run(
        text="ship release after pytest gate passes",
        state={
            "goal": "release",
            "status": "needs_gate",
            "risk": "pytest",
        },
        candidate_actions=[
            "run release gate",
            "defer release",
        ],
        action_feedback={
            "primary": 0.9,
            "counterfactual-1": -0.6,
        },
    )
    runtime_trace_digest = build_runtime_trace_digest(runtime_report)
    runtime_trace_comparison = compare_runtime_trace_digests(runtime_report, runtime_replay_report)
    runtime_trace_replay_consistency = 1.0 if runtime_trace_comparison["consistent"] else 0.0
    runtime_causal_candidate_trace_integrity = 1.0 if (
        runtime_report.get("selected_action", {}).get("causal_trace", {}).get("trace_complete")
        and runtime_report.get("counterfactual_action", {}).get("causal_trace", {}).get("trace_complete")
    ) else 0.0
    causal_candidate_trace_integrity = min(
        causal_candidate_trace_integrity,
        runtime_causal_candidate_trace_integrity,
    )
    runtime_candidates = (
        runtime_report.get("modules", {})
        .get("world_model", {})
        .get("trace", {})
        .get("candidates", [])
        if isinstance(runtime_report.get("modules", {}), dict)
        else []
    )
    selected_support_submodels = (
        runtime_report.get("selected_action", {}).get("support_submodels", [])
        if isinstance(runtime_report.get("selected_action", {}), dict)
        else []
    )
    counterfactual_support_submodels = (
        runtime_report.get("counterfactual_action", {}).get("support_submodels", [])
        if isinstance(runtime_report.get("counterfactual_action", {}), dict)
        else []
    )
    selected_submodel_route = (
        runtime_report.get("selected_action", {}).get("submodel_route", {})
        if isinstance(runtime_report.get("selected_action", {}), dict)
        else {}
    )
    counterfactual_submodel_route = (
        runtime_report.get("counterfactual_action", {}).get("submodel_route", {})
        if isinstance(runtime_report.get("counterfactual_action", {}), dict)
        else {}
    )
    world_model_trace = (
        runtime_report.get("modules", {}).get("world_model", {}).get("trace", {})
        if isinstance(runtime_report.get("modules", {}), dict)
        else {}
    )
    runtime_submodel_concept_trace = (
        world_model_trace.get("plastic_submodel_concept_trace", {})
        if isinstance(world_model_trace, dict)
        else {}
    )
    runtime_submodel_route_action_grounding = 1.0 if (
        bool(selected_support_submodels)
        and isinstance(selected_submodel_route, dict)
        and bool(selected_submodel_route.get("state_budget_ok", False))
        and bool(selected_submodel_route.get("connected_pairs", []))
    ) else 0.0
    runtime_submodel_counterfactual_route_separation = 1.0 if (
        bool(selected_support_submodels)
        and bool(counterfactual_support_submodels)
        and set(selected_support_submodels) != set(counterfactual_support_submodels)
        and isinstance(counterfactual_submodel_route, dict)
        and bool(counterfactual_submodel_route.get("state_budget_ok", False))
    ) else 0.0
    runtime_submodel_concept_trace_observed = 1.0 if (
        isinstance(runtime_submodel_concept_trace, dict)
        and bool(runtime_submodel_concept_trace.get("route_edges", []))
        and bool(runtime_submodel_concept_trace.get("trace", []))
    ) else 0.0
    feedback_report = (
        feedback_runtime_report.get("action_feedback", {})
        if isinstance(feedback_runtime_report, dict)
        else {}
    )
    feedback_records = (
        feedback_report.get("records", [])
        if isinstance(feedback_report.get("records", []), list)
        else []
    )
    runtime_submodel_local_credit_assignment = 1.0 if (
        bool(feedback_report.get("applied", False))
        and int(feedback_report.get("feedback_count", 0) or 0) == 2
        and all(int(record.get("updated_submodel_count", 0) or 0) > 0 for record in feedback_records)
        and bool(feedback_report.get("state_budget_ok", False))
    ) else 0.0
    runtime_submodel_feedback_trace = 1.0 if (
        isinstance(feedback_runtime_report.get("selected_action", {}), dict)
        and isinstance(feedback_runtime_report.get("counterfactual_action", {}), dict)
        and bool(feedback_runtime_report["selected_action"].get("feedback_trace", {}).get("state_budget_ok", False))
        and bool(
            feedback_runtime_report["counterfactual_action"].get("feedback_trace", {}).get(
                "state_budget_ok",
                False,
            )
        )
    ) else 0.0
    sparse_verifier = evaluate_sparse_verifier_trace(
        runtime_candidates,
        evidence_texts=[
            "release gate passes after pytest completes",
            "run release gate is the grounded low energy action",
        ],
        expected_branch_id="primary",
        max_energy_budget=6.0,
    )
    sparse_verifier_metrics = (
        sparse_verifier.get("metrics", {})
        if isinstance(sparse_verifier.get("metrics"), dict)
        else {}
    )
    sparse_best_of_n_candidates = list(runtime_candidates)
    if runtime_candidates:
        retrieval_heavy = dict(runtime_candidates[0])
        retrieval_heavy["branch_id"] = "retrieval-heavy"
        retrieval_heavy["action"] = "run release gate with retrieved pytest evidence"
        retrieval_heavy["score"] = 0.60
        retrieval_heavy["budget"] = {"budget": 4, "bounded": True}
        relation_trace = dict(retrieval_heavy.get("relation_trace", {}))
        causal_trace = dict(retrieval_heavy.get("causal_trace", {}))
        relation_trace["branch_id"] = "retrieval-heavy"
        causal_trace["branch_id"] = "retrieval-heavy"
        retrieval_heavy["relation_trace"] = relation_trace
        retrieval_heavy["causal_trace"] = causal_trace
        sparse_best_of_n_candidates.append(retrieval_heavy)
    sparse_best_of_n = evaluate_sparse_best_of_n_trace(
        sparse_best_of_n_candidates,
        evidence_texts=[
            "release gate passes after pytest completes",
            "run release gate is the grounded low energy action",
            "retrieved pytest evidence can support the same action but costs more",
        ],
        expected_branch_id="primary",
        summary_text="Selected primary branch: run release gate.",
        max_n=3,
        max_energy_budget=6.0,
    )
    sparse_best_of_n_metrics = (
        sparse_best_of_n.get("metrics", {})
        if isinstance(sparse_best_of_n.get("metrics"), dict)
        else {}
    )
    self_correction_trace = evaluate_self_correction_trace(
        {
            **dict(runtime_candidates[0] if runtime_candidates else {}),
            "branch_id": "draft",
            "action": "defer release",
            "score": 0.25,
            "budget": {"budget": 5, "bounded": True},
            "relation_trace": {
                **dict((runtime_candidates[0] if runtime_candidates else {}).get("relation_trace", {})),
                "branch_id": "draft",
            },
            "causal_trace": {
                **dict((runtime_candidates[0] if runtime_candidates else {}).get("causal_trace", {})),
                "branch_id": "draft",
            },
        },
        sparse_best_of_n_candidates,
        evidence_texts=[
            "release gate passes after pytest completes",
            "run release gate is the grounded low energy action",
        ],
        expected_branch_id="primary",
        max_loops=2,
        min_improvement=0.05,
        max_energy_budget=6.0,
    )
    self_correction_metrics = (
        self_correction_trace.get("metrics", {})
        if isinstance(self_correction_trace.get("metrics"), dict)
        else {}
    )
    tree_candidates = []
    for index, candidate in enumerate(sparse_best_of_n_candidates):
        if not isinstance(candidate, dict):
            continue
        tree_candidate = dict(candidate)
        tree_candidate["depth"] = 1 if index < 2 else 2
        tree_candidate["event_cost"] = 2
        if index >= 2:
            tree_candidate["parent_branch_id"] = "primary"
        tree_candidates.append(tree_candidate)
    if runtime_candidates:
        too_deep = dict(runtime_candidates[0])
        too_deep["branch_id"] = "too-deep-rollout"
        too_deep["action"] = "run recursive rollout"
        too_deep["score"] = 0.80
        too_deep["depth"] = 3
        too_deep["event_cost"] = 1
        too_deep["parent_branch_id"] = "retrieval-heavy"
        relation_trace = dict(too_deep.get("relation_trace", {}))
        causal_trace = dict(too_deep.get("causal_trace", {}))
        relation_trace["branch_id"] = "too-deep-rollout"
        causal_trace["branch_id"] = "too-deep-rollout"
        too_deep["relation_trace"] = relation_trace
        too_deep["causal_trace"] = causal_trace
        tree_candidates.append(too_deep)
    bounded_tree_search = evaluate_bounded_tree_search_trace(
        tree_candidates,
        evidence_texts=[
            "release gate passes after pytest completes",
            "run release gate is the grounded low energy action",
            "retrieved pytest evidence can support the same action but costs more",
        ],
        expected_branch_id="primary",
        max_depth=2,
        max_branch_factor=2,
        max_event_budget=6,
        max_energy_budget=6.0,
    )
    bounded_tree_metrics = (
        bounded_tree_search.get("metrics", {})
        if isinstance(bounded_tree_search.get("metrics"), dict)
        else {}
    )
    forest_lanes = []
    for index, candidate in enumerate(sparse_best_of_n_candidates[:3]):
        if not isinstance(candidate, dict):
            continue
        lane = dict(candidate)
        lane["lane_id"] = ["memory-prior", "counterfactual", "retrieval"][index]
        lane["snapshot"] = {"read_only": True, "mutation_count": 0}
        lane["selection_reason"] = (
            "primary branch has grounded release gate evidence"
            if str(lane.get("branch_id", "") or "") == "primary"
            else f"{lane['lane_id']} branch is retained for comparison"
        )
        forest_lanes.append(lane)
    reasoning_forest_lane = evaluate_reasoning_forest_lane_trace(
        forest_lanes,
        evidence_texts=[
            "release gate passes after pytest completes",
            "run release gate is the grounded low energy action",
            "retrieved pytest evidence can support the same action but costs more",
        ],
        expected_branch_id="primary",
        max_lanes=3,
        max_energy_budget=6.0,
    )
    reasoning_forest_metrics = (
        reasoning_forest_lane.get("metrics", {})
        if isinstance(reasoning_forest_lane.get("metrics"), dict)
        else {}
    )
    hierarchical_reasoning = evaluate_hierarchical_reasoning_trace(
        {
            "event_type": "instruction_event",
            "instruction_id": "stage-e-release-gate",
            "instruction": "run release gate",
            "target_branch_id": "primary",
        },
        sparse_best_of_n_candidates,
        evidence_texts=[
            "release gate passes after pytest completes",
            "run release gate is the grounded low energy action",
        ],
        expected_branch_id="primary",
        max_execution_steps=3,
        max_energy_budget=6.0,
    )
    hierarchical_metrics = (
        hierarchical_reasoning.get("metrics", {})
        if isinstance(hierarchical_reasoning.get("metrics"), dict)
        else {}
    )
    micro_turn_interaction = evaluate_micro_turn_interaction_trace(
        [
            {
                "time_bucket": 0,
                "event_type": "audio_tick",
                "stream": "audio",
                "lane": "foreground",
                "event_cost": 2,
                "policy": "listen",
            },
            {
                "time_bucket": 0,
                "event_type": "visual_change",
                "stream": "vision",
                "lane": "foreground",
                "event_cost": 2,
                "policy": "inspect",
            },
            {
                "time_bucket": 1,
                "event_type": "text_delta",
                "stream": "text",
                "lane": "foreground",
                "event_cost": 2,
                "handoff": True,
                "policy": "route",
            },
            {
                "time_bucket": 1,
                "event_type": "background_context_update",
                "stream": "memory",
                "lane": "background",
                "event_cost": 3,
                "handoff": True,
                "policy": "summarize",
            },
            {
                "time_bucket": 2,
                "event_type": "interrupt",
                "stream": "text",
                "lane": "foreground",
                "event_cost": 2,
                "policy": "yield",
            },
            {
                "time_bucket": 3,
                "event_type": "interrupt_recovery",
                "stream": "planner",
                "lane": "background",
                "event_cost": 2,
                "policy": "resume",
            },
            {
                "time_bucket": 3,
                "event_type": "backchannel",
                "stream": "audio",
                "lane": "foreground",
                "event_cost": 1,
                "backchannel": True,
                "latency_ms": 180,
                "policy": "acknowledge",
            },
        ],
        max_turns=8,
        max_event_budget=18,
    )
    micro_turn_metrics = (
        micro_turn_interaction.get("metrics", {})
        if isinstance(micro_turn_interaction.get("metrics"), dict)
        else {}
    )
    phase_assigned_blocks = evaluate_phase_assigned_submodel_block_trace(
        [
            {
                "phase": "memory_phase",
                "uncertainty_bucket": "low",
                "submodel": "memory_system",
                "event_cost": 4,
                "independent_update": True,
                "local_credit": True,
                "residual_reduction": 0.10,
            },
            {
                "phase": "prediction_error",
                "uncertainty_bucket": "medium",
                "submodel": "world_model",
                "event_cost": 5,
                "independent_update": True,
                "local_credit": True,
                "correction_event": True,
                "residual_reduction": 0.35,
            },
            {
                "phase": "correction",
                "uncertainty_bucket": "high",
                "submodel": "self_monitor",
                "event_cost": 5,
                "independent_update": True,
                "local_credit": True,
                "correction_event": True,
                "residual_reduction": 0.45,
            },
            {
                "phase": "planning",
                "uncertainty_bucket": "medium",
                "submodel": "value_system",
                "event_cost": 4,
                "independent_update": True,
                "local_credit": True,
                "residual_reduction": 0.20,
            },
        ],
        max_event_budget=24,
    )
    phase_assigned_metrics = (
        phase_assigned_blocks.get("metrics", {})
        if isinstance(phase_assigned_blocks.get("metrics"), dict)
        else {}
    )

    metrics = {
        "common_spike_space_integrity": common_spike_space_integrity,
        "temporal_compression_efficiency": temporal_compression_efficiency,
        "modality_temporal_budget_integrity": modality_temporal_budget_integrity,
        "dendritic_context_gate_stability": dendritic_context_gate_stability,
        "spiking_hjepa_latent_transition": spiking_hjepa_latent_transition,
        "reverse_reasoning_trace_integrity": reverse_reasoning_trace_integrity,
        "causal_candidate_trace_integrity": causal_candidate_trace_integrity,
        "module_orchestration_integrity": module_orchestration_integrity,
        "counterfactual_lane_integrity": counterfactual_lane_integrity,
        "action_trace_observability": action_trace_observability,
        "runtime_trace_replay_consistency": runtime_trace_replay_consistency,
        "manifold_trace_support_observed": float(
            manifold_trace_metrics.get("manifold_trajectory_case_coverage", 0.0)
        ),
        "manifold_trace_recall_observed": float(
            manifold_trace_metrics.get("manifold_average_case_recall", 0.0)
        ),
        "manifold_trace_scan_budget_observed": float(
            manifold_trace_metrics.get("manifold_scan_budget_integrity", 0.0)
        ),
        "manifold_trace_index_scan_reduction_observed": float(
            manifold_trace_metrics.get("manifold_index_scan_reduction", 0.0)
        ),
        "manifold_trace_candidate_guard_observed": float(
            manifold_trace_metrics.get("manifold_candidate_miss_guard", 0.0)
        ),
        "delta_memory_steering_integrity_observed": float(
            delta_memory_metrics.get("delta_memory_steering_integrity", 0.0)
        ),
        "delta_memory_counterfactual_isolation_observed": float(
            delta_memory_metrics.get("delta_memory_counterfactual_isolation", 0.0)
        ),
        "delta_memory_trace_observability_observed": float(
            delta_memory_metrics.get("delta_memory_trace_observability", 0.0)
        ),
        "predictive_spike_entropy_reduction_observed": predictive_spike_entropy_reduction,
        "phase_binding_coincidence_integrity_observed": float(
            phase_binding_metrics.get("phase_binding_coincidence_integrity", 0.0)
        ),
        "forward_only_local_update_stability_observed": float(
            forward_only_metrics.get("forward_only_local_update_stability", 0.0)
        ),
        "lejepa_linear_identifiability_proxy_observed": float(
            lejepa_latent_health_metrics.get("lejepa_linear_identifiability_proxy", 0.0)
        ),
        "lejepa_latent_whitening_health_observed": float(
            lejepa_latent_health_metrics.get("lejepa_latent_whitening_health", 0.0)
        ),
        "lejepa_factor_disentanglement_observed": float(
            lejepa_latent_health_metrics.get("lejepa_factor_disentanglement", 0.0)
        ),
        "lejepa_latent_planning_consistency_observed": float(
            lejepa_latent_health_metrics.get("lejepa_latent_planning_consistency", 0.0)
        ),
        "lejepa_positive_pair_alignment_observed": float(
            lejepa_latent_health_metrics.get("lejepa_positive_pair_alignment", 0.0)
        ),
        "plastic_submodel_registry_integrity_observed": float(
            plastic_submodel_metrics.get("plastic_submodel_registry_integrity", 0.0)
        ),
        "dynamic_submodel_route_integrity_observed": float(
            plastic_submodel_metrics.get("dynamic_submodel_route_integrity", 0.0)
        ),
        "submodel_relearning_trace_integrity_observed": float(
            plastic_submodel_metrics.get("submodel_relearning_trace_integrity", 0.0)
        ),
        "interpretable_submodel_concept_trace_observed": float(
            plastic_submodel_metrics.get("interpretable_submodel_concept_trace", 0.0)
        ),
        "runtime_submodel_route_action_grounding_observed": runtime_submodel_route_action_grounding,
        "runtime_submodel_counterfactual_route_separation_observed": (
            runtime_submodel_counterfactual_route_separation
        ),
        "runtime_submodel_concept_trace_observed": runtime_submodel_concept_trace_observed,
        "submodel_intervention_trace_integrity_observed": float(
            plastic_submodel_intervention_metrics.get("submodel_intervention_trace_integrity", 0.0)
        ),
        "submodel_ablation_effect_observed": float(
            plastic_submodel_intervention_metrics.get("submodel_ablation_effect_observed", 0.0)
        ),
        "submodel_reactivation_recovery_observed": float(
            plastic_submodel_intervention_metrics.get("submodel_reactivation_recovery_observed", 0.0)
        ),
        "submodel_credit_assignment_trace_integrity_observed": float(
            plastic_submodel_credit_metrics.get("submodel_credit_assignment_trace_integrity", 0.0)
        ),
        "submodel_credit_selectivity_observed": float(
            plastic_submodel_credit_metrics.get("submodel_credit_selectivity_observed", 0.0)
        ),
        "submodel_credit_state_budget_observed": float(
            plastic_submodel_credit_metrics.get("submodel_credit_state_budget_observed", 0.0)
        ),
        "runtime_submodel_local_credit_assignment_observed": runtime_submodel_local_credit_assignment,
        "runtime_submodel_feedback_trace_observed": runtime_submodel_feedback_trace,
        "submodel_structural_adaptation_trace_integrity_observed": float(
            plastic_submodel_structural_metrics.get("submodel_structural_adaptation_trace_integrity", 0.0)
        ),
        "submodel_structural_growth_bounded_observed": float(
            plastic_submodel_structural_metrics.get("submodel_structural_growth_bounded_observed", 0.0)
        ),
        "submodel_structural_pruning_observed": float(
            plastic_submodel_structural_metrics.get("submodel_structural_pruning_observed", 0.0)
        ),
        "submodel_scientific_hypothesis_trace_integrity_observed": float(
            plastic_submodel_scientific_metrics.get("submodel_scientific_hypothesis_trace_integrity", 0.0)
        ),
        "submodel_counterexample_revision_observed": float(
            plastic_submodel_scientific_metrics.get("submodel_counterexample_revision_observed", 0.0)
        ),
        "submodel_scientific_model_budget_observed": float(
            plastic_submodel_scientific_metrics.get("submodel_scientific_model_budget_observed", 0.0)
        ),
        "submodel_hypothesis_bank_integrity_observed": float(
            plastic_submodel_hypothesis_bank_metrics.get("submodel_hypothesis_bank_integrity", 0.0)
        ),
        "submodel_open_ended_selection_observed": float(
            plastic_submodel_hypothesis_bank_metrics.get("submodel_open_ended_selection_observed", 0.0)
        ),
        "submodel_hypothesis_bank_budget_observed": float(
            plastic_submodel_hypothesis_bank_metrics.get("submodel_hypothesis_bank_budget_observed", 0.0)
        ),
        "micro_turn_event_budget_observed": float(
            micro_turn_metrics.get("micro_turn_event_budget", 0.0)
        ),
        "foreground_background_context_handoff_observed": float(
            micro_turn_metrics.get("foreground_background_context_handoff", 0.0)
        ),
        "interrupt_recovery_trace_observed": float(
            micro_turn_metrics.get("interrupt_recovery_trace", 0.0)
        ),
        "simultaneous_stream_route_integrity_observed": float(
            micro_turn_metrics.get("simultaneous_stream_route_integrity", 0.0)
        ),
        "time_aligned_backchannel_policy_observed": float(
            micro_turn_metrics.get("time_aligned_backchannel_policy", 0.0)
        ),
        "phase_assigned_submodel_route_observed": float(
            phase_assigned_metrics.get("phase_assigned_submodel_route", 0.0)
        ),
        "uncertainty_bucket_specialization_observed": float(
            phase_assigned_metrics.get("uncertainty_bucket_specialization", 0.0)
        ),
        "denoising_correction_trace_integrity_observed": float(
            phase_assigned_metrics.get("denoising_correction_trace_integrity", 0.0)
        ),
        "block_independent_local_update_budget_observed": float(
            phase_assigned_metrics.get("block_independent_local_update_budget", 0.0)
        ),
        "sparse_verifier_grounding_observed": float(
            sparse_verifier_metrics.get("sparse_verifier_grounding_observed", 0.0)
        ),
        "sparse_verifier_trace_integrity_observed": float(
            sparse_verifier_metrics.get("sparse_verifier_trace_integrity_observed", 0.0)
        ),
        "sparse_verifier_energy_budget_observed": float(
            sparse_verifier_metrics.get("sparse_verifier_energy_budget_observed", 0.0)
        ),
        "sparse_verifier_uncertainty_observed": float(
            sparse_verifier_metrics.get("sparse_verifier_uncertainty_observed", 0.0)
        ),
        "sparse_verifier_selection_observed": float(
            sparse_verifier_metrics.get("sparse_verifier_selection_observed", 0.0)
        ),
        "sparse_best_of_n_bounded_count_observed": float(
            sparse_best_of_n_metrics.get("sparse_best_of_n_bounded_count_observed", 0.0)
        ),
        "sparse_best_of_n_branch_diversity_observed": float(
            sparse_best_of_n_metrics.get("sparse_best_of_n_branch_diversity_observed", 0.0)
        ),
        "sparse_best_of_n_verifier_selection_observed": float(
            sparse_best_of_n_metrics.get("sparse_best_of_n_verifier_selection_observed", 0.0)
        ),
        "sparse_best_of_n_summary_alignment_observed": float(
            sparse_best_of_n_metrics.get("sparse_best_of_n_summary_alignment_observed", 0.0)
        ),
        "self_correction_bounded_loop_observed": float(
            self_correction_metrics.get("self_correction_bounded_loop_observed", 0.0)
        ),
        "self_correction_improvement_observed": float(
            self_correction_metrics.get("self_correction_improvement_observed", 0.0)
        ),
        "self_correction_rollback_reason_observed": float(
            self_correction_metrics.get("self_correction_rollback_reason_observed", 0.0)
        ),
        "self_correction_verifier_failure_observed": float(
            self_correction_metrics.get("self_correction_verifier_failure_observed", 0.0)
        ),
        "bounded_tree_search_depth_observed": float(
            bounded_tree_metrics.get("bounded_tree_search_depth_observed", 0.0)
        ),
        "bounded_tree_search_branch_factor_observed": float(
            bounded_tree_metrics.get("bounded_tree_search_branch_factor_observed", 0.0)
        ),
        "bounded_tree_search_event_budget_observed": float(
            bounded_tree_metrics.get("bounded_tree_search_event_budget_observed", 0.0)
        ),
        "bounded_tree_search_verifier_selection_observed": float(
            bounded_tree_metrics.get("bounded_tree_search_verifier_selection_observed", 0.0)
        ),
        "reasoning_forest_lane_bounded_count_observed": float(
            reasoning_forest_metrics.get("reasoning_forest_lane_bounded_count_observed", 0.0)
        ),
        "reasoning_forest_lane_read_only_snapshot_observed": float(
            reasoning_forest_metrics.get("reasoning_forest_lane_read_only_snapshot_observed", 0.0)
        ),
        "reasoning_forest_lane_diversity_observed": float(
            reasoning_forest_metrics.get("reasoning_forest_lane_diversity_observed", 0.0)
        ),
        "reasoning_forest_lane_verifier_selection_observed": float(
            reasoning_forest_metrics.get("reasoning_forest_lane_verifier_selection_observed", 0.0)
        ),
        "reasoning_forest_lane_selection_reason_observed": float(
            reasoning_forest_metrics.get("reasoning_forest_lane_selection_reason_observed", 0.0)
        ),
        "hierarchical_reasoning_instruction_observed": float(
            hierarchical_metrics.get("hierarchical_reasoning_instruction_observed", 0.0)
        ),
        "hierarchical_reasoning_execution_trace_observed": float(
            hierarchical_metrics.get("hierarchical_reasoning_execution_trace_observed", 0.0)
        ),
        "hierarchical_reasoning_verification_trace_observed": float(
            hierarchical_metrics.get("hierarchical_reasoning_verification_trace_observed", 0.0)
        ),
        "hierarchical_reasoning_plan_alignment_observed": float(
            hierarchical_metrics.get("hierarchical_reasoning_plan_alignment_observed", 0.0)
        ),
    }
    gate_metrics = {
        metric_name: value
        for metric_name, value in metrics.items()
        if not str(metric_name).endswith("_observed")
    }
    observed_metrics = {
        metric_name: value
        for metric_name, value in metrics.items()
        if str(metric_name).endswith("_observed")
    }
    overall_score = sum(gate_metrics.values()) / max(len(gate_metrics), 1)
    return {
        "suite_name": "CognitiveRuntimeBenchmark",
        "passed": all(value >= 1.0 for value in gate_metrics.values()),
        "overall_score": overall_score,
        "metrics": metrics,
        "gate_metrics": gate_metrics,
        "observed_metrics": observed_metrics,
        "metric_policy": {
            "gate_score_source": "gate_metrics",
            "observed_metric_suffix": "_observed",
            "observed_metrics_excluded_from_overall_score": True,
            "observed_metrics_excluded_from_release_gate": True,
        },
        "details": {
            "event_counts": {
                "text": len(text_events),
                "state": len(state_events),
                "image_adapter": len(image_adapter_events),
                "combined": len(all_events),
                "compressed": len(compressed_events),
            },
            "compression": compression_report,
            "temporal_budgets": budget_reports,
            "dendritic_gate": {
                "first": first_gate_report,
                "second": second_gate_report,
            },
            "latent_transition": latent_transition,
            "counterfactual_transition": counterfactual_transition,
            "latent_branch_comparison": latent_branch_comparison,
            "lejepa_latent_health": lejepa_latent_health,
            "manifold_trace_support": manifold_trace_support,
            "delta_memory_steering": delta_memory_steering,
            "predictive_error_gated_spike": predictive_gated_spike,
            "phase_synchronized_binding": phase_binding_trace,
            "forward_only_local_update": forward_only_update,
            "plastic_submodel_registry": plastic_submodel_registry,
            "plastic_submodel_intervention": plastic_submodel_intervention,
            "plastic_submodel_credit_assignment": plastic_submodel_credit_assignment,
            "plastic_submodel_structural_adaptation": plastic_submodel_structural_adaptation,
            "plastic_submodel_scientific_model": plastic_submodel_scientific_model,
            "plastic_submodel_hypothesis_bank": plastic_submodel_hypothesis_bank,
            "micro_turn_interaction": micro_turn_interaction,
            "phase_assigned_submodel_blocks": phase_assigned_blocks,
            "event_relation_trace": relation_trace,
            "reverse_reasoning_trace": reverse_trace,
            "causal_candidate_trace": causal_trace,
            "modular_runtime": runtime_report,
            "feedback_modular_runtime": feedback_runtime_report,
            "runtime_trace_digest": runtime_trace_digest,
            "runtime_trace_replay": runtime_trace_comparison,
            "sparse_verifier": sparse_verifier,
            "sparse_best_of_n": sparse_best_of_n,
            "self_correction_trace": self_correction_trace,
            "bounded_tree_search": bounded_tree_search,
            "reasoning_forest_lane": reasoning_forest_lane,
            "hierarchical_reasoning": hierarchical_reasoning,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage E cognitive runtime benchmark.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "cognitive_runtime_benchmark.json"),
        help="Managed output path for the benchmark report.",
    )
    args = parser.parse_args()

    report = run_cognitive_runtime_benchmark()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("Cognitive runtime benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
