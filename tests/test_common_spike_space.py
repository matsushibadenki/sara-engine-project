# Directory Path: tests/test_common_spike_space.py
# English Title: Common Spike Space Tests
# Purpose/Content: Verifies lightweight multimodal sparse-event primitives for Stage E readiness.

from sara_engine.nn.common_spike_space import (
    CommonSpikeSpaceEncoder,
    DendriticContextGate,
    ModalityTemporalBudget,
    ModularCognitiveRuntime,
    TemporalCompressionPolicy,
    build_causal_candidate_trace,
    build_event_relation_trace,
    build_runtime_trace_digest,
    build_spiking_hjepa_multistep_trace,
    build_spiking_hjepa_transition_trace,
    compare_spiking_hjepa_transition_branches,
    compare_runtime_trace_digests,
    evaluate_lejepa_sparse_latent_health_trace,
    evaluate_micro_turn_interaction_trace,
    evaluate_phase_assigned_submodel_block_trace,
    build_reverse_reasoning_trace,
)


def test_common_spike_space_normalizes_multiple_modalities() -> None:
    encoder = CommonSpikeSpaceEncoder(dimension=512, active_bits=2)

    events = []
    events.extend(encoder.encode_text("release gate", confidence=0.9))
    events.extend(encoder.encode_structured_state({"goal": "release"}, confidence=0.8))
    events.extend(encoder.encode_adapter_features("audio", ["click", "tone"], confidence=0.6))

    assert events
    assert {event.modality for event in events} == {"text", "state", "audio"}
    assert all(0 <= event.spike_id < 512 for event in events)
    assert all(event.channel for event in events)
    adapter_events = [event for event in events if event.modality == "audio"]
    assert adapter_events
    assert all("adapter:v1" in event.tags for event in adapter_events)


def test_temporal_compression_keeps_bounded_window_and_modalities() -> None:
    encoder = CommonSpikeSpaceEncoder(dimension=1024, active_bits=4)
    events = (
        encoder.encode_text("one two three four five six", confidence=0.9)
        + encoder.encode_structured_state({"a": 1, "b": 2, "c": 3}, confidence=0.8)
    )

    compressed, report = TemporalCompressionPolicy(max_window=3, max_events_per_modality=5).compress(events)

    assert len(compressed) <= len(events)
    assert report["max_timestep"] <= 2.0
    assert {event.modality for event in compressed} == {"text", "state"}


def test_modality_temporal_budget_is_bounded_and_adaptive() -> None:
    budgeter = ModalityTemporalBudget(max_budget=5)

    normal_state = budgeter.allocate("state", confidence=0.9, surprise=0.1)
    uncertain_image = budgeter.allocate("image", confidence=0.3, surprise=0.9)

    assert normal_state["bounded"] is True
    assert uncertain_image["bounded"] is True
    assert uncertain_image["budget"] >= normal_state["budget"]


def test_dendritic_context_gate_separates_context_and_prediction_error() -> None:
    encoder = CommonSpikeSpaceEncoder(dimension=512, active_bits=2)
    events = encoder.encode_text("ship release", confidence=0.9)
    predicted = events[:1]
    gate = DendriticContextGate(short_term_limit=8, long_term_limit=8, error_limit=4)

    first = gate.update(events, consolidate=True)
    second = gate.update(events, prediction_events=predicted)

    assert first["long_term_count"] > 0
    assert second["prediction_error_count"] <= 4
    assert second["integrated_spikes"]
    assert second["context_stability"] == 1.0


def test_reasoning_traces_remain_complete_and_branch_observable() -> None:
    relation = build_event_relation_trace(
        cause="pytest passed",
        relation="enables",
        effect="release",
        branch_id="primary",
    )
    reverse = build_reverse_reasoning_trace(
        outcome="release blocked",
        candidate_causes=["pytest failed", "version mismatch"],
        selected_cause="pytest failed",
        branch_id="rollback-a",
    )

    assert relation["trace_complete"] is True
    assert reverse["trace_complete"] is True
    assert relation["branch_id"] != reverse["branch_id"]


def test_causal_candidate_trace_links_relation_and_reverse_evidence() -> None:
    relation = build_event_relation_trace(
        cause="needs_gate:run release gate",
        relation="projects",
        effect="release:ready",
        branch_id="primary",
    )
    reverse = build_reverse_reasoning_trace(
        outcome="release:blocked",
        candidate_causes=["run release gate:missing", "needs_gate:unchanged"],
        selected_cause="needs_gate:unchanged",
        branch_id="primary",
    )

    trace = build_causal_candidate_trace(
        relation_trace=relation,
        reverse_trace=reverse,
        selected_action="run release gate",
        branch_id="primary",
    )

    assert trace["trace_complete"] is True
    assert trace["causal_alignment"] is True
    assert trace["candidate_cause_count"] == 2


def test_modular_cognitive_runtime_keeps_module_and_counterfactual_traces() -> None:
    runtime = ModularCognitiveRuntime(
        encoder=CommonSpikeSpaceEncoder(dimension=1024, active_bits=2),
        compressor=TemporalCompressionPolicy(max_window=4, max_events_per_modality=8),
        budgeter=ModalityTemporalBudget(max_budget=6),
        gate=DendriticContextGate(short_term_limit=8, long_term_limit=12, error_limit=4),
    )

    report = runtime.run(
        text="ship release after gate passes",
        state={"goal": "release", "status": "needs_gate"},
        candidate_actions=[
            "run release gate",
            "defer release",
        ],
    )

    assert report["module_orchestration_complete"] is True
    assert report["counterfactual_lane_complete"] is True
    assert report["action_trace_complete"] is True
    assert report["selected_action"]["causal_trace"]["trace_complete"] is True
    assert report["counterfactual_action"]["causal_trace"]["trace_complete"] is True
    assert report["selected_action"]["submodel_route"]["state_budget_ok"] is True
    assert report["counterfactual_action"]["submodel_route"]["state_budget_ok"] is True
    assert report["selected_action"]["support_submodels"] != report["counterfactual_action"]["support_submodels"]
    assert report["module_order"] == ["encoder", "memory_controller", "world_model", "planner", "actor"]
    assert report["selected_action"]["branch_id"] == "primary"
    assert report["counterfactual_action"]["branch_id"] == "counterfactual-1"
    concept_trace = report["modules"]["world_model"]["trace"]["plastic_submodel_concept_trace"]
    assert concept_trace["schema"] == "sara-plastic-submodel-concept-trace-v1"


def test_runtime_trace_digest_is_stable_for_replayed_sparse_runtime() -> None:
    runtime = ModularCognitiveRuntime(
        encoder=CommonSpikeSpaceEncoder(dimension=1024, active_bits=2),
        compressor=TemporalCompressionPolicy(max_window=4, max_events_per_modality=8),
        budgeter=ModalityTemporalBudget(max_budget=6),
        gate=DendriticContextGate(short_term_limit=8, long_term_limit=12, error_limit=4),
    )
    first = runtime.run(
        text="ship release after gate passes",
        state={"goal": "release", "status": "needs_gate"},
        candidate_actions=["run release gate", "defer release"],
    )
    second = runtime.run(
        text="ship release after gate passes",
        state={"goal": "release", "status": "needs_gate"},
        candidate_actions=["run release gate", "defer release"],
    )

    first_digest = build_runtime_trace_digest(first)
    comparison = compare_runtime_trace_digests(first, second)

    assert first_digest["trace_digest"]
    assert comparison["consistent"] is True
    assert comparison["matching_fields"]["trace_digest"] is True


def test_modular_cognitive_runtime_applies_local_submodel_feedback() -> None:
    runtime = ModularCognitiveRuntime(
        encoder=CommonSpikeSpaceEncoder(dimension=1024, active_bits=2),
        compressor=TemporalCompressionPolicy(max_window=4, max_events_per_modality=8),
        budgeter=ModalityTemporalBudget(max_budget=6),
    )

    report = runtime.run(
        text="ship release after gate passes",
        state={"goal": "release", "status": "needs_gate"},
        candidate_actions=[
            "run release gate",
            "defer release",
        ],
        action_feedback={
            "primary": 0.9,
            "counterfactual-1": -0.6,
        },
    )

    assert report["action_feedback"]["applied"] is True
    assert report["action_feedback"]["feedback_count"] == 2
    assert report["action_feedback"]["state_budget_ok"] is True
    assert report["selected_action"]["feedback_trace"]["updated_submodel_count"] > 0
    assert report["counterfactual_action"]["feedback_trace"]["updated_submodel_count"] > 0


def test_spiking_hjepa_transition_trace_observes_error_and_correction() -> None:
    encoder = CommonSpikeSpaceEncoder(dimension=1024, active_bits=2)
    source = encoder.encode_structured_state({"status": "needs_gate"})
    predicted = encoder.encode_structured_state({"status": "release_ready"})
    observed = encoder.encode_structured_state({"status": "release_ready", "audit": "complete"})
    correction = encoder.encode_structured_state({"audit": "complete"})

    trace = build_spiking_hjepa_transition_trace(
        source_events=source,
        predicted_events=predicted,
        observed_events=observed,
        correction_events=correction,
        operator="release_gate.latent_transition",
    )

    assert trace["trace_complete"] is True
    assert trace["alignment_ratio"] == 1.0
    assert trace["prediction_error_ids"]
    assert trace["correction_coverage"] is True
    assert trace["anti_collapse_diversity"] is True


def test_spiking_hjepa_branch_comparison_requires_separable_counterfactual() -> None:
    encoder = CommonSpikeSpaceEncoder(dimension=1024, active_bits=2)
    source = encoder.encode_structured_state({"status": "needs_gate"})
    primary = build_spiking_hjepa_transition_trace(
        source_events=source,
        predicted_events=encoder.encode_structured_state({"status": "release_ready"}),
        observed_events=encoder.encode_structured_state({"status": "release_ready", "audit": "complete"}),
        correction_events=encoder.encode_structured_state({"audit": "complete"}),
        operator="release_gate.latent_transition",
        branch_id="primary",
    )
    counterfactual = build_spiking_hjepa_transition_trace(
        source_events=source,
        predicted_events=encoder.encode_structured_state({"status": "release_deferred"}),
        observed_events=encoder.encode_structured_state({"status": "release_deferred", "risk": "pytest_pending"}),
        correction_events=encoder.encode_structured_state({"risk": "pytest_pending"}),
        operator="release_gate.counterfactual_transition",
        branch_id="counterfactual-1",
    )

    comparison = compare_spiking_hjepa_transition_branches(primary, counterfactual)

    assert comparison["both_complete"] is True
    assert comparison["different_branch"] is True
    assert comparison["different_prediction"] is True
    assert comparison["different_observation"] is True
    assert comparison["separable"] is True


def test_lejepa_sparse_latent_health_reads_sparse_transition_quality() -> None:
    encoder = CommonSpikeSpaceEncoder(dimension=1024, active_bits=2)
    source = encoder.encode_structured_state({"status": "needs_gate"})
    primary = build_spiking_hjepa_transition_trace(
        source_events=source,
        predicted_events=encoder.encode_structured_state({"status": "release_ready"}),
        observed_events=encoder.encode_structured_state({"status": "release_ready", "audit": "complete"}),
        correction_events=encoder.encode_structured_state({"audit": "complete"}),
        operator="release_gate.latent_transition",
        branch_id="primary",
    )
    counterfactual = build_spiking_hjepa_transition_trace(
        source_events=source,
        predicted_events=encoder.encode_structured_state({"status": "release_deferred"}),
        observed_events=encoder.encode_structured_state({"status": "release_deferred", "risk": "pytest_pending"}),
        correction_events=encoder.encode_structured_state({"risk": "pytest_pending"}),
        operator="release_gate.counterfactual_transition",
        branch_id="counterfactual-1",
    )
    comparison = compare_spiking_hjepa_transition_branches(primary, counterfactual)

    health = evaluate_lejepa_sparse_latent_health_trace(primary, counterfactual, comparison)

    assert health["observed_only"] is True
    assert health["metrics"] == {
        "lejepa_linear_identifiability_proxy": 1.0,
        "lejepa_latent_whitening_health": 1.0,
        "lejepa_factor_disentanglement": 1.0,
        "lejepa_latent_planning_consistency": 1.0,
        "lejepa_positive_pair_alignment": 1.0,
    }
    assert health["trace"]["branch_comparison"]["separable"] is True


def test_micro_turn_interaction_trace_tracks_bounded_stream_handoff() -> None:
    trace = evaluate_micro_turn_interaction_trace(
        [
            {"time_bucket": 0, "event_type": "audio_tick", "stream": "audio", "lane": "foreground", "event_cost": 2},
            {"time_bucket": 0, "event_type": "visual_change", "stream": "vision", "lane": "foreground", "event_cost": 2},
            {
                "time_bucket": 1,
                "event_type": "text_delta",
                "stream": "text",
                "lane": "foreground",
                "handoff": True,
                "event_cost": 2,
            },
            {
                "time_bucket": 1,
                "event_type": "background_context_update",
                "stream": "memory",
                "lane": "background",
                "handoff": True,
                "event_cost": 3,
            },
            {"time_bucket": 2, "event_type": "interrupt", "stream": "text", "lane": "foreground", "event_cost": 2},
            {
                "time_bucket": 3,
                "event_type": "interrupt_recovery",
                "stream": "planner",
                "lane": "background",
                "event_cost": 2,
            },
            {
                "time_bucket": 3,
                "event_type": "backchannel",
                "stream": "audio",
                "lane": "foreground",
                "backchannel": True,
                "latency_ms": 180,
                "policy": "acknowledge",
                "event_cost": 1,
            },
        ],
        max_turns=8,
        max_event_budget=18,
    )

    assert trace["observed_only"] is True
    assert trace["metrics"] == {
        "micro_turn_event_budget": 1.0,
        "foreground_background_context_handoff": 1.0,
        "interrupt_recovery_trace": 1.0,
        "simultaneous_stream_route_integrity": 1.0,
        "time_aligned_backchannel_policy": 1.0,
    }
    assert trace["trace"]["simultaneous_bucket_count"] >= 1


def test_phase_assigned_submodel_block_trace_tracks_local_correction_blocks() -> None:
    trace = evaluate_phase_assigned_submodel_block_trace(
        [
            {
                "phase": "memory_phase",
                "uncertainty_bucket": "low",
                "submodel": "memory_system",
                "event_cost": 4,
                "independent_update": True,
                "local_credit": True,
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
            },
        ],
        max_event_budget=24,
    )

    assert trace["observed_only"] is True
    assert trace["metrics"] == {
        "phase_assigned_submodel_route": 1.0,
        "uncertainty_bucket_specialization": 1.0,
        "denoising_correction_trace_integrity": 1.0,
        "block_independent_local_update_budget": 1.0,
    }
    assert trace["trace"]["independent_block_count"] == trace["trace"]["block_count"]


def test_spiking_hjepa_multistep_trace_tracks_error_convergence() -> None:
    encoder = CommonSpikeSpaceEncoder(dimension=1024, active_bits=2)
    first = build_spiking_hjepa_transition_trace(
        source_events=encoder.encode_structured_state({"status": "needs_gate"}),
        predicted_events=encoder.encode_structured_state({"status": "release_ready"}),
        observed_events=encoder.encode_structured_state({"status": "release_ready", "audit": "complete"}),
        correction_events=encoder.encode_structured_state({"audit": "complete"}),
        operator="release_gate.latent_transition.step1",
        branch_id="primary-step1",
    )
    second = build_spiking_hjepa_transition_trace(
        source_events=encoder.encode_structured_state({"status": "release_ready", "audit": "complete"}),
        predicted_events=encoder.encode_structured_state({"deployment": "prepared"}),
        observed_events=encoder.encode_structured_state({"deployment": "prepared", "handoff": "documented"}),
        correction_events=encoder.encode_structured_state({"handoff": "documented"}),
        operator="release_gate.latent_transition.step2",
        branch_id="primary-step2",
    )

    trace = build_spiking_hjepa_multistep_trace([first, second])

    assert trace["trace_complete"] is True
    assert trace["step_count"] == 2
    assert trace["complete_steps"] == 2
    assert trace["chain_complete"] is True
    assert trace["correction_converged"] is True
    assert trace["total_corrections"] >= trace["total_prediction_errors"]
