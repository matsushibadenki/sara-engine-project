from sara_engine.learning.structural_plasticity import BoundedStructuralPlasticityController
from sara_engine.memory.concept_review_loop import ConceptReviewLoop
from sara_engine.risa import (
    RisaObservation,
    SARAAlignedRisaKernel,
    build_feedback_package,
    route_key_for_relation,
    run_risa_structural_plasticity_cycle,
)


def _toy_kernel() -> SARAAlignedRisaKernel:
    kernel = SARAAlignedRisaKernel(min_support=2, min_distinct_actors=2)
    kernel.ingest_observations(
        (
            RisaObservation(
                timestamp=1,
                event_id="e1",
                actor="dog",
                action="run",
                observed_effects=["fatigue_up"],
                verified=True,
                resonance_score=0.7,
            ),
            RisaObservation(
                timestamp=2,
                event_id="e2",
                actor="human",
                action="run",
                observed_effects=["fatigue_up"],
                verified=True,
                resonance_score=0.8,
            ),
            RisaObservation(
                timestamp=3,
                event_id="e3",
                actor="horse",
                action="run",
                observed_effects=["fatigue_up"],
                verified=True,
                resonance_score=0.8,
            ),
        )
    )
    return kernel


def test_risa_structural_cycle_stabilizes_supported_predictive_route() -> None:
    kernel = _toy_kernel()
    feedback = build_feedback_package(kernel, current_segment=6)
    review_result = ConceptReviewLoop().run(
        feedback.revalidation_entries,
        feedback.candidate_relations,
        current_segment=6,
    )
    controller = BoundedStructuralPlasticityController(
        min_stable_verified_support=2,
        min_stable_prediction_gain=0.6,
        max_rewrites_per_event=4,
    )

    result = run_risa_structural_plasticity_cycle(
        controller,
        kernel,
        review_result=review_result,
        feedback_package=feedback,
        current_segment=6,
    )

    route = route_key_for_relation("process:run", "state:fatigue_up", "predicts")
    assert result.structural_result.update_allowed is True
    assert route in controller.routes
    assert controller.routes[route].route_state in {"stable", "provisional"}
    assert result.support_route_count >= 1


def test_risa_structural_cycle_prunes_dormant_route_under_contradiction_pressure() -> None:
    kernel = _toy_kernel()
    concept_id = "concept:shared_run_fatigue_up"
    concept = kernel.state.graph.get_node(concept_id)
    assert concept is not None
    concept.dormant = True
    concept.energy = 0.01
    route = route_key_for_relation(concept_id, "state:fatigue_up", "predicts")
    controller = BoundedStructuralPlasticityController(
        prune_grace_steps=1,
        contradiction_prune_threshold=1,
        max_rewrites_per_event=4,
    )
    controller.register_route(
        route,
        weight=0.3,
        route_state="decaying",
        responsibility=0.01,
        prediction_gain_support=0.1,
        contradiction_count=1,
        support_count=1,
        verified_support_count=1,
        created_step=0,
        last_active_step=0,
    )
    feedback = build_feedback_package(kernel, current_segment=8, skip_dormant=False)
    review_result = ConceptReviewLoop().run(
        feedback.revalidation_entries,
        feedback.candidate_relations,
        current_segment=8,
    )

    result = run_risa_structural_plasticity_cycle(
        controller,
        kernel,
        review_result=review_result,
        feedback_package=feedback,
        current_segment=8,
    )

    assert route in result.structural_result.pruned_routes
    assert route not in controller.routes


def test_risa_structural_cycle_separates_relation_types_in_route_space() -> None:
    kernel = _toy_kernel()
    feedback = build_feedback_package(kernel, current_segment=6)
    review_result = ConceptReviewLoop().run(
        feedback.revalidation_entries,
        feedback.candidate_relations,
        current_segment=6,
    )
    controller = BoundedStructuralPlasticityController(max_rewrites_per_event=8)

    result = run_risa_structural_plasticity_cycle(
        controller,
        kernel,
        review_result=review_result,
        feedback_package=feedback,
        current_segment=6,
    )

    predicts_route = route_key_for_relation("concept:shared_run_fatigue_up", "state:fatigue_up", "predicts")
    participates_route = route_key_for_relation("concept:shared_run_fatigue_up", "process:run", "participates_in")
    assert predicts_route in controller.routes
    assert participates_route in controller.routes
    assert predicts_route != participates_route
    predicts_label = result.route_labels[f"{predicts_route[0]}:{predicts_route[1]}"]
    participates_label = result.route_labels[f"{participates_route[0]}:{participates_route[1]}"]
    assert predicts_label["relation_type"] == "predicts"
    assert participates_label["relation_type"] == "participates_in"


def test_risa_structural_cycle_keeps_nonpredictive_growth_more_conservative() -> None:
    kernel = _toy_kernel()
    controller = BoundedStructuralPlasticityController(
        provisional_growth_threshold=0.5,
        max_rewrites_per_event=8,
    )
    review_result = ConceptReviewLoop().run((), (), current_segment=6)

    result = run_risa_structural_plasticity_cycle(
        controller,
        kernel,
        review_result=review_result,
        feedback_package=None,
        current_segment=6,
    )

    predicts_route = route_key_for_relation("process:run", "state:fatigue_up", "predicts")
    observes_route = route_key_for_relation("event:e1", "state:fatigue_up", "observes")
    assert predicts_route in controller.routes
    assert observes_route in controller.routes
    assert controller.routes[predicts_route].prediction_gain_support >= controller.routes[observes_route].prediction_gain_support
    assert controller.routes[predicts_route].responsibility >= controller.routes[observes_route].responsibility
    assert result.support_route_count >= 2


def test_risa_structural_cycle_builds_relation_class_feedback_from_replay_and_phase() -> None:
    kernel = _toy_kernel()
    feedback = build_feedback_package(kernel, current_segment=6)
    review_result = ConceptReviewLoop().run(
        feedback.revalidation_entries,
        feedback.candidate_relations,
        current_segment=6,
    )
    controller = BoundedStructuralPlasticityController(max_rewrites_per_event=8)

    result = run_risa_structural_plasticity_cycle(
        controller,
        kernel,
        review_result=review_result,
        feedback_package=feedback,
        current_segment=6,
        idle_replay_report={
            "selected": (
                {
                    "entry_id": "memory-predicts",
                    "own_latent_id": "predicts:process:run->state:fatigue_up",
                    "replay_score": 0.94,
                    "event_cost": 4,
                },
                {
                    "entry_id": "memory-observes",
                    "own_latent_id": "observes:event:e1->state:fatigue_up",
                    "replay_score": 0.22,
                    "event_cost": 11,
                },
            ),
        },
        memory_phase_report={
            "phase_tracks": (
                {
                    "memory_id": "memory-predicts",
                    "final_phase": "crystal",
                    "final_plasticity": 0.18,
                    "final_retention": 0.92,
                },
                {
                    "memory_id": "memory-observes",
                    "final_phase": "liquid",
                    "final_plasticity": 0.91,
                    "final_retention": 0.28,
                },
            ),
        },
    )

    class_feedback = result.signals["relation_class_feedback"]
    predicts_feedback = class_feedback["predicts"]
    observes_feedback = class_feedback["observes"]
    assert predicts_feedback["phase_maturity_mean"] == 1.0
    assert predicts_feedback["replay_score_mean"] >= 0.9
    assert predicts_feedback["stability_support_multiplier"] > 1.0
    assert predicts_feedback["prune_pressure"] < observes_feedback["prune_pressure"]
    assert predicts_feedback["active_route_multiplier"] > observes_feedback["active_route_multiplier"]


def test_risa_structural_cycle_applies_persistent_contradiction_per_relation_class() -> None:
    kernel = _toy_kernel()
    route = route_key_for_relation("process:run", "state:fatigue_up", "predicts")
    controller = BoundedStructuralPlasticityController(
        contradiction_growth_block=0.5,
        contradiction_prune_threshold=2,
        max_rewrites_per_event=8,
    )
    controller.register_route(
        route,
        weight=0.7,
        route_state="provisional",
        responsibility=0.7,
        prediction_gain_support=0.7,
        contradiction_count=2,
        support_count=2,
        verified_support_count=2,
    )
    review_result = ConceptReviewLoop().run((), (), current_segment=6)

    result = run_risa_structural_plasticity_cycle(
        controller,
        kernel,
        review_result=review_result,
        current_segment=6,
    )

    predicts_feedback = result.signals["relation_class_feedback"]["predicts"]
    assert predicts_feedback["historical_contradiction_mean"] == 0.5
    assert predicts_feedback["historical_contradiction_peak"] == 1.0
    assert predicts_feedback["contradiction_persistence"] >= 0.5
    assert controller.routes[route].contradiction_count == 3
