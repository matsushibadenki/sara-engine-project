# Directory Path: tests/test_plastic_submodel_registry.py
# English Title: Plastic Submodel Registry Tests
# Purpose/Content: Verifies sparse specialist routing and local relearning traces for Stage E observed-only policy.

from sara_engine.nn.plastic_submodel_registry import (
    PlasticSubmodelRegistry,
    evaluate_plastic_submodel_credit_assignment_trace,
    evaluate_plastic_submodel_intervention_trace,
    evaluate_plastic_submodel_open_ended_hypothesis_bank_trace,
    evaluate_plastic_submodel_registry_trace,
    evaluate_plastic_submodel_scientific_model_trace,
    evaluate_plastic_submodel_structural_adaptation_trace,
)


def test_plastic_submodel_registry_routes_and_relearns_locally() -> None:
    registry = PlasticSubmodelRegistry(max_submodels=3, max_route_edges=2)
    registry.register("world_model", role="world", concepts=["release"], event_budget=4)
    registry.register("memory_system", role="memory", concepts=["gate"], event_budget=4)
    registry.connect("memory_system", "world_model", reason="memory supports world state")
    registry.relearn_local("world_model", positive_events=[101, 102], credit=1.0)

    route = registry.route([101], goal="world memory")
    trace = registry.concept_trace()

    assert route["state_budget_ok"] is True
    assert {"memory_system", "world_model"}.issubset(
        {item["submodel_id"] for item in route["selected_submodels"]}
    )
    assert route["connected_pairs"] == [{"source": "memory_system", "target": "world_model"}]
    assert any(event["event_type"] == "relearn" for event in trace["trace"])


def test_plastic_submodel_registry_evaluator_reports_observed_metrics() -> None:
    report = evaluate_plastic_submodel_registry_trace()

    assert report["observed_only"] is True
    assert report["metrics"]["plastic_submodel_registry_integrity"] == 1.0
    assert report["metrics"]["dynamic_submodel_route_integrity"] == 1.0
    assert report["metrics"]["submodel_relearning_trace_integrity"] == 1.0
    assert report["metrics"]["interpretable_submodel_concept_trace"] == 1.0
    assert report["route"]["state_budget_ok"] is True
    assert report["concept_trace"]["schema"] == "sara-plastic-submodel-concept-trace-v1"


def test_plastic_submodel_intervention_trace_tracks_ablation_and_recovery() -> None:
    report = evaluate_plastic_submodel_intervention_trace()

    assert report["observed_only"] is True
    assert report["metrics"]["submodel_intervention_trace_integrity"] == 1.0
    assert report["metrics"]["submodel_ablation_effect_observed"] == 1.0
    assert report["metrics"]["submodel_reactivation_recovery_observed"] == 1.0
    assert any(
        event["event_type"] == "deactivate"
        for event in report["concept_trace"]["trace"]
    )


def test_plastic_submodel_credit_assignment_updates_route_support_locally() -> None:
    report = evaluate_plastic_submodel_credit_assignment_trace()

    assert report["observed_only"] is True
    assert report["metrics"]["submodel_credit_assignment_trace_integrity"] == 1.0
    assert report["metrics"]["submodel_credit_selectivity_observed"] == 1.0
    assert report["metrics"]["submodel_credit_state_budget_observed"] == 1.0
    assert report["positive_feedback"]["updated_submodel_count"] > 0
    assert report["negative_feedback"]["updated_submodel_count"] > 0
    assert any(
        event["event_type"] == "route_credit"
        for event in report["concept_trace"]["trace"]
    )


def test_plastic_submodel_structural_adaptation_grows_and_prunes_bounded_edges() -> None:
    report = evaluate_plastic_submodel_structural_adaptation_trace()

    assert report["observed_only"] is True
    assert report["metrics"]["submodel_structural_adaptation_trace_integrity"] == 1.0
    assert report["metrics"]["submodel_structural_growth_bounded_observed"] == 1.0
    assert report["metrics"]["submodel_structural_pruning_observed"] == 1.0
    assert report["growth"]["created_edges"]
    assert report["pruning"]["pruned_edges"]
    assert report["growth"]["route_edge_count"] <= report["growth"]["route_edge_budget"]


def test_plastic_submodel_scientific_model_trace_revises_counterexample() -> None:
    report = evaluate_plastic_submodel_scientific_model_trace()

    assert report["observed_only"] is True
    assert report["metrics"]["submodel_scientific_hypothesis_trace_integrity"] == 1.0
    assert report["metrics"]["submodel_counterexample_revision_observed"] == 1.0
    assert report["metrics"]["submodel_scientific_model_budget_observed"] == 1.0
    assert report["prediction"]["trace_complete"] is True
    assert report["falsification"]["falsified"] is True
    assert report["revised_hypothesis"]["confidence"] < report["hypothesis"]["confidence"]
    assert report["revised_hypothesis"]["guard_condition"]


def test_plastic_submodel_hypothesis_bank_selects_and_prunes_bounded_models() -> None:
    report = evaluate_plastic_submodel_open_ended_hypothesis_bank_trace()

    assert report["observed_only"] is True
    assert report["metrics"]["submodel_hypothesis_bank_integrity"] == 1.0
    assert report["metrics"]["submodel_open_ended_selection_observed"] == 1.0
    assert report["metrics"]["submodel_hypothesis_bank_budget_observed"] == 1.0
    assert len(report["retained_hypotheses"]) == report["bank_capacity"]
    assert any(
        item["hypothesis_id"] == "language-only-release"
        for item in report["pruned_hypotheses"]
    )
