from sara_engine.learning.metabolic_budget import (
    MetabolicBudgetConfig,
    evaluate_structural_metabolic_budget,
)


def test_metabolic_budget_rejects_low_importance_growth_under_pressure():
    operations = [
        {"kind": "grow", "synapse_delta": 2, "event_cost": 0.70, "reserve_cost": 0.18, "importance": 0.82},
        {"kind": "rewire", "synapse_delta": 1, "event_cost": 0.55, "reserve_cost": 0.14, "importance": 0.76},
        {"kind": "grow", "synapse_delta": 2, "event_cost": 0.80, "reserve_cost": 0.20, "importance": 0.68},
        {"kind": "grow", "synapse_delta": 1, "event_cost": 0.70, "reserve_cost": 0.18, "importance": 0.24},
        {"kind": "prune", "synapse_delta": -2, "event_cost": 0.20, "reserve_cost": 0.02, "importance": 0.12, "reason": "low_importance_under_pressure"},
    ]

    report = evaluate_structural_metabolic_budget(
        operations,
        MetabolicBudgetConfig(max_synapses=6, event_budget=3.0, plasticity_reserve=0.72),
    )

    assert report["observed_only"] is True
    assert report["metrics"]["metabolic_budget_integrity"] == 1.0
    assert report["metrics"]["plasticity_reserve_integrity"] == 1.0
    assert report["metrics"]["structural_growth_bounded_observed"] == 1.0
    assert report["metrics"]["pruning_reason_trace_observed"] == 1.0
    assert report["rejected_operations"][0]["reason"] == "low_importance_under_resource_pressure"
    assert report["pruning_trace"][0]["reason"] == "low_importance_under_pressure"


def test_metabolic_budget_reports_synapse_budget_rejection():
    operations = [
        {"kind": "grow", "synapse_delta": 2, "event_cost": 0.10, "reserve_cost": 0.02, "importance": 0.90},
        {"kind": "grow", "synapse_delta": 2, "event_cost": 0.10, "reserve_cost": 0.02, "importance": 0.90},
    ]

    report = evaluate_structural_metabolic_budget(
        operations,
        MetabolicBudgetConfig(max_synapses=3, event_budget=2.0, plasticity_reserve=1.0),
    )

    assert report["synapse_count"] == 2
    assert report["rejected_operations"][0]["reason"] == "synapse_budget_limit"
