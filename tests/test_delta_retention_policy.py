from sara_engine.learning.delta_retention_policy import (
    DeltaRetentionPolicyConfig,
    evaluate_delta_erase_write_decoupling,
    evaluate_delta_retention_policy,
    evaluate_delta_retention_policy_stress,
)


def test_delta_retention_policy_preserves_crystal_and_forgets_liquid_context():
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

    assert report["observed_only"] is True
    assert report["metrics"]["delta_memory_phase_retention_policy_observed"] == 1.0
    assert report["metrics"]["delta_memory_crystal_retention_observed"] == 1.0
    assert report["metrics"]["delta_memory_liquid_forget_observed"] == 1.0
    assert report["metrics"]["delta_memory_astro_gate_alignment_observed"] == 1.0
    assert report["metrics"]["delta_memory_policy_state_budget_observed"] == 1.0
    assert report["traces"][0]["retention_gate"] > report["traces"][2]["retention_gate"]


def test_delta_retention_policy_reports_state_budget_pressure():
    memory_events = [
        {
            "phase": "crystal",
            "astro_stability": 1.0,
            "context_events": [index],
            "predicted_events": [],
            "observed_events": [900 + index],
        }
        for index in range(4)
    ]

    report = evaluate_delta_retention_policy(
        memory_events,
        DeltaRetentionPolicyConfig(capacity=2),
    )

    assert report["snapshot"]["state_units"] <= 2
    assert report["metrics"]["delta_memory_policy_state_budget_observed"] == 1.0


def test_delta_retention_policy_stress_preserves_multiple_histories_without_cross_leak():
    histories = [
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
            "branch_id": "temporary-topic",
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

    report = evaluate_delta_retention_policy_stress(
        histories,
        DeltaRetentionPolicyConfig(capacity=8),
    )

    assert report["observed_only"] is True
    assert report["metrics"]["delta_memory_multi_history_recall_observed"] == 1.0
    assert report["metrics"]["delta_memory_multi_history_noise_resilience_observed"] == 1.0
    assert report["metrics"]["delta_memory_multi_history_health_observed"] == 1.0
    assert report["metrics"]["delta_memory_multi_history_manifold_guard_observed"] == 1.0
    assert report["unrelated_probe"]["predicted_ids"] == []


def test_delta_erase_write_decoupling_preserves_stable_memory_and_commits_residual():
    events = [
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
    ]

    report = evaluate_delta_erase_write_decoupling(
        events,
        DeltaRetentionPolicyConfig(capacity=6),
    )

    assert report["observed_only"] is True
    assert report["metrics"]["delta_memory_erase_write_decoupling_observed"] == 1.0
    assert report["metrics"]["delta_memory_erase_preserves_stable_memory_observed"] == 1.0
    assert report["metrics"]["delta_memory_write_commits_residual_observed"] == 1.0
    assert report["traces"][1]["erase_gate"] < report["traces"][2]["write_gate"]
