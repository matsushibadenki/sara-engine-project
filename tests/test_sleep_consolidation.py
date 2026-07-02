from sara_engine.learning.sleep_consolidation import (
    SleepConsolidationConfig,
    evaluate_sleep_consolidation,
)


def test_sleep_consolidation_observes_retention_noise_and_energy_budget():
    replay_events = [
        {
            "memory_id": "anchor",
            "baseline_retention": 0.76,
            "post_retention": 0.86,
            "baseline_noise": 0.25,
            "post_noise": 0.18,
            "health_before": 0.74,
            "health_after": 0.82,
            "multimodal_bundle_affinity": 1.0,
            "event_cost": 0.40,
            "latent_branch_count": 3,
            "selected_branch": "stable-branch",
        },
        {
            "memory_id": "handoff",
            "baseline_retention": 0.72,
            "post_retention": 0.80,
            "baseline_noise": 0.22,
            "post_noise": 0.17,
            "health_before": 0.72,
            "health_after": 0.79,
            "event_cost": 0.35,
            "latent_branch_count": 2,
            "selected_branch": "handoff-branch",
        },
    ]

    report = evaluate_sleep_consolidation(
        replay_events,
        SleepConsolidationConfig(event_budget=1.0),
    )

    assert report["observed_only"] is True
    assert report["event_budget_ok"] is True
    assert report["metrics"]["sleep_consolidation_retention_observed"] == 1.0
    assert report["metrics"]["latent_replay_noise_resilience_observed"] == 1.0
    assert report["metrics"]["sleep_consolidation_memory_health_observed"] == 1.0
    assert report["metrics"]["latent_replay_counterfactual_branch_observed"] == 1.0
    assert report["metrics"]["multimodal_bundle_sleep_observed"] == 1.0
    assert report["metrics"]["sleep_consolidation_energy_budget_observed"] == 1.0
    assert report["traces"][0]["retention_delta"] > 0.0
    assert report["traces"][0]["noise_delta"] < 0.0
    assert report["traces"][0]["multimodal_bundle_affinity"] == 1.0


def test_sleep_consolidation_reports_energy_budget_failure():
    replay_events = [
        {
            "memory_id": "anchor",
            "baseline_retention": 0.76,
            "post_retention": 0.84,
            "baseline_noise": 0.25,
            "post_noise": 0.18,
            "health_before": 0.74,
            "health_after": 0.80,
            "event_cost": 1.2,
            "latent_branch_count": 2,
            "selected_branch": "stable-branch",
        }
    ]

    report = evaluate_sleep_consolidation(
        replay_events,
        SleepConsolidationConfig(event_budget=1.0),
    )

    assert report["event_budget_ok"] is False
    assert report["metrics"]["sleep_consolidation_energy_budget_observed"] == 0.0
