from sara_engine.learning.astro_structural_gate import (
    AstroStructuralGateConfig,
    evaluate_astro_structural_gate,
)


def test_astro_structural_gate_unlocks_then_locks_after_recovery():
    replay_steps = [
        {"world_model_event": "baseline", "prediction_error": 0.20, "replay_recovery": 0.20},
        {"world_model_event": "surprise", "prediction_error": 0.84, "replay_recovery": 0.10},
        {"world_model_event": "counterfactual", "prediction_error": 0.50, "replay_recovery": 0.70},
        {"world_model_event": "stable", "prediction_error": 0.14, "replay_recovery": 0.95},
    ]

    report = evaluate_astro_structural_gate(replay_steps)

    actions = [item["action"] for item in report["policy_trace"]]
    assert report["observed_only"] is True
    assert "unlock_structural_plasticity" in actions
    assert actions[-1] == "lock_to_bounded_stdp"
    assert report["final_structural_unlocked"] is False
    assert report["metrics"]["astro_structural_unlock_observed"] == 1.0
    assert report["metrics"]["astro_structural_lock_observed"] == 1.0
    assert report["metrics"]["astro_bounded_stdp_fallback_observed"] == 1.0
    assert report["metrics"]["world_model_replay_policy_trace_observed"] == 1.0


def test_astro_structural_gate_reports_policy_state_budget_pressure():
    replay_steps = [
        {"world_model_event": f"event-{index}", "prediction_error": 0.10, "replay_recovery": 0.80}
        for index in range(4)
    ]

    report = evaluate_astro_structural_gate(
        replay_steps,
        AstroStructuralGateConfig(max_policy_events=2),
    )

    assert len(report["policy_trace"]) == 4
    assert report["metrics"]["astro_policy_state_budget_observed"] == 0.0
