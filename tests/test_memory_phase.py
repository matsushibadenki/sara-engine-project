from sara_engine.learning.memory_phase import (
    MemoryPhaseConfig,
    build_memory_phase_observations,
    evaluate_memory_phase_transitions,
)


def test_memory_phase_tracks_liquid_glass_crystal_without_overfixing_noise():
    observations = [
        {"step": 1, "memory_id": "anchor", "stability": 0.18, "replay_success": 0.20, "interference": 0.20},
        {"step": 2, "memory_id": "anchor", "stability": 0.56, "replay_success": 0.62, "interference": 0.16},
        {"step": 3, "memory_id": "anchor", "stability": 0.84, "replay_success": 0.90, "interference": 0.07},
        {"step": 1, "memory_id": "fresh", "stability": 0.14, "replay_success": 0.10, "interference": 0.30},
        {"step": 2, "memory_id": "fresh", "stability": 0.20, "replay_success": 0.16, "interference": 0.28},
        {"step": 1, "memory_id": "noise", "stability": 0.50, "replay_success": 0.20, "interference": 0.72},
    ]

    report = evaluate_memory_phase_transitions(observations)

    anchor = next(track for track in report["phase_tracks"] if track["memory_id"] == "anchor")
    noise = next(track for track in report["phase_tracks"] if track["memory_id"] == "noise")
    assert report["observed_only"] is True
    assert anchor["phase_path"] == ["liquid", "glass", "crystal"]
    assert anchor["final_retention"] >= 0.75
    assert noise["final_phase"] != "crystal"
    assert report["metrics"]["memory_phase_transition_integrity"] == 1.0
    assert report["metrics"]["memory_phase_overfixation_guard_observed"] == 1.0


def test_memory_phase_reports_state_budget_pressure():
    observations = [
        {"step": 1, "memory_id": f"memory-{index}", "stability": 0.2, "replay_success": 0.1}
        for index in range(3)
    ]

    report = evaluate_memory_phase_transitions(observations, MemoryPhaseConfig(state_budget=2))

    assert len(report["phase_tracks"]) == 3
    assert report["metrics"]["memory_phase_state_budget_observed"] == 0.0


def test_memory_phase_can_project_replay_events_into_phase_observations():
    observations = build_memory_phase_observations(
        [
            {
                "memory_id": "anchor",
                "post_retention": 0.88,
                "post_noise": 0.08,
                "health_after": 0.86,
            },
            {
                "memory_id": "fresh",
                "post_retention": 0.28,
                "post_noise": 0.52,
                "health_after": 0.32,
            },
        ],
        step=10,
    )

    assert observations[0]["memory_id"] == "anchor"
    assert observations[0]["step"] == 10
    assert observations[0]["stability"] > observations[1]["stability"]
    assert observations[0]["interference"] < observations[1]["interference"]
