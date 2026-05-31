from sara_engine.nn.multi_timescale_leak_state import (
    MultiTimescaleLeakState,
    evaluate_multi_timescale_leak_state,
)


def test_multi_timescale_leak_state_retains_long_group_longer() -> None:
    state = MultiTimescaleLeakState(max_state_units=12)

    state.update(input_events=[101])
    state.step(count=3)
    values = state.read_event(101)

    assert values["long"] > values["mid"] > values["short"]
    assert values["long"] > 0.75
    assert values["short"] < 0.05


def test_multi_timescale_leak_state_uses_sparse_event_budget() -> None:
    state = MultiTimescaleLeakState(max_state_units=6)

    trace = state.update(input_events=[1, 2, 3, 4])
    snapshot = state.snapshot()

    assert trace["state_budget_ok"] is True
    assert trace["evicted_units"] == 6
    assert snapshot["state_units"] == 6
    assert snapshot["state_budget_ok"] is True
    assert all(len(group["entries"]) <= 4 for group in snapshot["groups"].values())


def test_multi_timescale_leak_state_tracks_active_events_by_threshold() -> None:
    state = MultiTimescaleLeakState(max_state_units=12)

    first = state.update(input_events=[7])
    state.step(count=2)
    active = state.active_events()

    assert first["active_events"]["short"] == [7]
    assert 7 not in active["short"]
    assert 7 in active["mid"]
    assert 7 in active["long"]


def test_multi_timescale_leak_state_evaluation_reports_observed_metrics() -> None:
    report = evaluate_multi_timescale_leak_state()

    assert report["observed_only"] is True
    assert report["metrics"]["multi_timescale_leak_retention"] == 1.0
    assert report["metrics"]["multi_timescale_long_state_activity"] == 1.0
    assert report["metrics"]["timescale_state_budget_integrity"] == 1.0


def test_multi_timescale_leak_state_is_exposed_from_nn_package() -> None:
    import sara_engine.nn as nn

    state = nn.MultiTimescaleLeakState(max_state_units=3)
    state.update(input_events=[1])

    assert state.snapshot()["state_units"] == 3
    assert nn.evaluate_multi_timescale_leak_state()["metrics"][
        "multi_timescale_leak_retention"
    ] == 1.0
