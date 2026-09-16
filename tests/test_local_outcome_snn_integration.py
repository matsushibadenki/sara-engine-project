from copy import deepcopy

import pytest

from sara_engine.evaluation.local_outcome_snn_integration import (
    FixedCoincidenceEncoder,
    decide,
    interval_bin,
    load_protocol,
    make_streams,
    route_id,
    run_arm,
)


def small_protocol():
    protocol = deepcopy(load_protocol())
    protocol["training_episodes"] = 192
    protocol["evaluation_episodes"] = 192
    return protocol


def test_address_space_distinguishes_order_and_three_intervals():
    values = {route_id(left, right, gap, 8) for left in range(8) for right in range(8) for gap in (2, 5, 10)}
    assert values == set(range(192))
    assert interval_bin(2) == 0
    assert interval_bin(5) == 1
    assert interval_bin(10) == 2
    with pytest.raises(ValueError):
        route_id(8, 0, 2, 8)
    with pytest.raises(ValueError):
        interval_bin(0)


def test_fresh_stream_is_balanced_deterministic_and_disjoint():
    protocol = small_protocol()
    training, evaluation = make_streams(protocol, 916103)
    assert (training, evaluation) == make_streams(protocol, 916103)
    assert {episode.identity for episode in training}.isdisjoint(episode.identity for episode in evaluation)
    assert sum(episode.label for episode in training) == 96
    assert sum(episode.label for episode in evaluation) == 96
    assert len({(episode.left, episode.right, episode.interval) for episode in training}) == 192


def test_fixed_encoder_uses_explicit_units_and_resets_refractory_state():
    protocol = small_protocol()
    episode = make_streams(protocol, 916103)[0][0]
    encoder = FixedCoincidenceEncoder(protocol)
    first = encoder.encode(episode)
    second = encoder.encode(episode)
    assert first == second
    assert len(encoder.units) == 192
    assert sum(unit.spike for unit in encoder.units) == 1


def test_timing_destruction_preserves_symbols_but_changes_long_route():
    protocol = small_protocol()
    episode = next(item for item in make_streams(protocol, 916103)[0] if item.interval == 10)
    intact = FixedCoincidenceEncoder(protocol).encode(episode)[0]
    destroyed = FixedCoincidenceEncoder(protocol, destroy_timing=True).encode(episode)[0]
    assert intact != destroyed
    assert intact // 3 == destroyed // 3


def test_actual_snn_residual_matches_equally_informed_scalar_control():
    protocol = small_protocol()
    training, evaluation = make_streams(protocol, 916103)
    snn = run_arm(protocol, "snn_residual", training, evaluation, 916103)
    scalar = run_arm(protocol, "scalar_residual", training, evaluation, 916103)
    assert snn["prediction_trace_sha256"] == scalar["prediction_trace_sha256"]
    assert snn["accuracy"] == scalar["accuracy"]
    assert snn["resources"]["units"] == 192
    assert scalar["resources"]["units"] == 0


def test_decision_fails_if_scalar_equivalence_or_resources_fail():
    protocol = small_protocol()
    def result(accuracy, brier, digest="same"):
        return {"accuracy": accuracy, "brier": brier, "prediction_trace_sha256": digest,
                "resources": {"contracts_passed": True}}
    arms = {
        "snn_residual": result(0.9, 0.05),
        "snn_three_factor": result(0.9, 0.10),
        "snn_timing_destroyed_residual": result(0.5, 0.25),
        "scalar_residual": result(0.9, 0.05),
        "snn_frozen": result(0.5, 0.25),
        "snn_shuffled_feedback": result(0.5, 0.25),
    }
    assert decide(protocol, [{"seed": 1, "arms": arms}])["passed"]
    arms["scalar_residual"]["prediction_trace_sha256"] = "different"
    assert not decide(protocol, [{"seed": 1, "arms": arms}])["passed"]
    arms["scalar_residual"]["prediction_trace_sha256"] = "same"
    arms["snn_residual"]["resources"]["contracts_passed"] = False
    assert not decide(protocol, [{"seed": 1, "arms": arms}])["passed"]
