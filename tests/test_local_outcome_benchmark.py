from copy import deepcopy

from sara_engine.evaluation.local_outcome_benchmark import load_protocol, make_stream, run_arm, select_rule


def development_protocol():
    protocol = load_protocol()
    protocol.update(seeds=[41], warmup=16, evaluation=32)
    return protocol


def test_stream_replays_and_shared_inputs_are_equal():
    protocol = development_protocol()
    stream = make_stream(protocol, 41, "stationary")
    assert stream == make_stream(protocol, 41, "stationary")
    results = {arm: run_arm(protocol, arm, stream, 41) for arm in protocol["arms"]}
    assert len({r["input_sha256"] for r in results.values()}) == 1
    assert results["residual"]["prediction_trace_sha256"] == results["scalar_ema"]["prediction_trace_sha256"]
    assert results["residual"]["prediction_trace_sha256"] == run_arm(protocol, "residual", stream, 41)["prediction_trace_sha256"]
    assert results["residual"]["scored"] == 32
    assert results["frozen"]["updates"] == 0


def test_expired_feedback_cannot_train_any_arm():
    protocol = development_protocol()
    stream = make_stream(protocol, 41, "expired_feedback")
    for arm in protocol["arms"]:
        result = run_arm(protocol, arm, stream, 41)
        assert result["updates"] == 0
        assert result["brier"] == 0.25


def test_selection_rejects_resource_failure_and_integrator_regression():
    protocol = development_protocol()
    def metric(brier):
        return {"accuracy": 0.8, "brier": brier, "max_state_bytes": 1000,
                "max_episode_ms": 0.01, "updates": 0, "input_sha256": "same",
                "prediction_trace_sha256": "same"}
    row = {"seed": 41, "scenario": "noisy", "arms": {arm: metric(0.2) for arm in protocol["arms"]}}
    row["arms"]["residual"]["brier"] = 0.15
    row["arms"]["integrated"]["brier"] = 0.16
    assert select_rule(protocol, [row])["selected_component_rule"] == "residual"
    broken = deepcopy(row)
    broken["arms"]["residual"]["max_state_bytes"] = 10**9
    assert select_rule(protocol, [broken])["selected_component_rule"] == "three_factor"
    row["arms"]["integrated"]["brier"] = 0.13
    assert select_rule(protocol, [row])["selected_component_rule"] == "integrated"
