from sara_engine.evaluation.r2_beijing_development import LocalArm, SensorPair, decide, load_protocol
from datetime import datetime


def pair(label=1):
    return SensorPair(datetime(2015, 1, 1), "training", "Aotizhongxin", 0, 1,
                      (50.0, 2.0, 3.0, 4.0, 10.0, 0.0, 1010.0, 2.0, 0.0, "NW", ()), label)


def test_spiking_and_scalar_local_arms_match_scores():
    protocol = load_protocol()
    snn = LocalArm(protocol, "snn_local_residual")
    scalar = LocalArm(protocol, "scalar_local_residual")
    for index in range(20):
        target = index % 2
        left, _ = snn.step(pair(target), target, index * 3)
        right, _ = scalar.step(pair(target), target, index * 3)
        assert left == right


def test_three_factor_arm_uses_its_own_update_receipt_contract():
    protocol = load_protocol()
    learner = LocalArm(protocol, "snn_three_factor")
    score, work = learner.step(pair(1), 1, 0)
    assert score == 0.0
    assert work > 0


def test_decision_blocks_frozen_test_when_a_gate_fails():
    protocol = load_protocol()
    metric = lambda ba, brier: {"balanced_accuracy": ba, "brier": brier}
    result = lambda ba, brier, digest="same": {"development": metric(ba, brier),
        "worst_station_balanced_accuracy": ba, "prediction_trace_sha256": digest,
        "resources": {"contracts_passed": True}}
    arms = {
        "snn_local_residual": result(.58, .23), "scalar_local_residual": result(.58, .23),
        "snn_three_factor": result(.55, .25), "snn_frozen": result(.50, .25),
        "snn_shuffled_outcomes": result(.50, .25), "snn_temporal_order_destroyed": result(.56, .24),
        "online_station_hour_direction_transition": result(.554, .246),
    }
    assert decide(protocol, arms)["frozen_test_authorized"]
    arms["scalar_local_residual"]["prediction_trace_sha256"] = "different"
    assert not decide(protocol, arms)["frozen_test_authorized"]
