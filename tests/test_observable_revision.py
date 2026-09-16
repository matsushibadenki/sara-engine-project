from dataclasses import replace
import math

import pytest

from sara_engine.learning.observable_revision import BoundedObservableRevisionReadout, ObservableRevisionConfig


def test_revision_signal_switches_for_exact_horizon_then_returns_to_stable_rule():
    learner = BoundedObservableRevisionReadout(ObservableRevisionConfig(adaptation_horizon=2))
    first = learner.predict([(1, 1.0)], time=0)
    assert first.mode == "three_factor"
    learner.observe(first, 1.0, time=1)
    learner.notify_revision(2, time=2)
    modes = []
    for time_value in (2, 3, 4):
        receipt = learner.predict([(1, 1.0)], time=time_value)
        modes.append(receipt.mode)
        learner.observe(receipt, -1.0, time=time_value)
    assert modes == ["residual", "residual", "three_factor"]
    assert learner.snapshot()["adaptive_remaining"] == 0


def test_revision_must_increase_and_cannot_interrupt_pending_prediction():
    learner = BoundedObservableRevisionReadout()
    with pytest.raises(ValueError):
        learner.notify_revision(1, time=0)
    receipt = learner.predict([(1, 1.0)], time=1)
    before = learner.snapshot()
    with pytest.raises(ValueError):
        learner.notify_revision(2, time=1)
    assert learner.snapshot() == before
    learner.discard(receipt, time=1)
    learner.notify_revision(2, time=2)
    with pytest.raises(ValueError):
        learner.notify_revision(2, time=2)


def test_feedback_receipt_identity_and_invalid_outcome_are_atomic():
    learner = BoundedObservableRevisionReadout()
    receipt = learner.predict([(1, 1.0)], time=0)
    before = learner.snapshot()
    with pytest.raises(ValueError):
        learner.observe(replace(receipt), 1.0, time=1)
    assert learner.snapshot() == before
    with pytest.raises(ValueError):
        learner.observe(receipt, math.nan, time=1)
    assert learner.snapshot() == before
    learner.observe(receipt, 1.0, time=1)
    with pytest.raises(ValueError):
        learner.observe(receipt, 1.0, time=2)


def test_expired_adaptive_feedback_does_not_consume_horizon_or_change_weight():
    learner = BoundedObservableRevisionReadout(ObservableRevisionConfig(adaptation_horizon=2, max_feedback_age=2))
    learner.notify_revision(2, time=0)
    receipt = learner.predict([(1, 1.0)], time=0)
    result = learner.observe(receipt, 1.0, time=3)
    assert result.decision == "expired"
    assert learner.snapshot()["adaptive_remaining"] == 2
    assert learner.snapshot()["weights"] == {}


def test_false_revision_at_correct_bound_has_no_weight_harm():
    learner = BoundedObservableRevisionReadout(ObservableRevisionConfig(adaptation_horizon=4))
    for time_value in range(10):
        receipt = learner.predict([(1, 1.0)], time=time_value)
        learner.observe(receipt, 1.0, time=time_value)
    assert learner.snapshot()["weights"][1] == 1.0
    learner.notify_revision(2, time=10)
    for time_value in range(10, 14):
        receipt = learner.predict([(1, 1.0)], time=time_value)
        learner.observe(receipt, 1.0, time=time_value)
    assert learner.snapshot()["weights"][1] == 1.0


def test_route_and_iterator_budgets_fail_before_persistent_mutation():
    learner = BoundedObservableRevisionReadout(ObservableRevisionConfig(max_active=2, max_routes=2))
    consumed = []
    def source():
        for route in range(4):
            consumed.append(route)
            yield route, 1.0
    with pytest.raises(ValueError):
        learner.predict(source(), time=0)
    assert consumed == [0, 1, 2]
    assert learner.snapshot()["weights"] == {}
    assert learner.snapshot()["pending"] is None


@pytest.mark.parametrize("kwargs", [
    {"adaptation_horizon": 0}, {"max_routes": True}, {"max_active": 0},
    {"max_feedback_age": 1.5}, {"learning_rate": math.inf},
    {"trace_decay": 1.1}, {"max_delta": 0.0},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ObservableRevisionConfig(**kwargs)


def test_status_supports_required_languages():
    learner = BoundedObservableRevisionReadout()
    assert "Revision readout" in learner.get_status("en")
    assert "改訂" in learner.get_status("ja")
    assert "修订" in learner.get_status("zh-CN")
    assert learner.get_status("unknown") == learner.get_status("en")
