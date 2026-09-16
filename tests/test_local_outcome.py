from dataclasses import replace
import math

import pytest

from sara_engine.learning.local_outcome import BoundedLocalOutcomeReadout, LocalOutcomeConfig


def test_receipt_binds_pre_feedback_prediction_and_only_active_routes_learn():
    learner = BoundedLocalOutcomeReadout()
    first = learner.predict([(3, 1.0), (7, 0.5)], time=10)
    assert first.score == 0.0
    assert learner.snapshot()["weights"] == {}
    result = learner.observe(first, 1.0, time=11)
    assert set(learner.snapshot()["weights"]) == {3, 7}
    assert dict(result.deltas)[3] == pytest.approx(2 * dict(result.deltas)[7])
    assert result.residual == 1.0
    with pytest.raises(ValueError):
        learner.observe(first, -1.0, time=12)


def test_copied_and_cross_instance_receipts_cannot_update_state():
    learner, other = BoundedLocalOutcomeReadout(), BoundedLocalOutcomeReadout()
    receipt = learner.predict([(1, 1.0)], time=0)
    other_receipt = other.predict([(1, 1.0)], time=0)
    before = learner.snapshot()
    for invalid in (replace(receipt), other_receipt):
        with pytest.raises(ValueError):
            learner.observe(invalid, 1.0, time=1)
        assert learner.snapshot() == before


def test_expiry_consumes_receipt_without_learning_or_integral_change():
    learner = BoundedLocalOutcomeReadout(LocalOutcomeConfig(integral_rate=0.25))
    receipt = learner.predict([(1, 1.0)], time=0)
    result = learner.observe(receipt, 1.0, time=65)
    assert result.decision == "expired"
    assert learner.snapshot()["weights"] == learner.snapshot()["integrals"] == {}
    assert learner.snapshot()["pending"] is None


def test_age_boundary_and_delayed_attenuation():
    scores = []
    for delay in (0, 64, 65):
        learner = BoundedLocalOutcomeReadout()
        receipt = learner.predict([(1, 1.0)], time=0)
        learner.observe(receipt, 1.0, time=delay)
        scores.append(learner.snapshot()["weights"].get(1, 0.0))
    assert scores[1] == pytest.approx(scores[0] * 0.99**64)
    assert scores[2] == 0.0


def test_pending_prediction_must_be_resolved_explicitly():
    learner = BoundedLocalOutcomeReadout()
    receipt = learner.predict([(1, 1.0)], time=2)
    with pytest.raises(ValueError):
        learner.predict([(2, 1.0)], time=3)
    learner.discard(receipt, time=3)
    assert learner.predict([(2, 1.0)], time=4).sequence == 2


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, True, "1", 2.0])
def test_bad_feedback_does_not_consume_pending_receipt(invalid):
    learner = BoundedLocalOutcomeReadout()
    receipt = learner.predict([(1, 1.0)], time=2)
    before = learner.snapshot()
    with pytest.raises(ValueError):
        learner.observe(receipt, invalid, time=3)
    assert learner.snapshot() == before
    assert learner.observe(receipt, 1.0, time=3).decision == "updated"


@pytest.mark.parametrize("invalid", [-1, True, 1.5, math.inf, 2**53])
def test_invalid_time_is_rejected_atomically(invalid):
    learner = BoundedLocalOutcomeReadout()
    before = learner.snapshot()
    with pytest.raises(ValueError):
        learner.predict([(1, 1.0)], time=invalid)
    assert learner.snapshot() == before


def test_regressive_feedback_time_does_not_mutate_state():
    learner = BoundedLocalOutcomeReadout()
    receipt = learner.predict([(1, 1.0)], time=5)
    before = learner.snapshot()
    with pytest.raises(ValueError):
        learner.observe(receipt, 1.0, time=4)
    assert learner.snapshot() == before


@pytest.mark.parametrize("active", [[], [(1, 0.0)], [(1, math.nan)], [(True, 1.0)], [(1, 1.0), (1, 1.0)], [(2**63, 1.0)]])
def test_invalid_active_routes_fail_without_mutation(active):
    learner = BoundedLocalOutcomeReadout()
    before = learner.snapshot()
    with pytest.raises(ValueError):
        learner.predict(active, time=1)
    assert learner.snapshot() == before


def test_infinite_active_input_is_bounded_and_rejected():
    learner = BoundedLocalOutcomeReadout(LocalOutcomeConfig(max_active=2))
    consumed = []
    def source():
        route = 0
        while True:
            consumed.append(route)
            yield route, 1.0
            route += 1
    with pytest.raises(ValueError):
        learner.predict(source(), time=0)
    assert consumed == [0, 1, 2]
    assert learner.snapshot()["pending"] is None


def test_state_capacity_rejects_new_route_but_existing_route_still_learns():
    learner = BoundedLocalOutcomeReadout(LocalOutcomeConfig(max_routes=1, max_active=1))
    receipt = learner.predict([(1, 1.0)], time=0)
    learner.observe(receipt, 1.0, time=1)
    before = learner.snapshot()
    with pytest.raises(ValueError):
        learner.predict([(2, 1.0)], time=2)
    assert learner.snapshot() == before
    receipt = learner.predict([(1, 1.0)], time=2)
    learner.observe(receipt, -1.0, time=3)
    assert learner.snapshot()["weights"][1] < before["weights"][1]


def test_learning_corrects_reversal_without_unbounded_saturation():
    learner = BoundedLocalOutcomeReadout()
    for t in range(100):
        receipt = learner.predict([(1, 1.0)], time=t)
        learner.observe(receipt, 1.0, time=t)
    for t in range(100, 106):
        receipt = learner.predict([(1, 1.0)], time=t)
        learner.observe(receipt, -1.0, time=t)
    assert learner.snapshot()["weights"][1] < 0.0


def test_integrator_has_matched_initial_gain_and_capped_local_history():
    base = BoundedLocalOutcomeReadout()
    integral = BoundedLocalOutcomeReadout(LocalOutcomeConfig(integral_rate=0.25))
    for learner in (base, integral):
        receipt = learner.predict([(1, 1.0)], time=0)
        learner.observe(receipt, 1.0, time=1)
    assert base.snapshot()["weights"] == integral.snapshot()["weights"]
    for t in range(2, 1000):
        receipt = integral.predict([(1, 1.0)], time=t)
        result = integral.observe(receipt, 1.0 if t % 3 else -1.0, time=t)
        assert all(abs(delta) <= 0.5 + 1e-15 for _, delta in result.deltas)
    assert abs(integral.snapshot()["weights"][1]) <= 1.0
    assert abs(integral.snapshot()["integrals"][1][0]) <= 1.0
    assert base.snapshot()["integrals"] == {}


def test_old_integral_expires_before_next_update():
    learner = BoundedLocalOutcomeReadout(LocalOutcomeConfig(integral_rate=0.25, integral_max_age=2))
    receipt = learner.predict([(1, 1.0)], time=0)
    learner.observe(receipt, 1.0, time=0)
    receipt = learner.predict([(1, 1.0)], time=3)
    learner.observe(receipt, -1.0, time=3)
    assert learner.snapshot()["integrals"][1][0] == pytest.approx(0.25 * (-1.0 - receipt.score))


def test_zero_error_is_quiet_without_integrator():
    learner = BoundedLocalOutcomeReadout()
    receipt = learner.predict([(1, 1.0)], time=0)
    assert learner.observe(receipt, receipt.score, time=0).deltas == ((1, 0.0),)


@pytest.mark.parametrize("kw", [{"max_active": 0}, {"max_routes": True}, {"learning_rate": math.nan}, {"max_age": 1.5}, {"trace_decay": 1.1}, {"integral_max_age": 0}, {"max_delta": 0.0}])
def test_bad_configuration_is_rejected(kw):
    with pytest.raises(ValueError):
        LocalOutcomeConfig(**kw)


def test_replay_and_input_permutation_are_deterministic():
    a, b = BoundedLocalOutcomeReadout(), BoundedLocalOutcomeReadout()
    for t in range(10):
        for learner, active in ((a, [(1, 0.3), (2, 0.7)]), (b, [(2, 0.7), (1, 0.3)])):
            receipt = learner.predict(active, time=t)
            learner.observe(receipt, 1.0, time=t)
    assert a.snapshot() == b.snapshot()
    copy = a.snapshot()
    copy["weights"].clear()
    assert a.snapshot()["weights"]


def test_status_supports_three_languages_and_english_fallback():
    learner = BoundedLocalOutcomeReadout()
    assert "Local readout" in learner.get_status()
    assert "局所" in learner.get_status("ja")
    assert "局部" in learner.get_status("zh-CN")
    assert learner.get_status("unknown") == learner.get_status("en")


def test_public_api_uses_selected_rule_and_accepts_actual_neuron_events():
    from sara_engine.learning import BoundedLocalOutcomeReadout as PublicReadout
    from sara_engine.learning import LocalOutcomeConfig as PublicConfig
    from sara_engine.neuro.neuron import Neuron

    assert PublicReadout is BoundedLocalOutcomeReadout
    assert PublicConfig is LocalOutcomeConfig
    neuron = Neuron(42, num_branches=2)
    neuron.add_input_to_branch(0, 0.8)
    neuron.add_input_to_branch(0, 0.8)
    assert neuron.step()
    learner = PublicReadout()
    receipt = learner.predict([(neuron.id, 1.0)], time=0)
    learner.observe(receipt, 1.0, time=1)
    assert learner.snapshot()["weights"][42] > 0.0
    assert learner.config.integral_rate == 0.0
    assert learner.snapshot()["integrals"] == {}
