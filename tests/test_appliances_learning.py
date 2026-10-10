"""Delayed credit, actual writes, sufficient controls and frozen replay tests."""
from dataclasses import replace
from datetime import datetime, timedelta
import json
import random
from hashlib import sha256
from pathlib import Path

import pytest

from sara_engine.evaluation.appliances_learning import (
    DelayedCounts, decide, deliver, load_protocol, project, run_phase, shuffled_targets,
)
from sara_engine.evaluation.appliances_prospective import DemandExample
from sara_engine.utils.project_paths import model_path, processed_data_path


def active(protocol):
    return project((40.,) * 36, datetime(2016, 2, 1), "ordinal", protocol)[0]


def test_frozen_protocol_loads():
    assert load_protocol()["active_routes"] == 40


def test_early_or_late_outcome_not_even_read():
    protocol = load_protocol()
    model = DelayedCounts(protocol)
    receipt = model.predict(active(protocol), datetime(2016, 2, 1))
    def forbidden():
        raise AssertionError("Outcome accessed outside due time")
    for now in (receipt.cutoff, receipt.due - timedelta(minutes=10), receipt.due + timedelta(minutes=10)):
        with pytest.raises(ValueError):
            deliver(model, receipt, now, forbidden, training=True)
    assert model.counts == {} and model.counter_writes == 0
    deliver(model, receipt, receipt.due, lambda: 1, training=True)
    assert model.counter_writes == model.changed_counters == 40


def test_forged_cross_owned_and_repeated_receipts_rejected():
    protocol = load_protocol()
    model, other = DelayedCounts(protocol), DelayedCounts(protocol)
    receipt = model.predict(active(protocol), datetime(2016, 2, 1))
    for owner, invalid in ((model, replace(receipt)), (other, receipt)):
        with pytest.raises(ValueError):
            owner.resolve(invalid, receipt.due, label=1, training=True)
    model.resolve(receipt, receipt.due, label=0, training=True)
    with pytest.raises(ValueError):
        model.resolve(receipt, receipt.due, label=0, training=True)
    assert model.counter_writes == 40


def test_pending_cap_rejects_before_mutation():
    protocol = load_protocol()
    model = DelayedCounts(protocol)
    start = datetime(2016, 2, 1)
    for tick in range(3):
        model.predict(active(protocol), start + timedelta(minutes=10 * tick))
    with pytest.raises(ValueError):
        model.predict(active(protocol), start + timedelta(minutes=30))
    assert model.serial == 3 and len(model.pending) == 3


@pytest.mark.parametrize("label", [True, 2, -1, float("nan")])
def test_invalid_outcome_is_atomic(label):
    protocol = load_protocol()
    model = DelayedCounts(protocol)
    receipt = model.predict(active(protocol), datetime(2016, 2, 1))
    with pytest.raises(ValueError):
        model.resolve(receipt, receipt.due, label=label, training=True)
    assert model.counts == {} and model.counter_writes == 0


def test_sufficient_ordinal_gap_and_histogram_information():
    protocol = load_protocol()
    now = datetime(2016, 2, 1)
    history = (20., 40., 60.) * 11 + (20., 40., 60.)
    permuted = (40., 20., 60.) * 11 + (20., 40., 60.)
    ordinal = project(history, now, "ordinal", protocol)[0]
    assert ordinal == project(history, now, "transition_gap", protocol)[0]
    assert ordinal != project(permuted, now, "ordinal", protocol)[0]
    assert project(history, now, "histogram", protocol)[0] == project(permuted, now, "histogram", protocol)[0]
    recovered = tuple(route % protocol["route_stride"] for route, _ in ordinal[:36])
    assert recovered == tuple(int(value // 20) for value in history)


def test_actual_neuron_matches_independent_scalar_recurrence():
    protocol = load_protocol()
    rng = random.Random(913)
    for _ in range(100):
        history = tuple(rng.randrange(150) * 10. for _ in range(36))
        assert project(history, datetime(2016, 2, 1), "joint", protocol) == project(
            history, datetime(2016, 2, 1), "joint", protocol, scalar_lif=True)


def examples(split, day, count=12):
    start = datetime(2016, 2 if split == "train" else 3, day)
    return [DemandExample(split, start + timedelta(minutes=10 * i),
                          start + timedelta(minutes=10 * i + 30),
                          tuple(float(20 + (i + j) % 5 * 20) for j in range(36)), i % 3 == 0 and 1 or 0)
            for i in range(count)]


def test_all_arms_have_equal_changed_writes_frozen_replay_and_three_pending():
    protocol = load_protocol()
    for arm in protocol["arms"]:
        model = DelayedCounts(protocol)
        train = run_phase(model, examples("train", 1), arm, protocol, training=True)
        assert train["actual_counter_writes"] == train["changed_counters_total"] == 12 * 40
        assert train["peak_pending"] == 3 and not model.pending
        checkpoint = json.loads(json.dumps(model.checkpoint()))
        evaluated = run_phase(model, examples("development", 1), arm, protocol, training=False)
        assert evaluated["actual_counter_writes"] == 0 and model.checkpoint() == checkpoint
        fresh = DelayedCounts(protocol)
        fresh.restore(checkpoint)
        replayed = run_phase(fresh, examples("development", 1), arm, protocol, training=False)
        assert replayed["prediction_sha256"] == evaluated["prediction_sha256"]


def test_feedback_order_is_causal_at_equal_timestamps():
    protocol = load_protocol()
    model = DelayedCounts(protocol)
    data = examples("train", 1, 4)
    original = model.predict
    writes_at_prediction = []
    def observed(active_routes, cutoff):
        writes_at_prediction.append(model.counter_writes)
        return original(active_routes, cutoff)
    model.predict = observed
    run_phase(model, data, "ordinal", protocol, training=True)
    assert writes_at_prediction == [0, 0, 0, 40]


def test_shuffle_preserves_each_training_day_support():
    protocol = load_protocol()
    data = examples("train", 1) + examples("train", 2)
    targets = shuffled_targets(data, protocol)
    for day in (1, 2):
        subset = [example for example in data if example.cutoff.day == day]
        assert sum(targets[example.cutoff] for example in subset) == sum(example.label for example in subset)
    assert targets == shuffled_targets(data, protocol)


def test_bad_checkpoint_does_not_mutate_state():
    model = DelayedCounts(load_protocol())
    with pytest.raises(ValueError):
        model.restore({"0": [0, 1], "05120": [0, 1]})
    assert model.counts == {}


def test_retained_real_counter_totals_hashes_and_negative_decision():
    protocol = load_protocol()
    retained = json.loads(Path(processed_data_path("benchmark_fixtures", "appliances_learning_result_v1.json")).read_text())
    for arm in protocol["arms"]:
        row = retained["arms"][arm]
        raw = Path(model_path("appliances", arm + "_v1.json")).read_bytes()
        assert sha256(raw).hexdigest() == row["checkpoint"]["sha256"]
        counts = json.loads(raw)
        assert sum(sum(pair) for pair in counts.values()) == row["training"]["actual_counter_writes"] == 282400
        assert sum(pair[1] for pair in counts.values()) == 1665 * 40
        assert row["development"]["actual_counter_writes"] == 0
    assert decide(retained["arms"], protocol) == retained["decision"]
    assert retained["decision"]["passed"] is False
    assert retained["decision"]["test_unsealing_authorized"] is False
    assert retained["sealed_numeric_values_read"] == 0
