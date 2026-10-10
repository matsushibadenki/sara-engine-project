"""Prospective sealing, strong-current information, local credit and replay tests."""
from datetime import datetime, timedelta
import json
from hashlib import sha256
from pathlib import Path

import pytest

from sara_engine.evaluation.traffic_temporal import (
    LocalConjunctionCounts, TrafficExample, contexts, extract, load_protocol,
    run_phase, shuffled_targets,
)
from sara_engine.utils.project_paths import processed_data_path, project_path

HEADER = ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main",
          "weather_description", "date_time", "traffic_volume"]
WEATHER = (56, 0, 0, 2, "None", (("Clouds", "scattered clouds"),))


def rows(start, volumes):
    yield HEADER
    for index, volume in enumerate(volumes):
        yield ["None", "280", "0", "0", "50", "Clouds", "scattered clouds",
               str(start + timedelta(hours=index)), str(volume)]


def test_sealed_and_outside_start_cells_not_accessed():
    class Spy:
        def __init__(self, stamp):
            self.stamp = stamp
        def __getitem__(self, index):
            assert index == HEADER.index("date_time"), "Forbidden numeric/category access"
            return self.stamp
    examples, audit = extract([HEADER, Spy("2017-01-01 00:00:00"), Spy("2011-01-01 00:00:00")], load_protocol())
    assert not examples and audit["access"]["sealed_rows_timestamp_only"] == 1
    assert audit["access"].get("admitted_numeric_rows_read", 0) == 0


def test_future_only_changes_label_not_features():
    protocol = load_protocol()
    start = datetime(2013, 1, 1)
    a, _ = extract(rows(start, [2000, 3000, 3500, 3900, 4000]), protocol)
    b, _ = extract(rows(start, [2000, 3000, 3500, 3900, 3000]), protocol)
    assert len(a) == len(b) == 1 and (a[0].label, b[0].label) == (1, 0)
    for arm in protocol["arms"]:
        assert contexts(a[0].volumes, a[0].weather, a[0].cutoff, arm, protocol) == contexts(
            b[0].volumes, b[0].weather, b[0].cutoff, arm, protocol)


def test_duplicate_weather_rows_do_not_duplicate_outcomes():
    source = list(rows(datetime(2013, 1, 1), [2000, 3000, 3500, 3900, 4000]))
    duplicate = source[-2].copy()
    duplicate[5:7] = ["Rain", "light rain"]
    source.append(duplicate)
    examples, audit = extract(source, load_protocol())
    assert len(examples) == 1 and len(examples[0].weather[-1]) == 2
    assert audit["access"]["duplicate_hour_rows"] == 1


def test_conflicting_duplicate_or_missing_future_abstains():
    source = list(rows(datetime(2013, 1, 1), [2000, 3000, 3500, 3900, 4000]))
    conflicting = source[-1].copy()
    conflicting[-1] = "3000"
    examples, audit = extract(source + [conflicting], load_protocol())
    assert not examples and audit["access"]["invalid_hours"] == 1
    examples, _ = extract(source[:-1], load_protocol())
    assert not examples


def test_purged_chronological_splits():
    data, _ = extract(rows(datetime(2015, 12, 31, 20), [3000] * 12), load_protocol())
    boundary = datetime(2016, 1, 1)
    for example in data:
        assert example.due < boundary if example.split == "train" else example.cutoff - timedelta(hours=3) >= boundary


@pytest.mark.parametrize("temperature", ["0", "nan", "inf", "-200"])
def test_invalid_current_measurements_abstain(temperature):
    source = list(rows(datetime(2013, 1, 1), [2000, 3000, 3500, 3900, 4000]))
    source[-2][1] = temperature
    examples, _ = extract(source, load_protocol())
    assert not examples


def test_same_strong_current_and_histogram_distinct_order_local_branch():
    protocol = load_protocol()
    cutoff = datetime(2013, 5, 6, 9)
    a, b = (3250, 3500, 2750, 3000), (2750, 3500, 3250, 3000)
    for arm in ("current_exact", "current_weather", "history_histogram"):
        assert contexts(a, WEATHER, cutoff, arm, protocol) == contexts(b, WEATHER, cutoff, arm, protocol)
    assert contexts(a, WEATHER, cutoff, "temporal_pair", protocol)[:-1] == contexts(b, WEATHER, cutoff, "temporal_pair", protocol)[:-1]
    assert contexts(a, WEATHER, cutoff, "temporal_pair", protocol)[-1] != contexts(b, WEATHER, cutoff, "temporal_pair", protocol)[-1]


def test_local_updates_learn_distinct_conjunctions_without_global_gradient():
    protocol = load_protocol()
    cutoff = datetime(2013, 5, 6, 9)
    a, b = (3250, 3500, 2750, 3000), (2750, 3500, 3250, 3000)
    keys = [contexts(history, WEATHER, cutoff, "temporal_pair", protocol) for history in (a, b)]
    model = LocalConjunctionCounts(protocol)
    for index in range(20):
        mode = index % 2
        receipt = model.predict(keys[mode], cutoff + timedelta(hours=index))
        model.deliver(receipt, receipt[1], lambda: 1 - mode, training=True)
    probabilities = []
    for index, key in enumerate(keys):
        receipt = model.predict(key, cutoff + timedelta(hours=20 + index))
        probabilities.append(receipt[3])
        model.deliver(receipt, receipt[1], lambda: 0, training=False)
    assert probabilities[0] > .5 > probabilities[1]
    assert model.writes == model.changed == 80


def test_early_cross_owned_copied_and_repeated_outcomes_rejected_before_access():
    protocol = load_protocol()
    now = datetime(2013, 1, 1)
    key = contexts((3000,) * 4, WEATHER, now, "current_exact", protocol)
    model, other = LocalConjunctionCounts(protocol), LocalConjunctionCounts(protocol)
    receipt = model.predict(key, now)
    def forbidden():
        raise AssertionError("Outcome was read")
    for owner, invalid, time in ((model, receipt, now), (other, receipt, receipt[1]),
                                  (model, tuple(list(receipt)), receipt[1])):
        with pytest.raises(ValueError):
            owner.deliver(invalid, time, forbidden, training=True)
    model.deliver(receipt, receipt[1], lambda: 1, training=True)
    with pytest.raises(ValueError):
        model.deliver(receipt, receipt[1], forbidden, training=True)
    assert model.writes == 4


@pytest.mark.parametrize("label", [True, -1, 2, float("nan")])
def test_bad_label_leaves_counters_and_pending_unchanged(label):
    protocol = load_protocol()
    now = datetime(2013, 1, 1)
    model = LocalConjunctionCounts(protocol)
    receipt = model.predict(contexts((3000,) * 4, WEATHER, now, "current_exact", protocol), now)
    with pytest.raises(ValueError):
        model.deliver(receipt, receipt[1], lambda: label, training=True)
    assert model.counts == {} and model.pending is receipt


def examples(split):
    start = datetime(2013 if split == "train" else 2016, 1, 1)
    return [TrafficExample(split, start + timedelta(hours=index), start + timedelta(hours=index + 1),
                           (3250, 3500, 2750, 3000) if index % 2 else (2750, 3500, 3250, 3000),
                           WEATHER, index % 2) for index in range(24)]


def test_all_arms_equal_changed_writes_frozen_replay_and_day_support_shuffle():
    protocol = load_protocol()
    targets = shuffled_targets(examples("train"), protocol)
    assert sum(targets.values()) == 12
    for arm in protocol["arms"]:
        model = LocalConjunctionCounts(protocol)
        train = run_phase(model, examples("train"), arm, protocol, training=True)
        assert train["actual_counter_writes"] == train["changed_counters_total"] == 96
        assert train["peak_pending"] == 1 and model.pending is None
        checkpoint = json.loads(json.dumps(model.checkpoint()))
        dev = run_phase(model, examples("development"), arm, protocol, training=False)
        assert dev["actual_counter_writes"] == 0 and model.checkpoint() == checkpoint
        fresh = LocalConjunctionCounts(protocol)
        fresh.restore(checkpoint)
        replay = run_phase(fresh, examples("development"), arm, protocol, training=False)
        assert replay["prediction_sha256"] == dev["prediction_sha256"]


def test_capacity_is_checked_before_allocation_and_restore_is_atomic():
    protocol = load_protocol()
    protocol["maximum_count_routes"] = 1
    model = LocalConjunctionCounts(protocol)
    with pytest.raises(ValueError):
        model.predict(contexts((3000,) * 4, WEATHER, datetime(2013, 1, 1), "current_exact", protocol), datetime(2013, 1, 1))
    assert not model.counts and model.pending is None
    with pytest.raises(ValueError):
        model.restore({"[0,[]]": [0, 1], "[8,[]]": [0, 0]})
    assert model.counts == {}


def test_registered_trend_saturation_collision_is_explicit():
    protocol = load_protocol()
    now = datetime(2013, 5, 6, 9)
    a, b = (2000, 3500, 2500, 3000), (2500, 3500, 2000, 3000)
    assert contexts(a, WEATHER, now, "temporal_pair", protocol) == contexts(b, WEATHER, now, "temporal_pair", protocol)


def test_retained_preflight_stops_before_scoring_and_preserves_old_evidence():
    protocol = load_protocol()
    root = Path(processed_data_path("benchmark_fixtures"))
    registered = json.loads((root / "traffic_temporal_registration_v1.json").read_text())
    retained = json.loads((root / "traffic_temporal_result_v1.json").read_text())
    assert registered["analytical_branch_audit"]["passed"] is True
    assert retained["audit"]["support_passed"] is True
    assert retained["contrasts"]["opposite_outcome_temporal_context_groups"] < protocol["minimum_current_contrast_groups"]
    assert retained["learner_scored"] is False and retained["arms"] == {}
    assert retained["sealed_scoring_authorized"] is False
    assert retained["audit"]["sealed_numeric_values_read"] == 0
    assert retained["preserved_negative_sha256"] == registered["preserved_negative_sha256"]
    for name, digest in registered["preserved_negative_sha256"].items():
        assert sha256(Path(project_path(name)).read_bytes()).hexdigest() == digest
    for name, digest in registered["code_sha256"].items():
        assert sha256(Path(project_path(name)).read_bytes()).hexdigest() == digest
