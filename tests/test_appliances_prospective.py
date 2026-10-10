"""Prospective extraction integrity and sufficient-control regression tests."""
from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.appliances_prospective import DemandExample, extract
from sara_engine.utils.project_paths import processed_data_path


def protocol():
    p = json.loads(Path(processed_data_path("benchmark_fixtures", "appliances_prospective_v1.json")).read_text())
    p.update(history_samples=3, horizon_samples=2)
    return p


def rows(start, values):
    yield ["date", "Appliances"]
    for index, value in enumerate(values):
        yield [(start + timedelta(minutes=10 * index)).isoformat(), str(value)]


def test_future_is_only_label():
    start = datetime(2016, 2, 1)
    a, _ = extract(rows(start, [10, 20, 30, 40, 50]), protocol())
    b, _ = extract(rows(start, [10, 20, 30, 40, 0]), protocol())
    assert len(a) == len(b) == 1
    assert a[0].history == b[0].history == (10, 20, 30)
    assert a[0].controls() == b[0].controls()
    assert (a[0].label, b[0].label) == (1, 0)
    assert a[0].outcome_time > a[0].cutoff


def test_sealed_numeric_cells_never_accessed():
    class Spy:
        def __getitem__(self, index):
            if index != 0:
                raise AssertionError("Sealed numeric cell accessed")
            return "2016-04-01 00:00:00"
    examples, audit = extract(iter([["date", "Appliances"], Spy()]), protocol())
    assert not examples
    assert audit["sealed_rows_timestamp_only"] == 1
    assert audit["numeric_rows_read"] == 0


def test_split_boundary_is_purged():
    examples, _ = extract(rows(datetime(2016, 2, 29, 23, 20), [10] * 12), protocol())
    for example in examples:
        lower = example.cutoff - timedelta(minutes=20)
        boundary = datetime(2016, 3, 1)
        assert example.outcome_time < boundary if example.split == "train" else lower >= boundary


def test_missing_followup_abstains():
    examples, _ = extract(rows(datetime(2016, 2, 1), [10] * 4), protocol())
    assert not examples


@pytest.mark.parametrize("values", [[10, "nan", 20], [10, "inf", 20], [10, -1, 20]])
def test_invalid_observation_rejected(values):
    with pytest.raises(ValueError):
        extract(rows(datetime(2016, 2, 1), values), protocol())


def test_duplicate_timestamp_rejected():
    with pytest.raises(ValueError):
        extract([["date", "Appliances"], ["2016-02-01", "10"], ["2016-02-01", "20"]], protocol())


def test_gap_control_reconstructs_complete_ordinal_history():
    history = (10., 10., 20., 20., 10.)
    example = DemandExample("train", datetime(2016, 2, 1), datetime(2016, 2, 1, 0, 30), history, 0)
    changes = dict(example.controls()["transition_gaps"])
    restored = []
    for index in range(len(history)):
        if index in changes:
            current = changes[index]
        restored.append(current)
    assert tuple(restored) == history
