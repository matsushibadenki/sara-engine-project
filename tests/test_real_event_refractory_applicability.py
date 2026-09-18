"""Training-only applicability invariants; no real dataset is scored here."""

from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
from pathlib import Path

import pytest

from sara_engine.evaluation import real_event_refractory_applicability as audit


def _write_xes(path: Path, traces: list[tuple[str, list[tuple[str | None, str]]]]) -> str:
    rows = ["<log>"]
    for case_id, events in traces:
        rows.append(f'<trace><string key="concept:name" value="{case_id}"/>')
        for activity, timestamp in events:
            rows.append('<event>')
            if activity is not None:
                rows.append(f'<string key="concept:name" value="{activity}"/>')
            rows.append(f'<date key="time:timestamp" value="{timestamp}"/></event>')
        rows.append('</trace>')
    rows.append('</log>')
    with gzip.open(path, "wb") as stream:
        stream.write("".join(rows).encode("utf-8"))
    return sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("dataset", ["bpi2012", "sepsis"])
def test_training_parser_ignores_nontraining_activity_labels(tmp_path, monkeypatch, dataset):
    source = tmp_path / "events.xes.gz"
    digest = _write_xes(source, [
        ("A", [("route", "2020-01-01T00:00:00+00:00"),
               ("next", "2020-01-01T01:00:00+00:00")]),
        ("Z", [(None, "2020-02-01T00:00:00+00:00"),
               (None, "2020-02-01T01:00:00+00:00")]),
    ])
    monkeypatch.setitem(audit.SOURCES, dataset, ("dummy", "events.xes.gz", digest))
    monkeypatch.setattr(audit, "raw_data_path", lambda *_: source)
    boundary = datetime(2020, 1, 2, tzinfo=timezone.utc)
    monkeypatch.setattr(audit, "DEV_BOUNDARY", boundary)
    monkeypatch.setattr(audit, "TRAIN_END", (boundary, "A"))
    assert [tuple(activity for activity, _ in trace)
            for trace in audit._training_traces(dataset)] == [("route", "next")]


def test_logical_two_step_mask_and_long_gap_are_counted(tmp_path, monkeypatch):
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    traces = [tuple(("A", start + timedelta(days=index)) for index in range(4))]
    monkeypatch.setattr(audit, "_training_traces", lambda _: iter(traces))
    monkeypatch.setitem(audit.EXPECTED_TRAINING, "bpi2012", (1, 3))
    result = audit.audit_training_dataset("bpi2012")
    assert result["masked_events"] == 2
    assert result["masked_predictions"] == 2
    assert result["masked_prediction_fraction"] == pytest.approx(2 / 3)
    assert result["masked_recurrence_over_one_day_fraction"] == 0.5
    assert result["training_predictions"] == 3
    assert not result["screen_passed"]


def test_total_variation_bounds():
    from collections import Counter

    assert audit._total_variation(Counter({"a": 2}), Counter({"a": 1})) == 0
    assert audit._total_variation(Counter({"a": 1}), Counter({"b": 1})) == 1
