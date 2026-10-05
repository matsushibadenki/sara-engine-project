"""Temporal information preservation and explicit lossy boundaries."""
import pytest
from hashlib import sha256
import json
from pathlib import Path
from sara_engine.evaluation.temporal_information import TemporalInformationObserver
from sara_engine.evaluation.independent_temporal_design import synthetic_pairs


def observe(events):
    observer = TemporalInformationObserver()
    for event in events:
        observer.observe(event)
    return observer.snapshot()


def test_first_order_survives_packet_and_projection():
    for pair in synthetic_pairs():
        left, right = observe(pair.mode_zero), observe(pair.mode_one)
        assert left != right
        assert left.readout_routes() != right.readout_routes()
        assert len(left.wire_bytes()) <= 101
        assert all(0 < value <= 1 for _, value in left.readout_routes())


def test_middle_order_is_not_losslessly_preserved():
    assert observe((0, 1, 0, 1, 0, 1)) == observe((0, 1, 1, 0, 0, 1))


def test_bounds_reject_before_mutation_and_snapshots_are_immutable():
    observer = TemporalInformationObserver()
    observer.observe(0)
    snapshot = observer.snapshot()
    with pytest.raises(ValueError):
        observer.observe(True)
    assert observer.snapshot() == snapshot
    for index in range(94):
        observer.observe(index % 24)
    final = observer.snapshot()
    assert len(final.records) == 24
    assert len(final.wire_bytes()) == 101
    assert len(final.readout_routes()) == 72
    with pytest.raises(ValueError):
        observer.observe(0)
    assert observer.snapshot() == final
    assert snapshot.event_count == 1


def test_frozen_contract_keeps_learning_closed():
    path = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures/temporal_information_contract_v1.json'
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == '5dd6daf07b44303d9de2827c43e46f6201394e5f70bbf0e246ed1ba154cf6123'
    contract = json.loads(raw)
    assert not contract['learning_authorized']
    assert not contract['real_data_access_authorized']
