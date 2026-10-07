"""Information loss, timestamp validation and bounded-state regression tests."""
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.temporal_capability_boundary import BoundedSequenceObserver, audit_boundary, signatures
from sara_engine.utils.project_paths import processed_data_path


def test_registered_boundary():
    protocol = json.loads(Path(processed_data_path('benchmark_fixtures', 'temporal_capability_boundary_v1.json')).read_text())
    report = audit_boundary(protocol)
    assert report['passed']
    assert all(f['pairs'] == 276 for f in report['families'].values())
    assert report['learning_executed'] is False


@pytest.mark.parametrize('route,gap', [(True, 1), (24, 1), (-1, 1), (1, True), (1, 256), (1, 0)])
def test_invalid_event_does_not_mutate(route, gap):
    observer = BoundedSequenceObserver()
    observer.observe(0, 0)
    before = observer.snapshot()
    with pytest.raises(ValueError):
        observer.observe(route, gap)
    assert observer.snapshot() == before


def test_capacity_and_immutable_snapshot():
    observer = BoundedSequenceObserver()
    observer.observe(0, 0)
    snapshot = observer.snapshot()
    for i in range(94):
        observer.observe(i % 24, 255)
    assert snapshot == ((0, 0),)
    assert len(observer.wire_bytes()) == 194
    before = observer.snapshot()
    with pytest.raises(ValueError):
        observer.observe(0, 1)
    assert observer.snapshot() == before


def test_middle_order_and_gap_are_distinct_requirements():
    a, _ = signatures(((0, 0), (1, 1), (0, 1), (1, 1), (0, 1), (1, 1)))
    b, _ = signatures(((0, 0), (1, 1), (1, 1), (0, 1), (0, 1), (1, 1)))
    assert a['first_last_count'] == b['first_last_count']
    assert a['full_order'] != b['full_order']
    c, _ = signatures(((0, 0), (1, 3), (0, 1), (1, 1), (0, 1), (1, 1)))
    assert a['full_order'] == c['full_order']
    assert a['full_order_gap'] != c['full_order_gap']


def test_first_timestamp_must_be_zero():
    with pytest.raises(ValueError):
        BoundedSequenceObserver().observe(0, 1)
