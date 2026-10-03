"""Fixed synthetic traces test matching, not predictive performance."""
import json
from hashlib import sha256
from pathlib import Path
import random

import pytest
from sara_engine.evaluation.temporal_control_contract import compare_controls, recency_selector


def test_parity_and_matching_across_fixed_synthetic_traces():
    generator = random.Random(31003)
    traces = [(0,) * 95, tuple(index % 24 for index in range(95))]
    traces.extend(tuple(generator.randrange(24) for _ in range(95)) for _ in range(100))
    for events in traces:
        result = compare_controls(events)
        assert result.reference_routes == result.equivalent_routes
        assert len(result.reference_routes) == len(result.recency_routes)
        assert set(result.recency_routes) <= set(events)
        assert result.reference_unit_ticks == result.equivalent_gate_ticks == 2280
        assert result.recency_event_visits == 95
        assert result.ranked_routes <= 24


def test_matched_quantity_does_not_imply_matched_identity():
    result = compare_controls((0,) * 94 + (1,))
    assert result.reference_routes == (0,)
    assert result.recency_routes == (1,)
    assert compare_controls((0,) * 94 + (1,)) == result


def test_zero_count_and_episode_reset():
    events = tuple(index % 24 for index in range(95))
    first = compare_controls(events)
    assert first.reference_routes == first.recency_routes == ()
    compare_controls((0,) * 95)
    assert compare_controls(events) == first


def test_selector_receives_only_count_and_bounds_input_consumption():
    visited = []
    def oversized():
        for index in range(1000000):
            visited.append(index)
            yield 0
    with pytest.raises(ValueError, match='95'):
        compare_controls(oversized())
    assert len(visited) == 96
    for count in (True, -1, 25, 2):
        with pytest.raises(ValueError):
            recency_selector((0,) * 95, count)
    with pytest.raises(ValueError, match='route'):
        compare_controls((True,) * 95)


def test_contract_does_not_authorize_scoring_or_real_access():
    path = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures/temporal_control_contract_v1.json'
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == 'd4082689b272672428cd030727572efb0e5f865c808e44137fc764963bab40e2'
    contract = json.loads(raw)
    assert not contract['real_data_access_authorized']
    assert not contract['candidate_scoring_authorized']
    assert not contract['promotion_authorized']
