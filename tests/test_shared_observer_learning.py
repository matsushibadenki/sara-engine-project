"""Synthetic boundaries before the registered one-shot learning run."""
from hashlib import sha256
import json
from pathlib import Path

import pytest
from sara_engine.evaluation.shared_observer_learning import (
    ARMS, episodes, features, readout, predict, evaluate_frozen)


def test_pair_features_and_input_only_ablations():
    rows = list(episodes(17, 1))
    for arm in ARMS:
        a, costs = features(rows[0][1], arm)
        b, _ = features(rows[1][1], arm)
        assert (a == b) == (arm in ('order_blind', 'backend_only_lif'))
        assert len(a) <= 97
        assert costs['observer_record_visits'] == 95
        assert costs['backend_unit_ticks'] <= 2280
    with pytest.raises(ValueError):
        features(iter(range(1000000)), 'observer_only')


def test_delayed_local_learning_and_frozen_replay():
    model = readout()
    _, events, target = next(episodes(18, 1))
    active, _ = features(events, 'observer_only')
    receipt, probability = predict(model, active, 0)
    assert probability == 0.5
    update = model.observe(receipt, 2 * target - 1, time=4)
    assert set(route for route, _ in update.deltas) == set(route for route, _ in active)
    weights = model.snapshot()['weights']
    result = evaluate_frozen(model, 'observer_only', 19, 2, start_time=5)
    assert model.snapshot()['weights'] == weights
    clone = readout()
    clone._weights = dict(weights)
    replay = evaluate_frozen(clone, 'observer_only', 19, 2)
    assert replay['prediction_trace_sha256'] == result['prediction_trace_sha256']


def test_frozen_registration_and_disjoint_seeds():
    root = Path(__file__).resolve().parents[1]
    raw = (root / 'data/processed/benchmark_fixtures/shared_observer_learning_v1.json').read_bytes()
    assert sha256(raw).hexdigest() == '7e999640944764a6b86873e61771fda4fdeb50197142ad0a2c240c43803dfbfa'
    protocol = json.loads(raw)
    assert not set(protocol['training_seeds']) & set(protocol['evaluation_seeds'])
    assert not protocol['promotion_authorized'] and not protocol['real_data_access_authorized']
