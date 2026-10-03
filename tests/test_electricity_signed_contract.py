"""Signed v2 repairs scoring explicitly while preserving v1 identity."""
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

import pytest
from sara_engine.evaluation.electricity_development import Episode, evaluate
from sara_engine.evaluation.electricity_learner import (
    SignedElectricityLearner, HistoryRoutes, encode_history)


def test_signed_endpoints_and_changed_route_subsets():
    learner = SignedElectricityLearner('scalar')
    for index, (route, target) in enumerate([(0, 1), (1, 0), (0, 0), (1, 1)] * 20):
        routes = HistoryRoutes((route,) * 95, (24, 31, 43))
        receipt = learner.predict(routes, time=index * 5, outcome_id=str(index))
        assert 0 <= learner.probability(receipt) <= 1
        if index == 0:
            assert learner.probability(receipt) == 0.5
        learner.resolve(receipt, target=target, time=index * 5 + 4, outcome_id=str(index), training=True)
    assert learner.learning_target(0) == -1
    assert learner.learning_target(1) == 1
    with pytest.raises(ValueError, match='pending'):
        learner.probability(receipt)


def test_v2_runner_cannot_promote():
    episodes = []
    for year in (2012, 2013):
        for month in range(1, 13):
            cutoff = datetime(year, month, 15, 12)
            routes = encode_history([Decimal(index + 1) for index in range(96)], cutoff)
            for client in range(2):
                episodes.append(Episode(f'{year}-{month}-{client}', client, cutoff,
                                        'training' if year == 2012 else 'development', routes, client, 2))
    result = evaluate(iter(episodes), scoring_contract='signed_v2')
    assert result['gates']['parity']
    assert result['adopted'] is False
    assert result['confirmatory'] is False
    with pytest.raises(ValueError, match='Unknown scoring'):
        evaluate([], scoring_contract='implicit_repair')


def test_disclosed_protocol_keeps_heldout_closed():
    path = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures/r2_electricity_learner_protocol_v2.json'
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == '11779a93583c66e510ddb3c639827aaff0d1985306d761d4ed9e52746d746533'
    protocol = json.loads(raw)
    assert protocol['numeric_access_years'] == [2011, 2012, 2013]
    assert not protocol['held_out_access_authorized']
    assert not protocol['promotion_authorized']
    assert not protocol['development_pristine']


def test_affine_probability_endpoints_and_development_freeze():
    routes = HistoryRoutes((0,) * 95, (24, 31, 43))
    for score, expected in [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)]:
        learner = SignedElectricityLearner('scalar')
        learner.readout._weights = {route: score for route in (0, 24, 31, 43)}
        receipt = learner.predict(routes, time=0, outcome_id='test')
        assert learner.probability(receipt) == expected
        before = learner.readout.snapshot()['weights']
        learner.resolve(receipt, target=1, time=4, outcome_id='test', training=False)
        assert learner.readout.snapshot()['weights'] == before


def test_retained_negative_result_cannot_be_relabelled_as_adoption():
    path = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures/r2_electricity_v2_result_receipt.json'
    result = json.loads(path.read_text())
    assert not result['adopted'] and not result['confirmatory']
    assert not result['gates']['primary']
    assert result['primary_improvement'] == pytest.approx(
        result['scalar_macro_client_brier'] - result['stateful_lif_macro_client_brier'])
    assert result['primary_improvement'] < result['required_primary_improvement']
    assert result['held_out_numeric_cells_parsed'] == 0
    assert result['development_episodes'] == 60721 + 53941
