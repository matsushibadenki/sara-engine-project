"""Synthetic boundary and parity checks before real candidate evaluation."""
from datetime import datetime
from decimal import Decimal

import pytest
from sara_engine.evaluation.electricity_learner import (
    ElectricityLearner, HistoryRoutes, encode_history, unit_activity, brier_score)


def trace():
    return encode_history((Decimal(index) for index in range(96)), datetime(2013, 1, 1, 12))


def test_history_only_encoding_and_prework_bound():
    routes = trace()
    assert len(routes.comparison_routes) == 95
    assert routes.comparison_routes[:12] == (2,) * 12
    assert routes.comparison_routes[-1] == 23
    assert routes.calendar_routes[-1] == 43
    with pytest.raises(ValueError, match='96'):
        encode_history(iter(Decimal(index) for index in range(1000000)), datetime(2013, 1, 1))
    with pytest.raises(ValueError, match='authorized'):
        encode_history([], datetime(2014, 1, 1))
    with pytest.raises(ValueError, match='budget'):
        unit_activity(HistoryRoutes((0,) * 96, (24, 31, 43)), 'stateful_lif')


def test_lif_state_reset_intervention_and_episode_reset():
    routes = trace()
    active, ticks = unit_activity(routes, 'stateful_lif')
    reset, reset_ticks = unit_activity(routes, 'stateful_reset_each_event')
    assert len(active) > len(reset)
    assert reset == ((25, 1.0), (31, 1.0), (43, 1.0))
    assert ticks == reset_ticks == 2280
    assert unit_activity(routes, 'stateful_lif') == (active, ticks)


def test_delayed_receipts_parity_and_frozen_development():
    scalar, compact = ElectricityLearner('scalar'), ElectricityLearner('compact_event')
    for index in range(4):
        left = scalar.predict(trace(), time=index * 5, outcome_id=str(index))
        right = compact.predict(trace(), time=index * 5, outcome_id=str(index))
        assert left == right
        for learner, receipt in [(scalar, left), (compact, right)]:
            with pytest.raises(ValueError, match='four intervals'):
                learner.resolve(receipt, target=1, time=index * 5 + 3, outcome_id=str(index), training=True)
        updates = [learner.resolve(receipt, target=index % 2, time=index * 5 + 4,
                                  outcome_id=str(index), training=True)
                   for learner, receipt in [(scalar, left), (compact, right)]]
        assert updates[0] == updates[1]
    before = scalar.readout.snapshot()['weights']
    receipt = scalar.predict(trace(), time=20, outcome_id='dev')
    with pytest.raises(ValueError, match='identity'):
        scalar.resolve(receipt, target=1, time=24, outcome_id='other', training=False)
    scalar.resolve(receipt, target=1, time=24, outcome_id='dev', training=False)
    assert scalar.readout.snapshot()['weights'] == before
    assert scalar.state_bytes() <= 65536


def test_metric_contract():
    assert brier_score(0.5, 1) == 0.25
    with pytest.raises(ValueError):
        brier_score(float('nan'), 0)
