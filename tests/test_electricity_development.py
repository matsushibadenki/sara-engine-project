"""End-to-end synthetic episode boundaries and metric contracts."""
from datetime import datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path

import pytest
from sara_engine.evaluation.electricity_development import Episode, Metrics, iter_episodes, evaluate
from sara_engine.evaluation.electricity_learner import encode_history
from sara_engine.learning.local_outcome import BoundedLocalOutcomeReadout, LocalOutcomeConfig


def protocol():
    return json.loads((Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures/r2_electricity_load_feasibility_protocol_v1.json').read_text())


def rows(outcome):
    start = datetime(2013, 1, 2, 12) - timedelta(minutes=95 * 15)
    return [f'{start + timedelta(minutes=index * 15):%Y-%m-%d %H:%M:%S};{index if index < 96 else outcome}\n'
            for index in range(100)]


def test_future_changes_label_not_history_routes():
    rising = list(iter_episodes(rows(200), protocol(), client_count=1))
    falling = list(iter_episodes(rows(0), protocol(), client_count=1))
    # Activate at the first row, not after the history begins.
    assert not rising
    up, down = rows(200), rows(0)
    up[0] = up[0].rsplit(';', 1)[0] + ';1\n'
    down[0] = down[0].rsplit(';', 1)[0] + ';1\n'
    a = list(iter_episodes(up, protocol(), client_count=1))[0]
    b = list(iter_episodes(down, protocol(), client_count=1))[0]
    assert a.routes == b.routes
    assert (a.target, b.target) == (1, 0)
    assert a.cutoff == datetime(2013, 1, 2, 12)
    assert not list(iter_episodes([line.replace('2013', '2014').replace('2012', '2013')
                                   for line in rows(200)], protocol(), client_count=1))


def episode(client, month, target, year=2013):
    cutoff = datetime(year, month, 15, 12)
    return Episode(f'{year}-{month}-{client}', client, cutoff,
                   'training' if year == 2012 else 'development',
                   encode_history([Decimal(index + 1) for index in range(96)], cutoff), target, 2)


def test_macro_clients_months_and_confusion():
    metric = Metrics()
    metric.add(episode(0, 1, 1), 0)
    metric.add(episode(1, 1, 0), 0)
    metric.add(episode(1, 1, 0), 0)
    result = metric.finish()
    assert result['macro_client_brier'] == 0.5
    assert result['micro_brier'] == pytest.approx(1 / 3)
    assert result['balanced_accuracy'] == 0.5
    assert result['monthly_macro_client_brier'] == {'1': 0.5}


def test_synthetic_runner_parity_calendar_and_stop_rule():
    episodes = [episode(client, month, client % 2, year)
                for year in (2012, 2013) for month in range(1, 13) for client in range(2)]
    report = evaluate(iter(episodes))
    assert report['gates']['parity']
    assert report['gates']['resources']
    assert len(report['monthly_improvements']) == 12
    assert report['held_out_numeric_cells_parsed'] == 0
    assert report['label_counts_fall_rise'] == {'training': [12, 12], 'development': [12, 12]}
    with pytest.raises(ValueError, match='ordering'):
        evaluate(iter([episodes[1], episodes[0]]))


def test_binary_targets_do_not_make_signed_readout_a_probability():
    readout = BoundedLocalOutcomeReadout(LocalOutcomeConfig(trace_decay=1.0))
    prediction = readout.predict([(0, 1.0)], time=0)
    readout.observe(prediction, 1, time=1)
    prediction = readout.predict([(0, 1.0), (1, 1.0)], time=2)
    readout.observe(prediction, 0, time=3)
    prediction = readout.predict([(1, 1.0)], time=4)
    assert prediction.score < 0
    from sara_engine.evaluation.electricity_learner import brier_score
    with pytest.raises(ValueError, match='unit interval'):
        brier_score(prediction.score, 0)
