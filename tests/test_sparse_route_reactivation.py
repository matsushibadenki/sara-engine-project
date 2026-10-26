"""Development-only mechanics; registered study seeds are not tuned here."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.sparse_route_reactivation import (
    ARMS, ReactivationMemory, dependencies, rows, state_digest, train,
)
from sara_engine.evaluation.sparse_factor_routing import ROUTES
from scripts.eval import sparse_route_reactivation as runner


def test_registered_seeds_do_not_overlap_prior_study_or_development_seeds():
    p = runner.read('selection')
    old = json.loads(Path(runner.path('selection').parent/'sparse_route_budget_v1.json').read_text())
    assert not set(p['seeds']) & set(old['seeds'])
    assert not set(p['seeds']) & set(p['development_test_seeds'])
    assert p['steps']*p['route_updates_per_tick'] == 9216
    assert p['actual_learning_field_changes_per_route'] == 4
    assert p['arms'] == list(ARMS)


@pytest.mark.parametrize('strategy', ARMS)
def test_actual_four_field_changes_and_unchanged_inactive_routes(strategy):
    model = ReactivationMemory(strategy, 81001)
    for tick, row in enumerate(rows(81001, 'interacting', False, 192)):
        _, trace = model.begin(row.events, tick)
        active = trace['updated_routes']
        assert len(set(active)) == 9
        assert all(sum(ROUTES[i][1] == slot for i in active) == 3 for slot in range(3))
        before = deepcopy((model.tables, model.errors, model.weights, model.observations))
        model.observe(row.outcome, tick+1)
        for i in range(45):
            if i not in active:
                assert (model.tables[i], model.errors[i], model.weights[i], model.observations[i]) == tuple(v[i] for v in before)
            else:
                assert model.observations[i] == before[3][i]+1
                assert model.errors[i] > before[1][i]
                assert model.weights[i] > before[2][i]
                assert sum(map(sum, model.tables[i].values())) == sum(map(sum, before[0][i].values()))+1
    assert model.updates == 192*9


def test_common_warmup_learning_state_is_identical():
    samples = rows(81001, 'independent', False, 180)
    models = [ReactivationMemory(a, 81001) for a in ARMS]
    for model in models:
        for tick, row in enumerate(samples):
            model.begin(row.events, tick)
            model.observe(row.outcome, tick+1)
    assert all(m.tables == models[0].tables and m.weights == models[0].weights and m.errors == models[0].errors for m in models)


@pytest.mark.parametrize('strategy', ARMS)
def test_rejection_is_atomic_and_no_future_result_can_be_read(strategy):
    model = ReactivationMemory(strategy, 81001)
    with pytest.raises(ValueError): model.observe((0, 1, 0), 1)
    assert model.begin((0, 0, 0, 0, 0), 0)[0] == (None, None, None)
    before = deepcopy(model.__dict__)
    for action in (lambda: model.begin((0, 0, 0, 0, 0), 1),
                   lambda: model.observe((0, 1, 0), 0),
                   lambda: model.observe((0, 1, 0), 2),
                   lambda: model.observe((0, True, 0), 1)):
        with pytest.raises(ValueError): action()
        assert model.__dict__ == before


def test_detector_receives_only_delivered_readout_loss_and_start_is_next_begin():
    model = ReactivationMemory('error_triggered', 81001)
    model.steps = 180
    model.detector = 0.1
    model.clock = 179
    _, trace = model.begin((0, 0, 0, 0, 0), 180)
    assert not model.starts
    before = model.detector
    model.observe((1, 1, 1), 181)
    assert model.detector == before+(0.25-before)/16
    model.detector = 0.21
    model.begin((0, 0, 0, 0, 0), 181)
    assert model.starts == [181]
    assert model.explore_until == 361
    assert len(trace['updated_routes']) == 9


def test_exploration_commits_only_after_full_observed_window_and_preserves_tables():
    model = ReactivationMemory('error_triggered', 81001)
    model.steps, model.clock, model.detector = 180, 179, 0.21
    samples = rows(81001, 'independent', False, 182)
    old_pool = None
    for tick in range(180, 360):
        model.begin(samples[tick-180].events, tick)
        if old_pool is None: old_pool = model.pool
        assert model.pool == old_pool
        model.observe(samples[tick-180].outcome, tick+1)
    before = deepcopy(model.tables)
    model.begin(samples[180].events, 360)
    assert model.tables == before
    assert model.commits[0]['tick'] == 360
    assert model.cooldown_until == 424
    assert model.explore_until is None
    assert len(model.pool) == 9


def test_periodic_trigger_is_fixed_and_permanent_cannot_reactivate():
    for strategy in ('periodic', 'permanent'):
        model = ReactivationMemory(strategy, 81002)
        model.steps, model.clock, model.detector = 436, 435, 1.0
        model.begin((0, 0, 0, 0, 0), 436)
        assert model.starts == ([436] if strategy == 'periodic' else [])


def test_generator_shares_input_prefix_and_hides_change_from_model():
    stationary = rows(81001, 'interacting', False)
    changed = rows(81001, 'interacting', True)
    assert [r.events for r in stationary] == [r.events for r in changed]
    assert stationary[:512] == changed[:512]
    assert any(a.outcome != b.outcome for a, b in zip(stationary[512:], changed[512:]))
    assert set(dependencies(81001, 'interacting', True)) <= set(ROUTES)
    assert not hasattr(ReactivationMemory('error_triggered', 81001), 'change_at')


def test_protocol_pins_replay_and_consumed_execution():
    if runner.path('registration').exists(): runner.verify()
    if runner.path('result').exists():
        result = runner.read('result')
        assert len(result['cases']) == 20
        assert result['execution_integrity_passed']
        assert not result['test_opened'] and not result['snn_advantage_tested']
        assert result['resource_measurements'] == result['new_real_data_reads'] == 0
        with pytest.raises(FileExistsError): runner.main('execute')
