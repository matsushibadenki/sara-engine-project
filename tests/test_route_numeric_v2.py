"""Saved-report and mocked launch-gate tests; no learner execution."""
from copy import deepcopy
import inspect
import math
from pathlib import Path
import sys

import pytest

from sara_engine.evaluation.route_numeric_v2 import (
    canonical, compare, payload, record, reference_frame, validate_record,
)
from scripts.eval import route_numeric_v2 as runner


@pytest.fixture(scope='module')
def saved():
    return runner.read_file(runner.old_path('sparse_route_reactivation_result_v1.json'))


def context(phase='primary'):
    return {'phase': phase, 'registration_sha256': 'a'*64, 'execution_sha256': 'b'*64,
            'runtime': runner.runtime(), 'preserved_count': 422, 'historical_report_sha256': 'c'*64}


def frame(saved):
    return reference_frame(saved, 'a'*64, 422)[0]


def test_scientific_spec_is_exact_original_without_new_thresholds():
    original = runner.read_file(runner.old_path('sparse_route_reactivation_selection_v1.json'))
    expected = {k: v for k, v in original.items()
                if k not in {'schema', 'registered_at_utc', 'authorization', 'preserved', 'stop'}}
    assert canonical(expected) == canonical(runner.read('selection')['frozen_scientific_spec'])
    assert runner.read('selection')['phases'] == ['primary', 'replication']


def test_metadata_rebinding_is_exactly_two_fields_and_never_numeric_results(saved):
    before = deepcopy(saved)
    adjusted, mapping = reference_frame(saved, 'a'*64, 422)
    assert set(mapping) == {'registration_sha256', 'preserved_artifacts'}
    assert saved == before
    assert adjusted['cases'] == saved['cases']
    assert adjusted['failed_cases'] == saved['failed_cases']
    assert adjusted['selection_sha256'] == saved['selection_sha256']
    assert not adjusted['component_passed']
    summary = {k: v for k, v in saved.items() if k not in {
        'schema', 'cases', 'selection_sha256', 'registration_sha256', 'preserved_artifacts'}}
    assert payload(deepcopy(saved['cases']), summary, saved, 'a'*64, 422) == adjusted


def test_tuple_normalization_matches_all_saved_values_and_preserves_negative(saved):
    adjusted = frame(saved)
    for case in adjusted['cases']:
        for arm in case['arms'].values():
            if arm['initial_pool'] is not None:
                arm['initial_pool'] = tuple(arm['initial_pool'])
    result = record(adjusted, saved, context())
    assert result['historical_values_exact_match']
    assert not result['payload']['component_passed']
    assert not result['old_failed_replay_reclassified']
    assert validate_record(result, saved, context()) == result
    replica = record(frame(saved), saved, context('replication'))
    assert compare(result, replica, saved)


@pytest.mark.parametrize('change', ['ulp', 'gate', 'count', 'state_digest', 'ledger', 'pool_order'])
def test_any_numeric_gate_digest_or_order_drift_fails_reference(saved, change):
    adjusted = frame(saved)
    arm = adjusted['cases'][0]['arms']['error_triggered']
    if change == 'ulp': arm['tail_brier'] = math.nextafter(arm['tail_brier'], math.inf)
    elif change == 'gate': adjusted['component_passed'] = True
    elif change == 'count': arm['actual_learning_field_changes'] += 1
    elif change == 'state_digest': arm['state_sha256'] = '0'*64
    elif change == 'ledger': arm['update_prediction_ledger_sha256'] = '0'*64
    else: arm['initial_pool'].reverse()
    result = record(adjusted, saved, context())
    assert result['historical_values_exact_match'] is False
    with pytest.raises(ValueError): validate_record(result, saved, context())


@pytest.mark.parametrize('change', ['phase', 'registration', 'marker', 'boolean_type', 'extra_field', 'runtime'])
def test_provenance_and_boolean_coercions_are_not_ignored(saved, change):
    result = record(frame(saved), saved, context())
    if change == 'phase': result['phase'] = 'replication'
    elif change == 'registration': result['registration_sha256'] = 'd'*64
    elif change == 'marker': result['execution_sha256'] = 'e'*64
    elif change == 'boolean_type': result['conditions_relaxed'] = 0
    elif change == 'extra_field': result['extra'] = False
    else: result['runtime'] = {'python': '3.10.20'}
    with pytest.raises(ValueError): validate_record(result, saved, context())


def forbidden(*args, **kwargs):
    pytest.fail('Learner launched before saved-result/one-shot gate')


def test_missing_primary_blocks_replication_before_marker_or_import(monkeypatch):
    monkeypatch.setattr(runner, 'verify', lambda: None)
    monkeypatch.setattr(runner, 'run_study', forbidden)
    def missing(phase):
        raise FileNotFoundError('Primary not retained')
    monkeypatch.setattr(runner, 'validated', missing)
    with pytest.raises(FileNotFoundError): runner.main('replicate')


def test_consumed_primary_refuses_restart_before_learner(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'path', lambda suffix: tmp_path/('route_numeric_v2_'+suffix+'_v1.json'))
    monkeypatch.setattr(runner, 'verify', lambda: None)
    monkeypatch.setattr(runner, 'digest', lambda file: 'a'*64)
    monkeypatch.setattr(runner, 'run_study', forbidden)
    runner.path('primary_execution').write_text('{}')
    with pytest.raises(FileExistsError): runner.main('execute')


def test_consumed_replication_refuses_restart_before_learner(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'path', lambda suffix: tmp_path/('route_numeric_v2_'+suffix+'_v1.json'))
    monkeypatch.setattr(runner, 'verify', lambda: None)
    monkeypatch.setattr(runner, 'digest', lambda file: 'a'*64)
    monkeypatch.setattr(runner, 'validated', lambda phase: None)
    monkeypatch.setattr(runner, 'marker', lambda phase: {'pid': -1})
    monkeypatch.setattr(runner, 'run_study', forbidden)
    runner.path('replication_execution').write_text('{}')
    with pytest.raises(FileExistsError): runner.main('replicate')


def test_same_process_cannot_replicate(monkeypatch):
    monkeypatch.setattr(runner, 'verify', lambda: None)
    monkeypatch.setattr(runner, 'validated', lambda phase: None)
    monkeypatch.setattr(runner, 'marker', lambda phase: {'pid': runner.os.getpid()})
    monkeypatch.setattr(runner, 'run_study', forbidden)
    with pytest.raises(ValueError, match='fresh process'): runner.main('replicate')


def test_lazy_execution_and_old_runner_remain_isolated():
    assert 'sara_engine.evaluation.sparse_route_reactivation' not in sys.modules
    source = inspect.getsource(runner)
    assert 'scripts.eval.sparse_route_reactivation' not in source
    assert 'def run_study()' in source
    assert not runner.old_path('sparse_route_reactivation_replay_v1.json').exists()


def test_registered_or_completed_artifacts_validate_without_rerun():
    if runner.path('registration').exists(): runner.verify()
    if runner.path('result').exists():
        assert canonical(runner.read('result')) == canonical(runner.proof())
        assert runner.read('result')['numerical_reproduction_passed']
        assert not runner.read('result')['component_passed']
        assert runner.marker('primary')['pid'] != runner.marker('replication')['pid']
