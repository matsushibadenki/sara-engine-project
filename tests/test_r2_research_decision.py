"""Retrospective scope, actual-write parity and incomplete-cost counterexamples."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.r2_research_decision import ROSTER, WORK_DIMENSIONS, evaluate, ledger, parity
from sara_engine.utils.project_paths import project_path, processed_data_path


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'r2_research_decision_' + suffix + '_v1.json'))


def sample():
    selection = {'roster': list(ROSTER), 'budgets': {'receipt_files': 19, 'per_receipt_bytes': 65536, 'aggregate_receipt_bytes': 1048576}}
    receipts = {name: json.loads(Path(processed_data_path('benchmark_fixtures', name + '.json')).read_text()) for name in ROSTER}
    return selection, receipts


def runner():
    spec = importlib.util.spec_from_file_location('r2_decision_runner', project_path('scripts/eval/r2_research_decision.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component_success_does_not_become_snn_increment_or_generalization():
    result = evaluate(*sample())
    shared, _, _, room = result['learning_rows'][:4]
    assert shared['component_identification_passed']
    assert room['component_identification_passed']
    assert shared['retained_gates']['backend_increment_passed'] is False
    assert result['reported_write_parity_passed']
    assert all(not row['independent_usefulness_passed'] for row in result['learning_rows'])
    assert all(not row['snn_predictive_increment_established'] for row in result['learning_rows'])
    assert result['new_predictions_scored'] == result['audit_learning_writes'] == 0


def test_confidence_supplement_is_included_in_actual_write_parity():
    result = evaluate(*sample())
    parity_row = result['learning_rows'][6]['write_parity']
    assert len(parity_row['actual_changed_writes']) == 10
    assert set(parity_row['actual_changed_writes'].values()) == {27972}
    assert set(parity_row['frozen_period_writes'].values()) == {0}
    args = sample()
    args[1][ROSTER[7]]['references']['histogram']['changed_writes'] -= 1
    assert not evaluate(*args)['reported_write_parity_passed']


def test_hypothetical_or_attempted_updates_do_not_prove_changed_write_parity():
    with pytest.raises(KeyError):
        parity({'a': {'hypothetical_writes': 10}, 'b': {'attempts': 10}}, 'changed_writes')
    assert not parity({'a': {'writes': 0}, 'b': {'writes': 0}}, 'writes')['equal_positive_actual_writes']
    assert not parity({'a': {'writes': 3}, 'b': {'writes': 4}}, 'writes')['equal_positive_actual_writes']
    assert not parity({'a': {'writes': 3, 'frozen': 1}, 'b': {'writes': 3, 'frozen': 0}}, 'writes', 'frozen')['frozen_period_zero']


@pytest.mark.parametrize('counter', [True, -1, 2.5, None, '10'])
def test_invalid_counter_is_not_silently_coerced(counter):
    with pytest.raises(ValueError):
        parity({'a': {'writes': counter}, 'b': {'writes': 10}}, 'writes')


def test_partial_work_and_instrumented_cpu_do_not_prove_resource_advantage():
    result = evaluate(*sample())
    work = result['learning_rows'][2]['work_ledger']
    assert set(work) == set(WORK_DIMENSIONS)
    assert work['deployment_cpu'] == {'coverage': 'unreported', 'observations': []}
    assert work['expiry_maintenance']['coverage'] == 'unreported'
    assert work['replay_instrumentation']['observations']
    units = {item['unit'] for item in work['live_and_transient_memory']['observations']}
    assert units == {'conservative_bound_bytes', 'readout_object_bytes'}
    assert not any(row['complete_work_accounting'] or row['resource_advantage_established'] for row in result['learning_rows'])
    assert all('total_work' not in row for row in result['learning_rows'])


def test_queue_peak_and_route_count_are_not_total_queue_work_or_memory_bytes():
    work = ledger({'arms': {'x': {'changed_writes': 2, 'routes': 999, 'peak_pending': 1}}}, 'room')
    assert work['feedback_enqueue_delivery']['observations'][0]['unit'] == 'peak_pending_receipts'
    assert work['live_and_transient_memory']['observations'][0]['unit'] == 'routes'
    assert work['energy']['observations'] == []
    assert work['input_decode']['coverage'] == 'unreported'


def test_scalar_parity_keeps_implementation_cost_hypothesis_open():
    result = evaluate(*sample())
    assert result['learning_rows'][5]['scalar_equivalence']
    assert 'unmeasured' in result['remaining_hypothesis']
    assert 'event-driven scalar' in result['required_work_contract']['comparator']
    assert not result['r3_optimization_authorized']
    assert not result['new_acquisition_or_search_authorized']


@pytest.mark.parametrize('flag', ['test_opened', 'original_development_opened', 'independent_periods_opened', 'conditions_relaxed', 'sealed_numeric_values_read'])
def test_seal_or_condition_change_is_rejected(flag):
    args = sample()
    args[1][ROSTER[0]][flag] = 1
    with pytest.raises(ValueError, match='seal/condition'):
        evaluate(*args)


def test_new_positive_receipt_cannot_be_given_a_canned_negative_decision():
    args = sample()
    args[1][ROSTER[5]]['comparison']['quality_passed'] = True
    with pytest.raises(ValueError, match='new audit protocol'):
        evaluate(*args)


@pytest.mark.parametrize('change', ['drop', 'add', 'reorder', 'budget'])
def test_finite_scope_and_budgets_are_not_extended(change):
    selection, receipts = sample()
    if change == 'drop':
        receipts.pop(ROSTER[-1])
    elif change == 'add':
        receipts['unregistered'] = {}
    elif change == 'reorder':
        selection['roster'].reverse()
    else:
        selection['budgets']['per_receipt_bytes'] *= 2
    with pytest.raises(ValueError):
        evaluate(selection, receipts)


def test_missing_counter_is_not_zero():
    args = sample()
    del args[1][ROSTER[2]]['arms']['joint']['training']['actual_counter_writes']
    with pytest.raises(KeyError):
        evaluate(*args)


def test_reader_checks_budget_before_parsing_or_following_receipt_paths():
    module = runner()
    selection, _ = sample()
    selection['budgets']['per_receipt_bytes'] = 1
    with pytest.raises(ValueError, match='budget'):
        module.load_receipts(selection)


def test_reader_rejects_changed_receipt_hash():
    selection, _ = sample()
    name = 'data/processed/benchmark_fixtures/' + ROSTER[0] + '.json'
    selection['preserved'] = {name: '0' * 64}
    with pytest.raises(ValueError, match='changed during read'):
        runner().load_receipts(selection)


def test_hypothetical_forecast_is_retained_without_becoming_new_learning():
    result = evaluate(*sample())
    forecast = result['retained_support_and_acquisition_limits']['closed_world_joint_forecast']
    assert forecast['required_extra_observations'] == 70212
    assert forecast['closed_world_budget_feasible']
    assert not forecast['conservative_state_budget_feasible']
    assert result['audit_learning_writes'] == 0
    assert not result['runtime_promotion_authorized']


def test_registered_code_drift_is_rejected(monkeypatch):
    if not path('registration').exists():
        pytest.skip('Audit not registered')
    module = runner()
    monkeypatch.setattr(module, 'digest', lambda _: '0' * 64)
    with pytest.raises(ValueError, match='Pinned evidence/code changed'):
        module.verify()


def test_consumed_result_replay_and_old_evidence_pins():
    if not path('result').exists():
        pytest.skip('Registered audit has not executed')
    module = runner()
    module.verify()
    selection = json.loads(path('selection').read_text())
    receipts, total = module.load_receipts(selection)
    result = evaluate(selection, receipts)
    result.update(receipt_bytes_read=total, preserved_artifacts=len(selection['preserved']))
    assert result == json.loads(path('result').read_text())
    for name, expected in selection['preserved'].items():
        assert sha256(Path(project_path(name)).read_bytes()).hexdigest() == expected
    assert json.loads(path('replay').read_text())['result_sha256'] == sha256(path('result').read_bytes()).hexdigest()


def test_consumed_execution_is_exclusive():
    if not path('execution').exists():
        pytest.skip('Registered audit has not executed')
    with pytest.raises(FileExistsError):
        runner().main('execute')
