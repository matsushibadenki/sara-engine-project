"""Exact saved-evidence arithmetic; no learning or outcome rescoring."""
import ast
from copy import deepcopy
from fractions import Fraction
import inspect
import math
import sys

import pytest

from sara_engine.evaluation import route_score_audit as audit
from sara_engine.evaluation.route_snapshot_contract import CATALOG,costs,diagnose,encode
from scripts.eval import route_score_audit as runner


@pytest.fixture(scope='module')
def inputs():return runner.load_inputs(runner.read('selection'))


@pytest.fixture(scope='module')
def result(inputs):return audit.evaluate(runner.read('selection'),inputs[0],inputs[1])


def test_fixed_population_and_no_new_results_claim(result):
    assert len(result['boundary_comparisons'])==480
    assert len(result['window_comparisons'])==30 and len(result['missing_dependency_focus'])==3
    assert result['learner_executions']==result['actual_new_snapshots']==result['new_predictions_scored']==result['new_measurements']==0
    assert result['causal_attribution'] is result['counterfactual_accuracy'] is None
    assert not result['conditions_relaxed'] and not result['component_passed'] and not result['test_opened']
    assert not result['old_failed_replay_reclassified'] and result['old_discarded_trace_equality'] is None


def test_all_exact_identities_include_rounding_and_do_not_double_count(result):
    for row in result['boundary_comparisons']:
        assert Fraction(row['cumulative_error_gap'])+Fraction(row['width_gap'])+Fraction(row['rounding_residual'])==Fraction(row['actual_float_margin_exact'])
        assert audit.f(row['actual_float_margin'])==Fraction(row['actual_float_margin_exact'])
    for row in result['window_comparisons']:
        base=result['boundary_comparisons'][row['boundary_row_index']]
        error=Fraction(base['cumulative_error_gap'])
        assert Fraction(row['pre_window_contribution_to_cumulative_error_gap'])+Fraction(row['window_contribution_to_cumulative_error_gap'])==error
        assert Fraction(row['window_only_error_gap_same_pair'])+Fraction(row['cumulative_minus_window_only_error_gap'])==error
        assert Fraction(row['window_only_error_gap_same_pair'])+Fraction(base['width_gap'])+Fraction(row['window_float_rounding_residual'])==audit.f(row['window_float_margin_same_pair'])
        assert row['counterfactual_accuracy'] is None
        assert row['fixed_cumulative_cutoff_route']==base['same_output_cumulative_cutoff_route']


def test_original_float_rank_and_cutoff_match_saved_diagnostics(inputs,result):
    snapshots=inputs[0][audit.INPUTS[0]]['snapshots']
    for row in result['boundary_comparisons']:
        snapshot=snapshots[row['snapshot_index']]
        dependency=row['dependency_route'];cutoff=row['same_output_cumulative_cutoff_route']
        diag=diagnose(snapshot,[i for i in inputs[0][audit.INPUTS[1]]['diagnostics']['evaluator_truth'][row['point']['case_id']][row['truth_phase']]])
        stored=next(d for d in diag['dependencies'] if d['route_id']==dependency)
        assert row['rank']==stored['rank']
        assert row['actual_float_margin']==stored['cost_margin_to_third']
        assert CATALOG[dependency]['output']==CATALOG[cutoff]['output']
        assert row['rank_excluded']==(row['rank']>3)


def test_exact_coefficient_and_dependency_self_pair(inputs):
    assert audit.COEFFICIENT==Fraction.from_float(0.005)
    rows=inputs[0][audit.INPUTS[0]]['snapshots'][0]['rows']
    pair=audit.terms(rows,0,0)
    assert all(Fraction(pair[k])==0 for k in ('cumulative_error_gap','width_gap','rounding_residual'))
    assert pair['actual_float_margin']==0.0
    with pytest.raises(ValueError):audit.terms(rows,0,1)


def test_ulp_difference_remains_visible_in_original_float_and_exact_terms(inputs):
    original=inputs[0][audit.INPUTS[0]]['snapshots'][0]['rows']
    changed=deepcopy(original)
    changed[0][1]=math.nextafter(changed[0][1],math.inf)
    before=audit.terms(original,0,3);after=audit.terms(changed,0,3)
    assert before['cumulative_error_gap']!=after['cumulative_error_gap']
    # Even when the final float addition rounds away a quotient change, the
    # explicit rounding residual keeps the exact stored-input accounting intact.
    assert Fraction(after['cumulative_error_gap'])+Fraction(after['width_gap'])+Fraction(after['rounding_residual'])==audit.f(after['actual_float_margin'])


def test_id_tie_break_not_replaced_by_rational_reranking(inputs):
    rows=deepcopy(inputs[0][audit.INPUTS[0]]['snapshots'][0]['rows'])
    rows[3]=deepcopy(rows[0])
    values=costs(rows)
    assert values[0]==values[3]
    # The ordering still uses the original float score and ID, not Fraction.
    rank0,_=audit.ordering(rows,0);rank3,_=audit.ordering(rows,3)
    assert rank0<rank3


@pytest.mark.parametrize('change',['input_hash','proof_hash','numeric','negative','old_unknown','bool_count','roster'])
def test_scope_provenance_negative_and_seals_cannot_change(inputs,change):
    reports,hashes,_=inputs
    reports,hashes=deepcopy(reports),dict(hashes);selection=deepcopy(runner.read('selection'))
    if change=='input_hash':hashes[audit.INPUTS[0]]='0'*64
    elif change=='proof_hash':reports[audit.INPUTS[2]]['collection_sha256']='0'*64
    elif change=='numeric':reports[audit.INPUTS[2]]['numerical_noninterference_passed']=False
    elif change=='negative':reports[audit.INPUTS[2]]['component_passed']=True
    elif change=='old_unknown':reports[audit.INPUTS[2]]['old_discarded_trace_equality']=True
    elif change=='bool_count':reports[audit.INPUTS[2]]['actual_snapshots']=True
    else:selection['inputs'].append('unregistered.json')
    with pytest.raises(ValueError):audit.evaluate(selection,reports,hashes)


@pytest.mark.parametrize('change',['row','truth','missing','state_hash'])
def test_malformed_actual_evidence_fails_closed(inputs,change):
    reports,hashes,_=inputs
    reports=deepcopy(reports);collection=reports[audit.INPUTS[0]]
    if change=='row':collection['snapshots'][0]['rows'][0][0]+=1
    elif change=='truth':
        truth=reports[audit.INPUTS[1]]['diagnostics']['evaluator_truth']
        first=next(iter(truth));truth[first]['initial']=[0,3,6]
    elif change=='missing':collection['snapshots'].pop()
    else:reports[audit.INPUTS[1]]['read_only_capture_checks'][0]['after_sha256']='0'*64
    with pytest.raises(ValueError):audit.evaluate(runner.read('selection'),reports,hashes)


def test_loader_refuses_new_sources_or_larger_budget():
    selection=deepcopy(runner.read('selection'));selection['budgets']['all_input_bytes']+=1
    with pytest.raises(ValueError):runner.load_inputs(selection)


def test_audit_does_not_mutate_saved_inputs(inputs):
    reports,hashes,_=inputs;before=encode(reports)
    audit.evaluate(runner.read('selection'),reports,hashes)
    assert encode(reports)==before


def test_no_learner_evaluator_seed_or_hook_execution_dependencies():
    assert 'sara_engine.evaluation.sparse_route_reactivation' not in sys.modules
    tree=ast.parse(inspect.getsource(audit))
    imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert imports=={'fractions','hashlib','sara_engine.evaluation.route_snapshot_contract'}
    assert not hasattr(audit,'evaluator_truth') and not hasattr(audit,'Observer')


def test_registered_and_consumed_static_artifacts():
    if runner.path('registration').exists():runner.verify()
    if runner.path('result').exists():
        assert encode(runner.read('result'))==encode(runner.compute())
        with pytest.raises(FileExistsError):runner.main('execute')
