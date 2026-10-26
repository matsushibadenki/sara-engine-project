"""Report-only timing arithmetic and lifecycle counterexamples, no learners."""
import ast
from copy import deepcopy
import inspect
import sys

import pytest

from sara_engine.evaluation import route_failure_audit as audit
from scripts.eval import route_failure_audit as runner


@pytest.fixture(scope='module')
def inputs():
    return runner.load_inputs(runner.read('selection'))


@pytest.fixture(scope='module')
def result(inputs):
    reports,hashes,_=inputs
    return audit.evaluate(runner.read('selection'),reports,hashes)


def test_retained_negative_and_unknown_boundaries(result):
    assert result['summary']=={'changed_cases':10,'pool_lifecycles_reconciled':100,
        'recovery_failures':10,'deadline_lacks_two_wholly_post_commit_blocks':10,
        'exact_retention_failures':3,'high_accuracy_exact_retention_failures':2,'stationary_false_starts':0}
    assert result['learner_executions']==result['new_predictions_scored']==result['new_measurements']==0
    assert not result['component_passed'] and not result['old_failed_replay_reclassified']
    assert result['old_discarded_trace_equality'] is None
    assert not result['new_experiment_authorized'] and not result['test_opened']
    for row in result['changed_case_detail']:
        assert set(row['unknowns'])==set(audit.UNKNOWN)
        assert all(value is None for value in row['unknowns'].values())
        assert row['candidate_timing']['original_gate_impossible'] is None


def test_pipeline_arithmetic_and_block_alignment_are_not_a_causal_claim(result):
    for row in result['changed_case_detail']:
        t=row['candidate_timing']
        assert 6<=t['trigger_delay_ticks']<=13
        assert t['exploration_receipts']==180
        assert t['commit_delay_ticks']==t['trigger_delay_ticks']+180
        assert t['deadline_remaining_after_commit']==256-t['commit_delay_ticks']
        assert t['deadline_remaining_after_commit']<128
        assert t['earliest_wholly_post_commit_pair_end_delay'] in (320,384)
        assert t['causal_effect_of_shorter_exploration'] is None
        p=row['periodic_timing']
        assert p['committed_window_start']==436 and p['first_post_change_commit']==616
        assert p['window_pre_change_receipts']==76 and p['window_post_change_receipts']==104
        assert p['two_wholly_post_commit_blocks_fit_deadline']
    # Enough timing room is not sufficient for the interacting periodic results.
    interacting=[r for r in result['changed_case_detail'] if r['task']=='interacting']
    assert all(r['periodic_timing']['reported_recovery_ticks'] is None or
               r['periodic_timing']['reported_recovery_ticks']>256 for r in interacting)


def test_sharp_subset_count_bounds_from_aggregate_blocks(result):
    for row in result['changed_case_detail']:
        for name in ('candidate_timing','periodic_timing'):
            for block in row[name]['deadline_last_pair']:
                old,new=block['pre_commit_predictions'],block['post_commit_predictions']
                total=block['stored_total_correct']
                low,high=block['post_commit_correct_bounds']
                assert old+new==64 and 0<=low<=high<=new
                assert 0<=total-low<=old and 0<=total-high<=old
                if low: assert total-(low-1)>old
                if high<new: assert total-(high+1)<0
                if new==0: assert block['post_commit_accuracy_bounds'] is None
    # A reported qualifying pair may include a pre-commit prediction.
    assert any(r['candidate_timing']['reported_pair_contains_pre_commit_predictions']
               for r in result['changed_case_detail'])


def test_high_accuracy_does_not_rescue_exact_dependency_gate(result):
    high=[r for r in result['changed_case_detail'] if r['retention_class']=='exact_retention_failure_with_high_reported_accuracy']
    low=[r for r in result['changed_case_detail'] if r['retention_class']=='retention_failure_with_lower_reported_accuracy']
    assert len(high)==2 and len(low)==1
    assert all(r['reported_tail_exact']==r['reported_frozen_exact']==1.0 and not r['reported_exact_retention'] for r in high)
    assert all(r['unknowns']['functional_substitution_proven'] is None for r in high+low)


@pytest.mark.parametrize('arm',audit.ARMS)
def test_pool_lifecycle_is_exact_and_pure(inputs,arm):
    reports,_,_=inputs
    sample=reports[audit.INPUTS[4]]['payload']['cases'][1]['arms'][arm]
    before=deepcopy(sample)
    assert audit.lifecycle(sample,arm)['final_pool_reconciled']
    assert sample==before


@pytest.mark.parametrize('change',['commit_tick','final_pool','added_existing','removed_absent','unfinished','start_repeat','cooldown','bool_route'])
def test_inconsistent_saved_lifecycle_is_rejected(inputs,change):
    arm=deepcopy(inputs[0][audit.INPUTS[4]]['payload']['cases'][1]['arms']['error_triggered'])
    if change=='commit_tick': arm['commits'][0]['tick']+=1
    elif change=='final_pool': arm['final_pool']=list(range(9)) if arm['final_pool']!=list(range(9)) else list(range(9,18))
    elif change=='added_existing': arm['commits'][0]['added']=[arm['initial_pool'][0]]
    elif change=='removed_absent': arm['commits'][0]['removed']=[next(i for i in range(45) if i not in arm['initial_pool'])]
    elif change=='unfinished': arm['unfinished_exploration']=True
    elif change=='start_repeat': arm['exploration_starts'].append(arm['exploration_starts'][0])
    elif change=='cooldown': arm['exploration_starts'].append(arm['exploration_starts'][0]+200)
    else: arm['final_pool'][0]=True
    with pytest.raises(ValueError): audit.lifecycle(arm,'error_triggered')


def test_window_ending_at_final_state_has_no_next_begin_commit(inputs):
    arm=deepcopy(inputs[0][audit.INPUTS[4]]['payload']['cases'][1]['arms']['error_triggered'])
    arm['exploration_starts']=[844]
    arm['commits']=[]
    arm['final_pool']=deepcopy(arm['initial_pool'])
    arm['unfinished_exploration']=True
    value=audit.lifecycle(arm,'error_triggered')
    assert value['commit_count']==0 and value['unfinished']


@pytest.mark.parametrize('change',['recovery','fractional_correct','block_end','bool_accuracy'])
def test_stored_block_summary_must_reconstruct_exact_original_rule(inputs,change):
    arm=deepcopy(inputs[0][audit.INPUTS[4]]['payload']['cases'][1]['arms']['error_triggered'])
    if change=='recovery': arm['recovery_ticks']=256
    elif change=='fractional_correct': arm['blocks'][0]['exact']=0.901
    elif change=='block_end': arm['blocks'][0]['end']=63
    else: arm['blocks'][0]['exact']=True
    with pytest.raises(ValueError): audit.blocks(arm)


@pytest.mark.parametrize('value',[0,1,None,'false'])
def test_boolean_metadata_is_not_truthy(value):
    with pytest.raises(ValueError): audit.boolean(value)


@pytest.mark.parametrize('change',['hash','proof','negative','old_failure','scope'])
def test_changed_proof_negative_or_scope_is_rejected(inputs,change):
    reports,hashes,_=inputs
    reports,hashes=deepcopy(reports),dict(hashes)
    selection=deepcopy(runner.read('selection'))
    if change=='hash': hashes[audit.INPUTS[4]]='0'*64
    elif change=='proof': reports[audit.INPUTS[6]]['numerical_reproduction_passed']=False
    elif change=='negative': reports[audit.INPUTS[1]]['component_passed']=True
    elif change=='old_failure': reports[audit.INPUTS[2]]['numeric_trace_equality']=True
    else: selection['inputs'].append('unregistered_report.json')
    with pytest.raises(ValueError): audit.evaluate(selection,reports,hashes)


def test_loader_refuses_budget_or_extra_input_before_reading():
    selection=deepcopy(runner.read('selection'))
    selection['budgets']['all_input_bytes']+=1
    with pytest.raises(ValueError): runner.load_inputs(selection)


def test_no_learner_generator_or_old_execution_path_is_loaded():
    tree=ast.parse(inspect.getsource(audit))
    imported={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert imported=={'sara_engine.evaluation.reactivation_codec','sara_engine.evaluation.route_numeric_v2'}
    assert 'sara_engine.evaluation.sparse_route_reactivation' not in sys.modules
    assert 'random' not in inspect.getsource(audit)
    assert not hasattr(audit,'run_case')


def test_registered_result_and_consumed_attempt_are_immutable():
    if runner.path('registration').exists(): runner.verify()
    if runner.path('result').exists():
        assert runner.canonical(runner.read('result'))==runner.canonical(runner.compute())
        with pytest.raises(FileExistsError): runner.main('execute')
