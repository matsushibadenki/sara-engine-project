"""Symbolic metadata tests only; no actual learner state is collected."""
from copy import deepcopy
from hashlib import sha256
import inspect
import json
import sys

import pytest

from sara_engine.evaluation import route_snapshot_contract as contract
from scripts.eval import route_snapshot_contract as runner


def points():
    return [{'case_id':'symbolic','arm':'error_triggered','tick':180,'stage':s}
            for s in ('pre_initial','post_initial')]


def fixture_pair():
    state=contract.symbolic_state()
    before=contract.capture(state,points()[0],'symbolic_contract_fixture')
    state['pool']=tuple(before['proposed_pool'])
    after=contract.capture(state,points()[1],'symbolic_contract_fixture')
    return state,before,after


def header(roster):
    return {'schema':'sara-route-evidence-collection-v1','contract_sha256':contract.PROTOCOL_SHA256,
            'catalog':deepcopy(contract.CATALOG),'roster_sha256':sha256(contract.encode(roster)).hexdigest(),
            'provenance':'symbolic_contract_fixture','executor_binding':None}


def test_read_only_capture_complete_candidates_and_pair():
    state=contract.symbolic_state();original=deepcopy(state)
    a=contract.capture(state,points()[0],'symbolic_contract_fixture')
    assert state==original and len(a['rows'])==45
    assert sum(r[0] for r in a['rows'])==9*180
    assert sum(r[2] for r in a['rows'])==3600
    state['pool']=tuple(a['proposed_pool'])
    b=contract.capture(state,points()[1],'symbolic_contract_fixture')
    assert contract.compare_pair(a,b)
    a['rows'][0][3][0]+=1
    assert state['tables'][0][(0,)]==[36,0]
    assert b['rows'][0][3][0]==36


@pytest.mark.parametrize('change',['pending','tick','extra_field','table_count','key','count_bool',
                                  'error_nan','weight_zero','observations','routes','route_bool','pool_bool',
                                  'missing_candidate','per_output_budget'])
def test_invalid_state_rejects_without_mutation(change):
    state=contract.symbolic_state();boundary=points()[0]
    if change=='pending':state['pending']=(0,)
    elif change=='tick':state['steps']=181
    elif change=='extra_field':state['true_route_ids']=[0,1,2]
    elif change=='table_count':state['tables'][0][(0,)]=[35,0]
    elif change=='key':state['tables'][0]={(2,):[36,0]}
    elif change=='count_bool':state['tables'][0][(0,)]=[True,35]
    elif change=='error_nan':state['errors'][0]=float('inf')
    elif change=='weight_zero':state['weights'][0]=0.0
    elif change=='observations':state['observations'][0]=35
    elif change=='routes':state['routes']=tuple(reversed(state['routes']))
    elif change=='route_bool':
        value=list(state['routes']);value[0]=((False,),0);state['routes']=tuple(value)
    elif change=='pool_bool':state['pool']=(True,*range(1,45))
    elif change=='missing_candidate':state['weights'].pop()
    else:
        for i,count,weight in ((0,35,79.0),(1,37,81.0)):
            key=tuple(0 for _ in contract.CATALOG[i]['inputs'])
            state['tables'][i][key]=[count,0];state['observations'][i]=count
            state['weights'][i]=weight;state['errors'][i]=weight/4
    before=deepcopy(state)
    with pytest.raises(ValueError):contract.capture(state,boundary,'symbolic_contract_fixture')
    assert state==before


@pytest.mark.parametrize('change',['rows','pool','tick','kind','provenance'])
def test_hook_pairs_cannot_mutate_learner_or_mismatch_source(change):
    _,a,b=fixture_pair()
    if change=='rows':
        b['rows'][0][1]+=1.0
        b['learner_sha256']=contract.learner_digest(b['rows'],180)
        b['proposed_pool']=contract.ranked_pool(b['rows'])
        b['pool']=deepcopy(b['proposed_pool'])
    elif change=='pool':b['pool']=list(range(9,18))
    elif change=='tick':b['point']['case_id']='other'
    elif change=='kind':b['point']['stage']='post_commit'
    else:b['provenance']='future_external_state'
    with pytest.raises(ValueError):contract.compare_pair(a,b)


def test_start_pair_preserves_pool_and_digest():
    _,a,b=fixture_pair()
    a['point']['stage']='pre_start';a['pool']=deepcopy(b['pool'])
    b['point']['stage']='post_start'
    assert contract.compare_pair(a,b)
    b['pool']=list(range(9,18))
    with pytest.raises(ValueError):contract.compare_pair(a,b)


def test_evaluator_ids_are_separate_and_rank_ties_not_causal():
    _,_,snapshot=fixture_pair()
    assert contract.diagnose(snapshot)['dependencies'] is None
    included=contract.diagnose(snapshot,[0,1,2])
    assert all(r['retained'] and r['support_ready'] for r in included['dependencies'])
    excluded=contract.diagnose(snapshot,[9,10,11])
    assert all(not r['retained'] and r['rank_exclusion_kind']=='tie_break' for r in excluded['dependencies'])
    assert excluded['causal_attribution'] is None
    assert all(r['functional_substitution_proven'] is None for r in excluded['dependencies'])
    with pytest.raises(ValueError):contract.diagnose(snapshot,[0,3,6])


def test_support_below16_does_not_prove_ranking_or_causal_failure():
    state=contract.symbolic_state()
    for i,count,weight in ((0,12,30.0),(3,60,130.0)):
        state['tables'][i][(0,)]=[count,0];state['observations'][i]=count
        state['weights'][i]=weight;state['errors'][i]=weight/4
    a=contract.capture(state,points()[0],'symbolic_contract_fixture')
    state['pool']=tuple(a['proposed_pool'])
    snapshot=contract.capture(state,points()[1],'symbolic_contract_fixture')
    result=contract.diagnose(snapshot,[0,1,2])['dependencies'][0]
    assert not result['support_ready'] and result['retained'] and result['rank']<=3
    assert result['support_is_causal_failure'] is None


def exploration_endpoints():
    state,_,post=fixture_pair()
    start_point={**points()[1],'stage':'pre_start'}
    start=contract.capture(state,start_point,'symbolic_contract_fixture')
    state['steps']=360
    state['observations']=[72]*45
    state['tables']=[{tuple(0 for _ in r['inputs']):[72,0]} for r in contract.CATALOG]
    weights=[]
    for index,route in enumerate(contract.CATALOG):
        group=[i for i,r in enumerate(contract.CATALOG) if r['output']==route['output']]
        weights.append(float(652+(group.index(index)<12)))
    state['weights']=weights;state['errors']=[w/4 for w in weights]
    end=contract.capture(state,{**start_point,'tick':360,'stage':'pre_commit'},'symbolic_contract_fixture')
    return start,end


def test_recent_key_support_is_endpoint_difference_not_a_causal_or_reset_claim():
    start,end=exploration_endpoints()
    before=deepcopy((start,end))
    result=contract.window_support(start,end)
    assert all(r[0]==sum(r[3])==36 for r in result['route_window_rows'])
    assert len(result['route_window_rows'])==45
    assert result['recent_support_is_causal_failure'] is None
    assert result['counterfactual_reset_or_windowed_learning_accuracy'] is None
    assert (start,end)==before


def test_window_support_rejects_wrong_end_boundary():
    start,end=exploration_endpoints()
    end['point']['stage']='post_commit'
    with pytest.raises(ValueError):contract.window_support(start,end)


def test_collection_roundtrip_and_conditional_capacity():
    _,a,b=fixture_pair();plan=points()
    raw=contract.encode_collection(header(plan),[a,b],plan)
    decoded=json.loads(raw)
    assert decoded['snapshots']==[a,b]
    assert all(len(contract.encode(s))<=8192 for s in (a,b))
    assert 160*8192+65536+161==1376417<2097152


@pytest.mark.parametrize('change',['missing','order','duplicate','binding','fixture_as_execution','catalog','extra_header'])
def test_collection_refuses_missing_clipped_reordered_or_misattributed_records(change):
    _,a,b=fixture_pair();plan=points();snapshots=[a,b];h=header(plan)
    if change=='missing':snapshots.pop()
    elif change=='order':snapshots.reverse()
    elif change=='duplicate':plan=[plan[0],plan[0]];snapshots=[a,a];h=header(plan)
    elif change=='binding':h['roster_sha256']='0'*64
    elif change=='fixture_as_execution':h['provenance']='future_external_state'
    elif change=='catalog':h['catalog'].pop()
    else:h['labels']=[0,1,2]
    with pytest.raises(ValueError):contract.encode_collection(h,snapshots,plan)


def test_record_byte_overflow_rejects_without_clipping(monkeypatch):
    state=contract.symbolic_state();before=deepcopy(state)
    monkeypatch.setitem(contract.LIMITS,'bytes_per_snapshot',16)
    with pytest.raises(ValueError):contract.capture(state,points()[0],'symbolic_contract_fixture')
    assert state==before


def test_static_plan_and_no_execution_authorization():
    result=runner.evaluate()
    assert result['planned_snapshots']==160
    assert result['planned_candidate_snapshots']==100 and result['planned_permanent_snapshots']==60
    assert result['actual_snapshots_collected']==result['learner_executions']==result['new_predictions_scored']==0
    assert not result['future_collection_authorized'] and not result['new_experiment_authorized']
    assert not result['prior_component_passed'] and result['old_discarded_trace_equality'] is None
    assert result['evaluator_ids_from_old_reports'] is None


def test_no_learner_seed_decoder_or_generator_import():
    assert 'sara_engine.evaluation.sparse_route_reactivation' not in sys.modules
    assert 'random' not in inspect.getsource(contract)
    assert 'dependencies(' not in inspect.getsource(contract)
    assert 'run_case' not in inspect.getsource(runner)
    assert 'true_route_ids' not in inspect.signature(contract.capture).parameters


def test_registered_and_completed_static_artifacts():
    if runner.path('registration').exists():runner.verify()
    if runner.path('result').exists():
        assert contract.encode(runner.read('result'))==contract.encode(runner.evaluate())
        with pytest.raises(FileExistsError):runner.main('execute')
