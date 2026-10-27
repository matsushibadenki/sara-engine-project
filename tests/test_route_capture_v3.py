"""Development hook integration and saved-report gates, no study-seed tuning."""
from copy import deepcopy
import inspect
import math

import pytest

from sara_engine.evaluation import route_capture_v3 as hooks
from sara_engine.evaluation.reactivation_codec import ReportCodec
from sara_engine.evaluation.route_numeric_v2 import reference_frame
from sara_engine.evaluation.route_snapshot_contract import encode,symbolic_state
from scripts.eval import route_capture_v3 as runner


def dev_roster():
    return [{'case_id':'dev','arm':'error_triggered','tick':180,'stage':s}
            for s in ('pre_initial','post_initial')]


def test_real192_tick_development_witness_matches_without_instance_metadata():
    from sara_engine.evaluation import sparse_route_reactivation as old
    samples=old.rows(81001,'independent',False,192);labels=[r.outcome for r in samples]
    plain=old.ReactivationMemory('error_triggered',81001)
    expected=old.train(plain,samples,labels)
    observed=old.ReactivationMemory('error_triggered',81001)
    observer=hooks.Observer(dev_roster());observer.set_case('dev')
    original_narrow,original_observe=old.ReactivationMemory.narrow,old.ReactivationMemory.observe
    with observer.install():
        actual=old.train(observed,samples,labels)
    assert encode(expected)==encode(actual)
    assert old.state_digest(plain)==old.state_digest(observed)
    assert set(plain.__dict__)==set(observed.__dict__)
    assert len(observer.finish())==len(observer.read_only_checks)==2
    assert old.ReactivationMemory.narrow is original_narrow
    assert old.ReactivationMemory.observe is original_observe


def test_failure_restores_original_methods(monkeypatch):
    from sara_engine.evaluation import sparse_route_reactivation as old
    model=old.ReactivationMemory('error_triggered',81001);model.steps=180
    observer=hooks.Observer(dev_roster());observer.set_case('dev')
    original_narrow,original_observe=old.ReactivationMemory.narrow,old.ReactivationMemory.observe
    def fail(*args,**kwargs):raise ValueError('Deliberate unit hook failure')
    monkeypatch.setattr(hooks,'capture',fail)
    with pytest.raises(ValueError):
        with observer.install():model.narrow()
    assert old.ReactivationMemory.narrow is original_narrow and old.ReactivationMemory.observe is original_observe
    assert not observer.installed


def test_unplanned_actual_event_rejects_instead_of_adding_records():
    from sara_engine.evaluation import sparse_route_reactivation as old
    model=old.ReactivationMemory('error_triggered',81001);model.steps=180
    observer=hooks.Observer([]);observer.set_case('dev')
    with pytest.raises(ValueError,match='Actual event'):
        with observer.install():model.narrow()
    assert not observer.snapshots


def test_capture_mutation_is_detected_by_complete_model_hash(monkeypatch):
    from sara_engine.evaluation import sparse_route_reactivation as old
    model=old.ReactivationMemory('error_triggered',81001)
    for key,value in symbolic_state().items():model.__dict__[key]=value
    observer=hooks.Observer(dev_roster());observer.set_case('dev')
    original_capture=hooks.capture
    def mutate(state,boundary,provenance):
        result=original_capture(state,boundary,provenance)
        state['observations'][0]+=1
        return result
    monkeypatch.setattr(hooks,'capture',mutate)
    with pytest.raises(ValueError,match='complete model state'):
        with observer.install():model.narrow()
    assert not observer.snapshots


def test_missing_duplicate_or_nested_observer_is_rejected():
    observer=hooks.Observer(dev_roster())
    with pytest.raises(ValueError):observer.finish()
    with pytest.raises(ValueError):hooks.Observer([dev_roster()[0],dev_roster()[0]])
    with observer.install():
        with pytest.raises(ValueError):
            with observer.install():pass


def numerical_fixture():
    saved=runner.reference()
    frame=reference_frame(saved,'a'*64,450)[0]
    context={'phase':'baseline','registration_sha256':'a'*64,'preserved_count':450,
             'execution_sha256':'b'*64,'runtime':runner.runtime()}
    return saved,frame,context


def test_full_numeric_frame_matches_and_negative_is_not_rescued():
    saved,frame,ctx=numerical_fixture()
    record=hooks.record(frame,saved,ctx)
    assert hooks.validate_record(record,saved,ctx)==record
    assert record['historical_values_exact_match'] and not record['payload']['component_passed']


@pytest.mark.parametrize('change',['float','gate','counter','state','envelope_bool'])
def test_numerical_or_provenance_drift_is_never_hidden(change):
    saved,frame,ctx=numerical_fixture()
    result=hooks.record(frame,saved,ctx)
    arm=result['payload']['cases'][0]['arms']['error_triggered']
    if change=='float':arm['tail_brier']=math.nextafter(arm['tail_brier'],math.inf)
    elif change=='gate':result['payload']['component_passed']=True
    elif change=='counter':arm['route_updates']+=1
    elif change=='state':arm['state_sha256']='0'*64
    else:result['conditions_relaxed']=0
    with pytest.raises(ValueError):hooks.validate_record(result,saved,ctx)


def forbidden(*args,**kwargs):pytest.fail('New study launched before baseline/one-shot gate')


def test_missing_baseline_blocks_collection(monkeypatch):
    monkeypatch.setattr(runner,'verify',lambda:None)
    monkeypatch.setattr(runner,'run_study',forbidden)
    def missing(phase):raise FileNotFoundError('Missing saved baseline')
    monkeypatch.setattr(runner,'validated',missing)
    with pytest.raises(FileNotFoundError):runner.main('collect')


def test_consumed_baseline_blocks_restart_before_learning(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'path',lambda suffix:tmp_path/('route_capture_v3_'+suffix+'_v1.json'))
    monkeypatch.setattr(runner,'verify',lambda:None)
    monkeypatch.setattr(runner,'digest',lambda f:'a'*64)
    monkeypatch.setattr(runner,'run_study',forbidden)
    runner.path('baseline_execution').write_text('{}')
    with pytest.raises(FileExistsError):runner.main('baseline')


def test_same_process_collection_is_forbidden(monkeypatch):
    monkeypatch.setattr(runner,'verify',lambda:None)
    monkeypatch.setattr(runner,'validated',lambda p:None)
    monkeypatch.setattr(runner,'marker',lambda p:{'pid':runner.os.getpid()})
    monkeypatch.setattr(runner,'run_study',forbidden)
    with pytest.raises(ValueError,match='Fresh capture'):runner.main('collect')


def test_no_evaluator_truth_is_part_of_hook_capture_interface():
    assert 'true_route_ids' not in inspect.signature(hooks.Observer.take).parameters
    source=inspect.getsource(hooks.Observer)
    assert 'dependencies(' not in source and 'evaluator_truth' not in source
    assert not runner.prior('sparse_route_reactivation_replay_v1.json').exists()


def test_scientific_conditions_roster_and_completed_evidence():
    selection=runner.read('selection')
    original=runner.read_file(runner.prior('route_numeric_v2_selection_v1.json'))
    assert runner.canonical(selection['frozen_scientific_spec'])==runner.canonical(original['frozen_scientific_spec'])
    assert len(selection['boundary_roster'])==160
    if runner.path('registration').exists():runner.verify()
    if runner.path('result').exists():
        assert runner.canonical(runner.read('result'))==runner.canonical(runner.proof())
        assert runner.read('result')['numerical_noninterference_passed']
        assert not runner.read('result')['component_passed']
        assert len(runner.read('collection')['snapshots'])==160
