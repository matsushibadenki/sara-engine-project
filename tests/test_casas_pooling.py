"""Causal pooling, opposing-future witnesses, matched writes and sealed extraction."""
from dataclasses import replace
from datetime import datetime, timedelta
import pytest
from sara_engine.evaluation.casas_longitudinal import Example
from sara_engine.evaluation.casas_pooling import PoolCounts, routes, phase, identification, ARMS
from sara_engine.evaluation.casas_180 import extract_lines


def row(i=0, label=0):
    cutoff = datetime(2011,6,15,10)+timedelta(seconds=i)
    return Example(cutoff, cutoff+timedelta(seconds=.5), (('Bedroom','ON'),)*8,
                   (2,)*6+((0 if label else 2),), label)


def test_pool_has_at_most_four_branches_and_preserves_exact_current():
    keys = set()
    for same in (False,True):
        for fast in (False,True):
            e = row()
            if not same:
                e = replace(e, history=e.history[:-2]+(('Other','ON'),)+e.history[-1:])
            e = replace(e,gaps=(2,)*6+((0 if fast else 2),))
            address = routes(e,'pool_evidence')
            assert address[:2] == routes(e,'current')[:2]
            assert address == routes(replace(e,label=1),'pool_evidence')
            keys.add(address[-1])
    assert len(keys) == 4
    assert routes(row(),'pool_evidence') != routes(replace(row(),history=(('Other','ON'),)*8),'pool_evidence')


def test_pool_erases_older_details_but_keeps_latest_timing_and_sensor_change():
    e = row()
    altered = replace(e,history=(('Other','OFF'),)*6+e.history[-2:], gaps=(3,)*6+e.gaps[-1:])
    assert routes(e,'pool_evidence') == routes(altered,'pool_evidence')
    assert routes(e,'pool_evidence') != routes(row(label=1),'pool_evidence')
    assert routes(e,'pool_constant') == routes(row(label=1),'pool_constant')


def test_same_current_opposing_futures_identified_and_scalar_replay_exact():
    rows = [row(i,i%2) for i in range(400)]
    model = PoolCounts('pool_evidence')
    phase(model,rows,True)
    audit = identification(model,rows)
    assert audit['opposing_evidence_contexts'] == 1
    assert audit['maximum_branches_per_current'] == 2
    assert audit['actual_routes'] <= audit['route_ceiling']
    state = model.snapshot()
    replay = PoolCounts.from_snapshot(state)
    predictions = phase(model,[row(401,0),row(402,1)],False)
    assert predictions[0] < .5 < predictions[1]
    assert predictions == phase(replay,[row(401,0),row(402,1)],False)
    assert model.snapshot() == state and model.changed_writes == 1200


def test_all_thirteen_arms_have_three_real_writes_per_feedback():
    rows = [row(i,i%2) for i in range(400)]
    for arm in ARMS:
        model = PoolCounts(arm)
        phase(model,rows,True)
        assert model.changed_writes == 1200
        assert sum(sum(v) for v in model.cells.values()) == 1200
    assert len(ARMS) == 13


def test_no_early_outcome_read_and_fallback_exactly_matches_current():
    a, b = PoolCounts('current'), PoolCounts('pool_evidence')
    for e in [row(i,i%2) for i in range(10)]:
        ra, rb = a.predict_example(e), b.predict_example(e)
        assert ra.probability == rb.probability
        assert b.pending[rb.identity] is rb
        with pytest.raises(ValueError):
            b.deliver(rb,e.cutoff,lambda:pytest.fail('Early outcome read'),training=True)
        a.deliver(ra,ra.due,lambda:e.label,training=True)
        b.deliver(rb,rb.due,lambda:e.label,training=True)
    assert a.changed_writes == b.changed_writes == 30


def test_older_confirmation_and_independent_periods_stay_date_only():
    examples,audit = extract_lines([
        '2011-10-13 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE',
        '2011-12-12 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE',
        '2012-02-10 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE'],
        {'train_start':'2011-06-15','development_start':'2011-09-03',
         'test_start':'2011-10-13','clock_regression_days':[]})
    assert audit['sealed_date_only_rows'] == 3
    assert examples == {'train':[],'development':[]}


def test_snapshot_rejects_invalid_policy_and_write_accounting():
    state = PoolCounts('pool_evidence').snapshot()
    state['policy'] = 'retuned'
    with pytest.raises(ValueError):PoolCounts.from_snapshot(state)
    state = PoolCounts('pool_evidence').snapshot()
    state['counts']['changed_writes'] = 3
    with pytest.raises(ValueError):PoolCounts.from_snapshot(state)


def test_registered_pooling_result_preserves_old_negatives_and_seals():
    import json
    from pathlib import Path
    from sara_engine.utils.project_paths import processed_data_path, project_path, model_path
    from sara_engine.evaluation.casas_180_source import file_sha256
    def read(suffix):
        return json.loads(Path(processed_data_path('benchmark_fixtures',
                          'casas_pooling_'+suffix+'_v1.json')).read_text())
    reg = read('registration')
    assert file_sha256(processed_data_path('benchmark_fixtures',
                       'casas_pooling_protocol_v1.json')) == reg['protocol_sha256']
    for name,digest in {**reg['code'],**reg['preserved']}.items():
        assert file_sha256(project_path(name)) == digest
    result = read('result')
    assert result['qualification']['passed']
    assert result['qualification']['contrast_groups'] == 50
    assert result['audit']['sealed_date_only_rows'] == 4249759
    assert not result['previous_confirmation_opened']
    assert not result['original_development_opened'] and not result['test_opened']
    c = result['comparison']
    assert set(c['arms']) == set(ARMS)
    assert {v['changed_writes'] for v in c['arms'].values()} == {52659}
    assert all(v['confirmation_writes'] == 0 for v in c['arms'].values())
    assert c['identification']['opposing_evidence_contexts'] == 0
    assert c['identification']['maximum_branches_per_current'] == 4
    assert c['identification']['actual_routes'] == 9219 <= c['identification']['route_ceiling']
    assert c['arms']['pool_evidence']['fallbacks'] == 9801
    assert c['traces']['pool_evidence'] == c['traces']['current'] == c['traces']['scalar_pool']
    assert not c['passed'] and not c['feasibility_gates']['opposing_contexts']
    for arm,digest in result['model_sha256'].items():
        assert file_sha256(model_path('casas_pooling',arm+'_v1.json.gz')) == digest


def test_constant_gap_control_erases_controlled_timing_witness():
    rows = [row(i,i%2) for i in range(400)]
    current, constant = PoolCounts('current'), PoolCounts('pool_constant')
    assert phase(current,rows,True) == phase(constant,rows,True)
    assert current.changed_writes == constant.changed_writes == 1200
    assert phase(current,[row(401,0),row(402,1)],False) == phase(constant,[row(401,0),row(402,1)],False)
    assert constant.overrides == 0
