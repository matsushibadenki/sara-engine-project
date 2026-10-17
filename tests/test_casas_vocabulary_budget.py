"""Vocabulary unions, finite state certificates and unchanged local addresses."""
from copy import deepcopy
from datetime import datetime,timedelta
from itertools import product
import pytest
from sara_engine.evaluation.casas_vocabulary_budget import analyze, universe_bound, certified_vocabulary_limit, CLOCK_CONTEXTS
from sara_engine.evaluation.casas_longitudinal import Example
from sara_engine.evaluation.casas_pooling import routes


def states():
    event = ('Bedroom','ON')
    context = (event,2,10,0)
    current = repr(('current',context))
    base = {'global':[10,10],current:[10,10]}
    pool = {**base,repr(('pool',current,True,False)):[10,0],
            repr(('pool',current,False,True)):[0,10]}
    joint = {**base,repr(('detail',(context,('Hall','ON'),event,(0,2)))):[10,0],
             repr(('detail',(context,('Kitchen','OFF'),event,(3,2)))):[0,10]}
    return {arm:{'arm':arm,'policy':'same_sensor_fast10_four_branches_v1',
                 'counts':{'schema':1,'capacity':100000,'changed_writes':60,'cells':cells}}
            for arm,cells in (('pool_evidence',pool),('joint_evidence',joint))}


def test_history_only_events_expand_shared_known_vocabulary_without_mutation():
    source = states()
    before = deepcopy(source)
    report = analyze(source)
    assert source == before
    assert report['current_event_cardinality'] == 1
    assert report['shared_known_event_cardinality'] == 3
    assert report['history_only_event_cardinality'] == 2
    assert report['arms']['pool_evidence']['known_event_new_combination_bound'] == 10081
    assert not report['arms']['joint_evidence']['known_event_capacity_certified']
    assert report['actual_learning_writes'] == report['actual_acquired_observations'] == 0


def test_original_route_enumeration_matches_envelopes_for_two_event_universe():
    events = (('A','ON'),('B','OFF'))
    keys = {arm:set() for arm in ('pool_evidence','joint_evidence')}
    for current,previous,older,g1,g4 in product(events,events,events,range(4),range(4)):
        history = [current]*8
        history[-2],history[-5] = previous,older
        gaps = [0]*7
        gaps[-1],gaps[-4] = g1,g4
        cutoff = datetime(2026,10,12,10)
        row = Example(cutoff,cutoff+timedelta(seconds=300),tuple(history),tuple(gaps),0)
        for arm in keys:
            keys[arm].update(routes(row,arm))
    for arm in keys:
        assert 1+CLOCK_CONTEXTS*(len(keys[arm])-1) == universe_bound(2,arm)


def test_unknown_type_frontiers_are_capacity_certificates_not_rejection_rules():
    for arm,limit in (('pool_evidence',29),('joint_evidence',2)):
        assert certified_vocabulary_limit(arm) == limit
        assert universe_bound(limit,arm) <= 100000 < universe_bound(limit+1,arm)
        assert universe_bound(limit+1,arm) > universe_bound(limit,arm)
    assert universe_bound(30,'pool_evidence') == 100801
    for value in (0,-1,True):
        with pytest.raises(ValueError):universe_bound(value,'pool_evidence')


def test_known_combinations_and_unrestricted_events_remain_distinct():
    report = analyze(states())
    pooled = report['arms']['pool_evidence']
    assert pooled['known_event_capacity_certified']
    assert pooled['additional_unknown_event_types_certified'] == 26
    assert not pooled['unrestricted_event_capacity_certified']
    assert pooled['unlimited_horizon_unrestricted_event_bound'] is None
    assert pooled['unrestricted_event_bound_at_acquisition_ceiling'] == pooled['existing_routes']+2*263295
    assert report['arms']['joint_evidence']['additional_unknown_event_types_certified'] is None


def test_invalid_saved_gap_and_mismatched_current_counts_rejected():
    source = states()
    cells = source['joint_evidence']['counts']['cells']
    key = next(k for k in cells if k.startswith("('detail'"))
    value = cells.pop(key)
    cells[key.replace('(0, 2)','(4, 2)')] = value
    with pytest.raises(ValueError,match='gap-bin'):
        analyze(source)
    source = states()
    source['joint_evidence']['counts']['changed_writes'] = 61
    with pytest.raises(ValueError):analyze(source)


def test_registered_known_vocabulary_capacity_and_original_seals():
    import json
    from pathlib import Path
    from sara_engine.evaluation.casas_180_source import file_sha256
    from sara_engine.utils.project_paths import project_path, processed_data_path
    def path(suffix):
        return processed_data_path('benchmark_fixtures','casas_vocabulary_budget_'+suffix+'_v1.json')
    reg = json.loads(Path(path('registration')).read_text())
    result = json.loads(Path(path('result')).read_text())
    assert file_sha256(path('protocol')) == reg['protocol_sha256']
    for name,digest in {**reg['code'],**reg['preserved']}.items():
        assert file_sha256(project_path(name)) == digest
    assert result['actual_acquired_observations'] == result['actual_learning_writes'] == 0
    assert not any(result[k] for k in ('raw_source_opened','confirmation_opened',
                                       'original_development_opened','test_opened'))
    assert result['current_event_cardinality'] == 19
    assert result['stored_history_event_cardinality'] == result['shared_known_event_cardinality'] == 20
    assert result['history_only_event_cardinality'] == 1
    assert file_sha256(path('inventory')) == result['inventory_sha256']
    pool,joint = result['arms']['pool_evidence'],result['arms']['joint_evidence']
    assert pool['current_counts_sha256'] == joint['current_counts_sha256']
    assert pool['existing_changed_writes'] == joint['existing_changed_writes'] == 52659
    assert pool['known_event_new_combination_bound'] == 67201 <= 100000
    assert pool['known_event_capacity_certified']
    assert joint['known_event_new_combination_bound'] == 86029441
    assert not joint['known_event_capacity_certified']
    assert pool['additional_unknown_event_types_certified'] == 9
    assert not pool['unrestricted_event_capacity_certified']
    assert not joint['unrestricted_event_capacity_certified']


def test_pool_bound_is_conservative_even_for_one_sensor_and_unknown_lag():
    assert universe_bound(1,'pool_evidence') == 3361
    # A new lag event can flip the repeat boolean without changing the current.
    cutoff = datetime(2026,10,12,10)
    base = Example(cutoff,cutoff+timedelta(seconds=300),(('A','ON'),)*8,(0,)*7,0)
    changed_history = list(base.history)
    changed_history[-2] = ('Unseen','OFF')
    changed = Example(base.cutoff,base.due,tuple(changed_history),base.gaps,base.label)
    a,b = routes(base,'pool_evidence'),routes(changed,'pool_evidence')
    assert a[:2] == b[:2] and a[2] != b[2]
