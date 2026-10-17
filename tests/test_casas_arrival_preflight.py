"""Delayed arrivals, source parity, unknown retention and whole-audit budget stops."""
from datetime import datetime,timedelta
from sara_engine.evaluation.casas_arrival_preflight import measure, Limits
from sara_engine.evaluation.casas_180 import extract_lines
from sara_engine.evaluation.casas_pooling import routes


def meta(end='2011-06-16'):
    return {'train_start':'2011-06-15','development_start':end,'test_start':end,
            'clock_regression_days':[],'valid_recorded_days':180,'train_recorded_days':1}


def lines(count=200, offset=0, varied=False):
    start = datetime(2011,6,15)+timedelta(seconds=offset)
    output = []
    for i in range(count):
        event = ('Known','ON') if i%6 == 0 else (('Other'+str((i//6)%3),'OFF') if varied else ('Unseen','OFF'))
        at = start+timedelta(seconds=10*i)
        output.append(f'{at.date()} {at.time()} {event[0]} {event[1]}')
    return output


def test_exact_original_extraction_and_address_growth_parity_without_labels():
    source = lines()
    examples,_ = extract_lines(source,meta())
    report = measure(source,meta(),[('Known','ON')])
    rows = examples['train']
    assert report['qualified_outcomes'] == len(rows) > 0
    assert report['hypothetical_readout_writes_per_arm'] == 3*len(rows)
    for arm in ('pool_evidence','joint_evidence'):
        expected = {key for row in rows for key in routes(row,arm)}
        assert report['routes'][arm] == len(expected)
        assert report['actual_learning_writes_per_arm'][arm] == 0
    assert report['new_reference_event_types'] == 1
    assert report['audit']['new_reference_event_occurrences'] > 0
    assert report['retained_event_types'] == 2
    assert report['delay_seconds']['minimum'] >= 300
    assert report['delay_seconds']['maximum'] <= 600
    assert not report['outcome_labels_computed']


def test_both_arms_stop_atomically_when_joint_would_exceed_routes():
    report = measure(lines(varied=True),meta(),[],Limits(routes=4))
    assert report['terminal_stop']['reason'] == 'route_capacity'
    assert report['qualified_outcomes'] == 2
    assert report['routes'] == {'pool_evidence':3,'joint_evidence':4}
    assert report['audit']['uncommitted_qualified_outcomes'] == 1
    assert report['audit']['uninspected_training_rows'] > 0
    assert not report['gates']['whole_training_window']


def test_qualified_budget_stops_without_reading_later_training_fields():
    source = lines(100)+['2011-06-15 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE']
    report = measure(source,meta(),[],Limits(qualified_outcomes=2))
    assert report['qualified_outcomes'] == 2
    assert report['hypothetical_readout_writes_per_arm'] == 6
    assert report['terminal_stop']['reason'] == 'qualified_or_write_budget'
    assert report['audit']['uninspected_training_rows'] > 0


def test_original_independent_splits_are_date_only():
    source = lines()+['2011-06-16 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE']
    report = measure(source,meta(),[])
    assert report['audit']['sealed_date_only_rows'] == 1
    assert not report['original_development_opened'] and not report['test_opened']


def test_growth_prefix_cannot_use_later_day_vocabulary():
    first = lines()
    second = [row.replace('Unseen','FutureOnly') for row in lines(offset=86400)]
    a = measure(first,meta('2011-06-17'),[])
    b = measure(first+second,meta('2011-06-17'),[])
    assert a['growth']['2011-06-15'] == b['growth']['2011-06-15']
    assert b['growth']['2011-06-16']['cumulative_stream_event_types'] > a['stream_event_types']


def test_opened_rows_and_vocabulary_caps_stop_entire_measurement():
    source = lines()+['2011-06-15 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE']
    a = measure(source,meta(),[],Limits(opened_rows=10))
    assert a['audit']['opened_rows'] == 10
    assert a['terminal_stop']['reason'] == 'opened_row_budget'
    b = measure(source,meta(),[],Limits(vocabulary=1))
    assert b['terminal_stop']['reason'] == 'audit_vocabulary_capacity'
    assert b['audit']['opened_rows'] == 2
    assert b['audit']['uninspected_training_rows'] > 0


def test_late_and_unresolved_observations_are_not_negative_labels():
    source = lines(20)+lines(count=20,offset=3600)
    report = measure(source,meta(),[])
    assert report['audit']['late_outcomes'] > 0
    assert report['audit']['unresolved_outcomes'] > 0
    assert not report['outcome_labels_computed']


def test_pending_and_hypothetical_write_limits_do_not_partially_learn():
    a = measure(lines(),meta(),[],Limits(pending=1))
    assert a['terminal_stop']['reason'] == 'pending_capacity'
    assert a['peak_pending'] == 1
    b = measure(lines(),meta(),[],Limits(hypothetical_writes=6))
    assert b['qualified_outcomes'] == 2
    assert b['hypothetical_readout_writes_per_arm'] == 6
    assert set(b['actual_learning_writes_per_arm'].values()) == {0}


def test_metadata_clock_exclusions_stay_date_only():
    metadata = meta()
    metadata['clock_regression_days'] = ['2011-06-15']
    report = measure(['2011-06-15 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE'],metadata,[])
    assert report['audit']['metadata_excluded_rows'] == 1
    assert report['qualified_outcomes'] == 0


def test_registered_hh105_arrival_negative_preserves_old_evidence_and_seals():
    import json
    from pathlib import Path
    from sara_engine.evaluation.casas_180_source import file_sha256
    from sara_engine.utils.project_paths import project_path,processed_data_path
    def path(suffix):
        return processed_data_path('benchmark_fixtures','casas_arrival_preflight_'+suffix+'_v1.json')
    def read(suffix):return json.loads(Path(path(suffix)).read_text())
    reg = read('registration')
    for name,digest in {**reg['code'],**reg['preserved']}.items():
        assert file_sha256(project_path(name)) == digest
    assert file_sha256(path('protocol')) == reg['protocol_sha256']
    execution = read('execution')
    for suffix in ('metadata','acquisition','integrity'):
        assert file_sha256(path(suffix)) == execution['inputs'][suffix]
    result,inventory,growth = read('result'),read('inventory'),read('growth')
    assert result['qualified_outcomes'] == 29552
    assert result['hypothetical_readout_writes_per_arm'] == 88656
    assert set(result['actual_learning_writes_per_arm'].values()) == {0}
    assert result['routes'] == {'pool_evidence':11327,'joint_evidence':20839}
    assert result['new_reference_event_types'] == 2
    assert result['stream_event_types'] == 18
    assert sum(r['occurrences'] for r in inventory['stream_events']) == result['audit']['binary_events'] == 695102
    assert sum(r['new_to_reference'] for r in inventory['stream_events']) == 2
    assert sum(r['occurrences'] for r in inventory['stream_events'] if r['new_to_reference']) == 43260
    assert result['recurrence']['raw_repeat_row_fraction'] == 1
    assert result['recurrence']['rich_contexts_support8'] == 1257
    assert result['terminal_stop'] is None
    assert result['audit']['sealed_date_only_rows'] == 5731389
    assert not result['original_development_opened'] and not result['test_opened']
    assert not result['preflight_passed'] and not result['gates']['arrival_target']
    assert not result['task_class_quality_gates_assessed'] and not result['outcome_labels_computed']
    assert len(growth) == 180
    assert sum(r['qualified_arrivals'] for r in growth.values()) == 29552
    assert sum(r['binary_events'] for r in growth.values()) == 695102
    assert list(sorted(growth.items()))[-1][1]['cumulative_qualified'] == 29552
    for name in ('inventory','growth'):
        assert file_sha256(path(name)) == result[name+'_sha256']
