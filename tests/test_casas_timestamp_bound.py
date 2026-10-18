"""Adversarial timestamp bounds, binary filtering and sealed-field isolation."""
from datetime import datetime,timedelta
import pytest
from sara_engine.evaluation.casas_timestamp_bound import bound_lines, synthetic_lines, validate_counterexamples, BoundLimits
from sara_engine.evaluation.casas_arrival_preflight import measure


def meta(end='2011-06-16'):
    return {'train_start':'2011-06-15','development_start':end,'clock_regression_days':[],
            'valid_recorded_days':180,'train_recorded_days':1}


def test_naive_all_raw_greedy_is_not_an_upper_bound():
    times = [0]*8+[60,119,700]
    naive = measure(synthetic_lines(times),meta(),[])['qualified_outcomes']
    flags = [1]*len(times)
    flags[8] = 0
    actual = measure(synthetic_lines(times,flags),meta(),[])['qualified_outcomes']
    result = bound_lines(synthetic_lines(times),meta())
    assert naive == 0 < actual == 1 <= result['timestamp_upper_bound'] == 1


def test_all_registered_binary_masks_are_bounded():
    result = validate_counterexamples()
    assert result['mask_cases_checked'] == 8192
    assert result['undercounts'] == 0 and result['passed']


def test_timestamp_bound_never_reads_sensor_or_message_tokens():
    times = [0]*8+[60,119,700]
    plain = synthetic_lines(times)
    poisoned = [' '.join(row.split(None,2)[:2])+' SECRET,INVALID,UNPARSED' for row in plain]
    assert bound_lines(plain,meta()) == bound_lines(poisoned,meta())
    csv = [','.join(row.split(None,2)[:2])+',INVALID_TIME_IN_PAYLOAD,UNKNOWN' for row in plain]
    assert bound_lines(plain,meta()) == bound_lines(csv,meta())


def test_independent_periods_and_clock_exclusions_are_date_only():
    source = synthetic_lines([0]*8+[300])+['2011-06-16 INVALID_TIME SECRET SECRET','2012-02-29']
    result = bound_lines(source,meta())
    assert result['audit']['sealed_date_only_rows'] == 2
    metadata = meta()
    metadata['clock_regression_days'] = ['2011-06-15']
    assert bound_lines(['2011-06-15 INVALID_TIME SECRET'],metadata)['audit']['metadata_excluded_rows'] == 1


def test_budget_failure_is_inconclusive_and_drains_date_only():
    source = synthetic_lines([0]*8+[60,119,700])+['2011-06-15 INVALID_TIME SECRET']
    result = bound_lines(source,meta(),BoundLimits(opened_timestamps=8))
    assert result['timestamp_upper_bound'] is None
    assert result['screen_decision'] == 'inconclusive'
    assert result['audit']['uninspected_training_rows'] == 4
    result = bound_lines(synthetic_lines([0]*8+[60,119,700]),meta(),BoundLimits(pending_buckets=1))
    assert result['timestamp_upper_bound'] is None
    assert result['screen_decision'] == 'inconclusive'


def test_history_gap_and_delay_inclusive_endpoints_and_late_result():
    # Eight raw rows at zero form a potential cutoff, even if the real greedy
    # clock would reject it; inclusion is intentionally conservative.
    assert bound_lines(synthetic_lines([0]*8+[300]),meta())['timestamp_upper_bound'] == 1
    assert bound_lines(synthetic_lines([0]*8+[600]),meta())['timestamp_upper_bound'] == 1
    assert bound_lines(synthetic_lines([0]*8+[601]),meta())['timestamp_upper_bound'] == 0
    assert bound_lines(synthetic_lines([0]*7+[300,600]),meta())['timestamp_upper_bound'] == 1
    assert bound_lines(synthetic_lines([0]*7+[301,601]),meta())['timestamp_upper_bound'] == 0


def test_dense_duplicate_rows_need_only_bounded_minute_storage():
    result = bound_lines(synthetic_lines([0]*10000+[300]),meta())
    assert result['timestamp_upper_bound'] == 1
    assert result['work']['candidate_buckets'] == 2
    assert result['work']['peak_pending_buckets'] == 1
    assert result['work']['potential_cutoff_rows'] == 9994


def test_midnight_results_can_certify_an_earlier_cutoff_day():
    source = synthetic_lines([86300]*8+[86600])
    result = bound_lines(source,meta('2011-06-17'))
    assert result['timestamp_upper_bound'] == 1
    assert result['daily']['2011-06-15']['eligible_buckets'] == 1
    assert not result['original_development_opened'] and not result['test_opened']


def test_clock_regression_is_rejected_without_a_false_exclusion():
    with pytest.raises(ValueError,match='regressed'):
        bound_lines(synthetic_lines([60,0]),meta())


def test_exact_target_is_not_excluded_and_one_below_is_excluded():
    origin = datetime(2011,6,15)
    def stream(n):
        for i in range(n):
            at = origin+timedelta(minutes=i)
            yield f'{at.date()} {at.time()} UNINTERPRETED_PAYLOAD'
    metadata = meta('2011-08-16')
    equal = bound_lines(stream(70224),metadata)
    below = bound_lines(stream(70223),metadata)
    assert equal['timestamp_upper_bound'] == 70212
    assert equal['screen_decision'] == 'cannot_exclude'
    assert below['timestamp_upper_bound'] == 70211
    assert below['screen_decision'] == 'exclude_below_target'


def test_bucket_uses_last_candidate_for_inclusive_future_window_end():
    assert bound_lines(synthetic_lines([0]*8+[60,119,719]),meta())['timestamp_upper_bound'] == 1
    assert bound_lines(synthetic_lines([0]*8+[60,119,719.000001]),meta())['timestamp_upper_bound'] == 0


def test_registered_timestamp_results_preserve_negatives_and_training_seals():
    import json
    from pathlib import Path
    from sara_engine.utils.project_paths import project_path,processed_data_path
    from sara_engine.evaluation.casas_180_source import file_sha256
    def path(suffix):
        return processed_data_path('benchmark_fixtures','casas_timestamp_bound_'+suffix+'_v1.json')
    reg = json.loads(Path(path('registration')).read_text())
    result = json.loads(Path(path('result')).read_text())
    validation = json.loads(Path(path('validation')).read_text())
    daily = json.loads(Path(path('daily')).read_text())
    for name,digest in {**reg['code'],**reg['preserved']}.items():
        assert file_sha256(project_path(name)) == digest
    assert file_sha256(path('protocol')) == reg['protocol_sha256']
    assert file_sha256(path('daily')) == result['daily_sha256']
    assert file_sha256(path('validation')) == result['validation_sha256']
    assert validation['mask_cases_checked'] == 8192 and validation['undercounts'] == 0
    assert not result['new_identity_inputs_opened']
    assert not result['sensor_message_fields_inspected']
    assert result['actual_learning_writes'] == 0
    assert not result['original_development_opened'] and not result['test_opened']
    for name,upper,actual,sealed in (('HH103',25334,18676,5008622),
                                    ('HH104',22411,17553,4497877),
                                    ('HH105',37748,29552,5731389)):
        row = result['sources'][name]
        assert row['timestamp_upper_bound'] == upper >= actual == row['existing_training_arrivals']
        assert row['screen_decision'] == 'exclude_below_target'
        assert row['audit']['sealed_date_only_rows'] == sealed
        assert row['work']['peak_pending_buckets'] <= 6
        assert not row['sensor_message_fields_inspected']
        assert row['work']['candidate_buckets'] == (row['work']['eligible_buckets']
              +row['work']['expired_buckets']+row['unresolved_buckets'])
        assert sum(d.get('eligible_buckets',0) for d in daily[name].values()) == upper
