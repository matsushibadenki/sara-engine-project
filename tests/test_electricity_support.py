"""Small synthetic windows exercise the preregistered eligibility rules."""
from datetime import datetime, timedelta
import pytest

from sara_engine.evaluation.electricity_support import audit_support


def protocol():
    return {'history_observations_including_cutoff': 2, 'outcome_delay_intervals': 1,
            'interval_minutes': 15, 'excluded_transition_dates': [],
            'training_years': [2011, 2012], 'development_years': [2013],
            'cutoff_local_time': '12:00:00', 'minimum_observations_per_class_per_qualifying_client': 1,
            'minimum_active_clients_per_split': 1, 'minimum_clients_with_both_classes_per_split': 1}


def rows(values, start='2013-01-01T11:45:00'):
    time = datetime.fromisoformat(start)
    return [f'{time + timedelta(minutes=15*i):%Y-%m-%d %H:%M:%S};{value}\n' for i, value in enumerate(values)]


def test_direction_ties_activation_and_invalid_windows():
    for values, expected in [([1, 2, 3], 1), ([0, 2, 3], 0), ([1, 'NaN', 3], 0)]:
        result = audit_support(rows(values), protocol(), client_count=1)
        assert result['splits']['development']['rise_observations'] == expected
        assert result['peak_buffer_rows'] <= 3
    result = audit_support(rows([1, 2, 2]), protocol(), client_count=1)
    assert result['splits']['development']['excluded_ties'] == 1
    result = audit_support(rows([1, 2, 0]), protocol(), client_count=1)
    assert result['splits']['development']['fall_observations'] == 1


def test_dst_and_sealed_payload_are_excluded():
    config = protocol()
    config['excluded_transition_dates'] = ['2013-01-01']
    result = audit_support(rows([1, 2, 3]), config, client_count=1)
    assert result['splits']['development']['rise_observations'] == 0
    result = audit_support(rows(['SECRET'] * 3, '2014-01-01T11:45:00'), config, client_count=1)
    assert result['numeric_cells_parsed'] == 0
    assert result['opaque_rows'] == 3


def test_split_crossing_and_timestamp_gaps():
    config = protocol()
    config['cutoff_local_time'] = '23:45:00'
    result = audit_support(rows([1, 2, 3], '2012-12-31T23:30:00'), config, client_count=1)
    assert result['splits']['training']['rise_observations'] == 0
    assert result['splits']['development']['rise_observations'] == 0
    with pytest.raises(ValueError, match='Non-contiguous'):
        audit_support(rows([1, 2, 3])[::2], config, client_count=1)
