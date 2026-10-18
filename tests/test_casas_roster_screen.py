"""Test finite source budgets and timestamp-only split preservation."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
import struct

import pytest

from sara_engine.evaluation.casas_180_source import metadata as original_metadata
from sara_engine.evaluation.casas_roster_screen import (
    Budget, CapacityError, validate_selection, validate_range, member_range,
    date_time_prefix, timestamp_metadata, screen)
from sara_engine.utils.project_paths import processed_data_path, raw_data_path


def selection():
    with open(processed_data_path('benchmark_fixtures', 'casas_roster_screen_selection_v1.json')) as stream:
        return json.load(stream)


def calendar(n=200):
    start = datetime(2011, 1, 1)
    return [str((start + timedelta(days=i)).date()) + ' 12:00:00 opaque,unparsed payload' for i in range(n)]


def test_original_split_and_clock_exclusion_parity():
    lines = calendar()
    lines.insert(70, lines[68])
    limits = selection()['budgets']
    got = timestamp_metadata(lines, Budget(limits), 1000)
    # The original delimiter heuristic requires comma-free synthetic payloads.
    expected = original_metadata([line.replace(',', '_') for line in lines])
    assert {k: got[k] for k in expected if k not in ('selection', 'column_counts')} == {
        k: v for k, v in expected.items() if k not in ('selection', 'column_counts')}
    assert got['clock_regression_days'] == ['2011-03-10', '2011-03-11']


def test_payload_cannot_change_metadata():
    limits = selection()['budgets']
    baseline = timestamp_metadata(calendar(), Budget(limits), 1000)
    altered = [line.split(' opaque', 1)[0] + ' INVALID SENSOR,\tON OFF ; : NaN' for line in calendar()]
    assert timestamp_metadata(altered, Budget(limits), 1000) == baseline


@pytest.mark.parametrize('cap,aggregate', [(10, 1000), (1000, 10)])
def test_metadata_capacity_stops_before_timestamp_parse(cap, aggregate):
    budget = Budget({'metadata_timestamps_total': aggregate})
    with pytest.raises(CapacityError):
        timestamp_metadata(calendar(10) + ['invalid timestamp must stay unopened'], budget, cap)
    assert budget.used['metadata_timestamps_total'] == 10


def test_short_source_never_becomes_exclusion():
    with pytest.raises(ValueError, match='180'):
        timestamp_metadata(calendar(179), Budget({'metadata_timestamps_total': 1000}), 1000)


def test_budget_reservations_are_shared_and_never_overshoot():
    budget = Budget({'network_bytes': 12})
    budget.charge('network_bytes', 8)
    with pytest.raises(CapacityError):
        budget.charge('network_bytes', 5)
    assert budget.used['network_bytes'] == 8
    budget.charge('network_bytes', 4)
    assert budget.used['network_bytes'] == 12


def test_training_cap_is_inconclusive_and_sealed_time_unparsed():
    limits = {**selection()['budgets'], 'training_timestamps_total': 2}
    budget = Budget(limits)
    meta = {'train_start': '2011-01-01', 'development_start': '2011-01-02', 'clock_regression_days': []}
    lines = ['2011-01-01 12:00:00 opaque'] * 3 + ['2011-01-02 invalid-independent-time invalid-payload']
    got = screen(lines, meta, budget, limits)
    assert got['screen_decision'] == 'inconclusive'
    assert got['timestamp_upper_bound'] is None
    assert got['audit']['sealed_date_only_rows'] == 1
    assert budget.used['training_timestamps_total'] == 2
    with pytest.raises(CapacityError):
        screen(lines, meta, budget, limits)


def test_exact_training_budget_at_complete_eof_is_usable():
    limits = {**selection()['budgets'], 'training_timestamps_total': 2}
    budget = Budget(limits)
    meta = {'train_start': '2011-01-01', 'development_start': '2011-01-02', 'clock_regression_days': []}
    got = screen(['2011-01-01 12:00:00 opaque'] * 2, meta, budget, limits)
    assert got['screen_decision'] == 'exclude_below_target'
    assert budget.used['training_timestamps_total'] == 2


@pytest.mark.parametrize('change', ['reorder', 'extend', 'replace', 'size', 'aggregate', 'expanded'])
def test_roster_cannot_be_adapted(change):
    spec = deepcopy(selection())
    with open(raw_data_path('r2_casas_180', 'directory.json')) as stream:
        directory = json.load(stream)
    validate_selection(spec, directory)
    if change == 'reorder':
        spec['roster'].reverse()
    elif change == 'extend':
        spec['roster'].append(spec['roster'][0])
    elif change == 'replace':
        spec['roster'][0]['identity'] = 'HH109'
    elif change == 'size':
        spec['roster'][0]['size'] += 1
    elif change == 'aggregate':
        spec['budgets']['network_bytes'] = 100
    else:
        spec['budgets']['expanded_bytes_all_passes'] = 100
    with pytest.raises(ValueError):
        validate_selection(spec, directory)


def test_http_response_must_match_exact_range():
    headers = b'HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 30-39/100\r\nPrivate: do-not-print\r\n'
    validate_range(headers, 30, 39, 100, 10)
    for args in [(31, 40, 100, 10), (30, 39, 101, 10), (30, 39, 100, 9)]:
        with pytest.raises(ValueError):
            validate_range(headers, *args)


def test_zip_local_header_checks_identity_and_sizes():
    entry = selection()['roster'][0]
    name = entry['name'].encode()
    header = struct.pack('<4s5H3I2H', b'PK\x03\x04', 20, 0, 8, 0, 0,
                         entry['crc32'], entry['compressed_size'], entry['size'], len(name), 0) + name
    start, end = member_range(header, entry)
    assert start == entry['offset'] + len(header)
    assert end - start + 1 == entry['compressed_size']
    with pytest.raises(ValueError):
        member_range(header, {**entry, 'name': 'data/hh109.csv'})


@pytest.mark.parametrize('line', ['2011-01-01', '2011-01-01 bad payload', '2011-01-01 00:00:00+09:00 payload'])
def test_invalid_timestamp_metadata_fails_closed(line):
    with pytest.raises(ValueError):
        date_time_prefix(line)


def test_frozen_roster_results_and_previous_evidence():
    from pathlib import Path
    from sara_engine.evaluation.casas_180_source import file_sha256
    from sara_engine.utils.project_paths import project_path
    def fixture(suffix):
        return Path(processed_data_path('benchmark_fixtures', 'casas_roster_screen_' + suffix + '_v1.json'))
    if not fixture('result').exists():
        pytest.skip('Real roster screen has not executed yet')
    registration = json.loads(fixture('registration').read_text())
    for name, digest in {**registration['code'], **registration['preserved']}.items():
        assert file_sha256(project_path(name)) == digest
    result = json.loads(fixture('result').read_text())
    details = json.loads(fixture('details').read_text())
    assert file_sha256(fixture('details')) == result['details_sha256']
    assert [r['identity'] for r in result['sources']] == ['HH106', 'HH107', 'HH108']
    assert result['target'] == 70212 and result['actual_learning_writes'] == 0
    assert not result['sensor_message_fields_inspected']
    assert not result['independent_predictive_information_opened']
    for key, used in result['work'].items():
        assert used <= selection()['budgets'][key]
    assert result['network_reserved_bytes'] <= selection()['budgets']['network_bytes']
    for row in result['sources']:
        if row['screen_decision'] == 'inconclusive':
            assert row['timestamp_upper_bound'] is None
            continue
        upper = row['timestamp_upper_bound']
        assert (upper < 70212) == (row['screen_decision'] == 'exclude_below_target')
        meta = details[row['identity']]['metadata']
        assert meta['valid_recorded_days'] >= 180
        assert row['work']['peak_pending_buckets'] <= 6
        assert row['audit']['streamed_rows'] == meta['rows']
        assert not row['source_quality_pass'] and not row['input_preflight_authorized']
        assert details[row['identity']]['integrity']['size'] == next(
            r['size'] for r in selection()['roster'] if r['identity'] == row['identity'])
