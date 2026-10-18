"""Verify conditional capacity arithmetic and refusal to promote metadata averages."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.source_metadata_feasibility import (
    calendar_envelope, training_date_ceiling, regular_grid, evaluate, evidence_budget)
from sara_engine.utils.project_paths import processed_data_path, project_path


def fixture(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'source_metadata_feasibility_' + suffix + '_v1.json'))


def read(suffix):
    return json.loads(fixture(suffix).read_text())


@pytest.mark.parametrize('n,training', [(180, 120), (245, 163), (269, 179), (270, 180), (659, 180)])
def test_original_training_split_is_not_silently_lengthened(n, training):
    assert training_date_ceiling(n) == training


def test_global_short_duration_bounds_every_household_without_pooling():
    assert calendar_envelope('2019-04-01', '2019-06-30') == 91
    assert calendar_envelope('2012-06-01', '2013-01-31') == 245
    result = evaluate(read('selection'), read('evidence'), read('inputs'))
    tihm = result['sources'][0]
    assert tihm['status'] == 'exclude_documented_duration'
    assert tihm['source_calendar_date_ceiling'] == 91
    assert tihm['training_recorded_date_ceiling'] == 60
    assert not tihm['dataset_acquisition_authorized']


@pytest.mark.parametrize('step', [1, 4, 12, 20, 59, 60, 61, 119, 120, 180, 221, 222, 240, 300, 301, 601])
def test_regular_witness_matches_unchanged_original_extractor(step):
    from sara_engine.evaluation.casas_timestamp_bound import synthetic_lines
    from sara_engine.evaluation.casas_arrival_preflight import measure
    meta = {'train_start': '2011-06-15', 'development_start': '2011-06-16',
            'clock_regression_days': [], 'valid_recorded_days': 180, 'train_recorded_days': 1}
    actual = measure(synthetic_lines(range(0, 86400, step)), meta, [])['qualified_outcomes']
    assert regular_grid(1, step)['qualified_arrivals_if_assumptions_hold'] == actual


def test_dense_stream_can_meet_target_but_exceed_unchanged_row_budget():
    dense = regular_grid(180, 1)
    bounded = regular_grid(180, 20)
    sparse = regular_grid(180, 240)
    assert dense['count_target_possible_in_scenario']
    assert dense['raw_rows'] == 15552000 and not dense['within_unchanged_input_row_cap']
    assert bounded['count_target_possible_in_scenario'] and bounded['within_unchanged_input_row_cap']
    assert bounded['within_unchanged_outcome_write_caps']
    assert not sparse['count_target_possible_in_scenario']


def test_nominal_electric_rate_and_aggregate_counts_do_not_certify_pir_arrivals():
    inputs = deepcopy(read('inputs'))
    inputs['sources'][3]['aggregate_electric_rows_lower_bound'] = 10**15
    inputs['sources'][3]['nominal_electric_cadence_seconds'] = 1
    result = evaluate(read('selection'), read('evidence'), inputs)
    eco = result['sources'][3]
    assert eco['status'] == 'unresolved_metadata'
    assert 'per_identity_training_arrivals_unverified' in eco['reasons']
    assert eco['training_recorded_date_ceiling'] == 163
    assert result['approved_payload_acquisitions'] == []


@pytest.mark.parametrize('change', ['queries', 'pages', 'bytes', 'preview', 'payload', 'casas', 'order', 'provenance'])
def test_frozen_budgets_and_identity_provenance_are_enforced(change):
    selection, evidence, inputs = deepcopy(read('selection')), deepcopy(read('evidence')), deepcopy(read('inputs'))
    if change == 'queries':
        evidence['search_queries'].append('Unregistered extra search')
    elif change == 'pages':
        evidence['pages'].append({**evidence['pages'][0], 'id': 'P17'})
    elif change == 'bytes':
        selection['budgets']['retained_evidence_utf8_bytes'] = 1
    elif change == 'preview':
        evidence['dataset_preview_rows'] = 1
    elif change == 'payload':
        evidence['dataset_payload_bytes'] = 1
    elif change == 'casas':
        evidence['pages'][0]['candidate'] = 'HH109'
    elif change == 'order':
        inputs['sources'].reverse()
    else:
        inputs['sources'][0]['evidence_pages'] = ['P03']
    with pytest.raises(ValueError):
        evaluate(selection, evidence, inputs)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_arithmetic_input_is_rejected(value):
    with pytest.raises(ValueError):
        regular_grid(180, value)


def test_full_bundle_size_and_native_schema_are_not_waived():
    result = evaluate(read('selection'), read('evidence'), read('inputs'))
    ideal = result['sources'][2]
    assert ideal['status'] == 'incompatible_documented_stream'
    assert 'documented_full_bundle_over_budget_subset_unverified' in ideal['reasons']
    assert all(not r['source_quality_pass'] for r in result['sources'])
    assert result['actual_learning_writes'] == 0


def test_stored_result_and_prior_evidence():
    from sara_engine.evaluation.casas_180_source import file_sha256
    if not fixture('result').exists():
        pytest.skip('Metadata arithmetic has not executed yet')
    result = read('result')
    assert result == evaluate(read('selection'), read('evidence'), read('inputs'))
    for name, digest in read('selection')['preserved'].items():
        assert file_sha256(project_path(name)) == digest
    for name, digest in read('registration')['files'].items():
        assert file_sha256(project_path(name)) == digest
    assert not result['independent_periods_opened'] and result['casas_candidates_added'] == 0
    assert result['work'] == evidence_budget(read('selection'), read('evidence'))
