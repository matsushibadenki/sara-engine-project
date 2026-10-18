"""Frozen metadata budgets, literal target compatibility and seal preservation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.sphere_eco_metadata import (
    REQUIRED, ECO_HOMES, validate_requests, evaluate, metadata_target)
from sara_engine.utils.project_paths import processed_data_path, project_path


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'sphere_eco_metadata_' + suffix + '_v1.json'))


def sample():
    selection = json.loads(path('selection').read_text())
    requests = [{'id': 1, 'source': 'SPHERE', 'kind': 'page', 'target': 'https://arxiv.org/html/1805.11907v2'},
                {'id': 2, 'source': 'ECO', 'kind': 'page', 'target': 'https://vs.inf.ethz.ch/res/show.html?what=eco-data'}]
    records = [{'request_id': r['id'], 'source': r['source'], 'url': r['target'], 'status': 'read', 'notes': 'Synthetic metadata evidence'} for r in requests]
    sources = []
    for source, ref, kind in [('SPHERE', 1, 'history_packet'), ('ECO', 2, 'entrance_pir_and_numeric_occupancy')]:
        homes = []
        for home in ECO_HOMES if source == 'ECO' else []:
            homes.append({'id': home, 'checks': {k: {'status': 'unresolved', 'value': None, 'refs': []} for k in REQUIRED}})
        sources.append({'id': source, 'households': homes, 'publisher_household_manifest_found': False,
                        'native_kitchen_on': {'status': 'incompatible', 'raw_kind': kind,
                                             'native_kitchen_sensor': None, 'native_ON_message': None, 'refs': [ref]},
                        **{f: {'status': 'unresolved', 'value': None, 'refs': []} for f in ('density', 'package', 'terms')}})
    evidence = {'records': records, 'dataset_payload_bytes': 0, 'dataset_preview_rows': 0,
                'archive_range_requests': 0, 'learning_writes': 0, 'independent_periods_opened': False,
                'conditions_relaxed': False, 'casas_candidates_added': 0}
    return selection, requests, evidence, {'sources': sources}


def test_packet_history_and_entrance_motion_do_not_become_kitchen_on():
    result = evaluate(*sample())
    assert [r['decision'] for r in result['sources']] == ['exclude_documented_stream_for_fixed_target'] * 2
    assert result['sources'][0]['household_metadata_entries'] == 0
    assert result['sources'][1]['household_metadata_entries'] == 6
    assert result['approved_payload_acquisitions'] == []
    assert not result['snn_assessment']['ann_exclusive_advantage_established']
    assert not result['snn_assessment']['snn_usefulness_established']


def test_claiming_compatible_history_packet_is_rejected():
    args = sample()
    args[3]['sources'][0]['native_kitchen_on']['status'] = 'compatible'
    with pytest.raises(ValueError, match='contradicts'):
        evaluate(*args)


def test_complete_compatible_schema_does_not_authorize_payload_or_quality():
    args = sample()
    args[3]['sources'][0]['native_kitchen_on'].update(
        status='compatible', raw_kind='scalar_binary_notification', native_kitchen_sensor=True, native_ON_message=True)
    result = evaluate(*args)
    assert result['sources'][0]['decision'] == 'unresolved_metadata'
    assert not result['sources'][0]['payload_acquisition_authorized']
    assert not result['sources'][0]['source_quality_pass']


@pytest.mark.parametrize('field', ['dataset_payload_bytes', 'dataset_preview_rows', 'archive_range_requests', 'learning_writes',
                                  'independent_periods_opened', 'conditions_relaxed', 'casas_candidates_added'])
def test_prohibited_work_or_seal_changes_are_rejected(field):
    args = sample()
    args[2][field] = 1
    with pytest.raises(ValueError):
        evaluate(*args)


@pytest.mark.parametrize('kind,budget', [('search', 'search_queries'), ('page', 'primary_metadata_page_attempts')])
def test_request_is_refused_before_budget_overshoot(kind, budget):
    selection = sample()[0]
    selection['budgets'][budget] = 1
    requests = [{'id': i, 'source': 'SPHERE', 'kind': kind, 'target': 'https://arxiv.org/abs/1805.11907'} for i in (1, 2)]
    with pytest.raises(ValueError, match='budget'):
        validate_requests(selection, requests)


@pytest.mark.parametrize('change', ['source_order', 'eco_add', 'eco_drop', 'missing_receipt', 'failed_proof',
                                   'wrong_owner', 'invent_zero', 'terms_without_proof', 'evidence_bytes'])
def test_scope_and_provenance_cannot_rescue_unknown_metadata(change):
    selection, requests, evidence, inputs = sample()
    if change == 'source_order':
        requests = [requests[1], requests[0]]
    elif change == 'eco_add':
        inputs['sources'][1]['households'].append({'id': '07', 'checks': {}})
    elif change == 'eco_drop':
        inputs['sources'][1]['households'].pop()
    elif change == 'missing_receipt':
        evidence['records'].pop()
    elif change == 'failed_proof':
        evidence['records'][0]['status'] = 'unavailable'
    elif change == 'wrong_owner':
        inputs['sources'][1]['native_kitchen_on']['refs'] = [1]
    elif change == 'invent_zero':
        inputs['sources'][1]['households'][0]['checks']['pir_timestamp_rows']['value'] = 0
    elif change == 'terms_without_proof':
        inputs['sources'][1]['terms']['status'] = 'verified'
    else:
        selection['budgets']['retained_evidence_utf8_bytes'] = 1
    with pytest.raises(ValueError):
        evaluate(selection, requests, evidence, inputs)


@pytest.mark.parametrize('url', ['https://vs.inf.ethz.ch/source.zip', 'https://example.org/occupancy.csv',
                               'http://example.org/metadata', 'https://app.data-archive.ethz.ch/delivery/DeliveryManagerServlet?dps_pid=IE594964'])
def test_bulk_urls_are_never_opened_by_metadata_workflow(url):
    with pytest.raises(ValueError):
        metadata_target(url)


def test_consumed_result_and_all_previous_evidence():
    from sara_engine.evaluation.casas_180_source import file_sha256
    if not path('result').exists():
        pytest.skip('Real metadata audit has not executed')
    def read(suffix):
        return json.loads(path(suffix).read_text())
    journal = Path(processed_data_path('benchmark_fixtures', 'sphere_eco_metadata_requests_v1.jsonl'))
    requests = [json.loads(line) for line in journal.read_text().splitlines()]
    result = evaluate(read('selection'), requests, read('evidence'), read('inputs'))
    assert result == read('result')
    for name, digest in read('selection')['preserved'].items():
        assert file_sha256(project_path(name)) == digest
    for name, digest in read('registration')['files'].items():
        assert file_sha256(project_path(name)) == digest
    execution = read('execution')
    for suffix, digest in execution['sha256'].items():
        p = journal if suffix == 'requests' else path(suffix)
        assert file_sha256(p) == digest
