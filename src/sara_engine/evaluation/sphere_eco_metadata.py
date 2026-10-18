"""Fail-closed metadata ledger and fixed Kitchen ON source contract audit."""
from collections import Counter
import json
from urllib.parse import urlparse

ORDER = ('SPHERE', 'ECO')
ECO_HOMES = ('01', '02', '03', '04', '05', '06')
REQUIRED = ('recorded_dates', 'pir_timestamp_rows', 'kitchen_on',
            'causal_availability', 'compressed_bytes', 'expanded_bytes', 'reuse_terms')


def metadata_target(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ValueError('HTTPS metadata URL required')
    forbidden = ('.zip', '.gz', '.csv', '.bson', '.jsonl', '.h5', '.hdf5', '.7z', '.deflate')
    if parsed.path.lower().endswith(forbidden) or 'deliverymanagerservlet' in parsed.path.lower():
        raise ValueError('Dataset or bulk delivery URL prohibited')


def validate_requests(selection, requests):
    limits = selection['budgets']
    if [c['id'] for c in selection['candidates']] != list(ORDER) or selection['order'] != list(ORDER):
        raise ValueError('Frozen source roster changed')
    if selection['candidates'][1]['household_ids'] != list(ECO_HOMES):
        raise ValueError('Frozen ECO household list changed')
    if len(ORDER) > limits['source_families']:
        raise ValueError('Source budget exceeded')
    counts = Counter()
    previous_source = 0
    for index, request in enumerate(requests, 1):
        if request['id'] != index or request['source'] not in ORDER:
            raise ValueError('Invalid request identity')
        position = ORDER.index(request['source'])
        if position < previous_source:
            raise ValueError('Frozen source processing order changed')
        previous_source = position
        kind = request['kind']
        if kind not in ('search', 'page'):
            raise ValueError('Only metadata searches and pages permitted')
        if kind == 'page':
            metadata_target(request['target'])
            host = urlparse(request['target']).hostname
            roots = ('bristol.ac.uk', 'bris.ac.uk', 'ucl.ac.uk', 'arxiv.org', 'github.com', 'doi.org') if position == 0 else ('ethz.ch', 'github.com', 'doi.org')
            if not any(host == root or host.endswith('.' + root) for root in roots):
                raise ValueError('Unregistered primary publisher host')
        counts[kind] += 1
    if counts['search'] > limits['search_queries'] or counts['page'] > limits['primary_metadata_page_attempts']:
        raise ValueError('Metadata request budget exceeded')
    return dict(counts)


def validate_evidence(selection, requests, evidence):
    work = validate_requests(selection, requests)
    ids = {r['id']: r for r in requests}
    seen = set()
    for page in evidence['records']:
        request = ids.get(page['request_id'])
        if request is None or page['request_id'] in seen or page['source'] != request['source']:
            raise ValueError('Evidence request provenance differs')
        seen.add(page['request_id'])
        if page['status'] not in ('read', 'search_results', 'unavailable'):
            raise ValueError('Unknown evidence status')
        if request['kind'] == 'page' and page['url'] != request['target']:
            raise ValueError('Evidence URL differs')
        size = len(json.dumps(page, sort_keys=True, ensure_ascii=True).encode())
        if size > selection['budgets']['retained_bytes_per_page']:
            raise ValueError('Per-page evidence budget exceeded')
    if seen != set(ids):
        raise ValueError('Every reserved request needs a receipt, including failures')
    for key in ('dataset_payload_bytes', 'dataset_preview_rows', 'archive_range_requests', 'learning_writes'):
        if evidence[key] != 0 or selection['budgets'][key] != 0:
            raise ValueError('Prohibited payload, preview, archive range or learning')
    if evidence['independent_periods_opened'] or evidence['conditions_relaxed'] or evidence['casas_candidates_added']:
        raise ValueError('Existing seals or conditions changed')
    retained = len((json.dumps(evidence, sort_keys=True, indent=2) + '\n').encode())
    if retained > selection['budgets']['retained_evidence_utf8_bytes']:
        raise ValueError('Aggregate evidence byte budget exceeded')
    return {**work, 'retained_evidence_utf8_bytes': retained}


def reference_check(refs, source, records):
    if not refs or any(r not in records or records[r]['source'] != source
                       or records[r]['status'] != 'read' for r in refs):
        raise ValueError('Successful source-specific primary metadata reference required')


def evaluate(selection, requests, evidence, inputs):
    if selection['fixed']['target_arrivals'] != 70212:
        raise ValueError('Fixed arrival target changed')
    work = validate_evidence(selection, requests, evidence)
    records = {r['request_id']: r for r in evidence['records']}
    if [s['id'] for s in inputs['sources']] != list(ORDER):
        raise ValueError('Assessment sources differ')
    total_homes = 0
    reports = []
    for source in inputs['sources']:
        identity = source['id']
        homes = source['households']
        keys = [h['id'] for h in homes]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate household metadata')
        if identity == 'ECO' and keys != list(ECO_HOMES):
            raise ValueError('All fixed ECO households must remain visible')
        if identity == 'SPHERE' and len(homes) > 100:
            raise ValueError('SPHERE manifest household capacity exceeded')
        total_homes += len(homes)
        for home in homes:
            if set(home['checks']) != set(REQUIRED):
                raise ValueError('Missing household metadata check')
            for check in home['checks'].values():
                if check['status'] not in ('verified', 'incompatible', 'unresolved'):
                    raise ValueError('Invalid tri-state metadata status')
                if check['status'] == 'unresolved':
                    if check['value'] is not None:
                        raise ValueError('Unresolved facts cannot invent values')
                else:
                    reference_check(check['refs'], identity, records)
                    if check['value'] is None:
                        raise ValueError('Resolved household check needs a value')
        native = source['native_kitchen_on']
        if native['status'] not in ('compatible', 'incompatible', 'unresolved'):
            raise ValueError('Invalid native target compatibility')
        if native['status'] != 'unresolved':
            reference_check(native['refs'], identity, records)
        kind = native['raw_kind']
        expected = ('incompatible' if kind in ('history_packet', 'entrance_pir_and_numeric_occupancy')
                    or native['native_kitchen_sensor'] is False or native['native_ON_message'] is False
                    else 'compatible' if kind == 'scalar_binary_notification'
                    and native['native_kitchen_sensor'] is True and native['native_ON_message'] is True
                    else 'unresolved')
        if native['status'] != expected:
            raise ValueError('Target compatibility contradicts documented native format')
        for field in ('density', 'package', 'terms'):
            if source[field]['status'] not in ('verified', 'partial', 'unresolved'):
                raise ValueError('Invalid scoped metadata status')
            if source[field]['status'] != 'unresolved' or source[field]['refs']:
                reference_check(source[field]['refs'], identity, records)
        unresolved = sum(check['status'] == 'unresolved' for h in homes for check in h['checks'].values())
        reports.append({'id': identity,
            'decision': 'exclude_documented_stream_for_fixed_target' if native['status'] == 'incompatible' else 'unresolved_metadata',
            'native_kitchen_on': native, 'household_metadata_entries': len(homes),
            'household_unresolved_checks': unresolved,
            'publisher_household_manifest_found': source['publisher_household_manifest_found'],
            'density': source['density'], 'package': source['package'], 'terms': source['terms'],
            'source_quality_pass': False, 'payload_acquisition_authorized': False})
    if total_homes > selection['budgets']['household_metadata_entries']:
        raise ValueError('Aggregate household metadata budget exceeded')
    return {'sources': reports, 'work': {**work, 'household_metadata_entries': total_homes},
        'decision': 'stop_finite_metadata_audit_no_payload_authorized',
        'target_arrivals': 70212, 'approved_payload_acquisitions': [],
        'snn_assessment': {
            'question': selection['snn_question'],
            'opportunity': 'Native asynchronous sparse input and bounded causal timing could reduce active work and support local delayed updates; this is an implementation hypothesis.',
            'shared_costs': 'Input decoding, packet history/duplicate handling, time availability, state expiry and observer/maintenance work must be charged to every backend.',
            'ann_exclusive_advantage_established': False,
            'snn_usefulness_established': False,
            'required_evidence': 'Same native input/current, actual changed local writes, strongest causal non-spiking history controls and identical scalar replay; independently registered quality and complete CPU/work accounting.'},
        'actual_learning_writes': 0, 'dataset_payload_bytes': 0, 'dataset_preview_rows': 0,
        'archive_range_requests': 0, 'casas_candidates_added': 0,
        'conditions_relaxed': False, 'independent_periods_opened': False}
