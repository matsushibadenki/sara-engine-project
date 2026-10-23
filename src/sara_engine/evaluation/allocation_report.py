"""Bounded report representation only: no executor, tracer or worker imports."""
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.utils.project_paths import ensure_allowed_output_path


LIMIT = 8388608
WORKERS, CALLS, SITES, SITE_BYTES, INCIDENCES = 144, 66, 1024, 256, 75000
U64 = 18446744073709551615
MODES = ('paired', 'neuron', 'scalar')
OWNERS = ('execution', 'measurement', 'other')
STAGES = ('initialize', 'input', 'advance', 'notify', 'close')
SCALARS = ('new_live_at_return_bytes', 'new_peak_bytes', 'peak_excess_over_return_bytes',
           'tracer_metadata_after_snapshot_bytes')
GRAPH = ('executor_only_bytes', 'output_only_bytes', 'shared_reachable_bytes',
         'union_bytes', 'executor_graph_bytes', 'output_graph_bytes')
COUNTS = ('created', 'delivered', 'cancelled', 'expired', 'stale_timer', 'rejected_result')
HASHES = ('input_sha256', 'state_trace_sha256', 'receipt_sha256', 'final_state_sha256', 'teardown_state_sha256')
COLUMNS = {'record': ['stage', *SCALARS, *[o + '.' + k for o in OWNERS for k in ('bytes', 'blocks')], *GRAPH, 'sites'],
           'worker': ['case', 'mode', 'hashes', 'shared_input_calls', 'service_receipts', 'records'],
           'site': ['owner', 'name'], 'incidence': ['site', 'bytes', 'blocks'],
           'hashes': list(HASHES), 'counts': list(COUNTS), 'owners': list(OWNERS), 'stages': list(STAGES), 'modes': list(MODES)}
SCHEMA = 'sara-bounded-allocation-report-v1'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def exact_keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('Unexpected or missing report fields')


def integer(value, maximum=U64):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('Integer outside report domain')
    return value


def sequence(value, size=None):
    if type(value) is not list or (size is not None and len(value) != size):
        raise ValueError('Invalid report array shape')
    return value


def digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid report hash')
    return value


def metadata(value):
    exact_keys(value, ('synthetic', 'registration_sha256', 'execution_sha256', 'prior_failure_sha256', 'runtime'))
    if type(value['synthetic']) is not bool:
        raise ValueError('Synthetic report marker required')
    for key in ('registration_sha256', 'execution_sha256', 'prior_failure_sha256'):
        digest(value[key])
    for item, cap in zip(sequence(value['runtime'], 4), (32, 32, 64, 8)):
        if type(item) is not str or len(canonical(item)) - 1 > cap:
            raise ValueError('Runtime string outside report domain')


def roster_check(roster):
    sequence(roster, 48)
    expected = [(f, m, s, h) for f in ('first_order', 'middle_order', 'gap_order') for m in (0, 1)
                for s in ('on_time', 'late', 'missing', 'mixed') for h in (16, 1000000000)]
    for row, identity in zip(roster, expected):
        exact_keys(row, ('family', 'member', 'scenario', 'horizon', 'input_sha256', 'input_events'))
        if tuple(row[k] for k in ('family', 'member', 'scenario', 'horizon')) != identity:
            raise ValueError('Fixed case roster changed')
        if type(row['member']) is not int or type(row['horizon']) is not int:
            raise ValueError('Invalid case identity type')
        digest(row['input_sha256'])
        if row['input_events'] != (6 if row['family'] == 'middle_order' else 4) or type(row['input_events']) is not int:
            raise ValueError('Fixed case input count changed')


def expected_counts(case):
    n, scenario = case['input_events'], case['scenario']
    cancelled = sum(i % 3 == 1 for i in range(n)) if scenario == 'mixed' else 0
    delivered = n if scenario == 'on_time' else n - cancelled if scenario == 'mixed' else 0
    expired = n - delivered - cancelled
    rejected = n if scenario == 'late' else 1 + sum(i % 3 != 0 for i in range(n)) if scenario == 'mixed' else 0
    return [n, delivered, cancelled, expired, delivered + cancelled, rejected]


def _envelope(meta, roster, workers, witnesses, sites):
    return {'schema': SCHEMA, 'metadata': meta, 'roster': roster, 'columns': COLUMNS,
            'workers': workers, 'witnesses': witnesses, 'sites': sites,
            'unmeasured': {'cpu_seconds': None, 'energy_joules': None, 'rss_bytes': None, 'total_transient_peak_bytes': None}}


def encode(payload):
    exact_keys(payload, ('metadata', 'roster', 'workers', 'witnesses'))
    meta, roster = payload['metadata'], payload['roster']
    metadata(meta)
    roster_check(roster)
    registry, indices, incidence_count = [], {}, 0

    def metric(value):
        nonlocal incidence_count
        exact_keys(value, (*SCALARS, 'post_query_live_by_owner', 'post_query_live_sites'))
        numbers = [integer(value[k]) for k in SCALARS]
        if numbers[0] > numbers[1] or numbers[2] != numbers[1] - numbers[0] or numbers[1] > 1048576:
            raise ValueError('Invalid allocation scalar conservation')
        exact_keys(value['post_query_live_by_owner'], OWNERS)
        totals = {o: [0, 0] for o in OWNERS}
        for owner in OWNERS:
            entry = value['post_query_live_by_owner'][owner]
            exact_keys(entry, ('bytes', 'blocks'))
            numbers.extend(integer(entry[k]) for k in ('bytes', 'blocks'))
        sites = value['post_query_live_sites']
        if type(sites) is not dict or any(type(name) is not str for name in sites):
            raise ValueError('Invalid live site table')
        packed_sites = []
        for name, item in sorted(sites.items()):
            if type(name) is not str or len(canonical(name)) - 1 > SITE_BYTES:
                raise ValueError('Escaped site name outside report domain')
            exact_keys(item, ('owner', 'bytes', 'blocks'))
            if item['owner'] not in OWNERS:
                raise ValueError('Unknown allocation owner')
            owner = item['owner']
            amount, blocks = integer(item['bytes']), integer(item['blocks'])
            totals[owner][0] += amount
            totals[owner][1] += blocks
            key = (owner, name)
            if key not in indices:
                if len(registry) >= SITES:
                    raise ValueError('Site registry capacity exceeded')
                indices[key] = len(registry)
                registry.append([OWNERS.index(owner), name])
            packed_sites.append([indices[key], amount, blocks])
        if any(totals[o] != [value['post_query_live_by_owner'][o][k] for k in ('bytes', 'blocks')] for o in OWNERS):
            raise ValueError('Live site ownership conservation failed')
        incidence_count += len(packed_sites)
        if incidence_count > INCIDENCES:
            raise ValueError('Site incidence capacity exceeded')
        return numbers, packed_sites

    workers = []
    for index, worker in enumerate(sequence(payload['workers'], WORKERS)):
        exact_keys(worker, ('mode', 'family', 'member', 'scenario', 'horizon', *HASHES, 'shared_input_calls',
                            'service_receipts', 'recorded_calls', 'records', 'all_semantic_gates_passed'))
        case_id, mode_id = divmod(index, 3)
        case, mode = roster[case_id], MODES[mode_id]
        integer(worker['member'], 1)
        integer(worker['horizon'])
        if worker['mode'] != mode or any(worker[k] != case[k] for k in ('family', 'member', 'scenario', 'horizon', 'input_sha256')):
            raise ValueError('Duplicate or incorrect worker identity')
        hashes = [digest(worker[k]) for k in HASHES]
        if worker['all_semantic_gates_passed'] is not True or integer(worker['shared_input_calls']) != case['input_events']:
            raise ValueError('Worker semantic/shared-input gate failed')
        arms = ('neuron', 'scalar') if mode == 'paired' else (mode,)
        exact_keys(worker['service_receipts'], arms)
        services = []
        for arm in arms:
            exact_keys(worker['service_receipts'][arm], COUNTS)
            counts = [integer(worker['service_receipts'][arm][k]) for k in COUNTS]
            if counts != expected_counts(case):
                raise ValueError('Fixed service receipt counts changed')
            services.append(counts)
        records = sequence(worker['records'])
        if not 2 <= len(records) <= CALLS or integer(worker['recorded_calls']) != len(records):
            raise ValueError('Recorded call capacity/count invalid')
        rows = []
        for record in records:
            exact_keys(record, ('stage', 'allocation', 'ownership'))
            if record['stage'] not in STAGES:
                raise ValueError('Invalid call stage')
            values, site_rows = metric(record['allocation'])
            graph = record['ownership']
            exact_keys(graph, GRAPH)
            a, b, shared, union, state, output = [integer(graph[k]) for k in GRAPH]
            if (union, state, output) != (a + b + shared, a + shared, b + shared):
                raise ValueError('Graph ownership partition failed')
            rows.append([STAGES.index(record['stage']), *values, a, b, shared, union, state, output, site_rows])
        if records[0]['stage'] != 'initialize' or records[-1]['stage'] != 'close' or any(r['stage'] in ('initialize', 'close') for r in records[1:-1]):
            raise ValueError('Incomplete call lifecycle')
        if sum(r['stage'] == 'input' for r in records) != case['input_events']:
            raise ValueError('Shared observer input calls missing')
        workers.append([case_id, mode_id, hashes, worker['shared_input_calls'], services, rows])
    witness = payload['witnesses']
    exact_keys(witness, ('buffer_bytes', 'passed', 'measurements'))
    if witness['buffer_bytes'] != 65536 or type(witness['buffer_bytes']) is not int or witness['passed'] is not True:
        raise ValueError('Invalid allocation witness metadata')
    names = ('empty', 'ephemeral', 'retained', 'preexisting')
    exact_keys(witness['measurements'], names)
    packed_witness = []
    for name in names:
        values, site_rows = metric(witness['measurements'][name])
        packed_witness.append([*values, site_rows])
    measures = witness['measurements']
    if (measures['ephemeral'][SCALARS[2]] < 65536 or measures['retained'][SCALARS[0]] < 65536
            or measures['preexisting'][SCALARS[0]] >= 65536):
        raise ValueError('Allocation witness gate failed')
    raw = canonical(_envelope(meta, roster, workers, packed_witness, registry))
    if len(raw) > LIMIT:
        raise ValueError('Bounded report exceeds fixed byte budget')
    return raw


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON field')
        value[key] = item
    return value


def decode(raw):
    if type(raw) is not bytes or len(raw) > LIMIT:
        raise ValueError('Saved report outside byte domain')
    try:
        report = json.loads(raw, object_pairs_hook=_unique)
        exact_keys(report, ('schema', 'metadata', 'roster', 'columns', 'workers', 'witnesses', 'sites', 'unmeasured'))
        if report['schema'] != SCHEMA or report['columns'] != COLUMNS or report['unmeasured'] != _envelope({}, [], [], [], [])['unmeasured']:
            raise ValueError('Report schema/unknown scope changed')
        roster_check(report['roster'])
        registry = sequence(report['sites'])
        if len(registry) > SITES:
            raise ValueError('Site registry capacity exceeded')
        for owner, name in [sequence(item, 2) for item in registry]:
            integer(owner, 2)
            if type(name) is not str or len(canonical(name)) - 1 > SITE_BYTES:
                raise ValueError('Invalid registry name')
        if len({tuple(item) for item in registry}) != len(registry):
            raise ValueError('Duplicate registry identity')

        def metric(numbers, entries):
            sequence(numbers, 10)
            sites = {}
            for entry in sequence(entries):
                site_id, amount, blocks = sequence(entry, 3)
                integer(site_id)
                if site_id >= len(registry):
                    raise ValueError('Invalid site index')
                owner_id, name = registry[site_id]
                if name in sites:
                    raise ValueError('Duplicate site incidence')
                sites[name] = {'owner': OWNERS[owner_id], 'bytes': amount, 'blocks': blocks}
            return {**dict(zip(SCALARS, numbers[:4])),
                    'post_query_live_by_owner': {o: dict(zip(('bytes', 'blocks'), numbers[4 + 2*i:6 + 2*i])) for i, o in enumerate(OWNERS)},
                    'post_query_live_sites': sites}

        workers = []
        for packed in sequence(report['workers'], WORKERS):
            case_id, mode_id, hashes, shared_calls, services, rows = sequence(packed, 6)
            integer(case_id, 47)
            integer(mode_id, 2)
            case, mode = report['roster'][case_id], MODES[mode_id]
            sequence(hashes, 5)
            arms = ('neuron', 'scalar') if mode == 'paired' else (mode,)
            sequence(services, len(arms))
            records = []
            for row in sequence(rows):
                sequence(row, 18)
                integer(row[0], 4)
                records.append({'stage': STAGES[row[0]], 'allocation': metric(row[1:11], row[-1]),
                                'ownership': dict(zip(GRAPH, row[11:17]))})
            workers.append({**{k: case[k] for k in ('family', 'member', 'scenario', 'horizon')}, 'mode': mode,
                            **dict(zip(HASHES, hashes)), 'shared_input_calls': shared_calls,
                            'service_receipts': {arm: dict(zip(COUNTS, sequence(values, 6))) for arm, values in zip(arms, services)},
                            'recorded_calls': len(records), 'records': records, 'all_semantic_gates_passed': True})
        witnesses = {}
        for name, row in zip(('empty', 'ephemeral', 'retained', 'preexisting'), sequence(report['witnesses'], 4)):
            sequence(row, 11)
            witnesses[name] = metric(row[:10], row[-1])
        payload = {'metadata': report['metadata'], 'roster': report['roster'], 'workers': workers,
                   'witnesses': {'buffer_bytes': 65536, 'passed': True, 'measurements': witnesses}}
        if encode(payload) != raw:
            raise ValueError('Saved report is not canonical or complete')
        return payload
    except (KeyError, IndexError, TypeError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('Malformed saved allocation report') from error


def saved_replication_gate(path, expected_sha256, registration_sha256, execution_sha256, launch):
    """Validate a complete retained result before even calling a launcher."""
    digest(expected_sha256)
    digest(registration_sha256)
    digest(execution_sha256)
    file = Path(ensure_allowed_output_path(str(path)))
    with file.open('rb') as stream:
        raw = stream.read(LIMIT + 1)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Saved result hash mismatch')
    result = decode(raw)
    meta = result['metadata']
    if meta['synthetic'] or meta['registration_sha256'] != registration_sha256 or meta['execution_sha256'] != execution_sha256:
        raise ValueError('Saved result is synthetic or bound to another execution')
    return launch(result)


def byte_proof(roster):
    """A dominating syntax witness relaxes semantic constraints, never executes AI."""
    roster_check(roster)
    meta = {'synthetic': False, 'runtime': ['X'*30, 'X'*30, 'X'*62, 'X'*6],
            **{k: 'f'*64 for k in ('registration_sha256', 'execution_sha256', 'prior_failure_sha256')}}
    base = len(canonical(_envelope(meta, roster, [], [], [])))
    record = [4, *([U64]*16), []]
    worker = [47, 2, ['f'*64]*5, U64, [[U64]*6, [U64]*6], []]
    witness = [*([U64]*10), []]
    row_bytes = len(canonical(record)) - 1
    worker_bytes = len(canonical(worker)) - 1
    witness_bytes = len(canonical(witness)) - 1
    incidence_bytes = len(canonical([1023, U64, U64])) - 1
    site_bytes = 4 + SITE_BYTES  # [owner,name] excluding the following comma.
    bound = base + WORKERS*(worker_bytes + 1 + CALLS*(row_bytes + 1)) + 4*(witness_bytes + 1) + SITES*(site_bytes + 1) + INCIDENCES*(incidence_bytes + 1)
    # Separate construction checks separator/escaping arithmetic, not feasibility
    # of impossible simultaneous numeric maxima or duplicated semantic identities.
    workers = []
    remaining = INCIDENCES
    for _ in range(WORKERS):
        rows = []
        for _ in range(CALLS):
            count = min(8, remaining)
            rows.append([4, *([U64]*16), [[1023, U64, U64] for _ in range(count)]])
            remaining -= count
        workers.append([*worker[:-1], rows])
    if remaining:
        raise ValueError('Saturation incidence construction incomplete')
    sites = [[2, 'X'*250 + str(i).zfill(4)] for i in range(SITES)]
    raw = canonical(_envelope(meta, roster, workers, [witness]*4, sites))
    if not len(raw) <= bound < LIMIT:
        raise ValueError('Serialized report bound does not fit fixed budget')
    return {'proved_upper_bound_bytes': bound, 'saturation_witness_bytes': len(raw),
            'saturation_sha256': sha256(raw).hexdigest(), 'budget_bytes': LIMIT,
            'remaining_budget_bytes': LIMIT - bound, 'conditional_on_domain_validation': True,
            'saturation_is_a_relaxed_syntax_witness_not_measurement': True}
