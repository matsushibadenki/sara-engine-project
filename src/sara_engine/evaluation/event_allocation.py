"""Isolated Python allocation diagnostics, never deployment peak/energy estimates."""
from collections import Counter, deque
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc

from sara_engine.evaluation.event_execution import Executor, ARMS
from sara_engine.evaluation.event_allocation_execution import SingleArmExecutor
from sara_engine.evaluation.event_work import fingerprint, validate_lifecycle
from sara_engine.evaluation.event_work_separation import schedule, snapshot, freeze
from sara_engine.utils.project_paths import project_path


EXECUTION_FILES = {
    'src/sara_engine/evaluation/event_execution.py',
    'src/sara_engine/evaluation/event_allocation_execution.py',
    'src/sara_engine/evaluation/temporal_information.py',
    'src/sara_engine/evaluation/temporal_capability_boundary.py',
    'src/sara_engine/neuro/neuron.py',
}


def allocation_site(filename, line):
    path = Path(filename)
    try:
        name = str(path.relative_to(project_path()))
    except ValueError:
        name = path.name
    owner = 'execution' if name in EXECUTION_FILES else 'measurement' if (
        name == 'src/sara_engine/evaluation/event_allocation.py' or name == 'tracemalloc.py') else 'other'
    return owner, name + ':' + str(line)


def measure(call, args=()):
    """Trace newly allocated blocks for one call; inspect graphs only afterward."""
    if tracemalloc.is_tracing() or sys.gettrace() is not None or sys.getprofile() is not None:
        raise ValueError('Allocation diagnostic requires an untraced process')
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        tracemalloc.start(1)
        value = call(*args)
        current, peak = tracemalloc.get_traced_memory()
        # The peak query ends the target window. Snapshot/metadata work below
        # is measurement work and cannot retroactively change that reading.
        live = tracemalloc.take_snapshot()
        metadata = tracemalloc.get_tracemalloc_memory()
    finally:
        tracemalloc.stop()
        if enabled:
            gc.enable()
    sites = {}
    owners = {owner: {'bytes': 0, 'blocks': 0} for owner in ('execution', 'measurement', 'other')}
    for trace in live.traces:
        frame = trace.traceback[-1]
        owner, site = allocation_site(frame.filename, frame.lineno)
        item = sites.setdefault(site, {'owner': owner, 'bytes': 0, 'blocks': 0})
        item['bytes'] += trace.size
        item['blocks'] += 1
        owners[owner]['bytes'] += trace.size
        owners[owner]['blocks'] += 1
    if not 0 <= current <= peak <= 1048576:
        raise ValueError('Traced call peak outside budget')
    if sum(item['bytes'] for item in sites.values()) != sum(item['bytes'] for item in owners.values()):
        raise ValueError('Live allocation attribution does not conserve bytes')
    return value, {'new_live_at_return_bytes': current, 'new_peak_bytes': peak,
                   'peak_excess_over_return_bytes': peak - current,
                   'tracer_metadata_after_snapshot_bytes': metadata,
                   'post_query_live_by_owner': owners, 'post_query_live_sites': sites}


def graph_objects(root):
    """External shallow-size inventory keyed by identity, excluding code/RSS."""
    objects, stack = {}, [root]
    while stack:
        value = stack.pop()
        identity = id(value)
        if identity in objects:
            continue
        objects[identity] = sys.getsizeof(value)
        if isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, (tuple, list, deque, set, frozenset)):
            stack.extend(value)
        elif hasattr(value, '__dict__'):
            stack.append(value.__dict__)
        else:
            stack.extend(getattr(value, slot) for slot in getattr(type(value), '__slots__', ()) if hasattr(value, slot))
    return objects


def ownership(executor, output):
    if tracemalloc.is_tracing():
        raise ValueError('Graph analysis must occur after tracing stops')
    state = graph_objects(executor)
    returned = {} if output is None else graph_objects(output)
    common = state.keys() & returned.keys()
    exclusive_state = sum(size for key, size in state.items() if key not in common)
    exclusive_output = sum(size for key, size in returned.items() if key not in common)
    shared = sum(state[key] for key in common)
    union = sum({**state, **returned}.values())
    if union != exclusive_state + exclusive_output + shared:
        raise ValueError('Owned graph partition does not conserve unique bytes')
    return {'executor_only_bytes': exclusive_state, 'output_only_bytes': exclusive_output,
            'shared_reachable_bytes': shared, 'union_bytes': union,
            'executor_graph_bytes': sum(state.values()), 'output_graph_bytes': sum(returned.values())}


def project(state, mode):
    if mode == 'paired':
        return state
    return {**state, 'backends': {mode: state['backends'][mode]}, 'queues': {mode: state['queues'][mode]}}


def state_of(core):
    if isinstance(core, Executor):
        return snapshot(core)
    arm = core.arm
    shared = core.shared
    return {'shared': freeze((shared.ordinal.event_count, shared.ordinal._records,
                              shared.sequence._events, shared.packet)),
            'backends': {arm: freeze(core.backends[arm].cells)},
            'queues': {arm: freeze((core.queues[arm].clock, core.queues[arm].pending,
                                   tuple(core.queues[arm].timers)))}, 'closed': core.closed}


def oracle(operations, golden):
    """Identical untraced paired prepass in every worker; retain outside window."""
    core = Executor()
    initial = snapshot(core)
    states, batches, traces = [], [], {arm: [] for arm in ARMS}
    for operation in operations:
        batch = core.apply(operation)
        batches.append(batch)
        for arm in ARMS:
            traces[arm].extend(batch[arm])
        states.append(snapshot(core))
    final = states[-1]
    core.close()
    closed = snapshot(core)
    if (fingerprint([fingerprint(state) for state in states]) != golden['state_trace_sha256']
            or fingerprint(final) != golden['final_state_sha256']
            or fingerprint(closed) != golden['teardown_state_sha256']
            or any(fingerprint(rows) != golden['semantic_trace_sha256'] for rows in traces.values())):
        raise ValueError('Frozen paired reference identity changed')
    return initial, states, batches, closed


def run_case(events, golden, mode):
    if mode not in ('paired', *ARMS):
        raise ValueError('Unknown diagnostic mode')
    _, operations = schedule(events, golden['scenario'], golden['horizon'])
    initial, expected_states, expected_batches, expected_closed = oracle(operations, golden)
    core, allocation = measure(Executor) if mode == 'paired' else measure(SingleArmExecutor, (mode,))
    if state_of(core) != project(initial, mode):
        raise ValueError('Initialization changed semantic state')
    records = [{'stage': 'initialize', 'allocation': allocation, 'ownership': ownership(core, None)}]
    traces = {arm: [] for arm in core.backends}
    state_hashes = []
    for operation, expected, expected_batch in zip(operations, expected_states, expected_batches):
        batch, allocation = measure(core.apply, (operation,))
        if state_of(core) != project(expected, mode):
            raise ValueError('Traced boundary state differs from paired reference')
        if batch != {arm: expected_batch[arm] for arm in traces}:
            raise ValueError('Traced receipts differ from paired reference')
        if any(len(rows) > 8 for rows in batch.values()):
            raise ValueError('Returned buffer budget exceeded')
        records.append({'stage': operation[0], 'allocation': allocation, 'ownership': ownership(core, batch)})
        state_hashes.append(fingerprint(state_of(core)))
        for arm in traces:
            traces[arm].extend(batch[arm])
        del batch, allocation
    final = state_of(core)
    _, allocation = measure(core.close)
    if state_of(core) != project(expected_closed, mode):
        raise ValueError('Traced teardown changed semantic state')
    records.append({'stage': 'close', 'allocation': allocation, 'ownership': ownership(core, None)})
    if len(records) > 66:
        raise ValueError('Recorded call budget exceeded')
    service = {}
    for arm, rows in traces.items():
        validate_lifecycle(rows, events, golden['scenario'])
        counts = Counter(row[1] for row in rows)
        if fingerprint(rows) != golden['semantic_trace_sha256']:
            raise ValueError('Previous receipt hash differs')
        service[arm] = {key: counts[key] for key in ('created', 'delivered', 'cancelled', 'expired', 'stale_timer', 'rejected_result')}
    return {'mode': mode, 'family': golden['family'], 'member': golden['member'],
            'scenario': golden['scenario'], 'horizon': golden['horizon'], 'input_sha256': fingerprint(events),
            'state_trace_sha256': fingerprint(state_hashes), 'receipt_sha256': golden['semantic_trace_sha256'],
            'final_state_sha256': fingerprint(final), 'teardown_state_sha256': fingerprint(state_of(core)),
            'shared_input_calls': len(events), 'service_receipts': service, 'recorded_calls': len(records),
            'records': records, 'all_semantic_gates_passed': True}


def empty():
    return None


def ephemeral(size):
    value = bytearray(size)
    return len(value)


def retained(size):
    return bytearray(size)


def preexisting(value):
    return value


def witnesses():
    size = 65536
    existing = bytearray(size)
    specs = [('empty', empty, ()), ('ephemeral', ephemeral, (size,)),
             ('retained', retained, (size,)), ('preexisting', preexisting, (existing,))]
    rows = {}
    for name, call, args in specs:
        value, metrics = measure(call, args)
        rows[name] = metrics
        del value
    if (rows['ephemeral']['peak_excess_over_return_bytes'] < size
            or rows['retained']['new_live_at_return_bytes'] < size
            or rows['preexisting']['new_live_at_return_bytes'] >= size):
        raise ValueError('Known allocation witness failed')
    return {'buffer_bytes': size, 'passed': True, 'measurements': rows}


def cases(protocol, source, previous):
    if (protocol['input']['route_pair'] != [0, 1] or protocol['horizons'] != [16, 1000000000]
            or protocol['input']['families'] != ['first_order', 'middle_order', 'gap_order']
            or protocol['input']['members'] != [0, 1] or protocol['scenarios'] != ['on_time', 'late', 'missing', 'mixed']
            or protocol['lifecycle']['ttl'] != 4 or protocol['backend']['current'] != 0.6
            or protocol['budgets']['worker_modes'] != ['paired', 'neuron', 'scalar']):
        raise ValueError('Fixed allocation contract changed')
    roster = []
    for golden in previous['cases']:
        family, member = golden['family'], golden['member']
        routes, gaps = source['templates'][family][member], source['gap_templates'][family][member]
        if len(routes) != len(gaps):
            raise ValueError('Template length changed')
        events = tuple(zip(routes, gaps))
        if fingerprint(events) != golden['input_sha256']:
            raise ValueError('Synthetic input identity changed')
        roster.append((events, golden))
    expected = {(f, m, s, h) for f in protocol['input']['families'] for m in (0, 1)
                for s in protocol['scenarios'] for h in protocol['horizons']}
    if len(roster) != 48 or {(g['family'], g['member'], g['scenario'], g['horizon']) for _, g in roster} != expected:
        raise ValueError('Finite case roster changed')
    return roster


def evaluate(protocol, source, previous):
    roster = cases(protocol, source, previous)
    calibration = witnesses()
    workers = []
    started = time.monotonic()
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': project_path('src') + os.pathsep + project_path()}
    for case_index, (_, golden) in enumerate(roster):
        for mode in protocol['budgets']['worker_modes']:
            remaining = protocol['budgets']['execution_timeout_seconds'] - (time.monotonic() - started)
            if remaining <= 0:
                raise ValueError('Total diagnostic time budget exceeded')
            completed = subprocess.run([sys.executable, project_path('scripts/eval/event_allocation.py'),
                                        'worker', '--case', str(case_index), '--arm', mode],
                                       cwd=project_path(), env=env, capture_output=True, text=True,
                                       timeout=min(remaining, protocol['budgets']['worker_timeout_seconds']), check=True)
            if time.monotonic() - started > protocol['budgets']['execution_timeout_seconds']:
                raise ValueError('Total diagnostic time budget exceeded')
            if len(completed.stdout.encode()) > protocol['budgets']['worker_stdout_bytes']:
                raise ValueError('Worker output budget exceeded')
            row = json.loads(completed.stdout)
            if row['mode'] != mode or row['input_sha256'] != golden['input_sha256'] or row['receipt_sha256'] != golden['semantic_trace_sha256']:
                raise ValueError('Worker identity differs from fixed roster')
            workers.append(row)
        if len(workers) % 24 == 0:
            print('Allocation diagnostic workers completed: ' + str(len(workers)) + '/144', flush=True)
    if len(workers) != protocol['budgets']['workers_per_execution'] or len(workers) != 144:
        raise ValueError('Worker budget changed')
    return {'schema': 'sara-event-allocation-result-v1', 'case_count': 48, 'worker_count': len(workers),
            'workers': workers, 'witnesses': calibration, 'all_allocation_gates_passed': True,
            'actual_learning_writes': 0, 'predictions_scored': 0, 'raw_or_learned_payload_reads': 0,
            'cpu_seconds': None, 'energy_joules': None, 'rss_bytes': None, 'total_transient_peak_bytes': None,
            'decision': 'isolated_python_allocation_scope_verified_no_snn_advantage_established',
            'snn_resource_advantage_established': False, 'independent_usefulness_established': False,
            'runtime_promotion_authorized': False, 'test_opening_authorized': False}


def semantic_result(result):
    """Exact replay contract excludes variable traced bytes, not owned graphs."""
    fixed = {key: value for key, value in result.items() if key not in ('workers', 'witnesses')}
    fixed['witnesses'] = {key: value for key, value in result['witnesses'].items() if key != 'measurements'}
    fixed['workers'] = [{**{key: value for key, value in worker.items() if key != 'records'},
                         'records': [{key: value for key, value in row.items() if key != 'allocation'}
                                     for row in worker['records']]} for worker in result['workers']]
    return fixed
