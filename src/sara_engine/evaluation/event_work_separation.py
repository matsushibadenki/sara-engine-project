"""External diagnostics for the frozen synthetic execution-boundary contract."""
from collections import Counter
from itertools import groupby
import sys

from sara_engine.evaluation import event_work as reference
from sara_engine.evaluation.event_execution import Executor, ARMS
from sara_engine.evaluation.temporal_information import TemporalInformationObserver
from sara_engine.evaluation.temporal_capability_boundary import BoundedSequenceObserver


MEMORY_SCOPE = {
    'boundary_graph': 'Shared observers/latest packet, executor and mapping shells, backend/queue controls, cells, pending features, timers and returned batches. Shared roots charged in full to each arm; physical roots deduplicated.',
    'between_boundary_sites': [
        {'site': 'TemporalInformationObserver.snapshot', 'owner': 'shared', 'bounded_items': 2, 'unit': 'sorted records and snapshot records', 'peak_bytes': None},
        {'site': 'TemporalInformation.__post_init__', 'owner': 'shared', 'bounded_items': 2, 'unit': 'entries per validation set', 'peak_bytes': None},
        {'site': 'TemporalInformation.readout_routes', 'owner': 'shared', 'bounded_items': 6, 'unit': 'channel pairs in constructed packet', 'peak_bytes': None},
        {'site': 'BoundedSequenceObserver.snapshot / Executor.apply', 'owner': 'shared', 'bounded_items': 6, 'unit': 'prefix references / gap-validation visits', 'peak_bytes': None},
        {'site': 'Neuron.step', 'owner': 'neuron', 'bounded_items': 1, 'unit': 'active branch indices in temporary list', 'peak_bytes': None},
        {'site': 'Backend.step', 'owner': 'each arm', 'bounded_items': 1, 'unit': 'arriving cell per call; scalar arithmetic intermediates not byte-bounded', 'peak_bytes': None},
        {'site': 'ReceiptQueue.advance', 'owner': 'each arm', 'bounded_items': 8, 'unit': 'returned receipt rows per batch, including stale timers', 'peak_bytes': None},
    ],
    'returned_buffers': 'Returned dict/list/row objects and referenced packets remain live until consumed and released by caller; sampled immediately after each call. Retained diagnostic histories are harness-owned.',
    'harness': 'Input tape and operation schedule, frozen reference/Meter/journal, accumulated receipts, snapshots, hashes, reports, graph root assembly and recursive walker seen-set/children scratch. Never charged as zero-cost deployment work.',
    'unknown': 'Unobserved intra-call overlap, allocation capacity changes, Python frames/iterators/generators/arithmetic temporaries, allocator arenas, interpreter/library/native memory and RSS. Structural item ceilings are not a complete transient byte bound.',
    'total_transient_peak_bytes': None,
}


def freeze(value):
    """Copy semantic state only at a caller-controlled diagnostic boundary."""
    if isinstance(value, dict):
        return tuple((key, freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(value))
    if hasattr(value, '__dict__'):
        return freeze(vars(value))
    return value


def snapshot(executor):
    shared = executor.shared
    return {'shared': freeze((shared.ordinal.event_count, shared.ordinal._records,
                              shared.sequence._events, shared.packet)),
            'backends': {arm: freeze(executor.backends[arm].cells) for arm in ARMS},
            'queues': {arm: freeze((executor.queues[arm].clock, executor.queues[arm].pending,
                                   tuple(executor.queues[arm].timers))) for arm in ARMS},
            'closed': executor.closed}


def schedule(events, scenario, horizon):
    if not 1 <= len(events) <= 6 or horizon not in (16, 1000000000):
        raise ValueError('Input/horizon outside fixed budget')
    tape = reference.make_tape(events, scenario)
    if tape[-1][0] > horizon:
        raise ValueError('Horizon precedes external work')
    operations = []
    for time, actions in groupby(tape, key=lambda row: row[0]):
        operations.append(('advance', (time, False)))
        for _, _, index, kind, payload in actions:
            operations.append(('input', (index, time, *payload)) if kind == 'input'
                              else ('notify', (index, time, kind)))
        operations.append(('advance', (time, True)))
    operations.append(('advance', (horizon, True)))
    if len(operations) > 64:
        raise ValueError('Boundary call budget exceeded')
    return tape, operations


class ReferenceExecutor:
    """Frozen instrumented components; diagnostics are not cost measurements."""
    def __init__(self, protocol):
        self.meter = reference.Meter(protocol['budgets']['journal_entries_per_case'])
        # Deliberately reconstruct the prior observer path independently.
        self.shared = type('ReferenceShared', (), {})()
        self.shared.ordinal = TemporalInformationObserver()
        self.shared.sequence = BoundedSequenceObserver()
        self.shared.packet = None
        self.backends = {arm: reference.Backend(arm, self.meter, 0.6, 2) for arm in ARMS}
        self.queues = {arm: reference.Lifecycle(arm, self.meter, protocol['budgets'], 4) for arm in ARMS}
        self.closed = False

    def apply(self, operation):
        before = {arm: len(self.queues[arm].trace) for arm in ARMS}
        kind, args = operation
        if kind == 'advance':
            for queue in self.queues.values():
                queue.advance(*args)
        elif kind == 'notify':
            for queue in self.queues.values():
                queue.notify(*args)
        else:
            index, time, route, gap = args
            self.shared.ordinal.observe(route)
            self.shared.sequence.observe(route, gap)
            info = self.shared.ordinal.snapshot()
            history = self.shared.sequence.snapshot()
            self.shared.packet = (info.readout_routes(), history)
            for arm in ARMS:
                state = self.backends[arm].step(route)
                self.queues[arm].create(index, time, (self.shared.packet, state))
        return {arm: self.queues[arm].trace[before[arm]:] for arm in ARMS}

    def close(self):
        if any(queue.pending or queue.timers for queue in self.queues.values()):
            raise ValueError('Reference did not drain')
        for backend in self.backends.values():
            backend.cells.clear()
        self.shared.ordinal._records.clear()
        self.shared.sequence._events.clear()
        self.shared.packet = None
        self.closed = True
        self.meter.reconcile()


def inventory(executor, batch):
    """Logical occupancy validation; this is not an intra-call byte peak."""
    shared = executor.shared
    channels, history = shared.packet if shared.packet is not None else ((), ())
    counts = {'observer_records': len(shared.ordinal._records), 'prefix_items': len(shared.sequence._events),
              'packet_channels': len(channels), 'packet_history': len(history)}
    limits = {'observer_records': 2, 'prefix_items': 6, 'packet_channels': 6, 'packet_history': 6}
    for arm in ARMS:
        counts.update({arm + '.cells': len(executor.backends[arm].cells),
                       arm + '.pending': len(executor.queues[arm].pending),
                       arm + '.timers': len(executor.queues[arm].timers),
                       arm + '.returned_rows': len(batch[arm])})
        limits.update({arm + '.cells': 2, arm + '.pending': 8, arm + '.timers': 8, arm + '.returned_rows': 8})
        for _, features in executor.queues[arm].pending.values():
            packet, _ = features
            if len(packet[0]) > 6 or len(packet[1]) > 6:
                raise ValueError('Pending packet outside memory scope')
    if any(counts[key] > limit for key, limit in limits.items()):
        raise ValueError('Memory occupancy budget exceeded')
    return counts


def memory_sample(executor, batch):
    """External deduplicated roots; charge the whole shared dispatcher to each arm."""
    # Container shells are counted without recursively including the other arm.
    # Temporary root tuples/seen sets below are measurement scratch, not roots.
    shells = sum(sys.getsizeof(item) for item in (executor, executor.backends, executor.queues, batch))
    shared = (executor.shared, executor.closed, *ARMS)
    own = {arm: (executor.backends[arm], executor.queues[arm], batch[arm]) for arm in ARMS}
    def size(roots):
        seen, total, visits = set(), shells, 0
        for root in roots:
            amount, traversed = reference.owned_size(root, seen)
            total += amount
            visits += traversed
        return total, visits
    measured = {arm: size((*shared, *own[arm])) for arm in ARMS}
    measured['physical'] = size((*shared, *own['neuron'], *own['scalar']))
    shared_bytes, shared_visits = size(shared)
    values = {key: item[0] for key, item in measured.items()}
    if not max(values[arm] for arm in ARMS) <= values['physical'] <= sum(values[arm] for arm in ARMS) - shared_bytes:
        raise ValueError('Shared memory ownership mismatch')
    if max(values.values()) > 1048576:
        raise ValueError('Sampled graph byte budget exceeded')
    return {**values, 'shared': shared_bytes}, shared_visits + sum(item[1] for item in measured.values())


def receipt_work(traces, operations, events):
    """Externally observable service volumes, not new instruction/call totals."""
    prefix = [len({route for route, _ in events[:i + 1]}) for i in range(len(events))]
    shared = {'input.decode': len(events), 'input.fields': 2 * len(events),
              'observer.ordinal': len(events), 'observer.sequence': len(events),
              'observer.snapshot': 2 * len(events), 'observer.projection': len(events),
              'observer.sort_input': sum(prefix), 'observer.channels': 3 * sum(prefix),
              'observer.prefix_copy': len(events) * (len(events) + 1) // 2}
    physical = {'shared': shared}
    for arm in ARMS:
        counts = Counter(row[1] for row in traces[arm])
        physical[arm] = {
            'backend.step': counts['created'], 'backend.spike': sum(row[3][1][2] for row in traces[arm] if row[1] == 'created'),
            'probe.create': counts['created'], 'expiry.push': counts['created'],
            'expiry.pop': counts['expired'] + counts['stale_timer'],
            'expiry.expired': counts['expired'], 'expiry.stale': counts['stale_timer'],
            'feedback.result': sum(op == 'notify' and args[2] == 'result' for op, args in operations),
            'feedback.cancel': sum(op == 'notify' and args[2] == 'cancel' for op, args in operations),
            'feedback.delivered': counts['delivered'], 'feedback.cancelled': counts['cancelled'],
            'feedback.rejected': counts['rejected_result'] + counts['rejected_cancel'],
            'wait.advance': sum(op == 'advance' for op, _ in operations),
        }
    keys = set(shared) | set(physical['neuron'])
    charged = {arm: {key: shared.get(key, 0) + physical[arm].get(key, 0) for key in sorted(keys)} for arm in ARMS}
    return physical, charged


def bare_run(operations):
    """Receipt-only external consumer: no per-call snapshots or graph probes."""
    core = Executor()
    traces = {arm: [] for arm in ARMS}
    for operation in operations:
        batch = core.apply(operation)
        for arm in ARMS:
            traces[arm].extend(batch[arm])
        del batch
    final = snapshot(core)
    core.close()
    return traces, final, snapshot(core)


def run_case(events, scenario, horizon, protocol, golden=None):
    tape, operations = schedule(events, scenario, horizon)
    core, old = Executor(), ReferenceExecutor(protocol)
    traces = {arm: [] for arm in ARMS}
    state_hashes, peaks, occupancy = [], {}, {}
    visits = 0
    # All collectors execute after apply returns; no callback enters the core.
    for operation in operations:
        batch = core.apply(operation)
        expected = old.apply(operation)
        if batch != expected:
            raise ValueError('Boundary receipt mismatch')
        actual_state = snapshot(core)
        if actual_state != snapshot(old):
            raise ValueError('Boundary state mismatch')
        state_hashes.append(reference.fingerprint(actual_state))
        for arm in ARMS:
            traces[arm].extend(batch[arm])
        counts = inventory(core, batch)
        samples, traversed = memory_sample(core, batch)
        visits += traversed
        for key, value in counts.items():
            occupancy[key] = max(occupancy.get(key, 0), value)
        for key, value in samples.items():
            peaks[key] = max(peaks.get(key, 0), value)
        del batch, expected, actual_state
    final = snapshot(core)
    core.close()
    old.close()
    closed = snapshot(core)
    if closed != snapshot(old):
        raise ValueError('Teardown state mismatch')
    empty = {arm: [] for arm in ARMS}
    inventory(core, empty)
    samples, traversed = memory_sample(core, empty)
    visits += traversed
    for key, value in samples.items():
        peaks[key] = max(peaks[key], value)
    for arm in ARMS:
        reference.validate_lifecycle(traces[arm], events, scenario)
    if traces['neuron'] != traces['scalar']:
        raise ValueError('Neuron/scalar receipt mismatch')
    bare_traces, bare_final, bare_closed = bare_run(operations)
    if (bare_traces, bare_final, bare_closed) != (traces, final, closed):
        raise ValueError('Diagnostic removal changed execution')
    physical, charged = receipt_work(traces, operations, events)
    trace_hash = reference.fingerprint(traces['neuron'])
    if golden is not None:
        if trace_hash != golden['semantic_trace_sha256']:
            raise ValueError('Prior saved receipt trace mismatch')
        for owner, metrics in physical.items():
            if any(value != golden['physical'][owner][key] for key, value in metrics.items()):
                raise ValueError('Prior service volume mismatch')
    for arm in ARMS:
        if any(value != old.meter.totals[arm][key] for key, value in physical[arm].items()):
            raise ValueError('Reference service volume mismatch')
    if 4 * (len(operations) + 1) > protocol['budgets']['diagnostic_samples_per_case']:
        raise ValueError('External diagnostic sample budget exceeded')
    nonclock_final = {**final, 'queues': {arm: final['queues'][arm][1:] for arm in ARMS}}
    return {'scenario': scenario, 'horizon': horizon, 'boundary_calls': len(operations),
            'state_trace_sha256': reference.fingerprint(state_hashes),
            'semantic_trace_sha256': trace_hash, 'final_state_sha256': reference.fingerprint(final),
            'nonclock_final_state_sha256': reference.fingerprint(nonclock_final),
            'teardown_state_sha256': reference.fingerprint(closed),
            'sampled_owned_bytes': peaks, 'peak_occupancy': occupancy,
            'service_volumes': physical, 'charged_service_volumes': charged,
            'diagnostics': {'boundary_state_comparisons': len(operations) + 1,
                            'graph_samples': 4 * (len(operations) + 1), 'graph_visits': visits,
                            'reference_journal_rows': len(old.meter.journal), 'tape_rows': len(tape)},
            'bare_matches_diagnostic': True, 'reference_matches_every_boundary': True,
            'prior_receipt_hash_matched': golden is not None, 'gates_passed': True}


def evaluate(protocol, source, previous):
    if (protocol['arms'] != list(ARMS) or protocol['scenarios'] != list(reference.SCENARIOS)
            or protocol['horizons'] != [16, 1000000000] or protocol['backend']['current'] != 0.6
            or protocol['lifecycle']['ttl'] != 4 or protocol['input']['route_pair'] != [0, 1]
            or protocol['input']['families'] != ['first_order', 'middle_order', 'gap_order']
            or protocol['input']['members'] != [0, 1] or protocol['budgets']['cases'] != 48
            or protocol['budgets']['passes_per_case'] != 3):
        raise ValueError('Fixed separation contract changed')
    goldens = {(row['family'], row['member'], row['scenario'], row['horizon']): row for row in previous['cases']}
    if len(goldens) != 48 or len(previous['cases']) != 48:
        raise ValueError('Prior finite case roster changed')
    cases = []
    for family in protocol['input']['families']:
        for member in protocol['input']['members']:
            routes, gaps = source['templates'][family][member], source['gap_templates'][family][member]
            if len(routes) != len(gaps):
                raise ValueError('Input template shape changed')
            events = tuple(zip(routes, gaps))
            for scenario in protocol['scenarios']:
                pair = []
                for horizon in protocol['horizons']:
                    old = goldens[(family, member, scenario, horizon)]
                    if old['input_sha256'] != reference.fingerprint(events):
                        raise ValueError('Prior input identity changed')
                    row = run_case(events, scenario, horizon, protocol, old)
                    row.update(family=family, member=member, input_sha256=reference.fingerprint(events))
                    pair.append(row)
                    cases.append(row)
                for key in ('semantic_trace_sha256', 'nonclock_final_state_sha256', 'boundary_calls', 'service_volumes', 'peak_occupancy'):
                    if pair[0][key] != pair[1][key]:
                        raise ValueError('Idle extension changed semantic work')
    return {'schema': 'sara-event-work-separation-result-v1', 'case_count': len(cases), 'passes_per_case': 3,
            'cases': cases, 'all_separation_gates_passed': True, 'memory_scope': MEMORY_SCOPE,
            'work_scope': 'External service volumes preserve the old typed counts. The old journal is reference-harness work. New validation, wrapper, return-buffer and interpreter costs are not measured CPU instructions or a complete new operation vector.',
            'actual_learning_writes': 0, 'predictions_scored': 0, 'raw_or_learned_payload_reads': 0,
            'cpu_seconds': None, 'energy_joules': None, 'total_transient_peak_bytes': None,
            'decision': 'measurement_boundary_verified_no_snn_advantage_established',
            'snn_resource_advantage_established': False, 'independent_usefulness_established': False,
            'runtime_promotion_authorized': False, 'test_opening_authorized': False}
