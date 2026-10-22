"""Input-only event lifecycle accounting; no learner, labels or score interface."""
from collections import Counter, deque
from hashlib import sha256
from itertools import groupby
import json
import math
import sys

from sara_engine.evaluation.temporal_information import TemporalInformationObserver
from sara_engine.evaluation.temporal_capability_boundary import BoundedSequenceObserver
from sara_engine.neuro.neuron import Neuron


ARMS = ('neuron', 'scalar')
SCENARIOS = ('on_time', 'late', 'missing', 'mixed')
OWNERS = ('shared', *ARMS, 'harness')
UNITS = {
    'input.decode': 'calls', 'input.fields': 'fields',
    'observer.initialize': 'calls', 'observer.ordinal': 'calls', 'observer.sequence': 'calls',
    'observer.snapshot': 'calls', 'observer.projection': 'calls',
    'observer.sort_input': 'records', 'observer.prefix_copy': 'items',
    'observer.channels': 'channels',
    'backend.initialize': 'calls', 'backend.lookup': 'calls', 'backend.allocate': 'records',
    'backend.input': 'events', 'backend.step': 'calls',
    'backend.transform': 'evaluations', 'backend.refractory': 'steps', 'backend.spike': 'spikes',
    'probe.create': 'calls', 'probe.feature_reference': 'references',
    'feedback.initialize': 'calls', 'feedback.lookup': 'calls', 'feedback.result': 'notifications',
    'feedback.cancel': 'requests', 'feedback.delivered': 'receipts',
    'feedback.cancelled': 'receipts', 'feedback.rejected': 'notifications',
    'feedback.pending_insert': 'records', 'feedback.pending_delete': 'records',
    'expiry.push': 'timers', 'expiry.head_check': 'checks', 'expiry.pop': 'timers',
    'expiry.expired': 'receipts', 'expiry.stale': 'timers',
    'wait.advance': 'calls', 'wait.timer_dispatch': 'timers', 'wait.elapsed': 'logical_time_units',
    'teardown.clear': 'containers', 'teardown.released': 'records', 'teardown.references': 'references',
    'harness.tape_sort': 'items', 'harness.dispatch': 'actions',
    'harness.memory_sample': 'calls', 'harness.memory_visit': 'objects',
    'harness.trace_append': 'rows', 'harness.reconcile': 'journal_rows',
    'harness.trace_hash': 'bytes', 'harness.finalize': 'calls',
}


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Meter:
    """Typed inclusive operation counts, with independently reconciled journal."""
    def __init__(self, cap):
        self.cap = cap
        self.totals = {owner: dict.fromkeys(UNITS, 0) for owner in OWNERS}
        self.journal = []

    def add(self, owner, metric, amount=1):
        if owner not in OWNERS or metric not in UNITS or type(amount) is not int or amount < 0:
            raise ValueError('Invalid accounting entry')
        stage = metric.split('.')[0]
        allowed = ('input', 'observer', 'teardown') if owner == 'shared' else ('harness',) if owner == 'harness' else ('backend', 'probe', 'feedback', 'expiry', 'wait', 'teardown')
        if stage not in allowed:
            raise ValueError('Accounting stage has the wrong owner')
        if len(self.journal) >= self.cap:
            raise ValueError('Accounting journal budget exceeded')
        self.totals[owner][metric] += amount
        self.journal.append((owner, metric, amount))

    def reconcile(self):
        expected = {owner: dict.fromkeys(UNITS, 0) for owner in OWNERS}
        for owner, metric, amount in self.journal:
            expected[owner][metric] += amount
        if expected != self.totals:
            raise ValueError('Accounting journal mismatch')

    def charged(self, owner):
        if owner not in ARMS:
            raise ValueError('Only deployed arms receive shared costs')
        return {key: self.totals['shared'][key] + self.totals[owner][key] for key in UNITS}


def verify_charges(physical, charged):
    """Shared execution occurs once physically, but is charged to each arm."""
    for key in UNITS:
        for arm in ARMS:
            if charged[arm][key] != physical['shared'][key] + physical[arm][key]:
                raise ValueError('Shared charge missing or duplicated')


def owned_size(value, seen=None):
    """Sample owned object graphs, including deque and slots; exclude code/RSS."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0, 0
    seen.add(id(value))
    size, visits = sys.getsizeof(value), 1
    if isinstance(value, dict):
        children = tuple(value.keys()) + tuple(value.values())
    elif isinstance(value, (tuple, list, deque, set, frozenset)):
        children = value
    elif hasattr(value, '__dict__'):
        children = (value.__dict__,)
    else:
        children = tuple(getattr(value, slot) for slot in getattr(type(value), '__slots__', ()) if hasattr(value, slot))
    for child in children:
        child_size, child_visits = owned_size(child, seen)
        size += child_size
        visits += child_visits
    return size, visits


class Backend:
    """Existing neuron calls versus an independent scalar recurrence."""
    def __init__(self, arm, meter, current, max_routes):
        self.arm, self.meter, self.current, self.max_routes = arm, meter, current, max_routes
        self.cells = {}
        meter.add(arm, 'backend.initialize')

    def step(self, route):
        meter, arm = self.meter, self.arm
        meter.add(arm, 'backend.lookup')
        if route not in self.cells:
            if len(self.cells) >= self.max_routes:
                raise ValueError('Backend route budget exceeded')
            self.cells[route] = Neuron(route, num_branches=1) if arm == 'neuron' else (0.0, 0, False)
            meter.add(arm, 'backend.allocate')
        meter.add(arm, 'backend.input')
        meter.add(arm, 'backend.step')
        if arm == 'neuron':
            cell = self.cells[route]
            refractory = cell.refractory_time > 0
            cell.add_input_to_branch(0, self.current)
            spike = cell.step()
            state = (cell.v, cell.refractory_time, spike)
        else:
            potential, remaining, _ = self.cells[route]
            refractory = remaining > 0
            if refractory:
                state = (0.0, remaining - 1, False)
            else:
                # Match the pinned one-branch positive-current Neuron arithmetic,
                # including strict > threshold and arrival-indexed refractory time.
                activation = 1.0 / (1.0 + math.exp(-5.0 * (abs(self.current) - 0.5)))
                output = math.copysign(activation * 2.0 * abs(self.current), self.current)
                potential = potential * 0.95 + output
                state = (0.0, 2, True) if potential > 1.0 else (potential, 0, False)
            self.cells[route] = state
        meter.add(arm, 'backend.refractory' if refractory else 'backend.transform')
        meter.add(arm, 'backend.spike', int(state[2]))
        return state


class Lifecycle:
    """Bounded pending receipts with monotone FIFO expiry timers and no learning."""
    def __init__(self, arm, meter, budgets, ttl):
        self.arm, self.meter, self.budgets, self.ttl = arm, meter, budgets, ttl
        self.pending, self.timers = {}, deque()
        self.clock = 0
        self.peak_pending = self.peak_timers = 0
        self.trace = []
        self.sample = lambda: None
        meter.add(arm, 'feedback.initialize')

    def record(self, row):
        self.trace.append(row)
        self.meter.add('harness', 'harness.trace_append')

    def create(self, sequence, time, features):
        if time != self.clock or sequence in self.pending:
            raise ValueError('Noncausal or duplicate receipt')
        if len(self.pending) >= self.budgets['pending_per_arm'] or len(self.timers) >= self.budgets['timers_per_arm']:
            raise ValueError('Receipt/timer budget exceeded')
        self.pending[sequence] = (time, features)
        self.timers.append((time + self.ttl, sequence))
        self.meter.add(self.arm, 'probe.create')
        self.meter.add(self.arm, 'probe.feature_reference', len(features))
        self.meter.add(self.arm, 'feedback.pending_insert')
        self.meter.add(self.arm, 'expiry.push')
        self.peak_pending = max(self.peak_pending, len(self.pending))
        self.peak_timers = max(self.peak_timers, len(self.timers))
        self.record((time, 'created', sequence, features))

    def notify(self, sequence, time, kind):
        if time != self.clock or kind not in ('result', 'cancel'):
            raise ValueError('Invalid external notification')
        self.meter.add(self.arm, 'feedback.' + kind)
        self.meter.add(self.arm, 'feedback.lookup')
        if sequence not in self.pending:
            self.meter.add(self.arm, 'feedback.rejected')
            self.record((time, 'rejected_' + kind, sequence))
            return
        cutoff, _ = self.pending[sequence]
        if not cutoff < time <= cutoff + self.ttl:
            raise ValueError('Noncausal or expired receipt reached delivery')
        del self.pending[sequence]
        self.meter.add(self.arm, 'feedback.pending_delete')
        terminal = 'delivered' if kind == 'result' else 'cancelled'
        self.meter.add(self.arm, 'feedback.' + terminal)
        self.record((time, terminal, sequence))

    def advance(self, time, inclusive):
        if type(time) is not int or time < self.clock or type(inclusive) is not bool:
            raise ValueError('Clock must be monotone')
        self.meter.add(self.arm, 'wait.advance')
        while self.timers:
            self.meter.add(self.arm, 'expiry.head_check')
            deadline, sequence = self.timers[0]
            if deadline > time or (deadline == time and not inclusive):
                break
            # Simulate internal wakeups at their causal deadlines before moving
            # to the next external timestamp or final horizon. Never backdate work.
            if deadline < self.clock:
                raise ValueError('Overdue timer was not dispatched causally')
            self.meter.add(self.arm, 'wait.elapsed', deadline - self.clock)
            self.clock = deadline
            self.meter.add(self.arm, 'wait.timer_dispatch')
            self.timers.popleft()
            self.meter.add(self.arm, 'expiry.pop')
            self.meter.add(self.arm, 'feedback.lookup')
            if sequence in self.pending:
                del self.pending[sequence]
                self.meter.add(self.arm, 'feedback.pending_delete')
                self.meter.add(self.arm, 'expiry.expired')
                self.record((deadline, 'expired', sequence))
            else:
                self.meter.add(self.arm, 'expiry.stale')
                self.record((deadline, 'stale_timer', sequence))
            self.sample()
        self.meter.add(self.arm, 'wait.elapsed', time - self.clock)
        self.clock = time


def make_tape(events, scenario):
    """Transport-only schedule; result notifications contain no target values."""
    if scenario not in SCENARIOS:
        raise ValueError('Unknown lifecycle scenario')
    tape, time = [], 0
    for index, (route, gap) in enumerate(events):
        time += gap
        tape.append((time, 2, index, 'input', (route, gap)))
        if scenario in ('on_time', 'late'):
            tape.append((time + (4 if scenario == 'on_time' else 5), 1, index, 'result', None))
        elif scenario == 'mixed':
            if index % 3 == 0:
                tape.append((time + 4, 1, index, 'result', None))
            elif index % 3 == 1:
                tape.extend(((time + 1, 0, index, 'cancel', None), (time + 2, 1, index, 'result', None)))
            else:
                tape.extend(((time + 1, 1, index, 'result', None), (time + 2, 1, index, 'result', None)))
    if scenario == 'mixed':
        tape.append((time + 2, 1, 99, 'result', None))
    return sorted(tape)


def validate_lifecycle(trace, events, scenario):
    """Independent receipt oracle; detect lost, double or incorrectly timed results."""
    times, time = [], 0
    for _, gap in events:
        time += gap
        times.append(time)
    created = [row[2] for row in trace if row[1] == 'created']
    terminal = [row[:3] for row in trace if row[1] in ('delivered', 'cancelled', 'expired')]
    if created != list(range(len(events))) or Counter(row[2] for row in terminal) != Counter(created):
        raise ValueError('Receipts do not terminate exactly once')
    for row in trace:
        if row[1] != 'created':
            continue
        index = row[2]
        channels, history = row[3][0]
        prefix = events[:index + 1]
        if tuple(history) != prefix or row[0] != times[index]:
            raise ValueError('Probe contains noncausal shared input')
        records = {}
        for ordinal, (route, _) in enumerate(prefix):
            positions = records.setdefault(route, [])
            positions.append(ordinal)
        expected_channels = tuple(item for route, positions in sorted(records.items()) for item in (
            (3 * route, 1.0 / (1 + positions[0])),
            (3 * route + 1, (1 + positions[-1]) / 95.0),
            (3 * route + 2, len(positions) / 95.0)))
        if channels != expected_channels:
            raise ValueError('Shared observer projection differs from prefix oracle')
    for time, kind, index in terminal:
        expected = ('delivered', 4) if scenario == 'on_time' else ('expired', 4)
        if scenario == 'mixed':
            expected = (('delivered', 4), ('cancelled', 1), ('delivered', 1))[index % 3]
        if (kind, time - times[index]) != expected:
            raise ValueError('Receipt lifecycle oracle mismatch')
    rejected = sum(row[1] == 'rejected_result' for row in trace)
    expected_rejected = len(events) if scenario == 'late' else 1 + sum(i % 3 != 0 for i in created) if scenario == 'mixed' else 0
    if rejected != expected_rejected:
        raise ValueError('Late/duplicate/foreign notification rejection mismatch')


def run_case(events, scenario, horizon, protocol):
    budgets = protocol['budgets']
    if not 1 <= len(events) <= budgets['events_per_case'] or horizon not in protocol['horizons']:
        raise ValueError('Input/horizon budget exceeded')
    if len({r for r, _ in events}) > budgets['routes_per_case']:
        raise ValueError('Input route budget exceeded')
    meter = Meter(budgets['journal_entries_per_case'])
    ordinal, sequence = TemporalInformationObserver(), BoundedSequenceObserver()
    meter.add('shared', 'observer.initialize', 2)
    backends = {arm: Backend(arm, meter, protocol['backend']['current'], budgets['routes_per_case']) for arm in ARMS}
    engines = {arm: Lifecycle(arm, meter, budgets, protocol['lifecycle']['ttl']) for arm in ARMS}
    peaks = dict.fromkeys((*ARMS, 'physical'), 0)
    packet = None

    def sample():
        shared = (ordinal, sequence, packet)
        states = {arm: (backends[arm].cells, engines[arm].pending, engines[arm].timers) for arm in ARMS}
        graphs = [(arm, (shared, states[arm])) for arm in ARMS]
        graphs.append(('physical', (shared, *states.values())))
        for arm, graph in graphs:
            size, visits = owned_size(graph)
            peaks[arm] = max(peaks[arm], size)
            meter.add('harness', 'harness.memory_sample')
            meter.add('harness', 'harness.memory_visit', visits)
            if size > budgets['sampled_owned_bytes']:
                raise ValueError('Sampled owned-state budget exceeded')

    for engine in engines.values():
        engine.sample = sample
    tape = make_tape(events, scenario)
    meter.add('harness', 'harness.tape_sort', len(tape))
    if tape[-1][0] > horizon:
        raise ValueError('Horizon precedes external work')
    for time, actions in groupby(tape, key=lambda row: row[0]):
        for engine in engines.values():
            engine.advance(time, False)
        for _, _, index, kind, payload in actions:
            meter.add('harness', 'harness.dispatch')
            if kind == 'input':
                route, gap = payload
                meter.add('shared', 'input.decode')
                meter.add('shared', 'input.fields', 2)
                # Existing observer validation is included in each charged call.
                ordinal.observe(route)
                meter.add('shared', 'observer.ordinal')
                sequence.observe(route, gap)
                meter.add('shared', 'observer.sequence')
                info = ordinal.snapshot()
                history = sequence.snapshot()
                meter.add('shared', 'observer.snapshot', 2)
                meter.add('shared', 'observer.sort_input', len(info.records))
                meter.add('shared', 'observer.prefix_copy', len(history))
                channels = info.readout_routes()
                meter.add('shared', 'observer.projection')
                meter.add('shared', 'observer.channels', len(channels))
                packet = (channels, history)
                for arm in ARMS:
                    state = backends[arm].step(route)
                    engines[arm].create(index, time, (packet, state))
            else:
                for engine in engines.values():
                    engine.notify(index, time, kind)
            sample()
        for engine in engines.values():
            engine.advance(time, True)
    for engine in engines.values():
        engine.advance(horizon, True)
    sample()
    meter.add('harness', 'harness.finalize')
    shared = meter.totals['shared']
    n = len(events)
    distinct_prefix_records = sum(len({r for r, _ in events[:i + 1]}) for i in range(n))
    if not (shared['input.decode'] == shared['observer.ordinal'] == shared['observer.sequence'] == shared['observer.projection'] == n
            and shared['input.fields'] == shared['observer.snapshot'] == 2 * n
            and shared['observer.sort_input'] == distinct_prefix_records
            and shared['observer.channels'] == 3 * distinct_prefix_records
            and shared['observer.prefix_copy'] == n * (n + 1) // 2):
        raise ValueError('Shared observer work conservation failed')
    for arm, engine in engines.items():
        validate_lifecycle(engine.trace, events, scenario)
        if engine.pending or engine.timers:
            raise ValueError('Undrained lifecycle state')
        counts = meter.totals[arm]
        n = len(events)
        if not (counts['backend.step'] == counts['probe.create'] == counts['expiry.push'] == counts['expiry.pop'] == n
                and counts['expiry.expired'] + counts['expiry.stale'] == n
                and counts['feedback.delivered'] + counts['feedback.cancelled'] + counts['expiry.expired'] == n):
            raise ValueError('Operation conservation failed')
    if engines['neuron'].trace != engines['scalar'].trace:
        raise ValueError('Neuron/scalar semantic trace differs')
    traces = {arm: engine.trace for arm, engine in engines.items()}
    # Deployment teardown has a charged owner, while measurement/hash work stays
    # in the harness. All persistent data containers are cleared explicitly.
    for arm in ARMS:
        for container in (backends[arm].cells, engines[arm].pending, engines[arm].timers):
            meter.add(arm, 'teardown.released', len(container))
            container.clear()
            meter.add(arm, 'teardown.clear')
        engines[arm].sample = None
        meter.add(arm, 'teardown.references')
    for container in (ordinal._records, sequence._events):
        meter.add('shared', 'teardown.released', len(container))
        container.clear()
        meter.add('shared', 'teardown.clear')
    packet = None
    meter.add('shared', 'teardown.references')
    raw_trace = json.dumps(traces['neuron'], sort_keys=True, separators=(',', ':')).encode()
    trace_digest = sha256(raw_trace).hexdigest()
    meter.add('harness', 'harness.trace_hash', len(raw_trace))
    meter.add('harness', 'harness.reconcile', len(meter.journal) + 1)
    meter.reconcile()
    charged = {arm: meter.charged(arm) for arm in ARMS}
    verify_charges(meter.totals, charged)
    return {'scenario': scenario, 'horizon': horizon, 'input_events': len(events),
            'physical': meter.totals, 'charged': charged,
            'journal_rows': len(meter.journal), 'journal_sha256': fingerprint(meter.journal),
            'semantic_trace_sha256': trace_digest,
            'sampled_owned_bytes': peaks,
            'peak_pending': {arm: engines[arm].peak_pending for arm in ARMS},
            'peak_timers': {arm: engines[arm].peak_timers for arm in ARMS},
            'gates_passed': True, 'actual_learning_writes': {arm: 0 for arm in ARMS},
            'predictions_scored': 0}


def evaluate(protocol, source):
    if (protocol['arms'] != list(ARMS) or protocol['scenarios'] != list(SCENARIOS)
            or protocol['horizons'] != [16, 1000000000] or protocol['lifecycle']['ttl'] != 4
            or protocol['backend']['current'] != 0.6
            or protocol['input']['families'] != ['first_order', 'middle_order', 'gap_order']
            or protocol['input']['route_pair'] != [0, 1] or protocol['input']['members'] != [0, 1]
            or protocol['budgets']['cases'] != 48):
        raise ValueError('Fixed accounting contract changed')
    rows = []
    for family in protocol['input']['families']:
        for member in protocol['input']['members']:
            if len(source['templates'][family][member]) != len(source['gap_templates'][family][member]):
                raise ValueError('Input template shape changed')
            events = tuple((protocol['input']['route_pair'][r], gap) for r, gap in zip(
                source['templates'][family][member], source['gap_templates'][family][member]))
            for scenario in SCENARIOS:
                pair = []
                for horizon in protocol['horizons']:
                    result = run_case(events, scenario, horizon, protocol)
                    result.update(family=family, member=member, input_sha256=fingerprint(events))
                    pair.append(result)
                    rows.append(result)
                for owner in OWNERS:
                    a, b = pair[0]['physical'][owner], pair[1]['physical'][owner]
                    if any(a[key] != b[key] for key in UNITS if key != 'wait.elapsed'):
                        raise ValueError('Idle tail caused additional operation work')
                if pair[0]['semantic_trace_sha256'] != pair[1]['semantic_trace_sha256']:
                    raise ValueError('Idle tail changed semantics')
    if len(rows) != protocol['budgets']['cases']:
        raise ValueError('Finite case budget changed')
    return {'schema': 'sara-event-work-result-v1', 'cases': rows, 'case_count': len(rows),
            'units': UNITS, 'all_accounting_gates_passed': all(row['gates_passed'] for row in rows),
            'idle_extension_changes_only_elapsed_units': True,
            'actual_learning_writes': 0, 'predictions_scored': 0, 'raw_or_learned_payload_reads': 0,
            'cpu_seconds': None, 'energy_joules': None, 'total_transient_peak_bytes': None,
            'cost_scope': 'Inclusive source-level calls and item volumes, not CPU instructions. Owned memory is sampled; interpreter frames, transient allocation peaks, tape/journal/validation/hash overhead and RSS are not deployment memory. Harness work is separately owned.',
            'decision': 'accounting_contract_feasible_no_snn_advantage_established',
            'snn_resource_advantage_established': False, 'independent_usefulness_established': False,
            'runtime_promotion_authorized': False, 'test_opening_authorized': False}
