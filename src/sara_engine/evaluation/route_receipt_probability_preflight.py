"""Finite literal preflight of frozen receipt allocation/probability."""
from dataclasses import replace, asdict
from hashlib import sha256
import json
from math import isfinite, nextafter, inf
from sara_engine.evaluation.route_receipt_probability import (
    View, State, Delivery, open_receipt, resolve,
)

SELECTION_SHA = 'a865e9a4c10a15fdd70a38d44b7135ad8fb6e2805ad67c9d205715b22183e610'
LOCAL = 'route_error_binding_preflight_selection_v1.json'
PLAN = 'route_independent_transition_selection_v1.json'


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value: raise ValueError(message)


def equal(actual, expected):
    require(type(actual) is type(expected), 'Exact probability/error type mismatch')
    if type(expected) is float:
        return isfinite(actual) and actual.hex() == expected.hex()
    return actual == expected


def fixture(row, role, plans):
    tick, history, weighting, allocated, n0, n1, side, _, _ = row
    owner = (1201, 'independent', 'changed', history, weighting, role)
    source = next(p['shuffled_source'] for p in plans if p['tick'] == tick) if role == 'shuffled_outcome' else tick
    view = View(owner, 0, 0, 0, allocated, n0, n1)
    initial = State(owner, tick, None, frozenset(), ('diagnostic', tuple(range(9))))
    pending = open_receipt(initial, tick, source, view)
    delivery = Delivery(owner, tick, source, tick + 1, side)
    changed = View(owner, 0, 0, 1, not allocated, 0, 2 if not allocated else 0)
    return initial, pending, delivery, view, changed


def fault_checks(spec, bases):
    # Trusted base fixtures come from the successful primary rows; no learner runs.
    initial, state, event, view, changed = bases[(5, 'shuffled_outcome')]
    absent = bases[(9, 'shuffled_outcome')][1]
    zero = bases[(10, 'shuffled_outcome')][1]
    snapshot = state.pending
    operations = [
        ('absent_positive_mass', lambda: replace(absent.pending.view, n0=1)),
        ('bool_allocation', lambda: replace(view, allocated=1)),
        ('bool_mass', lambda: replace(view, n0=True)),
        ('nan_probability', lambda: replace(snapshot, probability=float('nan'))),
        ('probability_ulp', lambda: replace(snapshot, probability=nextafter(snapshot.probability, inf))),
        ('absent_as_half', lambda: replace(absent.pending, probability=0.5)),
        ('zero_as_absent', lambda: replace(zero.pending, probability=None)),
        ('allocation_drift', lambda: replace(snapshot, view=replace(view, allocated=False, n0=0, n1=0))),
        ('source_as_receiver', lambda: resolve(state, replace(event, receiving_id=event.source_id), changed)),
        ('wrong_source', lambda: resolve(state, replace(event, source_id=(event.source_id + 1) % 512), changed)),
        ('early_delivery', lambda: resolve(state, replace(event, at=event.receiving_id), changed)),
        ('late_delivery', lambda: resolve(state, replace(event, at=event.receiving_id + 2), changed)),
    ]
    closed, _ = resolve(state, event, changed)
    operations.extend([
        ('duplicate_delivery', lambda: resolve(closed, event, changed)),
        ('foreign_owner', lambda: resolve(state, replace(event, owner=event.owner[:-1] + ('permanent',)), changed)),
        ('wrong_route', lambda: resolve(state, event, replace(changed, route=1))),
        ('wrong_key', lambda: resolve(state, event, replace(changed, key=1))),
        ('diagnostic_drift', lambda: replace(state, diagnostic=('diagnostic', tuple(range(1, 10))))),
    ])
    later_view = replace(view, revision=1)
    later_pending = open_receipt(initial, event.receiving_id, event.source_id, later_view)
    operations.extend([
        ('backward_revision', lambda: resolve(later_pending, event, view)),
        ('evaluation_delivery', lambda: resolve(state, replace(event, purpose='evaluation'), changed)),
        ('delivery_view_substitution', lambda: replace(snapshot, view=changed)),
        ('unsupported_side', lambda: replace(event, side='unknown')),
        ('receipt_clock_drift', lambda: replace(state, pending=replace(snapshot, receiving_id=snapshot.receiving_id + 1))),
        ('mutable_view', lambda: open_receipt(initial, event.receiving_id, event.source_id, asdict(view))),
        ('double_open', lambda: open_receipt(state, event.receiving_id, event.source_id, view)),
    ])
    rejected = []
    for name, operation in operations:
        before = repr((initial, state, event, view, changed, absent, zero))
        try: operation()
        except ValueError: rejected.append(name)
        else: raise ValueError('Undetected receipt fault: ' + name)
        require(repr((initial, state, event, view, changed, absent, zero)) == before, 'Rejected fault mutated trusted state')
    require(rejected == spec['faults'], 'Fault roster differs')
    return rejected


def analyze(spec, inputs):
    require(set(inputs) == set(spec['inputs']) and sum(map(len, inputs.values())) <= spec['bounds']['input_bytes'], 'Wrong input roster/budget')
    for name, raw in inputs.items(): require(sha256(raw).hexdigest() == spec['inputs'][name], 'Frozen input differs')
    rows = json.loads(inputs[LOCAL])['local_rows']
    plans = json.loads(inputs[PLAN])['plans']
    require(len(rows) == 11, 'Original hand record count differs')
    records = []; bases = {}
    for index, row in enumerate(rows):
        for role in ('candidate', 'permanent', 'shuffled_outcome'):
            initial, state, event, view, changed = fixture(row, role, plans)
            bases[(index, role)] = (initial, state, event, view, changed)
            require(equal(state.pending.probability, row[7]) and state.pending.view.allocated is row[3], 'Capture differs from fixed literal')
            signature = repr((initial, state, event, view, changed))
            outcomes = []
            for branch, current in (('unchanged_view', view), ('changed_view', changed)):
                closed, result = resolve(state, event, current)
                require(equal(result['probability_at_receipt'], row[7]) and equal(result['error_increment'], row[8]), 'Original frozen probability/error mismatch')
                require(result['allocated_at_receipt'] is row[3] and result['fallback_used'] is (not row[3]), 'Absence/zero conflation')
                require(closed.pending is None and closed.consumed == frozenset((event.receiving_id,))
                        and closed.diagnostic == initial.diagnostic, 'Closed receipt/diagnostic mismatch')
                outcomes.append({k: v for k, v in result.items() if k != 'delivery_revision'})
                records.append({'row': index, 'role': role, 'branch': branch, 'readout': result,
                                'delivery_view': asdict(current), 'snapshot_sha256': sha256(canonical(asdict(state.pending))).hexdigest()})
            require(canonical(outcomes[0]) == canonical(outcomes[1]), 'Delivery view changed frozen readout')
            require(repr((initial, state, event, view, changed)) == signature, 'Contract mutated original view/state')
    faults = fault_checks(spec, bases)
    require(len(records) == spec['bounds']['delivery_branches'] == 66, 'Delivery branch budget differs')
    return {'schema': 'sara-receipt-probability-result-v1', 'records': records,
            'receipt_snapshots': 33, 'delivery_branches': 66, 'frozen_branch_pairs_equal': 33,
            'faults_detected': faults, 'absence_supported_as_snapshot': True, 'unallocated_learning_implemented': False,
            'scope': 'nonlearning_frozen_receipt_contract_only', 'actual_learning_updates': 0,
            'learner_executions': 0, 'task_scores': 0, 'acquisitions': 0, 'resource_measurements': 0,
            'runtime_provenance_authenticated': False, 'runtime_equivalence_established': False,
            'snn_advantage_established': False, 'promotion_authorized': False,
            'conditions_relaxed': False, 'test_opened': False}
