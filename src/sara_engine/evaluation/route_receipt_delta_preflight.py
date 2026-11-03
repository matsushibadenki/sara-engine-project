"""Literal fixtures and fault checks for a read-only receipt/delta bridge."""
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
import json
from math import nextafter, inf
from sara_engine.evaluation.route_receipt_probability import View, State, Delivery, open_receipt
from sara_engine.evaluation.route_receipt_delta import Stored, Batch, Proposal, validate, canonical, persisted

SELECTION_SHA = '45e8a58f0ccb1795c3ab252fbc530611cd46c55472eef5ef3a07411366997bbf'
LOCAL = 'route_error_binding_preflight_selection_v1.json'
NUMERIC = 'route_numeric_delta_preflight_selection_v1.json'
PLAN = 'route_independent_transition_selection_v1.json'


def require(value, message):
    if not value: raise ValueError(message)


def fixture(local, manual, plans, golden, role, branch):
    index, stable_mass, drift_mass, error_after, weight_after = golden
    row = local[index]; tick, history, weighting, allocated, n0, n1, side, _, expected_error = row
    base = next(r for r in manual['rows'] if r[:3] == row[:3])
    plan = next(p for p in plans if p['tick'] == tick)
    credits = tuple(tuple(c) for c in plan['credits'])
    target = next(c for c in credits if c[2] == side)
    owner = (1201, 'independent', 'changed', history, weighting, role)
    source = plan['shuffled_source'] if role == 'shuffled_outcome' else tick
    widths = [len(key) for size in (1, 2) for key in combinations(range(5), size) for _ in range(3)]
    receipt_fields = {}; allocation = {}
    for r, width in enumerate(widths):
        for k in range(2 ** width):
            allocation[(r, k)] = True
            for field in ('mass_left', 'mass_right'): receipt_fields[(r, field, k)] = manual['before']['each_mass']
        receipt_fields[(r, 'observations', -1)] = manual['before']['route_observations_by_width'][str(width)]
        receipt_fields[(r, 'error_sum', -1)] = manual['before']['error_sum']
        receipt_fields[(r, 'weight_sum', -1)] = manual['before']['weight_sum']
    r, k, _ = target
    allocation[(r, k)] = allocated
    receipt_fields[(r, 'mass_left', k)] = n0; receipt_fields[(r, 'mass_right', k)] = n1
    diagnostic = ('diagnostic', tuple(c[0] for c in credits))
    pending = []; deliveries = []; errors = []
    for route, key, mass_side in credits:
        view = View(owner, route, key, 0, allocation[(route, key)],
                    receipt_fields[(route, 'mass_left', key)], receipt_fields[(route, 'mass_right', key)])
        pending.append(open_receipt(State(owner, tick, None, frozenset(), diagnostic), tick, source, view))
        deliveries.append(Delivery(owner, tick, source, tick + 1, mass_side))
        # Literal witnesses come from the old local or symmetric numeric tables.
        errors.append(expected_error if (route, key) == (r, k) else base[6])
    before = dict(receipt_fields); flags = dict(allocation)
    if branch == 'changed_view':
        flags[(r, k)] = not allocated
        before[(r, 'mass_left', k)] = 0
        before[(r, 'mass_right', k)] = 0 if allocated else 2
    after = dict(before)
    for route, key, mass_side in credits:
        after[(route, mass_side, key)] = base[3]
        after[(route, 'observations', -1)] = manual['after_route_observations_by_width'][str(widths[route])]
        after[(route, 'error_sum', -1)] = base[4]
        after[(route, 'weight_sum', -1)] = base[5]
    after[(r, side, k)] = drift_mass if branch == 'changed_view' else stable_mass
    after[(r, 'error_sum', -1)] = error_after
    after[(r, 'weight_sum', -1)] = weight_after
    new_flags = dict(flags); new_flags[(r, k)] = True
    old = persisted(Stored(owner, tuple(sorted(before.items())), tuple(sorted(flags.items())), diagnostic))
    new = persisted(Stored(owner, tuple(sorted(after.items())), tuple(sorted(new_flags.items())), diagnostic))
    targets = tuple(sorted([(route, mass_side, key) for route, key, mass_side in credits]
                           + [(route, f, -1) for route, key, side in credits for f in ('observations', 'error_sum', 'weight_sum')]))
    p = Proposal(old, new, targets, ((r, k),), tuple(errors), 36, int(not flags[(r, k)]), int(branch == 'changed_view'))
    return p, Batch(old, tuple(pending), tuple(deliveries), (r, k))


def changed_field(state, address, value):
    return replace(state, fields=tuple((a, value if a == address else v) for a, v in state.fields))


def fault_checks(spec, bases):
    proposal, batch = bases[(5, 'shuffled_outcome', 'changed_view')]
    p, b = proposal, batch; target = b.allocation_target; r, k = target
    error = (r, 'error_sum', -1)
    j = next(i for i, state in enumerate(b.pending) if state.pending.view.route == r)
    def claim(value):
        claims = list(p.error_claims); claims[j] = value
        return tuple(claims)
    inactive = next(a for a, _ in p.before.fields if a[1] == 'mass_left' and a[0] not in [s.pending.view.route for s in b.pending])
    inactive_meta = next(a for a, _ in p.before.allocation if a[0] == inactive[0])
    operations = [
        ('claim_ulp', lambda: validate(replace(p, error_claims=claim(nextafter(p.error_claims[j], inf))), b)),
        ('delivery_probability_claim', lambda: validate(replace(p, error_claims=claim(0.5)), b)),
        ('self_consistent_false_error', lambda: validate(replace(p, error_claims=claim(0.5), after=changed_field(p.after, error, 8.5)), b)),
        ('learning_noop', lambda: validate(replace(p, after=changed_field(p.after, error, 8.0)), b)),
        ('inactive_learning_change', lambda: validate(replace(p, after=changed_field(p.after, inactive, 3)), b)),
        ('extra_learning_target', lambda: validate(replace(p, learning_targets=p.learning_targets + (inactive,)), b)),
        ('duplicate_learning_target', lambda: validate(replace(p, learning_targets=p.learning_targets + (p.learning_targets[0],)), b)),
        ('missing_learning_target', lambda: validate(replace(p, learning_targets=p.learning_targets[:-1]), b)),
        ('metadata_in_learning', lambda: validate(replace(p, learning_targets=p.learning_targets[:-1] + (target,)), b)),
        ('learning_in_metadata', lambda: validate(replace(p, allocation_targets=(error,)), b)),
        ('missing_metadata_target', lambda: validate(replace(p, allocation_targets=()), b)),
        ('extra_metadata_target', lambda: validate(replace(p, allocation_targets=p.allocation_targets + (inactive_meta,)), b)),
        ('allocation_not_admitted', lambda: validate(replace(p, after=replace(p.after, allocation=tuple((a, False if a == target else v) for a, v in p.after.allocation))), b)),
        ('inactive_allocation_change', lambda: validate(replace(p, after=replace(p.after, allocation=tuple((a, False if a == inactive_meta else v) for a, v in p.after.allocation))), b)),
        ('metadata_count_as_learning', lambda: validate(replace(p, changed_learning_claim=37), b)),
        ('wrong_metadata_changed_count', lambda: validate(replace(p, changed_allocation_claim=0), b)),
        ('bool_claim_count', lambda: validate(replace(p, changed_allocation_claim=True), b)),
        ('stale_predecessor', lambda: validate(replace(p, before=changed_field(p.before, error, 7.0)), b)),
    ]
    def delivery(index, value):
        events = list(b.deliveries); events[j] = replace(events[j], **{index: value})
        return replace(b, deliveries=tuple(events))
    operations.extend([
        ('wrong_receiving_id', lambda: validate(p, delivery('receiving_id', 65))),
        ('wrong_source', lambda: validate(p, delivery('source_id', (b.deliveries[j].source_id + 1) % 512))),
        ('same_tick_delivery', lambda: validate(p, delivery('at', 64))),
        ('diagnostic_drift', lambda: validate(replace(p, after=replace(p.after, diagnostic=('diagnostic', tuple(range(9))))), b)),
        ('rounded_error_type', lambda: validate(replace(p, after=changed_field(p.after, error, int(dict(p.after.fields)[error]))), b)),
    ])
    late_p, late_b = bases[(7, 'shuffled_outcome', 'changed_view')]
    operations.append(('delivery_clock_error', lambda: validate(replace(late_p, error_claims=tuple(v * 2 for v in late_p.error_claims)), late_b)))
    rejected = []
    for name, operation in operations:
        signature = repr((proposal, batch, late_p, late_b))
        try: operation()
        except ValueError: rejected.append(name)
        else: raise ValueError('Undetected receipt/delta fault: ' + name)
        require(repr((proposal, batch, late_p, late_b)) == signature, 'Rejected fault mutated trusted fixtures')
    require(rejected == spec['faults'], 'Fixed fault roster differs')
    return rejected


def analyze(spec, inputs):
    require(set(inputs) == set(spec['inputs']) and sum(map(len, inputs.values())) <= spec['bounds']['input_bytes'], 'Input roster/budget differs')
    for name, raw in inputs.items(): require(sha256(raw).hexdigest() == spec['inputs'][name], 'Frozen input changed')
    local, manual, plans = json.loads(inputs[LOCAL])['local_rows'], json.loads(inputs[NUMERIC])['manual_fixture'], json.loads(inputs[PLAN])['plans']
    require(len(local) == 11 and [g[0] for g in spec['goldens']] == list(range(11)), 'Fixed local roster differs')
    records = []; bases = {}
    for golden in spec['goldens']:
        for role in ('candidate', 'permanent', 'shuffled_outcome'):
            for branch in ('stable_view', 'changed_view'):
                p, b = fixture(local, manual, plans, golden, role, branch)
                before = repr((p, b)); report = validate(p, b)
                require(repr((p, b)) == before, 'Read-only validator modified fixtures')
                bases[(golden[0], role, branch)] = (p, b)
                records.append({'local_row': golden[0], 'role': role, 'branch': branch, **report})
    totals = {'all_exact_transactions': len(records),
              'requested_learning': sum(r['requested_learning'] for r in records),
              'changed_learning': sum(r['serialized_changed_learning'] for r in records),
              'unchanged_learning': sum(r['unchanged_learning'] for r in records),
              'requested_metadata': sum(r['requested_allocation'] for r in records),
              'changed_metadata': sum(r['serialized_changed_allocation'] for r in records),
              'unallocated_receipt_transactions': sum(local[r['local_row']][3] is False for r in records)}
    require(totals == spec['gates'], 'Frozen aggregate gates differ')
    return {'schema': 'sara-receipt-delta-result-v1', 'totals': totals, 'records': records,
            'faults_detected': fault_checks(spec, bases), 'scope': 'read_only_serialized_fixture_delta_contract',
            'actual_learning_updates': 0, 'learner_executions': 0, 'task_scores': 0, 'acquisitions': 0,
            'resource_measurements': 0, 'unallocated_key_learning_implemented': False,
            'runtime_provenance_authenticated': False, 'runtime_equivalence_established': False,
            'snn_advantage_established': False, 'promotion_authorized': False,
            'test_opened': False, 'conditions_relaxed': False}
