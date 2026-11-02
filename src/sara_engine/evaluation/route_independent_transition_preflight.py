"""Literal-golden harness for the separately implemented scalar transition."""
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
from math import copysign, isfinite, nextafter, inf
import ast
import sys
from sara_engine.evaluation.route_independent_transition import transition

SELECTION_SHA = '3c1a49e6e997d0a792fb2ce6cb9f43c2ebc5a23d4769331c6e74fb77c79cfb43'


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(condition, message):
    if not condition: raise ValueError(message)


def independence(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): raise ValueError('Unregistered kernel import')
        if isinstance(node, ast.ImportFrom):
            require((node.module, tuple(n.name for n in node.names), node.level) in
                    (('itertools', ('combinations',), 0), ('math', ('copysign', 'isfinite'), 0)), 'Primary/helper import')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            require(node.func.id not in ('eval', 'exec', '__import__', 'open', 'compile'), 'Dynamic/I/O kernel call')
    for name in sys.modules:
        require(not (name.startswith('sara_engine.evaluation.') and name.rsplit('.', 1)[1] in
                     ('route_numeric_delta_preflight', 'route_error_binding_preflight', 'route_independent_scalar',
                      'route_delta_ledger_preflight', 'route_interface_preflight', 'sparse_route_reactivation',
                      'route_memory_factorial_v4', 'sparse_factor_routing')), 'Old helper loaded')
    return sha256(source.encode()).hexdigest()


def fixture(spec, manual, row, role):
    # This catalog and literal after-state construction do not use kernel helpers.
    widths = [len(inputs) for size in (1, 2) for inputs in combinations(range(5), size) for output in range(3)]
    addresses = []
    for r in range(45):
        for field in ('error_sum', 'mass_left', 'mass_right', 'observations', 'weight_sum'):
            for k in (range(2 ** widths[r]) if field.startswith('mass_') else (-1,)):
                addresses.append((r, field, k))
    tick, history, error_rule, mass_after, error_after, weight_after, _ = row
    owner = (1201, 'independent', 'changed', history, error_rule, role)
    plan = next(p for p in spec['plans'] if p['tick'] == tick)
    credits = tuple(tuple(c) for c in plan['credits'])
    source = plan['shuffled_source'] if role == 'shuffled_outcome' else tick
    before = []
    for address in addresses:
        r, field, _ = address
        value = (manual['before']['each_mass'] if field.startswith('mass_') else
                 manual['before']['route_observations_by_width'][str(widths[r])] if field == 'observations'
                 else manual['before'][field])
        before.append((address, value))
    expected = dict(before)
    for route, key, side in credits:
        expected[(route, side, key)] = mass_after
        expected[(route, 'error_sum', -1)] = error_after
        expected[(route, 'weight_sum', -1)] = weight_after
        expected[(route, 'observations', -1)] = manual['after_route_observations_by_width'][str(widths[route])]
    diagnostic = ('diagnostic', tuple(c[0] for c in credits))
    args = (owner, tuple(before), (owner, tick, source, credits),
            (owner, tick, source, tick + 1, 'training'), diagnostic, frozenset())
    return args, tuple((a, expected[a]) for a in addresses)


def typed_equal(actual, expected):
    require(type(actual) is type(expected), 'Type mismatch')
    if type(expected) is float:
        require(isfinite(actual) and not (actual == 0 and copysign(1, actual) < 0), 'Invalid float')
        return actual.hex() == expected.hex()
    return actual == expected


def compare(after, expected, before):
    require(type(after) is tuple and len(after) == 435, 'Wrong result catalog')
    require(tuple(a for a, _ in after) == tuple(a for a, _ in expected), 'Result address drift')
    changed = 0
    for (address, value), (_, golden), (_, old) in zip(after, expected, before):
        require(typed_equal(value, golden), 'Literal after mismatch: ' + str(address))
        if not typed_equal(value, old): changed += 1
    require(changed == 36, 'Wrong changed field count')
    return {'changed': changed, 'unchanged': len(after) - changed}


def alter(fields, address, value):
    return tuple((a, value if a == address else v) for a, v in fields)


def fault_checks(spec, args, expected, actual):
    owner, before, pending, delivery, diagnostic, consumed = args
    r, key, side = pending[3][0]
    failures = []
    cases = []
    def add(name, index, value):
        changed = list(args); changed[index] = value
        cases.append((name, lambda changed=changed: transition(*changed)))
    def event(index, value):
        data = list(delivery); data[index] = value
        return tuple(data)
    add('receiving_id', 3, event(1, delivery[1] + 1))
    add('shuffled_source', 3, event(2, (delivery[2] + 1) % 512))
    add('same_tick', 3, event(3, pending[1]))
    add('late_delivery', 3, event(3, pending[1] + 2))
    add('foreign_owner', 3, event(0, (1213,) + owner[1:]))
    add('evaluation_purpose', 3, event(4, 'evaluation'))
    add('duplicate_receipt', 5, frozenset(((owner, pending[1]),)))
    for name, index, value in (('route_drift', 0, (r + 1) % 45), ('key_drift', 1, 1 - key),
                               ('side_drift', 2, 'mass_right' if side == 'mass_left' else 'mass_left')):
        credits = list(pending[3]); first = list(credits[0]); first[index] = value; credits[0] = tuple(first)
        add(name, 2, pending[:3] + (tuple(credits),))
    add('diagnostic_alias', 4, before)
    add('missing_field', 1, before[:-1])
    add('duplicate_field', 1, before[:-1] + (before[0],))
    add('bool_mass', 1, alter(before, (r, side, key), True))
    add('float_mass', 1, alter(before, (r, side, key), 2.0))
    add('negative_zero', 1, alter(before, (r, 'error_sum', -1), -0.0))
    add('nan_error', 1, alter(before, (r, 'error_sum', -1), float('nan')))
    absorbed = before
    for address, value in (((r, 'mass_left', key), 1000000), ((r, 'mass_right', key), 0),
                           ((r, 'error_sum', -1), 1000000.0), ((r, 'weight_sum', -1), 2000000.0)):
        absorbed = alter(absorbed, address, value)
    add('absorbed_increment', 1, absorbed)
    bad = alter(actual, (r, 'error_sum', -1), nextafter(dict(actual)[(r, 'error_sum', -1)], inf))
    cases.append(('wrong_after_error', lambda: compare(bad, expected, before)))
    inactive = next(a for a, _ in before if a[1] == 'mass_left' and a[0] not in diagnostic[1])
    bad_inactive = alter(actual, inactive, 3)
    cases.append(('inactive_after_change', lambda: compare(bad_inactive, expected, before)))
    add('wrong_role', 3, event(0, owner[:-1] + ('permanent',)))
    add('source_as_receiving_id', 3, event(1, pending[2]))
    add('mutable_fields', 1, list(before))
    bad_nonfinite = alter(actual, (r, 'error_sum', -1), inf)
    cases.append(('nonfinite_after', lambda: compare(bad_nonfinite, expected, before)))
    for name, operation in cases:
        signature = repr(args)
        try: operation()
        except ValueError: failures.append(name)
        else: raise ValueError('Undetected transition fault: ' + name)
        require(repr(args) == signature, 'Rejected fault modified original inputs')
    require(failures == spec['faults'], 'Fault roster differs')
    return failures


def analyze(spec, source, input_raw):
    require(sha256(input_raw).hexdigest() == spec['input']['sha256'], 'Literal input changed')
    require(len(input_raw) <= spec['bounds']['input_bytes'], 'Input byte budget')
    manual = json.loads(input_raw)['manual_fixture']
    require(sha256(canonical(manual['rows'])).hexdigest() == spec['fixture_rows_sha256'], 'Row roster changed')
    source_sha = independence(source)
    records = []; fault_base = None
    for index, row in enumerate(manual['rows']):
        for role in manual['roles']:
            args, expected = fixture(spec, manual, row, role)
            signature = repr(args)
            after, diagnostic, consumed = transition(*args)
            require(diagnostic == args[4] and consumed == frozenset(((args[0], row[0]),)), 'Diagnostic/consumed drift')
            checks = compare(after, expected, args[1])
            require(repr(args) == signature, 'Transition mutated fixture input')
            records.append({'row': index, 'role': role, **checks,
                            'after_sha256': sha256(canonical(after)).hexdigest(), 'pending': args[2],
                            'diagnostic_unchanged': True, 'consumed_receipt_count': len(consumed)})
            if row[:3] == [64, 'discounted', 'public_exponential'] and role == 'shuffled_outcome':
                fault_base = (args, expected, after)
    require(len(records) == spec['bounds']['transitions_per_phase'] == 36, 'Transition budget differs')
    detected = fault_checks(spec, *fault_base)
    return {'schema': 'sara-independent-transition-result-v1', 'records': records,
            'transitions': len(records), 'changed_fixture_fields': sum(r['changed'] for r in records),
            'unchanged_fixture_fields': sum(r['unchanged'] for r in records),
            'faults_detected': detected, 'kernel_source_sha256': source_sha,
            'scope': 'independent_nonlearning_literal_transitions_only',
            'actual_learning_updates': 0, 'learner_executions': 0, 'task_scores': 0,
            'acquisitions': 0, 'resource_measurements': 0, 'test_opened': False,
            'runtime_equivalence_established': False, 'runtime_write_provenance_authenticated': False,
            'snn_advantage_established': False, 'promotion_authorized': False, 'conditions_relaxed': False}
