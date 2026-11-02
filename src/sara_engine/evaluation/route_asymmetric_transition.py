"""Finite asymmetric literal comparison; no task stream or learner execution."""
import ast
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite, nextafter, inf
import sys
from sara_engine.evaluation.route_independent_transition import transition

SELECTION_SHA = '9f08848f95b667fb4165471aa9ac239b674b4dcade1e30a8a3051ff258394a64'
LOCAL = 'route_error_binding_preflight_selection_v1.json'
NUMERIC = 'route_numeric_delta_preflight_selection_v1.json'
PLAN = 'route_independent_transition_selection_v1.json'


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value: raise ValueError(message)


def bind_inputs(spec, inputs):
    require(set(inputs) == set(spec['inputs']), 'Wrong input roster')
    require(sum(map(len, inputs.values())) <= spec['bounds']['input_bytes'], 'Input byte budget exceeded')
    for name, raw in inputs.items():
        require(sha256(raw).hexdigest() == spec['inputs'][name], 'Frozen input changed: ' + name)


def source_guard(spec, source):
    require(sha256(source.encode()).hexdigest() == spec['kernel']['sha256'], 'Frozen independent kernel changed')
    for node in ast.walk(ast.parse(source)):
        require(not isinstance(node, ast.Import), 'Unexpected kernel import')
        if isinstance(node, ast.ImportFrom):
            require(node.module in ('math', 'itertools') and node.level == 0, 'Primary/helper kernel import')
    # Package initialization exposes these two infrastructure modules only.
    allowed = {'route_asymmetric_transition', 'route_independent_transition', 'evaluator', 'phase3_tracking'}
    for name in sys.modules:
        if name.startswith('sara_engine.evaluation.'):
            require(name.rsplit('.', 1)[1] in allowed, 'Old calculation/fixture/learner module loaded')


def case(spec, local_rows, manual, plans, golden, role):
    index, mass_after, error_after, weight_after = golden
    row = local_rows[index]
    tick, history, error_rule, allocated, n0, n1, side, _, _ = row
    require(type(allocated) is bool and allocated is True, 'Unallocated row not supported by frozen transition')
    base = next(r for r in manual['rows'] if r[:3] == row[:3])
    plan = next(p for p in plans if p['tick'] == tick)
    credits = tuple(tuple(c) for c in plan['credits'])
    target = next(c for c in credits if c[2] == side)
    owner = (1201, 'independent', 'changed', history, error_rule, role)
    source = plan['shuffled_source'] if role == 'shuffled_outcome' else tick
    widths = [len(key) for size in (1, 2) for key in combinations(range(5), size) for output in range(3)]
    before = {}
    for r, width in enumerate(widths):
        for key in range(2 ** width):
            for field in ('mass_left', 'mass_right'): before[(r, field, key)] = manual['before']['each_mass']
        before[(r, 'observations', -1)] = manual['before']['route_observations_by_width'][str(width)]
        before[(r, 'error_sum', -1)] = manual['before']['error_sum']
        before[(r, 'weight_sum', -1)] = manual['before']['weight_sum']
    r, key, _ = target
    before[(r, 'mass_left', key)] = n0
    before[(r, 'mass_right', key)] = n1
    expected = dict(before)
    for route, key_slot, mass_side in credits:
        expected[(route, mass_side, key_slot)] = base[3]
        expected[(route, 'observations', -1)] = manual['after_route_observations_by_width'][str(widths[route])]
        expected[(route, 'error_sum', -1)] = base[4]
        expected[(route, 'weight_sum', -1)] = base[5]
    expected[(r, side, key)] = mass_after
    expected[(r, 'error_sum', -1)] = error_after
    expected[(r, 'weight_sum', -1)] = weight_after
    pending = (owner, tick, source, credits)
    args = (owner, tuple(sorted(before.items())), pending,
            (owner, tick, source, tick + 1, 'training'), ('diagnostic', tuple(c[0] for c in credits)), frozenset())
    return args, tuple(sorted(expected.items())), target


def equal(actual, expected):
    require(type(actual) is type(expected), 'Exact type mismatch')
    if type(expected) is float:
        return isfinite(actual) and actual.hex() == expected.hex()
    return actual == expected


def compare(actual, expected, before):
    require(type(actual) is tuple and len(actual) == len(expected) == len(before) == 435, 'Wrong immutable catalog')
    changed = 0
    for (a, value), (b, golden), (c, old) in zip(actual, expected, before):
        require(a == b == c and equal(value, golden), 'Exact after-state mismatch: ' + str(b))
        changed += not equal(value, old)
    require(changed == 36, 'Wrong changed field count')
    return {'changed': changed, 'unchanged': 435 - changed}


def mutants(source):
    original = 'probability = (right + prior) / ((left + right) + (prior + prior))'
    replacements = [
        ('constant_half', original, 'probability = 0.5'),
        ('reverse_class', original, 'probability = (left + prior) / ((left + right) + (prior + prior))'),
        ('unit_prior', "prior = public_scale if history == 'discounted' else 1", 'prior = 1'),
        ('post_count_probability', original, "probability = (right + (count_step if side == 'mass_right' else 0) + prior) / ((left + right + count_step) + (prior + prior))"),
        ('unit_error_weight', "weight = float(public_scale) if error_rule == 'public_exponential' else 1.0", 'weight = 1.0'),
        ('unit_count_step', "count_step = public_scale if history == 'discounted' else 1", 'count_step = 1'),
    ]
    result = []
    for name, old, new in replacements:
        require(source.count(old) == 1, 'Mutation source anchor differs')
        modified = source.replace(old, new)
        namespace = {}
        # Only the pinned, standard-library-only pure kernel and fixed edits execute.
        exec(compile(modified, '<fixed-fixture-mutant:' + name + '>', 'exec'), namespace)
        result.append((name, namespace['transition'], sha256(modified.encode()).hexdigest()))
    return result


def analyze(spec, source, inputs):
    bind_inputs(spec, inputs); source_guard(spec, source)
    local = json.loads(inputs[LOCAL])['local_rows']
    manual = json.loads(inputs[NUMERIC])['manual_fixture']
    plans = json.loads(inputs[PLAN])['plans']
    require([g[0] for g in spec['goldens']] == spec['allocation_scope']['included_rows'], 'Golden roster differs')
    require([i for i, row in enumerate(local) if row[3] is False] == spec['allocation_scope']['excluded_rows'] == [9], 'Absence scope changed')
    cases = []; records = []; before_signature = canonical(inputs_as_hashes(inputs))
    for golden in spec['goldens']:
        index = golden[0]
        for role in manual['roles']:
            args, expected, target = case(spec, local, manual, plans, golden, role)
            original = repr(args)
            after, diagnostic, consumed = transition(*args)
            counts = compare(after, expected, args[1])
            require(diagnostic == args[4] and consumed == frozenset(((args[0], args[2][1]),)), 'Diagnostic/receipt ledger drift')
            require(repr(args) == original, 'Fixture input mutated')
            cases.append((index, role, args, expected))
            values = dict(after); r, key, side = target
            records.append({'local_row': index, 'role': role, **counts, 'target': target,
                            'target_mass_after': values[(r, side, key)], 'target_error_after': values[(r, 'error_sum', -1)],
                            'target_weight_after': values[(r, 'weight_sum', -1)],
                            'asymmetric': local[index][4] != local[index][5],
                            'after_sha256': sha256(canonical(after)).hexdigest()})
    require(len(records) == 30 and sum(r['asymmetric'] for r in records) == 24, 'Fixed baseline budget differs')
    results = []
    for name, implementation, source_sha in mutants(source):
        rejected = []; matched = []
        for index, role, args, expected in cases:
            signature = repr(args)
            try:
                after, diagnostic, consumed = implementation(*args)
                compare(after, expected, args[1])
                require(diagnostic == args[4] and consumed == frozenset(((args[0], args[2][1]),)), 'Mutant metadata differs')
            except ValueError: rejected.append([index, role])
            else: matched.append([index, role])
            require(repr(args) == signature, 'Mutant modified trusted input')
        require(bool(rejected), 'Undetected formula mutant: ' + name)
        if name == 'constant_half':
            require(len(rejected) == 24 and len(matched) == 6, 'Constant-half discrimination failed')
        results.append({'mutant': name, 'source_sha256': source_sha, 'rejected': rejected, 'matched': matched})
    require([r['mutant'] for r in results] == spec['mutants'], 'Mutant roster differs')
    # Comparator/input faults require no additional transition calls.
    index, role, args, expected = cases[0]
    bad = list(expected); address = next(a for a, v in expected if a[1] == 'error_sum')
    bad = tuple((a, nextafter(v, inf) if a == address else v) for a, v in bad)
    mass = next(a for a, v in expected if a[1] == 'mass_left')
    boolean = tuple((a, True if a == mass else v) for a, v in expected)
    changed_inputs = dict(inputs); changed_inputs[LOCAL] += b' '
    operations = [('golden_ulp', lambda: compare(bad, expected, args[1])),
                  ('bool_after_mass', lambda: compare(boolean, expected, args[1])),
                  ('changed_input_bytes', lambda: bind_inputs(spec, changed_inputs)),
                  ('unallocated_row', lambda: case(spec, local, manual, plans, [9, 2, 8.25, 17.0], 'candidate'))]
    faults = []
    for name, operation in operations:
        try: operation()
        except ValueError: faults.append(name)
        else: raise ValueError('Undetected guard fault: ' + name)
    require(faults == spec['faults'], 'Guard fault roster differs')
    require(canonical(inputs_as_hashes(inputs)) == before_signature, 'Saved inputs mutated')
    return {'schema': 'sara-asymmetric-transition-result-v1', 'records': records, 'mutants': results,
            'baseline_transitions': 30, 'asymmetric_baselines': 24, 'symmetric_baselines': 6,
            'changed_fixture_fields': 1080, 'unchanged_fixture_fields': 11970,
            'mutant_transitions': 180, 'faults_detected': faults, 'unallocated_transitions_tested': False,
            'scope': 'allocated_nonlearning_asymmetric_fixture_transitions_only',
            'actual_learning_updates': 0, 'learner_executions': 0, 'task_scores': 0,
            'acquisitions': 0, 'resource_measurements': 0, 'test_opened': False, 'conditions_relaxed': False,
            'runtime_equivalence_established': False, 'runtime_write_provenance_authenticated': False,
            'snn_advantage_established': False, 'promotion_authorized': False}


def inputs_as_hashes(inputs): return {name: sha256(raw).hexdigest() for name, raw in inputs.items()}
