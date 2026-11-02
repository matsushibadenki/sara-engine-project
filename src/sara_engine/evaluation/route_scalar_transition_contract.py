"""Saved-evidence-only specification of an independent transition check."""
from copy import deepcopy
from hashlib import sha256
import json

SELECTION_SHA = 'b79bdd0f95bcb03f7fbdf90e85a85c095d43ba172eb8bca0c4ebd28c7f0716c7'
NUMERIC = 'route_numeric_delta_preflight_selection_v1.json'
SCALAR = 'route_independent_scalar_preflight_result_v1.json'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def analyze(selection_raw, inputs):
    """Inspect literal metadata only; do not compute or apply transitions."""
    if sha256(selection_raw).hexdigest() != SELECTION_SHA:
        raise ValueError('Frozen transition contract differs')
    spec = json.loads(selection_raw)
    if set(inputs) != set(spec['inputs']):
        raise ValueError('Wrong saved input roster')
    if sum(len(raw) for raw in inputs.values()) > spec['bounds']['input_bytes']:
        raise ValueError('Saved input budget exceeded')
    for name, raw in inputs.items():
        if sha256(raw).hexdigest() != spec['inputs'][name]:
            raise ValueError('Saved evidence changed: ' + name)
    numeric, scalar = json.loads(inputs[NUMERIC]), json.loads(inputs[SCALAR])
    fixture = numeric['manual_fixture']
    if len(fixture['rows']) != 12 or scalar['summary']['exact_matches'] != 11 or scalar['summary']['all_exact'] is not True:
        raise ValueError('Required saved evidence absent')
    # Copy literal expectations without deriving any after-state or increment.
    return {
        'schema': 'sara-scalar-transition-contract-result-v1',
        'literal_before': deepcopy(fixture['before']),
        'literal_after_observations': deepcopy(fixture['after_route_observations_by_width']),
        'literal_rows': deepcopy(fixture['rows']),
        'columns': list(fixture['columns']),
        'roles': list(fixture['roles']),
        'requirements': dict(spec['requirements']),
        'evidence': {'stateless_literal_arithmetic': 'checked_in_saved_report',
                     'before_after_literal_availability': 'checked_not_executed',
                     'independent_transition_implementation': 'missing',
                     'runtime_write_provenance': 'missing',
                     'same_role_runtime_scalar_equality': 'missing',
                     'independent_task': 'missing', 'snn_specific_causality': 'missing'},
        'next_step': spec['decision'],
        'new_transition_executions': 0, 'learner_executions': 0,
        'task_scores': 0, 'acquisitions': 0, 'resource_measurements': 0,
        'runtime_equivalence_established': False, 'promotion_authorized': False,
        'test_opened': False, 'conditions_relaxed': False,
    }


def faults(selection_raw, inputs):
    """Reject mutations of frozen contracts and saved proof bindings."""
    spec = json.loads(selection_raw)
    mutations = []
    changed = deepcopy(inputs)
    obj = json.loads(changed[NUMERIC]); obj['manual_fixture']['rows'][0][3] = 4
    changed[NUMERIC] = canonical(obj)
    mutations.append(('changed_row', selection_raw, changed))
    for name in ('missing_requirement', 'relaxed_budget', 'promoted_runtime_equality'):
        obj = deepcopy(spec)
        if name == 'missing_requirement': del obj['requirements']['delay']
        elif name == 'relaxed_budget': obj['bounds']['learner_executions'] = 1
        else: obj['promotion_authorized'] = True
        mutations.append((name, canonical(obj), inputs))
    changed = deepcopy(inputs); changed[SCALAR] += b' '
    mutations.append(('changed_input_binding', selection_raw, changed))
    result = []
    for name, raw, data in mutations:
        try: analyze(raw, data)
        except ValueError: result.append(name)
        else: raise ValueError('Undetected static-contract mutation: ' + name)
    if result != spec['faults']: raise ValueError('Fault roster changed')
    return result
