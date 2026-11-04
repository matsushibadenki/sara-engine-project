"""Saved-report-only validation of prospective same-role integration schemas."""
from copy import deepcopy
from hashlib import sha256
import json

SELECTION_SHA = '8214d46fc12dc32818eb37abcbf9c8d5039a1366b3b8f3a52a6f73c449cffacf'
REPORT = 'route_receipt_delta_result_v1.json'
CONTROL = 'route_fixed_control_contract_selection_v1.json'
CAPTURE = ['owner', 'receiving_id', 'source_id', 'route_key_ids', 'allocation', 'probability',
           'predecessor_revision', 'predecessor_digest', 'diagnostic_digest']
CASE_KEYS = {'role', 'engine', 'producer_id', 'stages', 'planned_route_updates', 'planned_changed_learning',
             'planned_unchanged_learning', 'planned_metadata_targets', 'metadata_accounting',
             'readback_origin', 'clock_origin', 'capture_schema', 'runtime_evidence'}


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value: raise ValueError(message)


def validate_plan(spec, plan, control):
    """Validate requirements only; no runtime event, commit or numeric update occurs."""
    require(type(plan) is dict and set(plan) == {'requirements', 'cases', 'pairs', 'diagnostic_policy',
            'original_adaptive_pass', 'launch_authorized'}, 'Protocol fields changed')
    require(set(plan['requirements']) == set(spec['required_requirements'])
            and all(value is True for value in plan['requirements'].values()), 'Missing integration requirement')
    require(plan['diagnostic_policy'] == 'isolated', 'Diagnostic may not control learning')
    require(plan['original_adaptive_pass'] is False and plan['launch_authorized'] is False, 'Static protocol cannot promote or launch')
    cases = plan['cases']
    require(type(cases) is list and len(cases) == 6, 'Fixed six case descriptors required')
    expected = {(role, engine) for role in control['roles'] for engine in control['engines']}
    identities = []
    for case in cases:
        require(type(case) is dict and set(case) == CASE_KEYS, 'Case schema changed')
        require((case['role'], case['engine']) in expected, 'Foreign role/engine descriptor')
        identities.append((case['role'], case['engine']))
        require(case['producer_id'] == case['engine'], 'Shared after-state producer declaration')
        require(case['stages'] == spec['stage_order'] and len(case['stages']) == 7, 'Missing, reordered or split commit stage')
        for key, value in (('planned_route_updates', 9), ('planned_changed_learning', 36),
                           ('planned_unchanged_learning', 399), ('planned_metadata_targets', 1)):
            require(type(case[key]) is int and case[key] == value, 'Planned count/type differs')
        require(case['metadata_accounting'] == 'separate' and case['readback_origin'] == 'committed_store'
                and case['clock_origin'] == 'receiving_id', 'Metadata/readback/clock boundary changed')
        require(case['capture_schema'] == CAPTURE, 'Capture must exclude future outcomes and preserve receipt context')
        evidence = case['runtime_evidence']
        require(type(evidence) is dict and set(evidence) == set(spec['runtime_slots'])
                and all(value is None for value in evidence.values()), 'Runtime evidence invented during static review')
    require(len(set(identities)) == 6 and set(identities) == expected, 'Duplicate/missing engine-role case')
    pairs = plan['pairs']; used = []
    require(type(pairs) is list and len(pairs) == 3, 'Three same-role comparison pairs required')
    for pair in pairs:
        require(type(pair) is dict and set(pair) == {'role', 'left', 'right'}, 'Comparison fields differ')
        for key in ('left', 'right'):
            require(type(pair[key]) is int and 0 <= pair[key] < 6, 'Invalid comparison case index')
        left, right = cases[pair['left']], cases[pair['right']]
        require(left['role'] == right['role'] == pair['role'], 'Cross-role comparison')
        require({left['engine'], right['engine']} == set(control['engines']), 'Pair must declare distinct engines')
        require(left['producer_id'] != right['producer_id'], 'Shared after-value producer')
        used.extend((pair['left'], pair['right']))
    require(sorted(used) == list(range(6)), 'Comparison reuses or omits case')
    return {'case_descriptors': 6, 'stage_bindings': 42, 'same_role_pairs': 3,
            'required_boundaries': len(spec['required_requirements']), 'static_contract_valid': True,
            'runtime_provenance_authenticated': False, 'atomicity_tested': False,
            'durable_write_evidence_observed': False, 'independent_runtime_execution_tested': False}


def faults(spec, control):
    original = spec['protocol']; result = []
    def check(name, change):
        candidate = deepcopy(original); change(candidate)
        try: validate_plan(spec, candidate, control)
        except ValueError: result.append(name)
        else: raise ValueError('Undetected protocol fault: ' + name)
    def case_field(key, value): return lambda p: p['cases'][0].__setitem__(key, value)
    def requirement(name): return lambda p: p['requirements'].__setitem__(name, False)
    check('missing_preoutcome_seal', requirement('preoutcome_seal'))
    check('missing_predecessor_authority', requirement('trusted_predecessor_readback'))
    check('source_used_as_clock', case_field('clock_origin', 'source_id'))
    check('duplicate_engine_pair', lambda p: p['cases'][0].__setitem__('engine', p['cases'][1]['engine']))
    check('cross_role_pair', lambda p: p['pairs'][0].__setitem__('right', 3))
    check('partial_credit_batch', case_field('planned_route_updates', 8))
    check('candidate_as_readback', case_field('readback_origin', 'candidate_after'))
    split = list(spec['stage_order']); split[4:5] = ['commit_fields', 'consume_receipt']
    check('split_commit_and_consume', case_field('stages', split))
    check('missing_consumption_binding', requirement('consume_with_state_revision'))
    check('diagnostic_changes_training', lambda p: p.__setitem__('diagnostic_policy', 'changes_training'))
    check('extra_learning_count', case_field('planned_changed_learning', 37))
    check('metadata_merged', case_field('metadata_accounting', 'merged'))
    check('shared_after_producer', lambda p: p['cases'][1].__setitem__('producer_id', p['cases'][0]['producer_id']))
    check('float_learning_count', case_field('planned_changed_learning', 36.0))
    check('bool_metadata_count', case_field('planned_metadata_targets', True))
    check('outcome_in_capture', case_field('capture_schema', CAPTURE + ['outcome_side']))
    stages = list(spec['stage_order']); stages[0], stages[1] = stages[1], stages[0]
    check('reversed_stages', case_field('stages', stages))
    check('missing_stage', case_field('stages', spec['stage_order'][:-1]))
    check('invented_runtime_evidence', lambda p: p['cases'][0]['runtime_evidence'].__setitem__('atomic_commit_and_recovery', 'passed'))
    check('adaptive_promotion', lambda p: p.__setitem__('original_adaptive_pass', True))
    check('automatic_launch', lambda p: p.__setitem__('launch_authorized', True))
    require(result == spec['faults'], 'Fault roster changed')
    return result


def analyze(spec, inputs):
    require(set(inputs) == set(spec['inputs']) and sum(map(len, inputs.values())) <= spec['bounds']['input_bytes'], 'Saved input roster/budget changed')
    for name, raw in inputs.items(): require(sha256(raw).hexdigest() == spec['inputs'][name], 'Saved evidence changed')
    report, control = json.loads(inputs[REPORT]), json.loads(inputs[CONTROL])
    require(report['totals'] == {'all_exact_transactions': 66, 'requested_learning': 2376,
            'changed_learning': 2376, 'unchanged_learning': 26334, 'requested_metadata': 66,
            'changed_metadata': 33, 'unallocated_receipt_transactions': 6}, 'Saved arithmetic scope differs')
    require(report['actual_learning_updates'] == report['learner_executions'] == 0
            and report['runtime_provenance_authenticated'] is False
            and report['runtime_equivalence_established'] is False, 'Saved fixture evidence was promoted')
    policy = control['endpoint_policy']
    require(policy['gain_over_permanent']['fixed_readout'] == 'structural_zero', 'Original applicability changed')
    require(all(row['original_adaptive_system'] == 'untested' for row in policy.values() if type(row) is dict), 'Adaptive endpoint promoted')
    before = canonical(spec['protocol'])
    checked = validate_plan(spec, spec['protocol'], control)
    rejected = faults(spec, control)
    require(canonical(spec['protocol']) == before, 'Static faults mutated original protocol')
    references = [{'role': role, 'source_local_row': 0, 'source_branch': 'stable_view',
                   'before_sha256': next(r['before_sha256'] for r in report['records'] if r['local_row'] == 0 and r['branch'] == 'stable_view' and r['role'] == role)}
                  for role in control['roles']]
    return {'schema': 'sara-runtime-integration-protocol-result-v1', 'static_checks': checked,
            'fixture_reference_scope': references, 'protocol': deepcopy(spec['protocol']),
            'atomic_rule': spec['atomic_rule'], 'independence_rule': spec['independence_rule'],
            'endpoint_policy_preserved': deepcopy(policy), 'faults_detected': rejected,
            'missing_runtime_evidence': list(spec['runtime_slots']),
            'next_step': 'preregister bounded nonlearning commit/consume metadata state-machine preflight; no live model run',
            'scope': 'saved_reports_and_static_schema_only', 'new_learner_executions': 0,
            'new_fixture_calculations': 0, 'task_scores': 0, 'acquisitions': 0,
            'resource_measurements': 0, 'launch_authorized': False, 'promotion_authorized': False,
            'conditions_relaxed': False, 'test_opened': False}
