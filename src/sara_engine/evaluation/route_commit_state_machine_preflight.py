"""Finite hand-metadata state-machine cases, not numeric/model transactions."""
from dataclasses import replace
from hashlib import sha256
import json
from sara_engine.evaluation.route_commit_state_machine import (
    MetadataState, Pending, Delivery, prepare, commit, classify, inspect_readonly, state_record, fingerprint,
)

SELECTION_SHA = '7d896061c86d1661187ffee599c0a5df3b46c62cc6cd5230a0b15727b64ba5d4'
PROTOCOL = 'route_runtime_protocol_result_v1.json'
PLAN = 'route_independent_transition_selection_v1.json'


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value: raise ValueError(message)


def fixture(spec, role, engine):
    hand = spec['hand_fixture']
    owner = (1201, 'independent', 'changed', 'cumulative', 'uniform', engine, role)
    pre = MetadataState(owner, hand['predecessor_revision'], hand['predecessor_core'],
                        hand['predecessor_allocation'], hand['diagnostic'], frozenset(), None, 0)
    source = hand['shuffled_source'] if role == 'shuffled_outcome' else hand['receiving_id']
    pending = Pending(pre, hand['receiving_id'], source, tuple(hand['routes']),
                      hand['successor_core'], hand['successor_allocation'])
    event = Delivery(owner, hand['receiving_id'], source, hand['delivery_at'])
    # The expected successor uses registered literal values, not the commit helper.
    expected = MetadataState(owner, hand['successor_revision'], hand['successor_core'], hand['successor_allocation'],
                             hand['diagnostic'], frozenset((hand['receiving_id'],)), hand['receiving_id'], 511)
    return pre, pending, event, expected


def observation(spec, owner, row):
    hand = spec['hand_fixture']; receiving = hand['receiving_id']
    return MetadataState(owner, row['revision'], row['core'] * 64, row['allocation'] * 64,
                         hand['diagnostic'], frozenset((receiving,)) if row['consumed'] else frozenset(),
                         receiving if row['last'] else None, row['mask'])


def faults(spec, base):
    pre, pending, event, post = base
    ticket = prepare(pre, pending, event, pending.routes, 10)
    foreign = pre.owner[:-1] + ('permanent',)
    altered_pending = replace(pending, successor_core='9' * 64)
    operations = [
        ('stale_revision', lambda: prepare(replace(pre, revision=9), pending, event, pending.routes, 10)),
        ('stale_core', lambda: prepare(replace(pre, core='9' * 64), pending, event, pending.routes, 10)),
        ('stale_allocation', lambda: prepare(replace(pre, allocation='9' * 64), pending, event, pending.routes, 10)),
        ('stale_diagnostic', lambda: prepare(replace(pre, diagnostic='9' * 64), pending, event, pending.routes, 10)),
        ('partial_routes', lambda: prepare(pre, pending, event, pending.routes[:-1], 10)),
        ('duplicate_routes', lambda: prepare(pre, pending, event, pending.routes[:-1] + pending.routes[:1], 10)),
        ('extra_routes', lambda: prepare(pre, pending, event, pending.routes + (0,), 10)),
        ('reordered_routes', lambda: prepare(pre, pending, event, tuple(reversed(pending.routes)), 10)),
        ('duplicate_commit', lambda: commit(post, pending, ticket, 11)),
        ('duplicate_delivery', lambda: prepare(post, pending, event, pending.routes, 11)),
        ('foreign_owner', lambda: prepare(pre, pending, replace(event, owner=foreign), pending.routes, 10)),
        ('wrong_receiver', lambda: prepare(pre, pending, replace(event, receiving_id=65), pending.routes, 10)),
        ('source_as_receiver', lambda: prepare(pre, pending, replace(event, receiving_id=event.source_id), pending.routes, 10)),
        ('wrong_source', lambda: prepare(pre, pending, replace(event, source_id=(event.source_id + 1) % 512), pending.routes, 10)),
        ('early_delivery', lambda: prepare(pre, pending, replace(event, at=64), pending.routes, 10)),
        ('late_delivery', lambda: prepare(pre, pending, replace(event, at=66), pending.routes, 10)),
        ('bool_revision', lambda: replace(pre, revision=True)),
        ('float_revision', lambda: replace(pre, revision=10.0)),
        ('bool_route', lambda: prepare(pre, pending, event, (True,) + pending.routes[1:], 10)),
        ('forged_predecessor_digest', lambda: replace(ticket, predecessor_digest='9' * 64)),
        ('changed_successor_ticket', lambda: commit(pre, pending, replace(ticket, pending=altered_pending), 10)),
        ('consumed_without_state', lambda: commit(replace(pre, consumed=frozenset((64,)), last_receipt=64, route_mask=511), pending, ticket, 10)),
        ('rollback_below_watermark', lambda: prepare(pre, pending, event, pending.routes, 11)),
        ('bool_watermark', lambda: classify(pre, pending, True)),
    ]
    rejected = []
    for name, operation in operations:
        before = repr((pre, pending, event, post, ticket))
        try: operation()
        except ValueError: rejected.append(name)
        else: raise ValueError('Undetected metadata fault: ' + name)
        require(repr((pre, pending, event, post, ticket)) == before, 'Rejected fault modified fixture state')
    require(rejected == spec['faults'], 'Fault roster changed')
    return rejected


def analyze(spec, inputs):
    require(set(inputs) == set(spec['inputs']) and sum(map(len, inputs.values())) <= spec['bounds']['input_bytes'], 'Input roster/budget differs')
    for name, raw in inputs.items(): require(sha256(raw).hexdigest() == spec['inputs'][name], 'Frozen input changed')
    parent, plan = json.loads(inputs[PROTOCOL]), json.loads(inputs[PLAN])
    require(parent['static_checks']['static_contract_valid'] is True
            and parent['static_checks']['atomicity_tested'] is False and parent['launch_authorized'] is False,
            'Parent runtime evidence promoted')
    frozen = next(p for p in plan['plans'] if p['tick'] == 64)
    require(spec['hand_fixture']['routes'] == [c[0] for c in frozen['credits']]
            and spec['hand_fixture']['shuffled_source'] == frozen['shuffled_source'], 'Old address/source conditions changed')
    records = []; readonly = 0; recovery_count = 0; base = None
    for descriptor in parent['protocol']['cases']:
        role, engine = descriptor['role'], descriptor['engine']
        pre, pending, event, expected = fixture(spec, role, engine)
        before = repr((pre, pending, event))
        a = prepare(pre, pending, event, pending.routes, 10)
        b = prepare(pre, pending, event, pending.routes, 10)
        require(a == b and repr((pre, pending, event)) == before, 'Prepare consumed/changed metadata')
        pre_read = inspect_readonly(pre, pending, 10)
        require(pre_read == inspect_readonly(pre, pending, 10) and pre_read['classification'] == 'predecessor_unconsumed', 'Read-only pre validation differs')
        actual = commit(pre, pending, a, 10)
        require(actual == expected and repr((pre, pending, event)) == before, 'Literal successor differs or predecessor mutated')
        post_read = inspect_readonly(actual, pending, 11)
        require(post_read == inspect_readonly(actual, pending, 11) and post_read['classification'] == 'successor_consumed', 'Read-only post validation differs')
        readonly += 4
        rows = []
        for row in spec['recovery_rows']:
            observed = observation(spec, pre.owner, row)
            fingerprint_before = fingerprint(observed)
            label = classify(observed, pending, row['minimum'])
            require(label == row['expected'] and fingerprint(observed) == fingerprint_before, 'Hand recovery classification differs')
            rows.append({'name': row['name'], 'classification': label})
            recovery_count += 1
        mask_labels = []
        for mask in range(512):
            label = classify(replace(actual, route_mask=mask), pending, 11)
            require(label == ('successor_consumed' if mask == 511 else 'reject_inconsistent'), 'Partial route mask accepted')
            mask_labels.append(label); recovery_count += 1
        records.append({'role': role, 'engine_descriptor': engine, 'predecessor': state_record(pre),
                        'successor': state_record(actual), 'repeat_prepare_equal': True,
                        'readonly_rechecks': 4, 'recovery_rows': rows,
                        'exhaustive_masks': 512, 'accepted_mask': 511,
                        'mask_classifications_sha256': sha256(canonical(mask_labels)).hexdigest(),
                        'runtime_atomicity_tested': False})
        if role == 'shuffled_outcome' and base is None: base = (pre, pending, event, actual)
    require(len(records) == 6 and readonly == 24 and recovery_count == 3126, 'Finite metadata budget differs')
    return {'schema': 'sara-commit-metadata-result-v1', 'records': records,
            'engine_role_cases': 6, 'baseline_metadata_commits': 6, 'readonly_checks': readonly,
            'recovery_classifications': recovery_count, 'faults_detected': faults(spec, base),
            'missing_runtime_evidence': list(parent['missing_runtime_evidence']),
            'scope': 'hand_metadata_reference_state_machine_only', 'learner_executions': 0,
            'numeric_fixture_executions': 0, 'actual_learning_updates': 0, 'task_scores': 0,
            'acquisitions': 0, 'resource_measurements': 0, 'storage_crash_tests': 0,
            'model_atomicity_established': False, 'durable_exactly_once_established': False,
            'independent_runtime_equivalence_established': False, 'snn_advantage_established': False,
            'promotion_authorized': False, 'launch_authorized': False, 'conditions_relaxed': False, 'test_opened': False}
