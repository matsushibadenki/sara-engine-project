"""Read-only cross-record checks on hand authority references; no runtime adapter."""
from copy import deepcopy
from hashlib import sha256
import json

SELECTION_SHA = 'a9e7cbafc6a8ff440d3e262060886c91eb4683e09f8e7fc2136662e651d76820'
BINDING = ('owner', 'receiving_id', 'source_id', 'receipt_revision', 'receipt_digest',
           'delivery_revision', 'delivery_digest', 'diagnostic_digest')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value:
        raise ValueError(message)


def exact(a, b, message):
    require(canonical(a) == canonical(b), message)


def validate(envelope, authority):
    """Validate declarations against a caller-trusted hand reference, without applying."""
    exact(sorted(envelope), ['commit', 'receipt', 'scalar'], 'Record roster differs')
    a = authority
    for name in ('receiving_id', 'source_id', 'receipt_revision', 'delivery_revision', 'successor_revision', 'minimum_revision'):
        require(type(a[name]) is int and a[name] >= 0, 'Invalid reference integer')
    require(a['receipt_revision'] <= a['delivery_revision'] and a['delivery_revision'] >= a['minimum_revision']
            and a['successor_revision'] == a['delivery_revision'] + 1, 'Invalid reference revisions')
    require(a['receipt_pre_outcome'] is True and a['receiving_id'] not in a['consumed_before'], 'Reference capture/consumption differs')
    credits = a['credits']; routes = [c['route'] for c in credits]
    require(len(routes) == len(set(routes)) == 9 and all(type(r) is int and 0 <= r < 45 for r in routes), 'Reference nine-route roster differs')
    for c in credits:
        require(type(c['allocated']) is bool and type(c['delivery_allocated']) is bool, 'Allocation must be explicit Boolean')
        require((c['probability'] is None) == (c['allocated'] is False), 'Absent probability differs')
    receipt = envelope['receipt']; scalar = envelope['scalar']; commit = envelope['commit']
    for record in (receipt, scalar, commit):
        exact(record['authority_ref'], a['ref'], 'Authority reference differs')
    expected_receipt = {k: a[k] for k in ('owner', 'receiving_id', 'source_id', 'receipt_revision', 'receipt_digest', 'diagnostic_digest')}
    expected_receipt.update(authority_ref=a['ref'], pre_outcome=True,
                            credits=[{k:c[k] for k in ('route', 'key', 'allocated', 'probability')} for c in credits])
    exact(receipt, expected_receipt, 'Receipt binding differs')
    binding = {k:a[k] for k in BINDING}
    expected_scalar = dict(binding, authority_ref=a['ref'], error_ref=a['error_ref'],
                           delivery_tick=a['receiving_id'] + 1, producer=a['owner'][5],
                           learning_clock=a['receiving_id'], outcome=[0, 0, 0],
                           credits=[{'route':c['route'], 'key':c['key'], 'receipt_allocated':c['allocated'],
                                     'receipt_probability':c['probability'], 'delivery_allocated':c['delivery_allocated'],
                                     'delivery_probability':c['delivery_probability'], 'fixed_error':c['fixed_error']} for c in credits])
    # Compare preregistered errors only. Never calculate a probability, error or after-state.
    exact(scalar, expected_scalar, 'Scalar input binding differs')
    expected_commit = dict(binding, authority_ref=a['ref'], successor_revision=a['successor_revision'],
                           successor_digest=a['successor_digest'], route_ids=routes,
                           consumed_after=a['consumed_before'] + [a['receiving_id']], pending_removed=True,
                           allocation_in_unit=True, requested_learning_fields=36,
                           declared_changed_learning_fields=36, declared_unchanged_learning_fields=399,
                           publication='all_nine_state_and_consumption', writer_executed=False)
    exact(commit, expected_commit, 'Commit declaration differs')
    return {'binding_valid': True, 'routes': 9, 'application_authorized': False,
            'actual_learning_writes_observed': False, 'authority_authenticated': False,
            'atomicity_established': False}


def mutations(case):
    a = case['authority']; receiver = a['receiving_id']
    changes = [
        ('foreign_authority', 'receipt', 'authority_ref', 'foreign'),
        ('wrong_receiver', 'scalar', 'receiving_id', receiver + 1),
        ('source_as_receiver', 'scalar', 'receiving_id', a['source_id']),
        ('wrong_source', 'scalar', 'source_id', a['source_id'] + 1),
        ('receipt_revision', 'scalar', 'receipt_revision', a['delivery_revision']),
        ('delivery_revision', 'commit', 'delivery_revision', a['receipt_revision']),
        ('stale_watermark', 'authority', 'minimum_revision', a['delivery_revision'] + 1),
        ('receipt_digest', 'receipt', 'receipt_digest', '9'*64),
        ('delivery_digest', 'scalar', 'delivery_digest', '9'*64),
        ('post_outcome', 'receipt', 'pre_outcome', False),
        ('error_reference', 'scalar', 'error_ref', 'foreign'),
        ('scalar_producer', 'scalar', 'producer', 'reference_validator'),
        ('early_delivery', 'scalar', 'delivery_tick', receiver),
        ('source_clock', 'scalar', 'learning_clock', a['source_id']),
        ('partial_routes', 'commit', 'route_ids', case['envelope']['commit']['route_ids'][:-1]),
        ('duplicate_routes', 'commit', 'route_ids', [a['credits'][0]['route']]*9),
        ('reordered_routes', 'commit', 'route_ids', list(reversed(case['envelope']['commit']['route_ids']))),
        ('split_consumption', 'commit', 'consumed_after', []),
        ('pending_retained', 'commit', 'pending_removed', False),
        ('diagnostic_change', 'commit', 'diagnostic_digest', '9'*64),
        ('successor_revision', 'commit', 'successor_revision', a['delivery_revision']),
        ('learning_count', 'commit', 'declared_changed_learning_fields', 35),
        ('writer_claim', 'commit', 'writer_executed', True),
        ('bool_revision', 'scalar', 'delivery_revision', True),
    ]
    for name, section, field, value in changes:
        changed = deepcopy(case)
        target = changed['authority'] if section == 'authority' else changed['envelope'][section]
        target[field] = value
        yield name, changed
    for name, field, value in (
            ('unallocated_as_zero', 'receipt_allocated', True),
            ('receipt_probability_overwritten', 'receipt_probability', 0.75),
            ('delivery_allocation', 'delivery_allocated', False),
            ('fixed_error', 'fixed_error', 0.0625)):
        changed = deepcopy(case)
        changed['envelope']['scalar']['credits'][0][field] = value
        yield name, changed


def analyze(spec, reports):
    exact(sorted(reports), sorted(spec['reports']), 'Report roster differs')
    require(sum(map(len, reports.values())) <= spec['bounds']['input_bytes'], 'Report budget exceeded')
    for name, raw in reports.items():
        require(sha256(raw).hexdigest() == spec['reports'][name], 'Frozen report changed')
    parent = json.loads(reports['route_adapter_spec_result_v1.json'])
    require(parent['launch_authorized'] is False, 'Parent promoted')
    before = canonical(spec['cases']); checks = []
    require(len(spec['cases']) == 6, 'Case roster differs')
    for case in spec['cases']:
        checks.append(validate(case['envelope'], case['authority']))
        require(validate(case['envelope'], case['authority']) == checks[-1], 'Readonly validation differs')
    # Source-as-receiver and source-clock faults require the deliberately distinct shuffled IDs.
    base = next(c for c in spec['cases'] if c['authority']['owner'][-1] == 'shuffled_outcome')
    rejected = []
    for name, changed in mutations(base):
        try:
            validate(changed['envelope'], changed['authority'])
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError('Undetected hand binding fault: ' + name)
    exact(sorted(rejected), sorted(spec['faults']), 'Fault roster differs')
    exact(canonical(spec['cases']).hex(), before.hex(), 'Input modified')
    return {'schema':'route-adapter-envelope-result-v1', 'checks':checks, 'hand_cases':6,
            'readonly_checks':12, 'faults_rejected':sorted(rejected),
            'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'learner_executions':0, 'calculator_executions':0, 'model_storage_executions':0,
            'task_scores':0, 'acquisitions':0, 'resource_measurements':0,
            'runtime_provenance_established':False, 'runtime_atomicity_established':False,
            'independent_runtime_equality_established':False, 'snn_advantage_established':False,
            'conditions_relaxed':False, 'test_opened':False, 'launch_authorized':False,
            'scope':'Hand-reference conformance only; fixed errors and declared counts are not recomputed or observed.'}
