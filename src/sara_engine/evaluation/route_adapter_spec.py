"""Static adapter specification checking only; no callable model/storage adapter."""
from copy import deepcopy
from hashlib import sha256
import json

SELECTION_SHA = 'f1f603f0f3e560d456f748578821cebd3f5375971a7dc57ebf8d253762c8cc29'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(contract, frozen):
    """Reject declarative drift, including type changes; does not prove implementation."""
    require(type(contract) is dict, 'Contract must be a JSON object')
    require(canonical(contract) == canonical(frozen), 'Frozen connection contract differs')
    fields = contract['scalar_inputs']
    require(len(fields) == len(set(fields)), 'Duplicate scalar field')
    require(set(contract['receipt_fields']) <= set(fields), 'Receipt-to-scalar binding incomplete')
    require(len(contract['stages']) == len(set(contract['stages'])) == 8, 'Stage ordering incomplete')
    require(len(contract['commit']['unit']) == len(set(contract['commit']['unit'])) == 6, 'Commit publication unit incomplete')
    require(all(type(v) is bool and v is False for v in contract['gates'].values()), 'Static specification promoted')
    return {'declarative_valid': True, 'live_adapter_executed': False}


def mutations(contract):
    changes = [
        ('predecessor_untrusted', 'authority', 'predecessor', 'caller_claim'),
        ('receipt_after_outcome', 'authority', 'receipt', 'capture_after_outcome'),
        ('identity_collapsed', 'authority', 'identity', ['owner', 'source_id']),
        ('source_clock', 'scalar', 'learning_clock', 'source_id'),
        ('probability_recomputed', 'scalar', 'probability', 'delivery_counts'),
        ('allocation_missing', None, 'scalar_inputs', [x for x in contract['scalar_inputs'] if x != 'allocated']),
        ('scalar_shared_producer', 'scalar', 'producer', 'local_sparse_reference'),
        ('oracle_producer', 'scalar', 'reference', 'after_state_producer'),
        ('partial_unit', 'commit', 'unit', ['one_route_learning_fields']),
        ('split_consumption', 'commit', 'unit', [x for x in contract['commit']['unit'] if x != 'receiving_id_consumption']),
        ('unconditional_commit', 'commit', 'precondition', 'none'),
        ('retry_unknown', 'commit', 'unknown_recovery', 'automatic_retry'),
        ('readonly_writer', 'commit', 'readonly_recovery', 'apply'),
        ('diagnostic_mutation', 'commit', 'diagnostic', 'update_with_training'),
        ('role_mismatch', 'comparison', 'same_role', 'candidate_vs_permanent'),
        ('update_budget', 'comparison', 'updates', 'requested_targets_only'),
        ('adaptive_promotion', 'comparison', 'original_adaptive', 'passed'),
        ('launch_enabled', 'gates', 'live_launch', True),
    ]
    for name, section, field, value in changes:
        changed = deepcopy(contract)
        (changed if section is None else changed[section])[field] = value
        yield name, changed


def analyze(spec, reports):
    require(set(reports) == set(spec['reports']), 'Report roster differs')
    require(sum(map(len, reports.values())) <= spec['bounds']['input_bytes'], 'Input bound exceeded')
    for name, raw in reports.items():
        require(sha256(raw).hexdigest() == spec['reports'][name], 'Frozen report changed')
    parent = json.loads(reports['route_source_evidence_result_v1.json'])
    require(parent['decision'] == 'stop_before_runtime_integration' and parent['launch_authorized'] is False, 'Parent promoted')
    contract = spec['contract']
    baseline = validate(contract, contract)
    rejected = []
    before = canonical(contract)
    for name, changed in mutations(contract):
        try:
            validate(changed, contract)
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError('Undetected declarative fault: ' + name)
    require(rejected == spec['faults'] and canonical(contract) == before, 'Fault roster/input changed')
    return {'schema': 'route-adapter-spec-result-v1', 'contract': contract, 'baseline': baseline,
            'faults_rejected': rejected, 'same_role_pairs': 3, 'stage_count': 8,
            'missing_runtime_evidence': parent['missing_runtime_evidence'],
            'limits': 'Exact declarative conformance only; no runtime authority, transaction, producer independence or arithmetic proof.',
            'next': 'Preregister a nonlearning adapter-envelope interface with hand-authored authority references and receipt/scalar/commit records; validate cross-record identity, revision and completeness without model/storage execution.',
            'learner_executions': 0, 'calculator_executions': 0, 'storage_writes': 0,
            'task_scores': 0, 'acquisitions': 0, 'resource_measurements': 0,
            'conditions_relaxed': False, 'test_opened': False, 'launch_authorized': False,
            'runtime_atomicity_established': False, 'snn_advantage_established': False}
