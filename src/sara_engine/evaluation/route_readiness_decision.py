"""Finite saved-report readiness decision; no runtime model/helper execution."""
from hashlib import sha256
import json

SELECTION_SHA = '6ec4b7441a867652c39407f7349aba5d1db8022972b477620367d7d9738e6720'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(spec, reports):
    require(set(reports) == set(spec['reports']) and len(reports) == 3, 'Frozen report roster differs')
    require(sum(map(len, reports.values())) <= spec['bounds']['input_bytes'], 'Report budget exceeded')
    parsed = {}
    for name, raw in reports.items():
        require(sha256(raw).hexdigest() == spec['reports'][name], 'Frozen evidence changed: ' + name)
        parsed[name] = json.loads(raw)
    requirements = spec['requirements']
    require(len(requirements) == 7, 'Runtime requirement roster differs')
    for name, report in parsed.items():
        require(set(report['missing_runtime_evidence']) == set(requirements)
                and len(report['missing_runtime_evidence']) == 7, 'Runtime evidence status conflicts: ' + name)
        for field in ('launch_authorized', 'runtime_atomicity_established', 'snn_advantage_established',
                      'conditions_relaxed', 'test_opened'):
            require(report[field] is False, 'Evidence promoted: ' + name + ':' + field)
        for field in ('learner_executions', 'calculator_executions', 'task_scores', 'acquisitions', 'resource_measurements'):
            require(type(report[field]) is int and report[field] == 0, 'Prohibited execution reported')
    envelope = parsed['route_adapter_envelope_result_v1.json']
    require(envelope['runtime_provenance_established'] is False
            and envelope['independent_runtime_equality_established'] is False
            and envelope['model_storage_executions'] == 0, 'Hand binding promoted')
    matrix = []
    for category, required in requirements.items():
        matrix.append({'category': category, 'status': 'missing', 'needed_evidence': required,
                       'evidence': [{'report': name, 'sha256':spec['reports'][name],
                                     'json_pointer':'/missing_runtime_evidence', 'contains':category}
                                    for name in sorted(parsed)],
                       'closed_by_metadata_success': False})
    return {'schema':'route-readiness-decision-result-v1', 'evidence_matrix':matrix,
            'decision':'hold_runtime_launch_and_scientific_promotion',
            'minimum_next_boundary':spec['minimum_boundary'], 'not_selected':spec['not_selected'],
            'completed_scope':'Source interpretation, declarative contract and hand-reference bindings only.',
            'independent_holds':['task_qualification','original_acceptance_criteria','snn_causal_evidence','resource_work'],
            'learner_executions':0, 'calculator_executions':0, 'model_storage_executions':0,
            'task_scores':0, 'acquisitions':0, 'resource_measurements':0,
            'new_hand_fixtures':0, 'conditions_relaxed':False, 'test_opened':False,
            'launch_authorized':False, 'promotion_authorized':False,
            'scope_limit':'Saved-report assertions only, not independently observed runtime or a proof of global source absence.'}
