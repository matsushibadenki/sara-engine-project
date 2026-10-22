"""Audit retained aggregate receipts without loading examples or learned state."""
from copy import deepcopy
import math


# Order and membership are fixed before this audit executes. Earlier results are known.
ROSTER = (
    'shared_observer_learning_result_receipt_v1',
    'temporal_boundary_learning_result_receipt_v1',
    'appliances_learning_result_v1',
    'room_local_identification_result_v1',
    'casas_notification_diagnostic_result_v1',
    'casas_180_result_v1',
    'casas_confidence_result_v1',
    'casas_confidence_references_result_v1',
    'casas_evidence_result_v1',
    'casas_pooling_result_v1',
    'traffic_temporal_result_v1',
    'room_temporal_result_v1',
    'casas_count_audit_result_v1',
    'casas_observation_budget_result_v1',
    'casas_vocabulary_budget_result_v1',
    'casas_arrival_preflight_result_v1',
    'casas_roster_screen_result_v1',
    'source_metadata_feasibility_result_v1',
    'sphere_eco_metadata_result_v1',
)

WORK_DIMENSIONS = (
    'input_decode', 'shared_observer_projection', 'active_backend',
    'prediction_readout', 'feedback_enqueue_delivery', 'expiry_maintenance',
    'changed_local_writes', 'live_and_transient_memory', 'deployment_cpu',
    'replay_instrumentation', 'energy',
)


def at(value, pointer):
    """Resolve a required JSON pointer; absent evidence must never become zero."""
    if not pointer.startswith('/'):
        raise ValueError('Expected an absolute JSON pointer')
    for key in pointer[1:].split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('Expected a nonnegative integer counter')
    return value


def parity(arms, write_key, frozen_key=None):
    """Compare actual changed writes, never scheduled/hypothetical updates."""
    if len(arms) < 2:
        raise ValueError('At least two arms are required')
    writes = {name: integer(arm[write_key]) for name, arm in arms.items()}
    frozen = ({name: integer(arm[frozen_key]) for name, arm in arms.items()}
              if frozen_key else None)
    return {'actual_changed_writes': writes,
            'equal_positive_actual_writes': len(set(writes.values())) == 1 and min(writes.values()) > 0,
            'frozen_period_writes': frozen,
            'frozen_period_zero': all(v == 0 for v in frozen.values()) if frozen is not None else None}


def check_seals(value):
    """Inspect summary flags only; do not open the periods they describe."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('test_opened', 'original_development_opened', 'previous_confirmation_opened',
                       'independent_periods_opened', 'sealed_numeric_values_read',
                       'test_unsealing_authorized', 'conditions_relaxed') and item not in (False, 0):
                raise ValueError('Retained seal/condition violation: ' + key)
            check_seals(item)
    elif isinstance(value, list):
        for item in value:
            check_seals(item)


def ledger(receipt, kind):
    """Keep incompatible units and phases separate; missing counters stay unknown."""
    result = {key: {'coverage': 'unreported', 'observations': []} for key in WORK_DIMENSIONS}

    def add(dimension, pointer, unit, scope, coverage='partial'):
        value = at(receipt, pointer)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid work counter: ' + pointer)
        result[dimension]['coverage'] = coverage
        result[dimension]['observations'].append(
            {'pointer': pointer, 'value': value, 'unit': unit, 'scope': scope})

    if kind == 'appliance':
        for arm in receipt['arms']:
            for phase in ('training', 'development'):
                base = '/arms/' + arm + '/' + phase
                scope = arm + ':' + phase
                for dimension, suffix, unit in (
                    ('shared_observer_projection', '/work/history_visits', 'history_visits'),
                    ('active_backend', '/work/neuron_or_analog_steps', 'backend_steps'),
                    ('active_backend', '/work/spikes', 'spikes'),
                    ('prediction_readout', '/prediction_route_visits', 'route_visits'),
                    ('feedback_enqueue_delivery', '/peak_pending', 'peak_pending_receipts'),
                    ('changed_local_writes', '/actual_counter_writes', 'changed_counter_writes'),
                    ('live_and_transient_memory', '/state_bound_bytes', 'conservative_bound_bytes'),
                    ('live_and_transient_memory', '/measured_readout_object_bytes_at_end', 'readout_object_bytes'),
                    ('replay_instrumentation', '/cpu_seconds_including_reference_and_instrumentation', 'cpu_seconds')):
                    add(dimension, base + suffix, unit, scope)
    elif kind in ('comparison', 'room', 'supplement'):
        prefix = '/comparison/arms' if kind == 'comparison' else '/references' if kind == 'supplement' else '/arms'
        for arm, summary in at(receipt, prefix).items():
            for dimension, key, unit in (
                ('changed_local_writes', 'changed_writes', 'changed_counter_writes'),
                ('live_and_transient_memory', 'routes', 'routes'),
                ('feedback_enqueue_delivery', 'peak_pending', 'peak_pending_receipts'),
                ('feedback_enqueue_delivery', 'delivered_outcomes', 'delivered_outcomes')):
                if key in summary:
                    add(dimension, prefix + '/' + arm + '/' + key, unit, arm)
    # A reported scalar or peak is not full stage accounting. No dimension is
    # upgraded to complete without a stage-ownership/measurement contract.
    return result


def audit_learning(name, receipt, receipts):
    row = {'id': name, 'component_scope': None, 'retained_gates': {},
           'write_parity': None, 'scalar_equivalence': None,
           'independent_usefulness_passed': False,
           'snn_predictive_increment_established': False}
    if name == ROSTER[0]:
        row.update(component_scope='Controlled first-order shared-observer local learning',
                   component_identification_passed=receipt['component_learning_passed'],
                   retained_gates={k: receipt[k] for k in ('execution_integrity_passed', 'component_learning_passed', 'backend_increment_passed')})
        kind = 'receipt_only'
    elif name == ROSTER[1]:
        row.update(component_scope='Controlled middle-order and gap local learning',
                   component_identification_passed=receipt['component_learning_passed'],
                   retained_gates={k: receipt[k] for k in ('execution_integrity_passed', 'component_learning_passed', 'strong_reference_superiority')})
        kind = 'receipt_only'
    elif name == ROSTER[2]:
        kind = 'appliance'
        row['component_scope'] = 'Independent prospective appliance comparison'
        row['retained_gates'] = deepcopy(receipt['decision']['gates'])
        arms = {arm: {'writes': r['training']['actual_counter_writes'],
                      'frozen': r['development']['actual_counter_writes']}
                for arm, r in receipt['arms'].items()}
        row['write_parity'] = parity(arms, 'writes', 'frozen')
        row['scalar_equivalence'] = receipt['decision']['gates']['ordinal_gap_equivalence']
        row['independent_usefulness_passed'] = receipt['decision']['passed']
    elif name == ROSTER[3]:
        kind = 'room'
        row.update(component_scope='Post-support training-only posterior identification',
                   component_identification_passed=receipt['learned_same_current_opposite_posterior_groups'] > 0,
                   scalar_equivalence=receipt['scalar_equivalent'],
                   write_parity=parity(receipt['arms'], 'changed_writes'))
    elif name == ROSTER[7]:
        kind = 'supplement'
        row.update(component_scope='HH103 confidence sufficient-reference supplement',
                   retained_gates=deepcopy(receipt['complete_reference_gates']),
                   write_parity=parity(receipt['references'], 'changed_writes', 'confirmation_writes'))
    else:
        kind = 'comparison'
        comparison = receipt['comparison']
        gates = comparison.get('gates', comparison.get('quality_gates'))
        row.update(component_scope={ROSTER[4]: 'Post-support HH102 diagnostic',
                                    ROSTER[5]: 'Independent HH103 comparison',
                                    ROSTER[6]: 'Reused HH103 training confidence diagnostic',
                                    ROSTER[8]: 'HH104 training evidence diagnostic',
                                    ROSTER[9]: 'Reused HH104 training pooling diagnostic'}[name],
                   retained_gates=deepcopy(gates),
                   scalar_equivalence=gates['scalar_equivalence'])
        frozen_key = 'development_writes' if name in (ROSTER[4], ROSTER[5]) else 'confirmation_writes'
        arms = deepcopy(comparison['arms'])
        if name == ROSTER[6]:
            supplement = receipts[ROSTER[7]]
            if not supplement['candidate_scores_unchanged']:
                raise ValueError('Confidence candidate scores changed')
            if set(arms) & set(supplement['references']):
                raise ValueError('Reference supplement duplicates an arm')
            arms.update(supplement['references'])
            row['retained_gates'] = deepcopy(supplement['complete_reference_gates'])
        row['write_parity'] = parity(arms, 'changed_writes', frozen_key)
        row['independent_usefulness_passed'] = comparison['quality_passed'] if name == ROSTER[5] else False
        if name == ROSTER[9]:
            row['retained_feasibility_gates'] = deepcopy(comparison['feasibility_gates'])
    row['work_ledger'] = ledger(receipt, kind)
    row['complete_work_accounting'] = all(v['coverage'] == 'complete' for v in row['work_ledger'].values())
    row['resource_advantage_established'] = False
    row['interpretation'] = (
        'Retain the original scope and gates. Shared temporal information/local updates are identifiable '
        'components, not evidence of an extra predictive effect over a sufficient scalar/history control. '
        'Scalar equivalence does not rule out a separately measured implementation cost benefit.')
    return row


def evaluate(selection, receipts):
    if tuple(selection['roster']) != ROSTER or tuple(receipts) != ROSTER:
        raise ValueError('Fixed receipt roster/order changed')
    if selection['budgets'] != {'receipt_files': 19, 'per_receipt_bytes': 65536, 'aggregate_receipt_bytes': 1048576}:
        raise ValueError('Fixed audit budget changed')
    for receipt in receipts.values():
        check_seals(receipt)
    # This is a retrospective audit of known outcomes, not a general classifier
    # for future successful studies. New outcomes require a new protocol.
    expected = (
        (at(receipts[ROSTER[0]], '/component_learning_passed'), True),
        (at(receipts[ROSTER[0]], '/backend_increment_passed'), False),
        (at(receipts[ROSTER[1]], '/component_learning_passed'), False),
        (at(receipts[ROSTER[1]], '/strong_reference_superiority'), False),
        (at(receipts[ROSTER[2]], '/decision/passed'), False),
        (at(receipts[ROSTER[4]], '/comparison/quality_passed'), False),
        (at(receipts[ROSTER[5]], '/comparison/quality_passed'), False),
        (at(receipts[ROSTER[6]], '/comparison/passed'), False),
        (at(receipts[ROSTER[7]], '/passed'), False),
        (at(receipts[ROSTER[8]], '/comparison/passed'), False),
        (at(receipts[ROSTER[9]], '/comparison/passed'), False),
    )
    if any(type(value) is not bool or value is not wanted for value, wanted in expected):
        raise ValueError('Known retained outcome changed; a new audit protocol is required')
    rows = [audit_learning(name, receipts[name], receipts) for name in ROSTER[:10]]
    matched = [row['write_parity'] for row in rows if row['write_parity'] is not None]
    # These are fixed saved aggregate fields, never new score thresholds.
    retained = {
        'traffic_pre_score_stop': at(receipts[ROSTER[10]], '/decision'),
        'room_pre_score_stop': at(receipts[ROSTER[11]], '/audit/passed'),
        'pool_count_categories': at(receipts[ROSTER[12]], '/arms/pool_evidence/categories'),
        'closed_world_joint_forecast': at(receipts[ROSTER[13]], '/arms/joint_evidence/scenarios/observed_rates'),
        'unrestricted_event_capacity_certified': at(receipts[ROSTER[14]], '/arms/pool_evidence/unrestricted_event_capacity_certified'),
        'hh105_arrival_gate': at(receipts[ROSTER[15]], '/gates/arrival_target'),
        'finite_household_screen': at(receipts[ROSTER[16]], '/decision'),
        'public_metadata_decision': at(receipts[ROSTER[17]], '/decision'),
        'sphere_eco_decision': at(receipts[ROSTER[18]], '/decision'),
    }
    return {
        'schema': 'sara-r2-research-decision-v1', 'receipt_count': len(receipts),
        'reported_write_parity_passed': all(p['equal_positive_actual_writes'] and p['frozen_period_zero'] is not False for p in matched),
        'learning_rows': rows, 'retained_support_and_acquisition_limits': deepcopy(retained),
        'decision': 'no_existing_receipt_licenses_snn_predictive_or_resource_advantage',
        'identifiable_component': 'Bounded causal history plus local delayed readout can identify controlled temporal contrasts; existing sufficient history/scalar controls recover that information.',
        'remaining_hypothesis': 'A sparse asynchronous SNN implementation may reduce complete deployment work at equal useful predictions versus an equally event-driven scalar implementation. This is unmeasured, not ANN-exclusive.',
        'required_work_contract': {
            'dimensions': list(WORK_DIMENSIONS),
            'ownership': 'Every native input, observer update/projection, prediction, delayed receipt, timer expiry and maintenance action has one stage/arm owner. Charge shared stages to every arm; report physical shared execution separately.',
            'units': 'Keep source-level visits, writes, bytes, CPU seconds and joules separate. Do not add heterogeneous counts or substitute wire bytes/routes/peak queue length for memory/total work.',
            'comparator': 'Same causal native input and shared current/history access; sufficient history and event-driven scalar emulator with identical receipt/expiry scheduling. No compulsory dense polling or free reference/observer computation.',
            'writes': 'Compare positive actual changed local writes, including every changed counter; count attempts, allocations, queue/state/metadata writes separately. Freeze evaluation learning.',
            'measurement': 'Separate deployment from data generation, reference checks, hashing, replay and instrumentation. Report idle baseline, wall/CPU time, live/transient memory and hardware/sampling scope. Energy remains pending independent measurement.',
            'gate': 'Quality/source/causal/state/write gates remain prerequisites for useful-system benefit; synthetic work feasibility alone never promotes runtime or establishes natural-data usefulness.',
        },
        'next_step': 'Preregister an input-only synthetic full-work-accounting feasibility contract using existing causal witnesses; validate stage ownership, idle gaps, delayed feedback and an event-driven scalar comparator. No new scoring or optimization.',
        'runtime_promotion_authorized': False, 'r3_optimization_authorized': False,
        'new_acquisition_or_search_authorized': False, 'seal_opening_authorized': False,
        'audit_learning_writes': 0, 'new_predictions_scored': 0,
        'raw_or_model_payload_reads': 0,
    }
