"""Symbolic contract counterexamples, not new prediction or resource trials."""
from copy import deepcopy
import pytest

from sara_engine.evaluation.local_credit_contract import ARMS, validate, witness
from scripts.eval import local_credit_contract as runner


def test_valid_boundary_proposal_is_pure_and_never_authorizes_execution():
    proposal = witness()
    before = deepcopy(proposal)
    result = validate(proposal)
    assert proposal == before
    assert result['proposed_changed_fields'] == {arm: 4 for arm in ARMS}
    assert result['actual_applied_writes'] == 0
    assert result['independent_task_qualified'] is None
    assert result['causal_effect_established'] is None
    assert not result['experiment_authorized']
    assert not result['resource_measurement_authorized']


@pytest.mark.parametrize('kind', [
    'future_event', 'old_event', 'early_outcome', 'late_outcome', 'before_outcome',
    'late_delivery', 'expiry', 'missing', 'cancelled', 'duplicate_receipt',
    'duplicate_consumed', 'history_cap', 'missing_arm', 'extra_arm', 'observer',
    'slots', 'unequal_slots', 'noop', 'counter_padding', 'duplicate_route',
    'stale_weight', 'stale_counter', 'foreign_receipt', 'private_event',
    'unowned_event', 'unknown_route', 'state_overflow', 'bool_time',
    'bool_weight', 'bool_counter', 'empty_updates', 'excess_writes',
    'unequal_updates', 'emulator', 'extra_field', 'duplicate_membership',
    'slots_omission', 'invalid_digest',
])
def test_unsafe_or_unfair_proposal_is_rejected(kind):
    p = witness()
    arm = p['arms']['current']
    update = arm['updates'][0]
    if kind == 'future_event': p['events']['e0'] = 301
    elif kind == 'old_event': p['events']['e0'] = -1
    elif kind == 'early_outcome': p['receipt']['outcome'] = 599
    elif kind == 'late_outcome': p['receipt']['outcome'] = 901
    elif kind == 'before_outcome': p['receipt']['delivered'] = 599
    elif kind == 'late_delivery': p['receipt']['delivered'] = 901
    elif kind == 'expiry': p['receipt']['expires'] = 899
    elif kind in ('missing', 'cancelled'): p['receipt']['status'] = kind
    elif kind == 'duplicate_receipt': p['consumed'] = ['r0']
    elif kind == 'duplicate_consumed': p['consumed'] = ['old', 'old']
    elif kind == 'history_cap': p['events'].update({f'e{i}': 0 for i in range(9)})
    elif kind == 'missing_arm': del p['arms']['history']
    elif kind == 'extra_arm': p['arms']['extra'] = deepcopy(arm)
    elif kind == 'observer': arm['observer'] = 'b'*64
    elif kind == 'slots': arm['state_slots'] = 4097
    elif kind == 'unequal_slots': arm['state_slots'] = 15
    elif kind == 'noop': update['new_weight'] = 0
    elif kind == 'counter_padding': update['new_observations'] = 2
    elif kind == 'duplicate_route': arm['updates'].append(deepcopy(update))
    elif kind == 'stale_weight': update['old_weight'] = 5
    elif kind == 'stale_counter': update['old_observations'] = 5
    elif kind == 'foreign_receipt': update['receipt'] = 'r1'
    elif kind == 'private_event': arm['routes']['left']['events'] = ['private']
    elif kind == 'unowned_event': update['event'] = 'e1'
    elif kind == 'unknown_route': update['route'] = 'unknown'
    elif kind == 'state_overflow': update['new_weight'] = 1000001
    elif kind == 'bool_time': p['receipt']['created'] = True
    elif kind == 'bool_weight': update['old_weight'] = False
    elif kind == 'bool_counter': update['old_observations'] = False
    elif kind == 'empty_updates': arm['updates'] = []
    elif kind == 'excess_writes': arm['updates'] *= 5
    elif kind == 'unequal_updates': arm['updates'].pop()
    elif kind == 'emulator': p['arms']['scalar_emulator']['updates'][0]['new_weight'] = 2
    elif kind == 'extra_field': update['dummy'] = 0
    elif kind == 'duplicate_membership': arm['routes']['left']['events'] = ['e0', 'e0']
    elif kind == 'slots_omission': arm['state_slots'] = 3
    elif kind == 'invalid_digest': p['receipt']['observer'] = 'x'*64
    with pytest.raises(ValueError): validate(p)


def test_eligibility_contract_does_not_claim_a_learning_rule_or_real_state_measurement():
    p = witness()
    for arm in p['arms'].values():
        arm['updates'][0]['new_weight'] = 17
    result = validate(p)
    assert result['actual_applied_writes'] == 0
    assert 'learning-rule proof' in result['scope']


def test_retained_decision_and_contract_code_pins():
    result = runner.evaluate()
    assert result['prior_resource_prerequisites']['independent_usefulness'] is False
    assert result['independent_task']['selected_source'] is None
    assert not result['test_opened']
    if runner.path('registration').exists(): runner.verify()
    if runner.path('result').exists():
        assert runner.read('result') == result
        with pytest.raises(FileExistsError): runner.main('execute')
