"""Read-only evidence consistency, conditional mechanism design and resource hold."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from sara_engine.evaluation import r2_mechanism_decision as audit
from scripts.eval import r2_mechanism_decision as runner


@pytest.fixture(scope='module')
def evidence():
    return runner.load_receipts(runner.read('selection'))[0]


def test_retained_audit_closes_measurement_branch_and_keeps_unknowns(evidence):
    result=audit.evaluate(runner.read('selection'),evidence)
    assert result['receipt_count']==26
    assert result['historical_prediction_audit_exactly_reconstructed']
    assert result['historical_write_parity_passed']
    assert result['additional_resource_work']['status']=='hold'
    assert result['resource_prerequisites']['independent_usefulness'] is False
    assert result['resource_prerequisites']['causal_mechanism_effect_with_matched_controls'] is None
    assert result['resource_prerequisites']['positive_actual_write_and_state_fairness'] is None
    assert result['independent_task']['selected_source'] is None
    assert not result['new_resource_measurement_authorized']
    assert result['measurement_evidence']['replicated_windows']==3840
    assert result['measurement_evidence']['deployment_cpu'] is None
    assert result['measurement_evidence']['physical_energy'] is None


def test_conditional_credit_candidate_is_not_a_newly_proven_snn_effect(evidence):
    result=audit.evaluate(runner.read('selection'),evidence)
    candidate=result['mechanism_candidates'][-1]
    assert candidate['id']=='branch_local_delayed_eligibility'
    assert candidate['increment_established_by_selected_receipts'] is None
    assert candidate['independently_qualified_task'] is None
    assert not candidate['experiment_authorized']
    assert any('equation-matched' in c for c in candidate['required_controls'])
    assert any('dummy/no-op/duplicate' in c for c in candidate['constraints'])


def test_component_success_and_joint_failure_are_both_retained(evidence):
    rows=audit.evaluate(runner.read('selection'),evidence)['learning_evidence']
    assert rows[0]['component_identification_passed'] is True
    assert rows[1]['component_identification_passed'] is False
    assert all(row['independent_usefulness_passed'] is False for row in rows)
    # Unknown frozen-period counters stay unknown, not a claimed zero.
    assert rows[3]['write_parity']['frozen_period_zero'] is None
    # Complete confidence references survive, rather than only easy candidate arms.
    assert {'current_fine','histogram'} <= rows[6]['write_parity']['actual_changed_writes'].keys()


@pytest.mark.parametrize('value,status',[(True,'eligible_for_separate_protocol'),(False,'hold'),(None,'hold')])
def test_conjunction_does_not_authorize_measurement_even_when_all_evidenced(value,status):
    predicates={k:True for k in audit.RESOURCE_PREREQUISITES}
    predicates['fresh_independent_task_qualified']=value
    result=audit.triage(predicates)
    assert result['status']==status
    assert not result['measurement_authorized']
    assert ('fresh_independent_task_qualified' in result['unknown'])==(value is None)


@pytest.mark.parametrize('value',[0,1,'true',[],{}])
def test_prerequisites_are_typed_not_truthy(value):
    predicates={k:True for k in audit.RESOURCE_PREREQUISITES}
    predicates['independent_usefulness']=value
    with pytest.raises(ValueError):audit.triage(predicates)


@pytest.mark.parametrize('kind',['parity','quality','seal','source','task','missing','roster'])
def test_changed_predictive_or_source_evidence_requires_new_protocol(evidence,kind):
    receipts=deepcopy(evidence)
    selection=deepcopy(runner.read('selection'))
    if kind=='parity':
        receipts['casas_confidence_references_result_v1']['references']['current_fine']['changed_writes']+=1
    elif kind=='quality':
        receipts['appliances_learning_result_v1']['decision']['passed']=True
    elif kind=='seal':
        receipts['event_allocation_v2_summary_v1']['test_opening_authorized']=True
    elif kind=='source':
        receipts['source_metadata_feasibility_result_v1']['approved_payload_acquisitions']=['unregistered']
    elif kind=='task':
        selection['fixed_original_task']['target_arrivals']=1
    elif kind=='missing':
        del receipts['r2_research_decision_result_v1']['learning_rows']
    else:
        receipts=dict(reversed(list(receipts.items())))
    with pytest.raises((ValueError,KeyError)):audit.evaluate(selection,receipts)


@pytest.mark.parametrize('kind',['shared_charge','service','case','replication','unknown_cpu','v1_failure','promotion'])
def test_measurement_feasibility_cannot_silently_become_advantage(evidence,kind):
    receipts=deepcopy(evidence)
    if kind=='shared_charge':
        receipts['event_work_result_v1']['cases'][0]['charged']['scalar']['observer.ordinal']=0
    elif kind=='service':
        receipts['event_allocation_v2_summary_v1']['modes']['scalar']['shared_input_calls']=0
    elif kind=='case':
        receipts['event_work_separation_result_v1']['cases'][0]['horizon']=99
    elif kind=='replication':
        receipts['event_allocation_v2_replay_v1']['replication']['semantic_and_ownership_exact_match']=False
    elif kind=='unknown_cpu':
        receipts['event_allocation_v2_summary_v1']['cpu_seconds']=0
        receipts['event_allocation_v2_replay_v1']['cpu_seconds']=0
    elif kind=='v1_failure':
        receipts['event_allocation_failure_v1']['measured_allocation_statistics']={'invented':0}
    else:
        for name in ('event_allocation_v2_summary_v1','event_allocation_v2_replay_v1'):
            receipts[name]['snn_resource_advantage_established']=True
    with pytest.raises(ValueError):audit.evaluate(runner.read('selection'),receipts)


def test_a_smaller_hypothetical_sample_still_does_not_unlock_resources(evidence):
    receipts=deepcopy(evidence)
    for name in ('event_allocation_v2_summary_v1','event_allocation_v2_replay_v1'):
        mode=receipts[name]['modes']['neuron']
        mode['max_new_traced_bytes']['new_peak_bytes']=2000
        mode['max_new_traced_bytes']['new_live_at_return_bytes']=2000
        mode['max_by_stage_bytes']['input']['new_peak_bytes']=2000
        mode['max_by_stage_bytes']['input']['new_live_at_return_bytes']=2000
    result=audit.evaluate(runner.read('selection'),receipts)
    assert result['additional_resource_work']['status']=='hold'
    assert not result['new_resource_measurement_authorized']


def test_receipt_loader_rejects_extra_files_and_budget_changes():
    selection=deepcopy(runner.read('selection'))
    selection['roster'].append('event_allocation_v2_result_v1')
    with pytest.raises(ValueError,match='roster'):runner.load_receipts(selection)
    selection=deepcopy(runner.read('selection'))
    selection['budgets']['aggregate_receipt_bytes']*=2
    with pytest.raises(ValueError,match='budget'):runner.load_receipts(selection)


def test_decision_module_has_no_measurement_or_execution_dependencies():
    tree=ast.parse(inspect.getsource(audit))
    imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
    assert set(imports)=={'copy','hashlib','json','sara_engine.evaluation'}
    assert [alias.name for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module=='sara_engine.evaluation'
            for alias in node.names]==['r2_research_decision']


def test_preserved_conditions_and_immutable_audit_result():
    selection=runner.read('selection')
    assert len(selection['preserved'])==386
    old=json.loads(Path(runner.path('selection').parent/'event_allocation_v2_selection_v1.json').read_text())
    assert selection['fixed_original_task']==old['fixed_original_task']
    if not runner.path('registration').exists():pytest.skip('Audit code not yet registered')
    runner.verify()
    if not runner.path('result').exists():pytest.skip('Finite audit not yet consumed')
    result=runner.read('result')
    assert result['new_measurements']==result['new_predictions_scored']==result['audit_learning_writes']==0
    assert not result['new_acquisition_or_search_authorized']
    with pytest.raises(FileExistsError):runner.main('execute')
