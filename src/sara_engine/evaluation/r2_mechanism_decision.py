"""Retrospective receipt-only decision gate; never execute a model or probe."""
from copy import deepcopy
from hashlib import sha256
import json

from sara_engine.evaluation import r2_research_decision as prior


ROSTER = (*prior.ROSTER, 'r2_research_decision_result_v1', 'event_work_result_v1',
          'event_work_separation_result_v1', 'event_allocation_failure_v1',
          'allocation_report_result_v1', 'event_allocation_v2_summary_v1',
          'event_allocation_v2_replay_v1')
AUTH_FLAGS = {'test_opened','original_development_opened','previous_confirmation_opened',
              'independent_periods_opened','independent_predictive_information_opened',
              'sealed_numeric_values_read','test_unsealing_authorized','test_opening_authorized',
              'seal_opening_authorized','conditions_relaxed','runtime_promotion_authorized',
              'production_loading_authorized','promotion_authorized'}
RESOURCE_PREREQUISITES = ('fresh_independent_task_qualified', 'independent_usefulness',
                          'causal_mechanism_effect_with_matched_controls',
                          'positive_actual_write_and_state_fairness', 'decision_changing_resource_question')


def triage(predicates):
    """Unknown is not false evidence and never permits another measurement."""
    if set(predicates) != set(RESOURCE_PREREQUISITES):
        raise ValueError('Missing or unexpected decision prerequisite')
    if any(v is not None and type(v) is not bool for v in predicates.values()):
        raise ValueError('Prerequisites require bool or explicit unknown')
    return {'status': 'eligible_for_separate_protocol' if all(v is True for v in predicates.values()) else 'hold',
            'failed': [k for k in RESOURCE_PREREQUISITES if predicates[k] is False],
            'unknown': [k for k in RESOURCE_PREREQUISITES if predicates[k] is None],
            'measurement_authorized': False}


def check_seals(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in AUTH_FLAGS and not (type(item) in (bool,int) and item == 0):
                raise ValueError('Retained seal/authorization violation: '+key)
            check_seals(item)
    elif isinstance(value, list):
        for item in value: check_seals(item)


def flag(value, expected=None):
    if type(value) is not bool or (expected is not None and value is not expected):
        raise ValueError('Invalid or changed retained gate')
    return value


def integer(value):
    return prior.integer(value)


def observation(receipts, name, pointer, scope, unit=None):
    value = deepcopy(prior.at(receipts[name], pointer))
    return {'receipt':name,'pointer':pointer,'scope':scope,'unit':unit,'value':value}


def work_reconciliation(receipts):
    work, separated = receipts[ROSTER[20]], receipts[ROSTER[21]]
    flag(work['all_accounting_gates_passed'],True)
    flag(separated['all_separation_gates_passed'],True)
    for value in (work,separated):
        if integer(value['case_count']) != 48 or len(value['cases']) != 48:
            raise ValueError('Fixed measurement roster changed')
        if integer(value['actual_learning_writes']) or integer(value['predictions_scored']):
            raise ValueError('Input-only evidence contains learning/scoring')
    totals = {'shared':{},'neuron':{},'scalar':{}}
    boundary_calls = 0
    identities = []
    for old,new in zip(work['cases'],separated['cases']):
        identity = tuple(old[k] for k in ('family','member','scenario','horizon'))
        if identity != tuple(new[k] for k in ('family','member','scenario','horizon')):
            raise ValueError('Measurement case identities differ')
        identities.append(identity)
        flag(old['gates_passed'],True)
        flag(new['gates_passed'],True)
        flag(new['bare_matches_diagnostic'],True)
        flag(new['reference_matches_every_boundary'],True)
        if old['semantic_trace_sha256'] != new['semantic_trace_sha256']:
            raise ValueError('Receipt hashes differ across measurement contracts')
        for owner,metrics in new['service_volumes'].items():
            for key,count in metrics.items():
                if integer(count) != integer(old['physical'][owner][key]):
                    raise ValueError('Saved service counters disagree')
                totals[owner][key] = totals[owner].get(key,0)+count
        for arm in ('neuron','scalar'):
            for key,value in old['charged'][arm].items():
                if integer(value) != integer(old['physical']['shared'][key])+integer(old['physical'][arm][key]):
                    raise ValueError('Shared charge omitted or doubled')
            for key,value in new['charged_service_volumes'][arm].items():
                if integer(value) != new['service_volumes']['shared'].get(key,0)+new['service_volumes'][arm].get(key,0):
                    raise ValueError('Separated shared charge disagrees')
        boundary_calls += integer(new['boundary_calls'])
    expected={(f,m,s,h) for f in ('first_order','middle_order','gap_order') for m in (0,1)
              for s in ('on_time','late','missing','mixed') for h in (16,1000000000)}
    if len(set(identities)) != 48 or set(identities) != expected:
        raise ValueError('Finite measurement identities changed')
    summary, replay = receipts[ROSTER[24]], receipts[ROSTER[25]]
    for key,value in summary.items():
        if key != 'replication' and replay[key] != value:
            raise ValueError('Primary/replication aggregate evidence disagrees')
    if integer(summary['case_count']) != 48 or integer(summary['workers_per_phase']) != 144:
        raise ValueError('Allocation roster changed')
    for key in ('actual_learning_writes','predictions_scored'):
        if integer(summary[key]):raise ValueError('Allocation diagnostic contains learning/scoring')
    flag(summary['snn_resource_advantage_established'],False)
    flag(summary['independent_usefulness_established'],False)
    for key in ('cpu_seconds','energy_joules','rss_bytes','total_transient_peak_bytes'):
        if summary[key] is not None:
            raise ValueError('Diagnostic scope changed; no deployment measurement is registered')
    comparison = replay['replication']
    flag(comparison['semantic_and_ownership_exact_match'],True)
    if integer(comparison['compared_windows']) != sum(integer(v['recorded_calls']) for v in summary['modes'].values()):
        raise ValueError('Replication window count does not reconcile')
    if integer(comparison['byte_identical_windows']) != comparison['compared_windows']:
        raise ValueError('Known raw-window replication outcome changed')
    mapping={'created':'probe.create','delivered':'feedback.delivered','cancelled':'feedback.cancelled',
             'expired':'expiry.expired','stale_timer':'expiry.stale','rejected_result':'feedback.rejected'}
    for mode,value in summary['modes'].items():
        if mode not in ('paired','neuron','scalar'):raise ValueError('Unexpected measurement mode')
        if integer(value['workers']) != 48 or integer(value['recorded_calls']) != boundary_calls+96:
            raise ValueError('Recorded lifecycle calls disagree')
        if integer(value['shared_input_calls']) != totals['shared']['input.decode']:
            raise ValueError('Standalone shared observation was omitted')
        arms=('neuron','scalar') if mode=='paired' else (mode,)
        if set(value['service_receipts']) != set(arms):raise ValueError('Comparator arms changed')
        for arm in arms:
            if any(integer(value['service_receipts'][arm][k]) != totals[arm][metric] for k,metric in mapping.items()):
                raise ValueError('Allocation receipt lifecycle disagrees')
        for size in value['max_new_traced_bytes'].values():integer(size)
        for size in value['max_boundary_graph_bytes'].values():integer(size)
    if set(summary['modes']) != {'paired','neuron','scalar'}:
        raise ValueError('Missing sufficient scalar comparator')
    failure, preflight = receipts[ROSTER[22]], receipts[ROSTER[23]]
    flag(failure['all_allocation_gates_passed'],False)
    if failure['status']!='failed_result_serialization_budget' or integer(failure['saved_worker_measurements'])!=0:
        raise ValueError('V1 failure was erased or reinterpreted')
    if failure['measured_allocation_statistics'] is not None or flag(failure['semantic_replay_verified']) is not False:
        raise ValueError('Unsaved v1 measurements became evidence')
    flag(preflight['all_report_preflight_gates_passed'],True)
    flag(preflight['constructed_report_is_measurement'],False)
    proof=preflight['byte_proof']
    if integer(proof['proved_upper_bound_bytes']) >= integer(proof['budget_bytes']):
        raise ValueError('Report representation bound failed')
    return {'aggregate_service_volumes':totals,'source':{'receipt':ROSTER[20],'pointer':'/cases',
            'aggregation':'sum identical typed metrics; no heterogeneous cost sum'},
            'state_and_receipt_contracts_reconciled':True,'allocation_replication_reconciled':True,
            'per_mode_calls':boundary_calls+96,'replicated_windows':comparison['compared_windows'],
            'v1_failure_retained':True,'report_preflight_is_resource_measurement':False,
            'deployment_cpu':None,'physical_energy':None,'total_process_transient_peak':None,
            'new_call_peak_observations':observation(receipts,ROSTER[24],'/modes',
                'Input-only instrumented per-call/new-block and boundary-graph maxima, not full deployment cost'),
            'neuron_not_smaller_in_selected_peak_and_graph_samples':(
                summary['modes']['neuron']['max_new_traced_bytes']['new_peak_bytes'] >= summary['modes']['scalar']['max_new_traced_bytes']['new_peak_bytes']
                and summary['modes']['neuron']['max_boundary_graph_bytes']['union_bytes'] >= summary['modes']['scalar']['max_boundary_graph_bytes']['union_bytes'])}


def evaluate(selection, receipts):
    if tuple(selection['roster'])!=ROSTER or tuple(receipts)!=ROSTER:
        raise ValueError('Registered finite receipt roster/order changed')
    task_digest=sha256(json.dumps(selection['fixed_original_task'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if task_digest!='f49f7f544c038018576b855baffa01b5f5ba3a9d24501217bfd2e71a055613f3':
        raise ValueError('Original task/current/evidence conditions changed')
    for receipt in receipts.values():check_seals(receipt)
    originals={name:receipts[name] for name in prior.ROSTER}
    rebuilt=prior.evaluate({'roster':list(prior.ROSTER),'budgets':{'receipt_files':19,'per_receipt_bytes':65536,'aggregate_receipt_bytes':1048576}},originals)
    historical=receipts[ROSTER[19]]
    if any(historical[key]!=value for key,value in rebuilt.items()):
        raise ValueError('Original predictive/mechanism audit no longer reconstructs')
    if not rebuilt['reported_write_parity_passed']:
        raise ValueError('Historical actual changed-write parity failed')
    learning=[]
    for index,row in enumerate(rebuilt['learning_rows']):
        learning.append({k:deepcopy(v) for k,v in row.items() if k not in ('work_ledger','interpretation')})
        learning[-1]['provenance']={'receipt':ROSTER[19],'pointer':'/learning_rows/'+str(index),
                                    'reconstructed_from_original_receipts':True}
    resource=work_reconciliation(receipts)
    # Independent efficacy is drawn only from original independent comparisons.
    independent=any(row['independent_usefulness_passed'] for row in learning)
    predicates={'fresh_independent_task_qualified':None,'independent_usefulness':independent,
                'causal_mechanism_effect_with_matched_controls':None,
                'positive_actual_write_and_state_fairness':None,'decision_changing_resource_question':None}
    decision=triage(predicates)
    sources={
        'finite_unused_household_survivors':observation(receipts,prior.ROSTER[16],'/survivors','Fixed exhausted roster, not all possible sources'),
        'public_acquisitions_approved':observation(receipts,prior.ROSTER[17],'/approved_payload_acquisitions','No ready source from the fixed review'),
        'sphere_eco_acquisitions_approved':observation(receipts,prior.ROSTER[18],'/approved_payload_acquisitions','Documented streams fail unchanged target semantics'),
        'hh105_arrival_gate':observation(receipts,prior.ROSTER[15],'/gates/arrival_target','Consumed training-only input preflight'),
        'source_dates_for_180_fit':observation(receipts,prior.ROSTER[17],'/minimum_valid_source_dates_for_180_fit_dates','Original fit-split prerequisite','recorded dates')}
    if sources['source_dates_for_180_fit']['value']!=270:
        raise ValueError('Original fit-date prerequisite changed')
    if any(sources[k]['value']!=[] for k in ('finite_unused_household_survivors','public_acquisitions_approved','sphere_eco_acquisitions_approved')):
        raise ValueError('Known finite source decision changed; a new protocol is required')
    hypotheses=deepcopy(selection['hypotheses'])
    if [h['id'] for h in hypotheses]!=['shared_temporal_information','arrival_indexed_lif_retuning','event_driven_scheduling_only','branch_local_delayed_eligibility']:
        raise ValueError('Finite mechanism candidate roster changed')
    candidate=hypotheses[-1]
    candidate.update(selection_status='conditional_design_candidate_only',increment_established_by_selected_receipts=None,
                     independently_qualified_task=None,experiment_authorized=False)
    hypotheses[0].update(status='retained_component_not_snn_increment', evidence=[
        observation(receipts,prior.ROSTER[0],'/component_learning_passed','Controlled shared-observer learning'),
        observation(receipts,prior.ROSTER[0],'/backend_increment_passed','Fixed LIF increment gate')])
    hypotheses[1].update(status='closed_known_formulation', evidence=[
        observation(receipts,prior.ROSTER[1],'/strong_reference_superiority','Joint middle-order/gap comparison'),
        observation(receipts,prior.ROSTER[1],'/decision','Preserve joint failure and reported gap-only positive scope')])
    hypotheses[2].update(status='shared_implementation_opportunity', evidence=[
        observation(receipts,ROSTER[24],'/modes/scalar/shared_input_calls','Equally event-driven scalar input service','calls')])
    return {'schema':'sara-r2-mechanism-decision-result-v1','receipt_count':len(receipts),
        'historical_prediction_audit_exactly_reconstructed':True,'learning_evidence':learning,
        'historical_write_parity_passed':True,'historical_parity_scope':'Per-comparison positive actual writes; overlapping confidence supplement rows are not summed as independent experiments.',
        'measurement_evidence':resource,'resource_prerequisites':predicates,'additional_resource_work':decision,
        'prerequisite_scope':'Independent usefulness is false for the retained comparisons. Fresh-task/candidate causal effect and fairness are unknown; the untested candidate is not labeled a failed experiment.',
        'independent_task':{'selected_source':None,'readiness':'not_established_by_selected_receipts',
            'fixed_definition':deepcopy(selection['fixed_original_task']),'source_constraints':sources,
            'required_conditions':['fresh identity/fit support qualified before labels or scores',
                'native target-compatible source, 180-day minimum and original fit/development split; 270 dates for 180 fit dates',
                'at least 70212 eligible arrivals within fixed acquisition/state budgets; unknown events retained',
                'exact-current/time recurrence plus sufficient opposing-history, both-class and date support under unchanged evidence rules',
                'native 300-600 second delayed outcomes; absent outcomes are not negative; no relabeling or backdating',
                'all original quality/gap/state/positive changed-write/frozen-period gates, test sealed',
                'matching positive actual writes must be possible without dummy/no-op/duplicate updates or private history']},
        'mechanism_candidates':hypotheses,
        'decision':'hold_additional_resource_work_until_causal_mechanism_and_independent_task_prerequisites',
        'new_measurements':0,'new_predictions_scored':0,'audit_learning_writes':0,'new_acquisitions':0,
        'raw_example_or_model_semantic_reads':0,'diagnostic_worker_launches':0,
        'runtime_promotion_authorized':False,'new_resource_measurement_authorized':False,
        'new_acquisition_or_search_authorized':False,'seal_opening_authorized':False,'r3_optimization_authorized':False,
        'next_step':'Prerequisite-only mechanism/task design or external compatible-source evidence; no implementation/measurement restart without a separately registered qualifying contract. Keep synthetic measurement feasibility closed.'}
