"""Finite saved-proof classification only; no fixture/calculator/model imports."""
import json

INPUTS=(
 'route_fixed_allocation_contract_result_v1.json','route_fixed_control_contract_selection_v1.json',
 'route_fixed_control_contract_result_v1.json','route_interface_preflight_result_v1.json',
 'route_delta_ledger_preflight_result_v1.json','route_numeric_delta_preflight_result_v1.json',
 'route_error_binding_preflight_result_v1.json','route_two_phase_preflight_result_v1.json',
 'route_memory_factorial_v4_primary_v1.json','route_memory_factorial_v4_result_v1.json',
 'sparse_route_reactivation_failure_v1.json','r2_mechanism_decision_result_v1.json',
 'r2_research_decision_result_v1.json','sphere_eco_metadata_result_v1.json')


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def pointer(document,path):
    try:
        current=document
        for raw in path.split('/')[1:]:
            key=raw.replace('~1','/').replace('~0','~')
            current=current[int(key)] if type(current) is list else current[key]
        return current
    except (KeyError,IndexError,ValueError,TypeError) as exc:raise ValueError('Required saved evidence absent: '+path) from exc


def evidence(reports,file,path,expected,scope):
    value=pointer(reports[file],path)
    require(canonical(value)==canonical(expected),'Saved evidence changed: '+file+path)
    return {'file':file,'pointer':path,'value':value,'scope':scope}


def analyze(spec,reports,hashes):
    require(spec['schema']=='sara-route-readiness-audit-selection-v1','Wrong readiness protocol')
    require(tuple(spec['inputs'])==INPUTS and set(reports)==set(INPUTS),'Finite input roster changed')
    require(spec['budgets']=={'input_files':14,'per_input_bytes':8388608,'all_input_bytes':8388608,
        'result_bytes':262144,'all_artifact_bytes':2097152,'new_learner_executions':0,'new_prediction_scores':0,
        'new_acquisitions':0,'new_measurements':0},'Readiness budget changed')
    require(spec['categories']==['fixture_api_invariants','runtime_provenance','independent_route_scalar_computation',
        'independent_task','snn_specific_causal_evidence','resource_readiness'],'Evidence categories changed')
    proofs=[]
    fixture_files=(INPUTS[0],INPUTS[2],*INPUTS[3:8])
    for file in fixture_files:
        for key in ('learner_executions','actual_learning_updates','prediction_scores','resource_measurements'):
            proofs.append(evidence(reports,file,'/status/'+key,0,'saved fixture phase, not actual runtime validation'))
        for key in ('test_opened','execution_authorized','promotion_authorized'):
            proofs.append(evidence(reports,file,'/status/'+key,False,'no promotion or new execution'))
    verified_specs=[(INPUTS[0],'/address_plan/planned_route_updates',9216),
       (INPUTS[2],'/contract_valid',True),(INPUTS[3],'/summary/receipt_metadata_pairs',15360),
       (INPUTS[3],'/summary/comparison_fixture_pairs',240),
       (INPUTS[4],'/summary/symbolic_transactions',144),
       (INPUTS[4],'/summary/symbolic_stamp_changes',5184),
       (INPUTS[5],'/summary/numeric_fixture_transactions',72),
       (INPUTS[5],'/summary/active_field_checks',2592),
       (INPUTS[6],'/summary/route_error_bindings',648),
       (INPUTS[7],'/summary/route_bindings',648),(INPUTS[7],'/summary/faults_rejected',26)]
    verified=[evidence(reports,f,p,v,'verified only in declared saved fixture/API scope') for f,p,v in verified_specs]
    failures=[]
    for file in INPUTS[3:8]:
        proofs.append(evidence(reports,file,'/status/runtime_scalar_equality_tested',False,'route computational independence not tested'))
    proofs.append(evidence(reports,INPUTS[7],'/status/runtime_causal_provenance_authenticated',False,'API order does not authenticate real observations'))
    control=reports[INPUTS[2]]
    require(len(control['endpoint_matrix'])==30,'Endpoint roster incomplete')
    require(all(r['value'] is None and r['original_adaptive_pass'] is False for r in control['endpoint_matrix']),'Fixture endpoint promoted to adaptive acceptance')
    require(all(r['status']=='untested' for r in control['endpoint_matrix'] if r['channel']=='original_adaptive_system'),'Original adaptive endpoint reclassified')
    proofs.append(evidence(reports,INPUTS[2],'/structural_identifiability/primary_gain_over_permanent',0,'fixed primary gain is structurally zero'))
    proofs.append(evidence(reports,INPUTS[1],'/original_gates/changed/recovery_max_ticks',256,'original gate unchanged'))
    proofs.append(evidence(reports,INPUTS[1],'/original_gates/changed/tail_gain_over_permanent_min',.10,'fixed primary cannot establish original gain gate'))
    primary,proof=reports[INPUTS[8]],reports[INPUTS[9]]
    require(proof['exact_replay_equal'] is True and hashes[INPUTS[8]]==proof['primary_sha256']==proof['replication_sha256'],'Saved v4 numeric proof mismatch')
    replica='data/processed/benchmark_fixtures/route_memory_factorial_v4_replication_v1.json'
    require(spec['preserved'][replica]==proof['replication_sha256'],'Preserved replication binding differs')
    require(len(primary['cases'])==20,'Saved v4 case count differs')
    expected_cases={(seed,task,condition) for seed in (1201,1213,1229,1237,1249) for task in ('independent','interacting') for condition in ('stationary','changed')}
    require({(c['seed'],c['task'],c['condition']) for c in primary['cases']}==expected_cases,'Saved case identities differ')
    for case in primary['cases']:
        require(len(case['cells'])==4,'Saved v4 cell count differs')
        for cell in case['cells'].values():
            require(cell['training']['route_updates']==9216 and cell['training']['actual_learning_field_changes']==36864,'Historical actual-write evidence changed')
            if case['condition']=='changed':require(cell['recovery_within_256'] is False,'Original recovery negative changed')
    for name in ('cumulative__uniform','cumulative__public_exponential','discounted__uniform','discounted__public_exponential'):
        failures.append(evidence(reports,INPUTS[8],'/summary/cells/'+name+'/changed_recovery_within_256_passes',0,'saved original 256-tick gate; no new scoring'))
    gains=[evidence(reports,INPUTS[8],'/summary/cells/cumulative__public_exponential/changed_retention_passes',7,'known synthetic cases only'),
           evidence(reports,INPUTS[8],'/summary/cells/discounted__public_exponential/changed_retention_passes',10,'known synthetic cases only'),
           evidence(reports,INPUTS[8],'/summary/cells/discounted__public_exponential/changed_tail_mean',1.0,'saved metric copied, not rescored'),
           evidence(reports,INPUTS[8],'/summary/cells/discounted__public_exponential/changed_frozen_mean',1.0,'not independent-task usefulness')]
    failures.append(evidence(reports,INPUTS[10],'/status','failed_exact_report_comparison','old consumed replay remains failed'))
    failures.append(evidence(reports,INPUTS[10],'/numeric_trace_equality',None,'discarded old numerical equality stays unknown'))
    r2=reports[INPUTS[11]]
    proofs.append(evidence(reports,INPUTS[11],'/decision','hold_additional_resource_work_until_causal_mechanism_and_independent_task_prerequisites','authoritative resource hold'))
    proofs.append(evidence(reports,INPUTS[12],'/decision','no_existing_receipt_licenses_snn_predictive_or_resource_advantage','historical research decision, not new measurement'))
    task=r2['independent_task']
    require(task['readiness']=='not_established_by_selected_receipts' and task['selected_source'] is None,'Independent task readiness changed')
    require(task['fixed_definition']['target_arrivals']==70212 and task['fixed_definition']['minimum_valid_recorded_dates']==180,'Independent task conditions changed')
    proofs.append(evidence(reports,INPUTS[13],'/approved_payload_acquisitions',[],'finite SPHERE/ECO review; no source expansion'))
    proofs.append(evidence(reports,INPUTS[13],'/decision','stop_finite_metadata_audit_no_payload_authorized','not universal source impossibility'))
    for key in ('runtime_promotion_authorized','new_resource_measurement_authorized','new_acquisition_or_search_authorized','seal_opening_authorized'):
        proofs.append(evidence(reports,INPUTS[11],'/'+key,False,'existing research hold'))
    branch=next((m for m in r2['mechanism_candidates'] if m['id']=='branch_local_delayed_eligibility'),None)
    require(branch is not None and branch['selection_status']=='conditional_design_candidate_only','Conditional SNN mechanism evidence absent')
    require(branch['increment_established_by_selected_receipts'] is None and branch['experiment_authorized'] is False and branch['independently_qualified_task'] is None,'Conditional mechanism promoted')
    historical_scalar=[{'id':row['id'],'scalar_equivalence':row['scalar_equivalence'],
                        'independent_usefulness_passed':row['independent_usefulness_passed'],
                        'scope':'historical component only; not route engine equality'}
                       for row in r2['learning_evidence'] if 'scalar_equivalence' in row]
    require(all(row.get('snn_predictive_increment_established') is False for row in r2['learning_evidence']),'Historical SNN increment evidence changed')
    gaps={
      'fixture_api_invariants':{'status':'verified_in_fixture_scope_only','evidence':verified},
      'runtime_provenance':{'established':None,'status':'missing_authenticated_observation_and_result_evidence',
         'reason':'Trusted hand inputs and API ordering are not real clock, allocation, label secrecy or event provenance.'},
      'independent_route_scalar_computation':{'established':None,'status':'not_demonstrated_by_symbolic_engine_descriptors',
         'historical_other_component_equivalence':historical_scalar,
         'reason':'Do not transfer other component scalar checks or matching opaque fixtures to route numerical engines.'},
      'independent_task':{'qualified':None,'status':task['readiness'],'selected_source':None,
         'fixed_definition':task['fixed_definition'],'required_conditions':task['required_conditions'],
         'source_constraints':task['source_constraints']},
      'snn_specific_causal_evidence':{'established':None,'status':'conditional_mechanism_not_demonstrated',
         'mechanism':branch,'retained_other_mechanism_statuses':[{'id':m['id'],'status':m['status']} for m in r2['mechanism_candidates'] if m['id']!=branch['id']]},
      'resource_readiness':{'ready':False,'status':'retain_original_resource_hold','prerequisites':r2['resource_prerequisites']}}
    choices=[
      {'id':'more_fixture_scaffolding','selected':False,'reason':'Verified fixture interfaces already exist; more metadata alone closes no numerical-independence gap.'},
      {'id':'independent_stateless_scalar_fixture_check','selected':True,'reason':'Smallest no-learning computational check: independent lookup/error arithmetic on existing eleven hand records.'},
      {'id':'live_runtime_provenance_trial','selected':False,'reason':'Larger live execution/provenance scope; unavailable and unauthorised in this audit.'},
      {'id':'new_independent_source_acquisition','selected':False,'reason':'No selected eligible source under unchanged native target/support/budget gates; retain finite source holds.'},
      {'id':'snn_causal_learning_experiment','selected':False,'reason':'Conditional mechanism/task prerequisites unmet; no authorised learning or positive causal evidence.'},
      {'id':'additional_resource_measurement','selected':False,'reason':'Independent usefulness, causal effect and decision-changing resource question remain missing.'}]
    require([c['id'] for c in choices]==spec['alternatives'],'Finite choice roster changed')
    return {'schema':'sara-route-readiness-audit-result-v1','readiness':'fixture_preflight_passed_runtime_research_not_ready',
       'evidence_categories':gaps,'scope_checks':proofs,'retained_scientific_improvements':gains,
       'retained_failures_and_unknowns':failures,'alternatives':choices,
       'selected_next_check':{'id':'independent_stateless_scalar_fixture_check','execution_performed':False,
          'requires_new_registration':True,'required_input':'route_error_binding_preflight_selection_v1.json /local_rows',
          'scope':'independent stateless CPU scalar lookup/error arithmetic on11 literal records; no state updates, models, task scoring or new data',
          'independence':'Separate implementation must not import/call primary local_check, bind_numeric or learner calculators. Compare against literal goldens with original float order; retain mismatches without retuning.',
          'does_not_establish':['independent runtime learner equality','authenticated causal provenance','independent-task usefulness','SNN-specific causal advantage','adaptive-system acceptance','resource benefit'],
          'selection_basis':'registered prerequisite/scope ordering, no measured cost or performance ranking'},
       'holds':{'additional_learning':True,'additional_resource_measurement':True,'source_acquisition':True,
                'seal_opening':True,'original_adaptive_promotion':True,'snn_advantage_claim':True},
       'status':{'new_learner_executions':0,'new_prediction_scores':0,'new_acquisitions':0,'new_measurements':0,
                 'fixture_executions':0,'stateless_scalar_executions':0,'test_opened':False,
                 'conditions_relaxed':False,'training_authorized':False,'promotion_authorized':False}}
