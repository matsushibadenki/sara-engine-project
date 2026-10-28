"""Static matched controls and endpoint namespaces; no model or outcome data."""
from hashlib import sha256
from itertools import product
import random

from sara_engine.evaluation.route_fixed_allocation_contract import canonical, require, analyze as parent_analyze

ROLES=('candidate','permanent','shuffled_outcome')
ENGINES=('local_sparse_reference','independent_event_driven_scalar')
SEEDS=(1201,1213,1229,1237,1249)
CHANNELS=('fixed_readout','shadow_selection','original_adaptive_system')
PRIMARY_STATUS={'changed_contrast':'not_applicable','tail_accuracy':'descriptive_only',
                'gain_over_permanent':'structural_zero','recovery_within_256':'descriptive_only',
                'dependencies_retained':'not_applicable','outcome_alignment':'descriptive_only',
                'frozen_accuracy':'descriptive_only','false_starts':'not_applicable',
                'stationary_noninferiority':'structural_zero','integrity':'runtime_required'}
COMPARATORS={'gain_over_permanent':['candidate','permanent'],
             'stationary_noninferiority':['permanent','candidate'],
             'outcome_alignment':['candidate','shuffled_outcome']}


def permutation_indices(seed):
    """Compile an offline vector-label address map; never reads any labels."""
    require(type(seed) is int and seed in SEEDS,'Unregistered permutation seed')
    rng=random.Random(seed+590000)
    result=[]
    for start in (0,512):
        indices=list(range(start,start+512))
        rng.shuffle(indices)
        result.extend(indices)
    return result


def validate_permutation(indices):
    require(len(indices)==1024 and all(type(i) is int for i in indices),'Invalid source index vector')
    for start in (0,512):
        require(sorted(indices[start:start+512])==list(range(start,start+512)),'Phase bijection or phase boundary violated')
    # Fixed points are deliberately allowed; no outcome-dependent rejection.
    return {'fixed_points':sum(i==source for i,source in enumerate(indices)),
            'forward_source_indices':sum(source>i for i,source in enumerate(indices)),
            'backward_source_indices':sum(source<i for i,source in enumerate(indices))}


def endpoint_policy(gate,channel):
    require(gate in PRIMARY_STATUS and channel in CHANNELS,'Unknown endpoint namespace')
    if channel=='original_adaptive_system':return 'untested'
    if channel=='shadow_selection':return 'runtime_required' if gate=='integrity' else 'diagnostic_only'
    return PRIMARY_STATUS[gate]


def endpoint_record(gate,channel):
    return {'gate':gate,'channel':channel,'status':endpoint_policy(gate,channel),
            'roles':COMPARATORS.get(gate,['candidate']),
            'value':None,'original_adaptive_pass':False}


def validate_endpoint_record(record):
    require(set(record)=={'gate','channel','status','roles','value','original_adaptive_pass'},'Endpoint fields changed')
    require(record['original_adaptive_pass'] is False and record['value'] is None,'Static metric or adaptive pass invented')
    require(record==endpoint_record(record['gate'],record['channel']),'Comparator/channel applicability substituted')
    return True


def validate_contract(spec,parent):
    require(spec['schema']=='sara-route-fixed-control-contract-v1','Wrong control schema')
    parent_analyze(parent)
    require(canonical(spec['conditions'])==canonical(parent['scientific_conditions']),'Scientific conditions changed')
    require(canonical(spec['original_gates'])==canonical(parent['unchanged_gates']),'Original gates changed')
    require(spec['parent']['all_parent_conditions_unchanged'] is True,'Parent condition change authorized')
    require(spec['roles']==list(ROLES) and spec['engines']==list(ENGINES),'Missing matched role or engine')
    require(spec['factors']=={'count_history':['cumulative','discounted'],'error_weight':['uniform','public_exponential']},'Factor scope changed')
    require(spec['permanent']['primary_candidate_identity'] is True and spec['permanent']['core_candidate_identity'] is True and spec['permanent']['shadow_pool_identity_required'] is False,'Permanent identity/isolation changed')
    shuffle=spec['shuffle']
    require(shuffle['type']=='offline_joint_outcome_vector_permutation_v1' and shuffle['seed_offset']==590000 and shuffle['phase_bounds']==[[0,512],[512,1024]],'Shuffle definition changed')
    require(shuffle['permutation']=='random.Random(seed+590000); sequential rng.shuffle on index lists [0..511] then [512..1023]','Shuffle RNG/order changed')
    for gate in PRIMARY_STATUS:
        require(spec['endpoint_policy'][gate]=={ch:endpoint_policy(gate,ch) for ch in CHANNELS},'Endpoint applicability changed: '+gate)
    compare=spec['comparators']
    require(all(compare[k]==v for k,v in COMPARATORS.items()),'Comparator direction changed')
    require(compare['matching']==['same seed','same task','same stationary/changed condition','same count factor','same error factor','same engine','same metric channel','same true evaluation labels'],'Comparator matching changed')
    require(compare['no_cross_channel_substitution'] is True and compare['no_historical_control_as_new_matched_run'] is True,'Forbidden comparator substitution')
    assertions=spec['structural_assertions']
    for key,value in {'candidate_permanent_same_core_state':True,'candidate_permanent_same_primary_predictions':True,
                      'candidate_permanent_primary_tail_gain':0,'candidate_permanent_primary_stationary_difference':0,
                      'shuffled_count_probabilities_need_not_match':True,'all_roles_same_update_addresses':True,
                      'all_roles_same_actual_update_budget':True,'all_same_role_cell_scalar_equivalence_required':True,
                      'gate_thresholds_are_not_tuned':True}.items():
        require(type(assertions[key]) is type(value) and assertions[key]==value,'Structural assertion changed')
    for key in ('scalar_control','delayed_outcomes','actual_update_contract'):
        require(canonical(spec[key])==canonical(parent[key]),'Parent local/event contract changed: '+key)
    budgets={'static_phases':2,'current_learner_executions':0,'current_sample_generation':0,
             'current_prediction_scores':0,'current_actual_updates':0,'permutation_maps':5,'per_map_indices':1024,
             'contract_bytes':262144,'result_bytes':262144,'all_artifact_bytes':2097152,'future_cases':20,
             'future_roles_per_factor_engine':3,'future_cells_per_case':24,'future_models_per_phase':480,
             'future_training_receipts_per_phase':491520,'future_route_updates_per_model':9216,
             'future_field_changes_per_model':36864,'future_total_route_updates_per_phase':4423680,
             'future_total_field_changes_per_phase':17694720,'future_primary_replication_phases':2}
    require(canonical(spec['budgets'])==canonical(budgets),'Finite control budget changed')
    return True


def analyze(spec,parent,parent_saved):
    validate_contract(spec,parent)
    plan=parent_analyze(parent)
    for key in ('address_plan','receipt_metadata_proof','finite_non_no_op_bound'):
        require(canonical(plan[key])==canonical(parent_saved[key]),'Preserved parent static proof differs')
    maps=[]
    for seed in SEEDS:
        indices=permutation_indices(seed)
        maps.append({'seed':seed,'indices':indices,'sha256':sha256(canonical(indices)).hexdigest(),
                     **validate_permutation(indices)})
    cells=[{'history':history,'error':error,'engine':engine,'role':role,
            'training_address_sha256':plan['address_plan']['schedule_and_readout_sha256'],
            'planned_route_updates':9216,'planned_learning_field_changes':36864,
            'label_source':'phase_permutation' if role=='shuffled_outcome' else 'own_true_label',
            'main_readout':'parent_fixed','shadow_pool_policy':'freeze_after_initial' if role=='permanent' else 'original_candidate_lifecycle'}
           for history,error,engine,role in product(('cumulative','discounted'),('uniform','public_exponential'),ENGINES,ROLES)]
    matrix=[endpoint_record(gate,ch) for gate,ch in product(PRIMARY_STATUS,CHANNELS)]
    for record in matrix:validate_endpoint_record(record)
    require(len(cells)==24 and len(matrix)==30,'Control or endpoint coverage changed')
    return {'schema':'sara-route-fixed-control-static-result-v1','contract_valid':True,
            'parent_static_plan_equal':True,'planned_cells_per_case':cells,'permutation_maps':maps,
            'endpoint_matrix':matrix,'prospective_budget':{'cases':20,'models_per_phase':480,
              'receipts_per_phase':491520,'route_updates_per_phase':4423680,'learning_field_changes_per_phase':17694720,
              'phases':2,'actual_current_model_runs':0},
            'structural_identifiability':{'primary_gain_over_permanent':0,
               'original_gain_requirement':.10,'original_gain_cannot_be_established_by_primary':True,
               'primary_stationary_difference':0,'zero_is_structural_not_accuracy_evidence':True,
               'shadow_gain_does_not_imply_original_adaptive_pass':True,
               'offline_shuffle_is_not_live_causal_label_delivery':True},
            'status':{'learner_executions':0,'samples_generated':0,'outcome_payloads_read':0,
                      'prediction_scores':0,'new_snapshots':0,'actual_learning_updates':0,
                      'resource_measurements':0,'test_opened':False,'original_adaptive_component_passed':False,
                      'original_adaptive_gates_evaluated':False,'runtime_scalar_equality_tested':False,
                      'execution_authorized':False,'snn_advantage_tested':False,'promotion_authorized':False},
            'pending_runtime_checks':['same-role/cell independent scalar equivalence',
                                      'candidate/permanent core and primary exact identity',
                                      'own pending identity distinct from shuffled source provenance',
                                      'equal actual per-route/key counts and four non-no-op fields',
                                      'no true-evaluation label leakage to shuffled learner or detector',
                                      'strict separation of primary, shadow and adaptive-system reports']}
