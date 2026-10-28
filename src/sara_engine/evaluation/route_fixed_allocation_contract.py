"""Nonexecuting allocation/receipt contract; no samples, models or updates."""
from hashlib import sha256
from itertools import combinations
import json
from math import ulp

ROUTES=tuple((key,slot) for width in (1,2) for key in combinations(range(5),width) for slot in range(3))
GROUPS=tuple(tuple(i for i,(_,s) in enumerate(ROUTES) if s==slot) for slot in range(3))


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def active_routes(tick):
    """Public-clock-only address plan; no event, label, score or phase input."""
    require(type(tick) is int and 0<=tick<1024,'Invalid static receipt tick')
    return tuple(group[(3*tick+j)%15] for group in GROUPS for j in range(3))


def fixed_readout(tick):
    return tuple(active_routes(tick)[i] for i in (0,3,6))


def validate_trace(events,expected_receipts=1024):
    """Validate receipt metadata only, without outcome values or a learner."""
    require(type(expected_receipts) is int and 0<expected_receipts<=1024,'Invalid trace bound')
    require(len(events)==2*expected_receipts,'Incomplete or excess trace')
    pending=None
    receipts=0
    for kind,tick,identity in events:
        require(type(tick) is int and type(identity) is int,'Boolean/noninteger event identity')
        if kind=='receipt':
            require(pending is None and tick==receipts and identity==tick,'Duplicate, nonmonotonic or overlapping receipt')
            pending=tick
        elif kind=='outcome':
            require(pending is not None and identity==pending and tick==pending+1,'Missing, wrong, early or late delayed outcome')
            pending=None
            receipts+=1
        else:raise ValueError('Unknown receipt metadata event')
    require(pending is None and receipts==expected_receipts,'Unresolved pending receipt')
    return {'receipts':receipts,'delayed_outcomes':receipts,'max_pending':1,
            'event_dispatches':2*receipts,'idle_dispatches':0,'expired_outcomes':0}


def validate_contract(contract):
    require(contract['schema']=='sara-route-fixed-allocation-contract-v1','Wrong contract schema')
    conditions=contract['scientific_conditions']
    expected={'seeds':[1201,1213,1229,1237,1249],'tasks':['independent','interacting'],
              'conditions':['stationary','changed'],'steps':1024,'change_at':512,'warmup':180,
              'input_bits':5,'outputs':3,'minimum_route_observations':16,'width_cost':.005,
              'trigger_threshold':.20,'detector_rate':'1/16','cooldown':64,'exploration_length':180,
              'fixed_block_length':64,'frozen_rows':256,'frozen_query_tick':1023}
    require(canonical(conditions)==canonical(expected),'Scientific condition changed')
    allocation=contract['allocation']
    require(allocation['type']=='exogenous_canonical_cyclic_v1' and allocation['group']=='ascending catalog IDs for each output','Exogenous catalog order changed')
    require(allocation['active']=='for each output, group[(3*t+j)%15], j=0,1,2','Schedule rule changed')
    require(allocation['readout']=='for each output, group[(3*t)%15], provided route observations >=16; unseen key abstains','Fixed readout rule changed')
    require(allocation['updates_per_tick']==9 and allocation['outputs_per_tick']==[3,3,3] and allocation['receipt_count']==1024 and allocation['first_readout_supported_tick']==80,'Allocation budget changed')
    require(all(allocation[k] is False for k in ('phase_hint_allowed','labels_or_predictions_allowed','donor_trace_allowed','shared_input_drop_allowed')),'Forbidden scheduler feedback/input removal')
    factors=contract['factors']
    require(factors['count_history']==['cumulative','discounted'] and factors['error_weight']==['uniform','public_exponential'] and factors['public_weight']=='2**floor(t/64)','Factor conditions changed')
    require(contract['engines']==['local_sparse_reference','independent_event_driven_scalar'],'Strong scalar control removed')
    require(contract['scalar_control']['snn_claim'] is False,'Unsupported spiking claim')
    separation=contract['separation']
    require(separation['primary_predictions_to_shadow'] is False and separation['shadow_predictions_to_shadow_only'] is True and separation['not_a_replay_of_old_adaptive_allocations'] is True,'Shadow feedback isolation changed')
    delayed=contract['delayed_outcomes']
    require(delayed['max_pending']==1 and delayed['outcome_time']=='t+1, own receipt only' and delayed['reject_before_mutation'] is True,'Delay/atomicity contract changed')
    updates=contract['actual_update_contract']
    require(updates['fields']==['observed_class_mass','route_observations','error_sum','weight_sum'],'Four-field contract changed')
    require(all(updates[k] is True for k in ('same_route_at_same_receipt','same_observed_key_and_label','no_dummy_writes','no_global_decay_writes','static_proof_not_actual_execution')),'Actual-update identity/no-op scope changed')
    budget=contract['budgets']
    for key,value in {'current_learner_executions':0,'current_input_payload_reads':0,'current_resource_measurements':0,
                      'static_schedule_rows':1024,'static_route_visits':9216,'actual_route_updates_per_future_cell':9216,
                      'genuine_learning_fields_per_route_update':4,'actual_learning_fields_per_future_cell':36864,
                      'future_cells_per_case':8,'future_cases':20,'future_model_runs_per_phase':160,
                      'future_training_receipts_per_phase':163840,'future_replication_phases':2,
                      'contract_bytes':262144,'result_bytes':262144,'all_artifacts_bytes':2097152}.items():
        require(type(budget[key]) is int and budget[key]==value,'Budget changed: '+key)
    gates=contract['unchanged_gates']
    require(gates['changed']=={'initial_new_dependency_absent':True,'tail_exact_min':.90,
           'tail_gain_over_permanent_min':.10,'recovery_max_ticks':256,
           'recovery':'first end of two consecutive 64-receipt post-change blocks each exact>=0.90, minus512; absent=None',
           'all_true_dependencies_retained':True,'tail_gain_over_shuffled_min':.20,'frozen_exact_min':.90},'Original changed gates altered')
    require(gates['stationary']=={'false_starts_max':1,'permanent_tail_minus_candidate_max':.02} and gates['acceptance']=='not evaluated; design only','Original stationary gate/acceptance altered')
    return True


def analyze(contract):
    validate_contract(contract)
    counts=[0]*45
    ledger=sha256()
    first_supported=None
    for tick in range(1024):
        active=active_routes(tick)
        readout=fixed_readout(tick)
        require(len(set(active))==9 and all(sum(ROUTES[i][1]==s for i in active)==3 for s in range(3)),'Unbalanced static schedule')
        require(len(readout)==3 and all(ROUTES[i][1]==s for s,i in enumerate(readout)),'Fixed readout output mismatch')
        if min(counts)>=16 and first_supported is None:first_supported=tick
        ledger.update(canonical([tick,active,readout])+b'\n')
        for i in active:counts[i]+=1
    require(sum(counts)==9216 and min(counts)==204 and max(counts)==205,'Route receipt plan failed')
    require(first_supported==80,'Eligibility timing changed')
    trace=[event for t in range(1024) for event in (('receipt',t,t),('outcome',t+1,t))]
    lifecycle=validate_trace(trace)
    max_observations=max(counts)
    max_weight=2**15
    max_sum=max_observations*max_weight
    # Effective evidence cannot exceed observations because all past weights <= current weight.
    min_probability=1/(max_observations+2)
    min_error_increment=min_probability**2
    require(max_sum<2**53 and min_error_increment>2*ulp(float(max_sum)),'Finite binary64 non-no-op bound failed')
    return {'schema':'sara-route-fixed-allocation-static-result-v1','contract_valid':True,
            'address_plan':{'rows':1024,'planned_route_updates':sum(counts),'per_route_receipts':counts,
                            'planned_changed_learning_fields':4*sum(counts),'min_per_route':min(counts),'max_per_route':max(counts),
                            'first_all_routes_support_16_tick':first_supported,'schedule_and_readout_sha256':ledger.hexdigest(),
                            'first_five_active_rows':[list(active_routes(t)) for t in range(5)]},
            'receipt_metadata_proof':lifecycle,
            'finite_non_no_op_bound':{'max_route_observations':max_observations,'max_public_weight':max_weight,
                                     'max_class_mass_weight_or_error_sum':max_sum,
                                     'minimum_smoothed_probability':min_probability,'minimum_positive_error_increment':min_error_increment,
                                     'maximum_sum_ulp':ulp(float(max_sum)),'runtime_changes_measured':False},
            'status':{'learner_executions':0,'task_samples_generated':0,'input_payload_reads':0,'prediction_scores':0,
                      'new_snapshots':0,'actual_learning_updates':0,'resource_measurements':0,'test_opened':False,
                      'static_contract_only':True,'execution_authorized':False,'component_gates_evaluated':False,
                      'snn_advantage_tested':False,'promotion_authorized':False},
            'pending_runtime_checks':['identical route/key/label receipt digests across engines and cells',
                                      'four actual non-no-op learning changes per active route',
                                      'independent scalar numerical equivalence and read-only evaluation',
                                      'complete matched permanent/shuffled-control scope before execution',
                                      'no training feedback from shadow lifecycle to fixed addresses'],
            'limitations':contract['separation']['does_not_isolate']}
