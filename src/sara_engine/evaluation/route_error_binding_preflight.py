"""Hand-record probability/error binding; no learner or prediction scoring."""
from dataclasses import dataclass,replace
from itertools import product
from math import isfinite
from hashlib import sha256

from sara_engine.evaluation.route_numeric_delta_preflight import (
    NumericView,MANUAL_ROWS,LIMIT,fixture as numeric_fixture,validate_numeric,altered,
)
from sara_engine.evaluation.route_delta_ledger_preflight import Credit,MASS
from sara_engine.evaluation.route_interface_preflight import integer
from sara_engine.evaluation.route_fixed_allocation_contract import ROUTES,canonical,require

OUTPUT_SIDES=('mass_left','mass_right','mass_left')


def probability_type(p):
    require(p is None or (type(p) is float and isfinite(p) and 0<=p<=1),'Invalid hand-record probability')


def local_check(tick,history,error_weight,allocated,n0,n1,side,probability,error_increment):
    integer(tick)
    require(history in ('cumulative','discounted') and error_weight in ('uniform','public_exponential'),'Unknown factor')
    require(type(allocated) is bool and side in MASS,'Invalid allocation/class side')
    require(all(type(n) is int and 0<=n<=LIMIT for n in (n0,n1)),'Invalid prior masses')
    probability_type(probability)
    require(type(error_increment) is float and isfinite(error_increment) and error_increment>0,'Invalid error increment')
    prior=2**(tick//64) if history=='discounted' else 1
    if allocated:
        expected=(n1+prior)/(n0+n1+2*prior)
        require(type(probability) is float and probability.hex()==expected.hex(),'Pre-outcome local probability mismatch')
        residual_p=expected
    else:
        require(n0==n1==0 and probability is None,'Absent key cannot invent probability or positive mass')
        residual_p=.5
    weight=float(2**(tick//64)) if error_weight=='public_exponential' else 1.0
    y=MASS.index(side)
    expected_increment=weight*(residual_p-y)**2
    require(error_increment.hex()==expected_increment.hex(),'Class/public-weight error binding mismatch')
    return {'prior':prior,'probability':probability,'residual_fallback':not allocated,
            'error_weight':weight,'error_increment':error_increment}


@dataclass(frozen=True,slots=True)
class LookupCredit:
    credit:Credit
    allocated:bool
    probability:object
    error_increment:float

    def __post_init__(self):
        require(type(self.credit) is Credit and type(self.allocated) is bool,'Invalid lookup-credit metadata')
        probability_type(self.probability)
        require(type(self.error_increment) is float and isfinite(self.error_increment) and self.error_increment>0,'Invalid lookup-credit error')


def bind_numeric(proposal,trusted_before,witness,records,allocation_flags,output_sides):
    require(type(trusted_before) is NumericView,'Missing trusted prior numeric view')
    require(type(records) is tuple and len(records)==9 and all(type(r) is LookupCredit for r in records),'Invalid lookup-record catalog')
    require(type(allocation_flags) is tuple and len(allocation_flags)==9 and all(type(a) is bool for a in allocation_flags),'Missing explicit allocation witnesses')
    require(type(output_sides) is tuple and len(output_sides)==3 and all(s in MASS for s in output_sides),'Invalid common output-class witness')
    require(proposal.receipt==witness.receipt and proposal.credits==witness.credits,'Pinned receipt/key/class witness mismatch')
    values=dict(trusted_before.fields)
    case=proposal.receipt.case
    for record,credit,allocated,error_delta in zip(records,witness.credits,allocation_flags,witness.error_increments):
        require(record.credit==credit and record.allocated is allocated,'Key/order/allocation credit drift')
        require(credit.mass_side==output_sides[ROUTES[credit.route][1]],'Same-output routes received inconsistent class credit')
        require(record.error_increment.hex()==error_delta.hex(),'Independent numeric error witness drift')
        n0=values[(credit.route,'mass_left',credit.key_slot)]
        n1=values[(credit.route,'mass_right',credit.key_slot)]
        local_check(proposal.receipt.receipt_id,case.history,case.error_weight,allocated,n0,n1,
                    credit.mass_side,record.probability,record.error_increment)
    result=validate_numeric(proposal,trusted_before,witness)
    return {'scope':'hand_record_probability_error_binding','route_bindings':9,
            'numeric_active_field_checks':result['active_field_checks'],'actual_learning_updates':0,
            'runtime_outcome_provenance_authenticated':False,'runtime_scalar_equality_tested':False,
            'original_adaptive_pass':False}


def fixture(row,engine,role):
    p,b,w=numeric_fixture(row,engine,role)
    credits=tuple(replace(c,mass_side=OUTPUT_SIDES[ROUTES[c.route][1]]) for c in p.credits)
    after=dict(p.after.fields);before=dict(b.fields)
    for c in p.credits:after[(c.route,c.mass_side,c.key_slot)]=before[(c.route,c.mass_side,c.key_slot)]
    for c in credits:after[(c.route,c.mass_side,c.key_slot)]=row[3]
    view=NumericView(p.after.case,tuple((a,after[a]) for a,_ in p.after.fields))
    p=replace(p,credits=credits,after=view);w=replace(w,credits=credits)
    records=tuple(LookupCredit(c,True,.5,delta) for c,delta in zip(credits,w.error_increments))
    return p,b,w,records,(True,)*9,OUTPUT_SIDES


def fault_preflight():
    args=fixture(MANUAL_ROWS[0],'local_sparse_reference','shuffled_outcome')
    p,b,w,records,flags,sides=args
    before=(b.typed_signature(),p.after.typed_signature(),w,records)
    faults=[]
    def rejected(name,operation):
        try:operation()
        except (ValueError,TypeError) as exc:
            require(before==(b.typed_signature(),p.after.typed_signature(),w,records),'Binding fault mutated fixture')
            faults.append({'fault':name,'rejected':True,'inputs_unchanged':True,'exception':type(exc).__name__})
        else:raise ValueError('Binding fault accepted: '+name)
    rejected('wrong_probability',lambda:local_check(0,'cumulative','uniform',True,0,2,'mass_left',.5,.5625))
    rejected('after_state_probability',lambda:bind_numeric(p,b,w,(replace(records[0],probability=3/7),)+records[1:],flags,sides))
    rejected('wrong_prior_clock',lambda:local_check(64,'discounted','uniform',True,0,2,'mass_left',.75,.5625))
    rejected('wrong_class_side',lambda:local_check(0,'cumulative','uniform',True,0,2,'mass_right',.75,.5625))
    rejected('wrong_error_weight',lambda:local_check(64,'discounted','public_exponential',True,0,4,'mass_left',.75,.5625))
    rejected('wrong_error_increment',lambda:bind_numeric(p,b,w,(replace(records[0],error_increment=.5),)+records[1:],flags,sides))
    rejected('bool_probability',lambda:replace(records[0],probability=True))
    rejected('nan_probability',lambda:replace(records[0],probability=float('nan')))
    rejected('missing_seen_probability',lambda:local_check(0,'cumulative','uniform',True,2,2,'mass_left',None,.25))
    rejected('invented_absent_probability',lambda:local_check(64,'discounted','uniform',False,0,0,'mass_left',.5,.25))
    rejected('absent_with_positive_mass',lambda:local_check(64,'discounted','uniform',False,0,1,'mass_left',None,.25))
    rejected('wrong_key_slot',lambda:bind_numeric(p,b,w,(replace(records[0],credit=replace(records[0].credit,key_slot=1-records[0].credit.key_slot)),)+records[1:],flags,sides))
    rejected('allocation_flag_drift',lambda:bind_numeric(p,b,w,(replace(records[0],allocated=False),)+records[1:],flags,sides))
    rejected('witness_increment_drift',lambda:bind_numeric(p,b,replace(w,error_increments=(.5,)+w.error_increments[1:]),records,flags,sides))
    def inconsistent():
        old,before_view,witness=numeric_fixture(MANUAL_ROWS[0],'local_sparse_reference','shuffled_outcome')
        old_records=tuple(LookupCredit(c,True,.5,d) for c,d in zip(witness.credits,witness.error_increments))
        bind_numeric(old,before_view,witness,old_records,flags,sides)
    rejected('inconsistent_same_output_credit',inconsistent)
    rejected('wrong_source',lambda:bind_numeric(replace(p,delivery=replace(p.delivery,source_index=(p.delivery.source_index+1)%512)),b,w,records,flags,sides))
    rejected('same_tick_delivery',lambda:bind_numeric(replace(p,delivery=replace(p.delivery,at=0)),b,w,records,flags,sides))
    c=w.credits[0]
    rejected('stale_before',lambda:bind_numeric(p,altered(b,{(c.route,'mass_left',c.key_slot):3}),w,records,flags,sides))
    rejected('reordered_records',lambda:bind_numeric(p,b,w,tuple(reversed(records)),flags,sides))
    rejected('wrong_output_class_vector',lambda:bind_numeric(p,b,w,records,flags,('mass_right',)*3))
    return faults


def analyze(spec):
    require(spec['schema']=='sara-route-error-binding-preflight-selection-v1','Wrong binding contract')
    require(spec['bounds']=={'faults':20,'local_rows':11,'transactions':72,'route_bindings':648,
             'result_bytes':262144,'all_artifact_bytes':2097152,'learner_executions':0,'actual_learning_updates':0,
             'outcome_payload_reads':0,'prediction_scores':0},'Binding budget changed')
    require(spec['integrated_fixtures']['output_sides']==list(OUTPUT_SIDES),'Output class witness changed')
    local_results=[local_check(*row) for row in spec['local_rows']]
    require(len(local_results)==11,'Hand-record roster changed')
    digest=sha256();bindings=0
    for row,engine,role in product(MANUAL_ROWS,('local_sparse_reference','independent_event_driven_scalar'),
                                   ('candidate','permanent','shuffled_outcome')):
        result=bind_numeric(*fixture(row,engine,role));bindings+=result['route_bindings']
        digest.update(canonical([row,engine,role,result])+b'\n')
    faults=fault_preflight()
    require([f['fault'] for f in faults]==spec['faults'] and bindings==648,'Binding fault/fixture coverage changed')
    return {'schema':'sara-route-error-binding-preflight-result-v1','summary':{'local_records':11,
            'numeric_fixture_transactions':72,'route_error_bindings':648,'faults_rejected':20,'inputs_unchanged':True},
            'local_results':local_results,'fixture_digest':digest.hexdigest(),'fault_injections':faults,
            'status':{'learner_executions':0,'actual_learning_updates':0,'task_samples_generated':0,
                      'outcome_payload_reads':0,'prediction_scores':0,'new_snapshots':0,'resource_measurements':0,
                      'test_opened':False,'runtime_provenance_authenticated':False,'runtime_scalar_equality_tested':False,
                      'original_adaptive_component_passed':False,'execution_authorized':False,
                      'snn_advantage_tested':False,'promotion_authorized':False},
            'pending':['authenticated allocation/key/class provenance','actual runtime probability/error observation',
                       'independent scalar numerical equality','independent-task/SNN-specific causal evidence']}
