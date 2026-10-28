"""Read-only numeric snapshot checks over literal fixtures, never learning."""
from dataclasses import dataclass,replace
from hashlib import sha256
from itertools import product
from math import isfinite,copysign

from sara_engine.evaluation.route_delta_ledger_preflight import (
    ADDRESSES,MASS,CoreView,Credit,DeltaProposal,DeltaLedger,validate_credits,
)
from sara_engine.evaluation.route_interface_preflight import (
    CaseKey,Receipt,Delivery,DiagnosticHandle,make_receipt,token,
)
from sara_engine.evaluation.route_fixed_allocation_contract import ROUTES,canonical,require

LIMIT=6717440
MANUAL_ROWS=[
 [0,'cumulative','uniform',3,8.25,17.0,0.25],
 [0,'cumulative','public_exponential',3,8.25,17.0,0.25],
 [0,'discounted','uniform',3,8.25,17.0,0.25],
 [0,'discounted','public_exponential',3,8.25,17.0,0.25],
 [64,'cumulative','uniform',3,8.25,17.0,0.25],
 [64,'cumulative','public_exponential',3,8.5,18.0,0.5],
 [64,'discounted','uniform',4,8.25,17.0,0.25],
 [64,'discounted','public_exponential',4,8.5,18.0,0.5],
 [1023,'cumulative','uniform',3,8.25,17.0,0.25],
 [1023,'cumulative','public_exponential',3,8200.0,32784.0,8192.0],
 [1023,'discounted','uniform',32770,8.25,17.0,0.25],
 [1023,'discounted','public_exponential',32770,8200.0,32784.0,8192.0]]


def typed_value(field,value):
    if field in MASS or field=='observations':
        require(type(value) is int,'Exact integer field required; no bool/float coercion')
        bound=205 if field=='observations' else LIMIT
        require(0<=value<=bound,'Integer field outside finite fixture bound')
        return ('int',str(value))
    require(type(value) is float and isfinite(value),'Exact finite float required')
    require(0<=value<=LIMIT and not (value==0 and copysign(1,value)<0),'Negative/negative-zero/out-of-bound float')
    return ('float',value.hex())


@dataclass(frozen=True,slots=True)
class NumericView:
    case:CaseKey
    fields:tuple

    def __post_init__(self):
        require(type(self.fields) is tuple and len(self.fields)==435,'Incomplete/mutable numeric field catalog')
        stamps=[]
        for cell in self.fields:
            require(type(cell) is tuple and len(cell)==2,'Mutable numeric field cell')
            address,value=cell
            require(type(address) is tuple and len(address)==3,'Invalid numeric address')
            stamps.append((address,sha256(canonical(typed_value(address[1],value))).hexdigest()))
        CoreView(self.case,tuple(stamps))
        values=dict(self.fields)
        require(all(values[(r,'error_sum',-1)]<=values[(r,'weight_sum',-1)] for r in range(45)),'Error exceeds its weight normalizer')

    def typed_signature(self):
        return tuple((a,typed_value(a[1],v)) for a,v in self.fields)

    def symbolic(self):
        return CoreView(self.case,tuple((a,sha256(canonical(typed_value(a[1],v))).hexdigest()) for a,v in self.fields))


@dataclass(frozen=True,slots=True)
class NumericWitness:
    receipt:Receipt
    credits:tuple
    error_increments:tuple

    def __post_init__(self):
        require(type(self.receipt) is Receipt,'Invalid witness receipt')
        validate_credits(self.credits,self.receipt)
        require(type(self.error_increments) is tuple and len(self.error_increments)==9,'Invalid error witness catalog')
        require(all(type(v) is float and isfinite(v) and 0<v<=LIMIT for v in self.error_increments),'Error witnesses must be positive finite floats')


@dataclass(frozen=True,slots=True)
class NumericProposal:
    receipt:Receipt
    delivery:Delivery
    credits:tuple
    before:NumericView
    after:NumericView
    diagnostic_before:DiagnosticHandle
    diagnostic_after:DiagnosticHandle

    def __post_init__(self):
        require(type(self.receipt) is Receipt and type(self.delivery) is Delivery,'Invalid numeric proposal provenance')
        validate_credits(self.credits,self.receipt)
        require(type(self.before) is NumericView and type(self.after) is NumericView,'Invalid numeric views')
        require(type(self.diagnostic_before) is DiagnosticHandle and type(self.diagnostic_after) is DiagnosticHandle,'Invalid numeric diagnostic witnesses')


def validate_numeric(proposal,trusted_before,witness):
    """Inspect values without changing views, witnesses, or a learner."""
    require(type(proposal) is NumericProposal and type(trusted_before) is NumericView and type(witness) is NumericWitness,'Invalid numeric validator arguments')
    case=proposal.receipt.case
    require(proposal.before.case==case and proposal.after.case==case and trusted_before.case==case,'Foreign numeric owner')
    require(proposal.receipt==witness.receipt and proposal.credits==witness.credits,'Pinned receipt/key/side drift')
    require(proposal.before.typed_signature()==trusted_before.typed_signature(),'Stale trusted numeric predecessor')
    before,after=dict(proposal.before.fields),dict(proposal.after.fields)
    tick=proposal.receipt.receipt_id
    count_weight=2**(tick//64) if case.history=='discounted' else 1
    error_weight=float(2**(tick//64)) if case.error_weight=='public_exponential' else 1.0
    for c,error_increment in zip(witness.credits,witness.error_increments):
        require(error_increment<=error_weight,'Error witness exceeds bounded squared residual')
        increments=(((c.route,c.mass_side,c.key_slot),count_weight),((c.route,'observations',-1),1),
                    ((c.route,'error_sum',-1),error_increment),((c.route,'weight_sum',-1),error_weight))
        for address,delta in increments:
            old,new=before[address],after[address]
            require(new>old,'Required numeric change was a no-op or decrease')
            expected=old+delta
            if type(old) is float:
                require(new.hex()==expected.hex(),'Wrong exact rounded float increment')
            else:require(new==expected,'Wrong exact integer increment')
    # Reuse the complete structural/provenance check on typed value fingerprints.
    structural=DeltaProposal(proposal.receipt,proposal.delivery,proposal.credits,
                            proposal.before.symbolic(),proposal.after.symbolic(),
                            proposal.diagnostic_before,proposal.diagnostic_after)
    ledger=DeltaLedger(case,trusted_before.symbolic(),proposal.diagnostic_before,tick)
    ledger.open(witness.receipt,witness.credits)
    ledger.accept(structural)
    for names,expected in ((MASS,9*count_weight),(('observations',),9),(('weight_sum',),9*error_weight)):
        actual=sum(v for a,v in proposal.after.fields if a[1] in names)-sum(v for a,v in proposal.before.fields if a[1] in names)
        require(actual==expected,'Independent numeric mass/observation/weight conservation differs')
    return {'scope':'hand_authored_numeric_fixture','active_field_checks':36,'unchanged_field_checks':399,
            'actual_learning_updates':0,'runtime_numeric_updates_observed':False,
            'runtime_scalar_equality_tested':False,'original_adaptive_pass':False}


def altered(view,changes):
    require(all(a in ADDRESSES for a in changes),'Foreign numeric address')
    return replace(view,fields=tuple((a,changes.get(a,v)) for a,v in view.fields))


def fixture(row,engine,role):
    tick,history,error,mass_after,error_after,weight_after,error_delta=row
    case=CaseKey(1201,'independent','changed',history,error,engine,role)
    receipt=make_receipt(case,tick)
    credits=tuple(Credit(i,tick%(2**len(ROUTES[i][0])),MASS[j%2]) for j,i in enumerate(receipt.active))
    # Both sides are literal templates; no predictor or update function creates them.
    before_values={a:(2 if a[1] in MASS else {1:8,2:16}[len(ROUTES[a[0]][0])] if a[1]=='observations' else 8.0 if a[1]=='error_sum' else 16.0) for a in ADDRESSES}
    after_values=dict(before_values)
    for c in credits:
        after_values[(c.route,c.mass_side,c.key_slot)]=mass_after
        after_values[(c.route,'observations',-1)]={1:9,2:17}[len(ROUTES[c.route][0])]
        after_values[(c.route,'error_sum',-1)]=error_after
        after_values[(c.route,'weight_sum',-1)]=weight_after
    before=NumericView(case,tuple((a,before_values[a]) for a in ADDRESSES))
    after=NumericView(case,tuple((a,after_values[a]) for a in ADDRESSES))
    diagnostic=DiagnosticHandle(token('numeric-fixture-diagnostic'),tuple(range(9)))
    p=NumericProposal(receipt,Delivery(case,tick,receipt.source_index,tick+1),credits,before,after,diagnostic,diagnostic)
    return p,before,NumericWitness(receipt,credits,(error_delta,)*9)


def fault_preflight():
    p,trusted,w=fixture(MANUAL_ROWS[0],'local_sparse_reference','shuffled_outcome')
    c=p.credits[0];mass=(c.route,c.mass_side,c.key_slot)
    obs=(c.route,'observations',-1);err=(c.route,'error_sum',-1);weight=(c.route,'weight_sum',-1)
    before_values=dict(p.before.fields);faults=[]
    signature=(p.before.typed_signature(),p.after.typed_signature(),trusted.typed_signature(),w)

    def rejected(name,operation):
        try:operation()
        except (ValueError,TypeError) as exc:
            require(signature==(p.before.typed_signature(),p.after.typed_signature(),trusted.typed_signature(),w),'Rejected numeric fault mutated inputs')
            faults.append({'fault':name,'rejected':True,'inputs_unchanged':True,'exception':type(exc).__name__})
        else:raise ValueError('Numeric fault accepted: '+name)

    for name,address,value in [('bool_observation',obs,True),('float_observation',obs,8.0),
            ('bool_mass',mass,True),('float_mass',mass,2.0),('int_error',err,8),('int_weight',weight,16),
            ('string_error',err,'8.0')]:
        rejected(name,lambda a=address,v=value:altered(p.after,{a:v}))
    rejected('nan_before',lambda:altered(p.before,{err:float('nan')}))
    rejected('nan_after',lambda:altered(p.after,{err:float('nan')}))
    for name,address,value in [('positive_infinity',err,float('inf')),('negative_infinity',err,-float('inf')),
            ('negative_mass',mass,-1),('negative_observation',obs,-1),('negative_error',err,-1.0),
            ('negative_weight',weight,-1.0),('negative_zero',err,-0.0)]:
        rejected(name,lambda a=address,v=value:altered(p.after,{a:v}))
    for name,address,value in [('mass_increment_wrong',mass,4),('observation_increment_wrong',obs,before_values[obs]+2),
            ('error_increment_wrong',err,8.5),('weight_increment_wrong',weight,18.0),
            ('mass_no_op',mass,2),('observation_no_op',obs,before_values[obs]),
            ('error_no_op',err,8.0),('weight_no_op',weight,16.0)]:
        rejected(name,lambda a=address,v=value:validate_numeric(replace(p,after=altered(p.after,{a:v})),trusted,w))
    rejected('opposite_mass_changed',lambda:validate_numeric(replace(p,after=altered(p.after,{(c.route,MASS[1],c.key_slot):3})),trusted,w))
    inactive=next(i for i in range(45) if i not in p.receipt.active)
    address=(inactive,'observations',-1)
    rejected('inactive_route_changed',lambda:validate_numeric(replace(p,after=altered(p.after,{address:before_values[address]+1})),trusted,w))
    rejected('inactive_key_changed',lambda:validate_numeric(replace(p,after=altered(p.after,{(c.route,c.mass_side,1-c.key_slot):3})),trusted,w))
    rejected('credit_key_drift',lambda:validate_numeric(replace(p,credits=(replace(c,key_slot=1-c.key_slot),)+p.credits[1:]),trusted,w))
    rejected('wrong_source',lambda:validate_numeric(replace(p,delivery=replace(p.delivery,source_index=(p.delivery.source_index+1)%512)),trusted,w))
    rejected('same_tick_delivery',lambda:validate_numeric(replace(p,delivery=replace(p.delivery,at=0)),trusted,w))
    rejected('diagnostic_drift',lambda:validate_numeric(replace(p,diagnostic_after=DiagnosticHandle(token('bad-diag'),tuple(range(9)))),trusted,w))
    rejected('numeric_absorption_no_op',lambda:validate_numeric(replace(p,after=altered(p.after,{err:8.0})),trusted,replace(w,error_increments=(float.fromhex('0x0.0000000000001p-1022'),)+w.error_increments[1:])))
    rejected('nonfinite_error_witness',lambda:replace(w,error_increments=(float('nan'),)+w.error_increments[1:]))
    rejected('nonpositive_error_witness',lambda:replace(w,error_increments=(0.0,)+w.error_increments[1:]))
    rejected('stale_trusted_before',lambda:validate_numeric(p,altered(trusted,{mass:3}),w))
    rejected('missing_field',lambda:replace(p.after,fields=p.after.fields[:-1]))
    rejected('duplicate_field',lambda:replace(p.after,fields=(p.after.fields[1],)+p.after.fields[1:]))
    rejected('mutable_values',lambda:replace(p.after,fields=list(p.after.fields)))
    rejected('foreign_owner',lambda:validate_numeric(replace(p,after=replace(p.after,case=replace(p.after.case,seed=1213))),trusted,w))
    rejected('witness_greater_than_weight',lambda:validate_numeric(p,trusted,replace(w,error_increments=(2.0,)+w.error_increments[1:])))
    return faults


def analyze(spec):
    require(spec['schema']=='sara-route-numeric-delta-preflight-selection-v1','Wrong numeric preflight schema')
    require(canonical(spec['manual_fixture']['rows'])==canonical(MANUAL_ROWS),'Hand-authored numeric rows changed')
    require(canonical(spec['manual_fixture']['before'])==canonical({'each_mass':2,'route_observations_by_width':{'1':8,'2':16},'error_sum':8.0,'weight_sum':16.0}),'Manual predecessor changed')
    require(canonical(spec['manual_fixture']['after_route_observations_by_width'])==canonical({'1':9,'2':17}),'Manual successor observations changed')
    require(canonical({k:spec['manual_fixture'][k] for k in ('seed','task','condition','engines','roles','transactions')})==canonical({'seed':1201,'task':'independent','condition':'changed','engines':['local_sparse_reference','independent_event_driven_scalar'],'roles':['candidate','permanent','shuffled_outcome'],'transactions':72}),'Manual case/role scope changed')
    require(spec['bounds']=={'fault_cases':40,'transactions':72,'numeric_active_field_checks':2592,'result_bytes':262144,
              'all_artifact_bytes':2097152,'learner_executions':0,'actual_learning_updates':0,'outcome_payload_reads':0,
              'prediction_scores':0,'resource_measurements':0},'Numeric fixture budget changed')
    receipts=[];ledger=sha256()
    for row,engine,role in product(MANUAL_ROWS,spec['manual_fixture']['engines'],spec['manual_fixture']['roles']):
        p,trusted,w=fixture(row,engine,role)
        signature=(p.before.typed_signature(),p.after.typed_signature())
        result=validate_numeric(p,trusted,w)
        require(signature==(p.before.typed_signature(),p.after.typed_signature()),'Numeric validation mutated views')
        ledger.update(canonical([row,engine,role,result])+b'\n');receipts.append(result)
    faults=fault_preflight()
    require([f['fault'] for f in faults]==spec['faults'],'Numeric fault roster changed')
    require(len(receipts)==72 and sum(r['active_field_checks'] for r in receipts)==2592,'Numeric fixture coverage changed')
    return {'schema':'sara-route-numeric-delta-preflight-result-v1',
            'summary':{'numeric_fixture_transactions':72,'active_field_checks':2592,'unchanged_field_checks':28728,
                       'faults_rejected':40,'inputs_preserved':True,'equality':'exact typed/float.hex; no tolerance'},
            'fixture_ledger_sha256':ledger.hexdigest(),'fault_injections':faults,
            'status':{'learner_executions':0,'actual_learning_updates':0,'samples_generated':0,'outcome_payload_reads':0,
                      'prediction_scores':0,'new_snapshots':0,'resource_measurements':0,'test_opened':False,
                      'runtime_numeric_updates_observed':False,'runtime_scalar_equality_tested':False,
                      'original_adaptive_component_passed':False,'execution_authorized':False,
                      'snn_advantage_tested':False,'promotion_authorized':False},
            'pending':['authenticated real receipt/key/outcome provenance','actual runtime delta observation',
                       'independent scalar numerical implementation/equality','independent-task/SNN causal evidence']}
