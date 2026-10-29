"""Two-phase hand-witness sealing; no learners or outcome payloads."""
from dataclasses import dataclass,replace,FrozenInstanceError
from hashlib import sha256
from itertools import product

from sara_engine.evaluation.route_error_binding_preflight import (
    fixture,bind_numeric,LookupCredit,probability_type,
)
from sara_engine.evaluation.route_numeric_delta_preflight import NumericView,MANUAL_ROWS,altered
from sara_engine.evaluation.route_interface_preflight import Receipt,Delivery,make_receipt,integer,digest_value
from sara_engine.evaluation.route_delta_ledger_preflight import MASS
from sara_engine.evaluation.route_fixed_allocation_contract import ROUTES,canonical,require


@dataclass(frozen=True,slots=True)
class PreLookup:
    route:int
    key_slot:int
    allocated:bool
    probability:object

    def __post_init__(self):
        integer(self.route,45);integer(self.key_slot,2**len(ROUTES[self.route][0]))
        require(type(self.allocated) is bool,'Invalid allocation witness')
        probability_type(self.probability)


@dataclass(frozen=True,slots=True)
class PreOutcome:
    receipt:Receipt
    before:NumericView
    lookups:tuple

    def __post_init__(self):
        require(type(self.receipt) is Receipt and type(self.before) is NumericView,'Invalid pre-outcome packet')
        require(self.before.case==self.receipt.case,'Foreign pre-state')
        require(type(self.lookups) is tuple and len(self.lookups)==9 and all(type(l) is PreLookup for l in self.lookups),'Prelookup must not carry post-outcome credit')
        require(tuple(l.route for l in self.lookups)==self.receipt.active,'Prelookup route ordering differs')


@dataclass(frozen=True,slots=True)
class DelayedClass:
    delivery:Delivery
    output_sides:tuple

    def __post_init__(self):
        require(type(self.delivery) is Delivery,'Invalid delayed result identity')
        require(type(self.output_sides) is tuple and len(self.output_sides)==3 and all(type(s) is str and s in MASS for s in self.output_sides),'Invalid delayed output-side vector')


@dataclass(frozen=True,slots=True)
class SealCapability:
    digest:str

    def __post_init__(self):digest_value(self.digest)


@dataclass(frozen=True,slots=True)
class PendingSeal:
    pre:PreOutcome
    capability:SealCapability


@dataclass(frozen=True,slots=True)
class PhaseState:
    next_tick:int
    pending:object=None


def fingerprint(pre):
    r=pre.receipt;c=r.case
    data=[list(c.matching_key())+[c.engine],r.receipt_id,r.source_index,r.active,r.readout,
          pre.before.typed_signature(),[(l.route,l.key_slot,l.allocated,None if l.probability is None else l.probability.hex()) for l in pre.lookups]]
    return sha256(canonical(data)).hexdigest()


def verify_pre(pre):
    require(type(pre) is PreOutcome,'Only a class-free pre-outcome packet may be sealed')
    r=pre.receipt
    require(r==make_receipt(r.case,r.receipt_id),'Pre receipt/source/fixed addresses differ')
    values=dict(pre.before.fields)
    prior=2**(r.receipt_id//64) if r.case.history=='discounted' else 1
    for l in pre.lookups:
        n0=values[(l.route,'mass_left',l.key_slot)];n1=values[(l.route,'mass_right',l.key_slot)]
        if l.allocated:
            p=(n1+prior)/(n0+n1+2*prior)
            require(type(l.probability) is float and l.probability.hex()==p.hex(),'Pre-state probability mismatch')
        else:require(n0==n1==0 and l.probability is None,'Absent-key witness mismatch')


class TwoPhaseGate:
    """Process-local capability and phase order, never actual learning."""
    __slots__=('_case','_state')

    def __init__(self,case,start_tick=0):
        integer(start_tick)
        # Fixed receipt construction validates the case descriptor without class data.
        make_receipt(case,start_tick)
        self._case=case;self._state=PhaseState(start_tick)

    @property
    def state(self):return self._state

    def seal(self,pre):
        state=self._state
        require(state.pending is None,'Duplicate pending seal')
        verify_pre(pre)
        require(pre.receipt.case==self._case and pre.receipt.receipt_id==state.next_tick,'Foreign or completed receipt')
        cap=SealCapability(fingerprint(pre))
        self._state=PhaseState(state.next_tick,PendingSeal(pre,cap))
        return cap

    def finish(self,cap,result,proposal,witness):
        state=self._state
        require(state.pending is not None,'Result has no prior seal')
        pending=state.pending;pre=pending.pre;r=pre.receipt
        require(type(cap) is SealCapability and cap is pending.capability,'Copied, foreign or altered seal capability')
        require(cap.digest==fingerprint(pre),'Sealed pre-witness changed')
        require(type(result) is DelayedClass,'Missing delayed class record')
        d=result.delivery
        require(d.case==self._case and d.purpose=='training','Foreign/evaluation result')
        require(d.receipt_id==r.receipt_id and d.source_index==r.source_index,'Receipt/source substitution')
        require(d.at==r.receipt_id+1,'Result must arrive at own tick+1')
        require(proposal.receipt==r and proposal.delivery==d,'Post proposal identity mismatch')
        require(proposal.before.case==pre.before.case and proposal.before.typed_signature()==pre.before.typed_signature(),'Post-outcome prior-state replacement')
        require(len(proposal.credits)==9 and len(witness.error_increments)==9,'Post credit count differs')
        records=[]
        for l,c,increment in zip(pre.lookups,proposal.credits,witness.error_increments):
            require(c.route==l.route and c.key_slot==l.key_slot,'Post-outcome key replacement')
            require(c.mass_side==result.output_sides[ROUTES[c.route][1]],'Class credit does not match own result')
            records.append(LookupCredit(c,l.allocated,l.probability,increment))
        bound=bind_numeric(proposal,pre.before,witness,tuple(records),tuple(l.allocated for l in pre.lookups),result.output_sides)
        receipt={'scope':'hand_fixture_two_phase','seal_sha256':cap.digest,'route_bindings':bound['route_bindings'],
                 'actual_learning_updates':0,'runtime_causal_provenance_authenticated':False,
                 'runtime_scalar_equality_tested':False,'original_adaptive_pass':False}
        self._state=PhaseState(state.next_tick+1)
        return receipt


def frame_for(args):
    p,b,w,records,flags,sides=args
    pre=PreOutcome(p.receipt,b,tuple(PreLookup(c.route,c.key_slot,a,r.probability) for c,a,r in zip(w.credits,flags,records)))
    return pre,DelayedClass(p.delivery,sides)


def fault_preflight():
    args=fixture(MANUAL_ROWS[0],'local_sparse_reference','shuffled_outcome');p,b,w,records,flags,sides=args
    pre,result=frame_for(args);case=p.receipt.case;faults=[]
    def rejected(name,operation,pending=True,finished=False):
        gate=TwoPhaseGate(case)
        cap=gate.seal(pre) if pending or finished else SealCapability('0'*64)
        if finished:gate.finish(cap,result,p,w)
        state=gate.state
        try:operation(gate,cap)
        except (ValueError,TypeError,FrozenInstanceError) as exc:
            require(gate.state is state,'Rejected two-phase request changed state')
            faults.append({'fault':name,'rejected':True,'state_unchanged':True,'exception':type(exc).__name__})
        else:raise ValueError('Fault accepted: '+name)
    rejected('premature_class_keyword',lambda g,c:PreOutcome(pre.receipt,b,pre.lookups,output_sides=sides),False)
    rejected('premature_error_keyword',lambda g,c:PreLookup(0,0,True,.5,error_increment=.25),False)
    rejected('post_credit_in_prelookup',lambda g,c:replace(pre,lookups=(records[0],)+pre.lookups[1:]),False)
    rejected('result_before_seal',lambda g,c:g.finish(c,result,p,w),False)
    rejected('duplicate_seal',lambda g,c:g.seal(pre))
    rejected('copied_token',lambda g,c:g.finish(replace(c),result,p,w))
    def foreign(g,c):
        other=TwoPhaseGate(case);other_cap=other.seal(pre);g.finish(other_cap,result,p,w)
    rejected('foreign_token',foreign)
    rejected('post_outcome_reseal',lambda g,c:g.seal(pre),finished=True)
    rejected('replacement_pre_argument',lambda g,c:g.finish(c,result,p,w,pre_lookup=pre.lookups))
    rejected('frozen_pre_probability',lambda g,c:setattr(pre.lookups[0],'probability',.75))
    a=(pre.lookups[0].route,'mass_left',pre.lookups[0].key_slot)
    rejected('changed_before_view',lambda g,c:g.finish(c,result,replace(p,before=altered(b,{a:3})),w))
    rejected('changed_key_credit',lambda g,c:g.finish(c,result,replace(p,credits=(replace(p.credits[0],key_slot=1-p.credits[0].key_slot),)+p.credits[1:]),w))
    rejected('replacement_allocation_argument',lambda g,c:g.finish(c,result,p,w,allocation_flags=(False,)*9))
    rejected('wrong_receipt',lambda g,c:g.finish(c,replace(result,delivery=replace(result.delivery,receipt_id=1)),p,w))
    rejected('source_as_receipt',lambda g,c:g.finish(c,replace(result,delivery=replace(result.delivery,receipt_id=p.receipt.source_index)),p,w))
    rejected('wrong_source',lambda g,c:g.finish(c,replace(result,delivery=replace(result.delivery,source_index=(p.receipt.source_index+1)%512)),p,w))
    rejected('same_tick',lambda g,c:g.finish(c,replace(result,delivery=replace(result.delivery,at=0)),p,w))
    rejected('late_result',lambda g,c:g.finish(c,replace(result,delivery=replace(result.delivery,at=2)),p,w))
    rejected('evaluation_result',lambda g,c:g.finish(c,replace(result,delivery=replace(result.delivery,purpose='evaluation')),p,w))
    rejected('foreign_result_case',lambda g,c:g.finish(c,replace(result,delivery=replace(result.delivery,case=replace(case,seed=1213))),p,w))
    rejected('inconsistent_class_vector',lambda g,c:g.finish(c,replace(result,output_sides=('mass_right',)*3),p,w))
    rejected('stale_error_witness',lambda g,c:g.finish(c,result,p,replace(w,error_increments=(.5,)+w.error_increments[1:])))
    rejected('malformed_class_vector',lambda g,c:DelayedClass(result.delivery,list(sides)))
    rejected('duplicate_finish',lambda g,c:g.finish(c,result,p,w),finished=True)
    rejected('wrong_after_value',lambda g,c:g.finish(c,result,replace(p,after=altered(p.after,{(p.credits[0].route,'error_sum',-1):8.5})),w))
    rejected('tampered_token_digest',lambda g,c:g.finish(replace(c,digest='0'*64),result,p,w))
    return faults


def analyze(spec):
    require(spec['schema']=='sara-route-two-phase-preflight-selection-v1','Wrong two-phase contract')
    require(spec['bounds']=={'fault_cases':26,'result_bytes':262144,'all_artifact_bytes':2097152,
             'learner_executions':0,'actual_learning_updates':0,'outcome_payload_reads':0,'prediction_scores':0},'Two-phase budget changed')
    sealed=[];digest=sha256();bindings=0
    for row,engine,role in product(MANUAL_ROWS,('local_sparse_reference','independent_event_driven_scalar'),
                                   ('candidate','permanent','shuffled_outcome')):
        args=fixture(row,engine,role);p,b,w,_,_,_=args;pre,result=frame_for(args)
        gate=TwoPhaseGate(p.receipt.case,row[0]);cap=gate.seal(pre)
        receipt=gate.finish(cap,result,p,w)
        require(gate.state.pending is None and gate.state.next_tick==row[0]+1,'Seal not consumed')
        bindings+=receipt['route_bindings'];digest.update(canonical([row,engine,role,receipt])+b'\n')
    for row in spec['hand_pre_rows']:
        tick,history,error,allocated,n0,n1,side,probability,increment=row
        manual=next(r for r in MANUAL_ROWS if r[:3]==[tick,history,error])
        pre,_=frame_for(fixture(manual,'local_sparse_reference','candidate'))
        first=pre.lookups[0];key=first.key_slot;r=first.route
        before=altered(pre.before,{(r,'mass_left',key):n0,(r,'mass_right',key):n1})
        # Current-class and error fields from the fixture manifest are deliberately unused.
        pre=replace(pre,before=before,lookups=(replace(first,allocated=allocated,probability=probability),)+pre.lookups[1:])
        gate=TwoPhaseGate(pre.receipt.case,tick);sealed.append(gate.seal(pre).digest)
    faults=fault_preflight()
    require([f['fault'] for f in faults]==spec['faults'] and len(sealed)==11 and bindings==648,'Two-phase coverage changed')
    return {'schema':'sara-route-two-phase-preflight-result-v1','summary':{'finished_fixture_transactions':72,
            'route_bindings':648,'pre_only_records':11,'faults_rejected':26,'rejection_preserves_state_identity':True},
            'fixture_digest':digest.hexdigest(),'pre_only_seals':sealed,'fault_injections':faults,
            'status':{'learner_executions':0,'actual_learning_updates':0,'task_samples_generated':0,'outcome_payload_reads':0,
                'prediction_scores':0,'new_snapshots':0,'resource_measurements':0,'test_opened':False,
                'runtime_causal_provenance_authenticated':False,'runtime_scalar_equality_tested':False,
                'original_adaptive_component_passed':False,'execution_authorized':False,'snn_advantage_tested':False,'promotion_authorized':False},
            'pending':['trusted real observation clock/allocation/label provenance','actual learner and scalar numerical equality',
                       'independent-task/SNN-specific causal evidence']}
