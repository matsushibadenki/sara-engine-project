"""Symbolic delta-ledger adapter. Never applies or computes learning updates."""
from dataclasses import dataclass, replace, FrozenInstanceError
from itertools import product
from hashlib import sha256

from sara_engine.evaluation.route_interface_preflight import (
    CaseKey,Receipt,Delivery,DiagnosticHandle,make_receipt,integer,digest_value,token,
)
from sara_engine.evaluation.route_fixed_allocation_contract import ROUTES,canonical,require

MASS=('mass_left','mass_right')
SCALARS=('observations','error_sum','weight_sum')
ADDRESSES=tuple(sorted([(i,f,k) for i,(inputs,_) in enumerate(ROUTES)
                        for k in range(2**len(inputs)) for f in MASS]+
                       [(i,f,-1) for i in range(45) for f in SCALARS]))


@dataclass(frozen=True,slots=True)
class CoreView:
    case:CaseKey
    fields:tuple
    channel:str='core'

    def __post_init__(self):
        require(type(self.case) is CaseKey and self.channel=='core','Wrong core owner/channel')
        require(type(self.fields) is tuple and len(self.fields)==435,'Incomplete or mutable core catalog')
        for cell in self.fields:
            require(type(cell) is tuple and len(cell)==2,'Mutable field cell')
            address,stamp=cell
            require(type(address) is tuple and len(address)==3,'Invalid field address')
            require(type(address[0]) is int and type(address[1]) is str and type(address[2]) is int,'Coerced field address')
            digest_value(stamp)
        require(tuple(c[0] for c in self.fields)==ADDRESSES,'Missing/duplicate/foreign field address')


@dataclass(frozen=True,slots=True)
class Credit:
    route:int
    key_slot:int
    mass_side:str

    def __post_init__(self):
        integer(self.route,45);integer(self.key_slot,2**len(ROUTES[self.route][0]))
        require(type(self.mass_side) is str and self.mass_side in MASS,'Unknown symbolic mass side')


def validate_credits(credits,receipt):
    require(type(credits) is tuple and all(type(c) is Credit for c in credits),'Mutable/invalid credits')
    require(tuple(c.route for c in credits)==receipt.active,'Active route order/credit budget differs')


def changed_addresses(credits):
    return frozenset([(c.route,c.mass_side,c.key_slot) for c in credits]+
                     [(c.route,f,-1) for c in credits for f in SCALARS])


@dataclass(frozen=True,slots=True)
class DeltaProposal:
    receipt:Receipt
    delivery:Delivery
    credits:tuple
    before:CoreView
    after:CoreView
    diagnostic_before:DiagnosticHandle
    diagnostic_after:DiagnosticHandle

    def __post_init__(self):
        require(type(self.receipt) is Receipt and type(self.delivery) is Delivery,'Invalid proposal provenance')
        validate_credits(self.credits,self.receipt)
        require(type(self.before) is CoreView and type(self.after) is CoreView,'Non-core delta views')
        require(type(self.diagnostic_before) is DiagnosticHandle and type(self.diagnostic_after) is DiagnosticHandle,'Invalid diagnostics')


@dataclass(frozen=True,slots=True)
class DiagnosticProposal:
    before:CoreView
    after:CoreView
    diagnostic_before:DiagnosticHandle
    diagnostic_after:DiagnosticHandle

    def __post_init__(self):
        require(type(self.before) is CoreView and type(self.after) is CoreView,'Invalid diagnostic core witness')
        require(type(self.diagnostic_before) is DiagnosticHandle and type(self.diagnostic_after) is DiagnosticHandle,'Invalid diagnostic witness')


@dataclass(frozen=True,slots=True)
class LedgerState:
    tick:int
    pending:object
    credits:tuple
    core:CoreView
    diagnostic:DiagnosticHandle


class DeltaLedger:
    """Accept immutable symbolic views after validation; no learner callback."""
    __slots__=('_case','_state')

    def __init__(self,case,core,diagnostic,start_tick=0):
        integer(start_tick)
        require(type(case) is CaseKey and type(core) is CoreView and core.case==case,'Wrong initial core owner')
        require(type(diagnostic) is DiagnosticHandle,'Wrong initial diagnostic owner')
        self._case=case
        self._state=LedgerState(start_tick,None,(),core,diagnostic)

    @property
    def state(self):return self._state

    def open(self,receipt,credits):
        state=self._state
        require(type(receipt) is Receipt and state.pending is None,'Duplicate or invalid pending receipt')
        require(receipt==make_receipt(self._case,state.tick),'Receipt/source/fixed allocation mismatch')
        validate_credits(credits,receipt)
        self._state=replace(state,pending=receipt,credits=credits)

    def accept(self,proposal):
        state=self._state
        require(type(proposal) is DeltaProposal and state.pending is not None,'No pending delta transaction')
        require(proposal.receipt==state.pending and proposal.credits==state.credits,'Receipt/key/mass credit drift')
        delivery=proposal.delivery
        require(delivery.case==self._case and delivery.purpose=='training','Foreign/evaluation delivery')
        require(delivery.receipt_id==state.pending.receipt_id and delivery.source_index==state.pending.source_index,'Receipt identity/source confusion')
        require(delivery.at==state.pending.receipt_id+1,'Wrong one-tick outcome time')
        require(proposal.before==state.core and proposal.after.case==self._case,'Stale view or foreign owner')
        require(proposal.diagnostic_before==state.diagnostic and proposal.diagnostic_after==state.diagnostic,'Core transaction changed diagnostic namespace')
        changed=frozenset(a[0] for a,b in zip(proposal.before.fields,proposal.after.fields) if a[1]!=b[1])
        expected=changed_addresses(state.credits)
        require(len(expected)==36 and changed==expected,'Missing/no-op/inactive/opposite/key field delta')
        receipt={'scope':'symbolic_delta_fixture','symbolic_changed_slots':len(changed),
                 'symbolic_unchanged_slots':435-len(changed),'actual_learning_updates':0,
                 'runtime_numeric_delta_tested':False,'original_adaptive_pass':False}
        # One bookkeeping replacement after all validation; no learner state writes.
        self._state=LedgerState(state.tick+1,None,(),proposal.after,state.diagnostic)
        return receipt

    def accept_diagnostic(self,proposal):
        state=self._state
        require(type(proposal) is DiagnosticProposal and state.pending is None,'Diagnostic transaction during pending receipt')
        require(proposal.before==state.core and proposal.after==state.core,'Diagnostic proposal changed core fields')
        require(proposal.diagnostic_before==state.diagnostic,'Stale diagnostic view')
        updated=replace(state,diagnostic=proposal.diagnostic_after)
        require(updated.core is state.core,'Diagnostic replacement lost core ownership')
        self._state=updated


def altered(view,changes):
    require(all(a in ADDRESSES for a in changes),'Foreign symbolic address')
    return replace(view,fields=tuple((a,changes.get(a,stamp)) for a,stamp in view.fields))


def fixture(case,tick):
    receipt=make_receipt(case,tick)
    # Slots are schema placeholders, not event keys or sampled outcome classes.
    credits=tuple(Credit(i,tick%(2**len(ROUTES[i][0])),MASS[(tick+j)%2]) for j,i in enumerate(receipt.active))
    root=canonical(case.matching_key()).decode()
    before=CoreView(case,tuple((a,token(root+':before:'+str(a))) for a in ADDRESSES))
    after=altered(before,{a:token(root+':after:'+str(tick)+':'+str(a)) for a in changed_addresses(credits)})
    diagnostic=DiagnosticHandle(token('symbolic-diagnostic'),tuple(range(9)))
    proposal=DeltaProposal(receipt,Delivery(case,tick,receipt.source_index,tick+1),credits,before,after,diagnostic,diagnostic)
    return DeltaLedger(case,before,diagnostic,tick),proposal


def fault_preflight():
    case=CaseKey(1201,'independent','changed','cumulative','uniform','local_sparse_reference','shuffled_outcome')
    _,p=fixture(case,0)
    require(p.receipt.source_index!=0,'Source/receipt fault fixture aliases')
    first=p.credits[0]
    mass=(first.route,first.mass_side,first.key_slot)
    faults=[]

    def rejected(name,operation,pending=True,committed=False):
        ledger,proposal=fixture(case,0)
        if pending or committed:ledger.open(proposal.receipt,proposal.credits)
        if committed:ledger.accept(proposal)
        before=ledger.state
        try:operation(ledger,proposal)
        except (ValueError,TypeError,FrozenInstanceError) as exc:
            require(ledger.state is before,'Rejected proposal changed adapter state')
            faults.append({'fault':name,'rejected':True,'adapter_state_unchanged':True,'exception':type(exc).__name__})
        else:raise ValueError('Injected fault accepted: '+name)

    rejected('source_as_receipt',lambda l,q:l.accept(replace(q,delivery=replace(q.delivery,receipt_id=q.receipt.source_index))))
    rejected('wrong_source',lambda l,q:l.accept(replace(q,delivery=replace(q.delivery,source_index=(q.delivery.source_index+1)%512))))
    rejected('same_tick',lambda l,q:l.accept(replace(q,delivery=replace(q.delivery,at=0))))
    rejected('late_delivery',lambda l,q:l.accept(replace(q,delivery=replace(q.delivery,at=2))))
    rejected('evaluation_delivery',lambda l,q:l.accept(replace(q,delivery=replace(q.delivery,purpose='evaluation'))))
    rejected('foreign_delivery_case',lambda l,q:l.accept(replace(q,delivery=replace(q.delivery,case=replace(case,role='candidate')))))
    rejected('missing_pending',lambda l,q:l.accept(q),pending=False)
    rejected('duplicate_pending',lambda l,q:l.open(q.receipt,q.credits))
    rejected('stale_before',lambda l,q:l.accept(replace(q,before=altered(q.before,{mass:token('stale')}))))
    rejected('foreign_after_owner',lambda l,q:l.accept(replace(q,after=replace(q.after,case=replace(case,seed=1213)))))
    rejected('diagnostic_as_core_channel',lambda l,q:replace(q.after,channel='diagnostic'))
    rejected('changed_credit_key',lambda l,q:l.accept(replace(q,credits=(replace(q.credits[0],key_slot=1-q.credits[0].key_slot),)+q.credits[1:])))
    rejected('changed_credit_side',lambda l,q:l.accept(replace(q,credits=(replace(q.credits[0],mass_side=MASS[1]),)+q.credits[1:])))
    rejected('reordered_credits',lambda l,q:replace(q,credits=tuple(reversed(q.credits))))
    def restore(q,addresses):
        old=dict(q.before.fields)
        return replace(q,after=altered(q.after,{a:old[a] for a in addresses}))
    rejected('missing_active_write',lambda l,q:l.accept(restore(q,changed_addresses(q.credits[:1]))))
    for name,field in (('mass_no_op',first.mass_side),('observation_no_op','observations'),('error_no_op','error_sum'),('weight_no_op','weight_sum')):
        address=(first.route,field,first.key_slot if field in MASS else -1)
        rejected(name,lambda l,q,a=address:l.accept(restore(q,[a])))
    rejected('unselected_mass_write',lambda l,q:l.accept(replace(q,after=altered(q.after,{(first.route,MASS[1],first.key_slot):token('opposite')}))))
    inactive=next(i for i in range(45) if i not in p.receipt.active)
    rejected('inactive_route_write',lambda l,q:l.accept(replace(q,after=altered(q.after,{(inactive,'observations',-1):token('inactive-route')}))))
    rejected('inactive_key_write',lambda l,q:l.accept(replace(q,after=altered(q.after,{(first.route,first.mass_side,1-first.key_slot):token('inactive-key')}))))
    rejected('extra_field',lambda l,q:replace(q.after,fields=q.after.fields+(q.after.fields[0],)))
    rejected('missing_field',lambda l,q:replace(q.after,fields=q.after.fields[:-1]))
    rejected('duplicate_field',lambda l,q:replace(q.after,fields=(q.after.fields[1],)+q.after.fields[1:]))
    rejected('mutable_field_alias',lambda l,q:replace(q.after,fields=list(q.after.fields)))
    rejected('diagnostic_drift_during_core_commit',lambda l,q:l.accept(replace(q,diagnostic_after=DiagnosticHandle(token('changed-diagnostic'),tuple(range(9))))))
    rejected('duplicate_commit',lambda l,q:l.accept(q),committed=True)
    rejected('numeric_stamp_payload',lambda l,q:altered(q.after,{mass:42}))
    rejected('diagnostic_proposal_changes_core',lambda l,q:l.accept_diagnostic(DiagnosticProposal(q.before,q.after,q.diagnostic_before,q.diagnostic_after)),pending=False)
    rejected('frozen_core_mutation',lambda l,q:setattr(q.before,'fields',()))
    return faults


def analyze(spec):
    require(spec['schema']=='sara-route-delta-ledger-preflight-selection-v1','Wrong delta-ledger preflight')
    require(spec['bounds']=={'fault_cases':31,'per_result_bytes':262144,'all_artifact_bytes':2097152,
                            'learner_executions':0,'outcome_payload_reads':0,'actual_learning_updates':0,
                            'prediction_scores':0,'resource_measurements':0},'Finite ledger budget changed')
    require(spec['fixtures']=={'seed':1201,'task':'independent','condition':'changed','ticks':[0,63,64,511,512,1023],
              'count_factors':['cumulative','discounted'],'error_factors':['uniform','public_exponential'],
              'engines':['local_sparse_reference','independent_event_driven_scalar'],'roles':['candidate','permanent','shuffled_outcome'],
              'transactions':144,'four_symbolic_fields_per_route':4,'symbolic_stamp_changes':5184,
              'note':'Each boundary is a fresh synthetic metadata fixture, not continuation of any training run.'},'Fixture scope changed')
    receipts=[];ledger_hash=sha256()
    for history,error,engine,role,tick in product(spec['fixtures']['count_factors'],spec['fixtures']['error_factors'],
                                                spec['fixtures']['engines'],spec['fixtures']['roles'],spec['fixtures']['ticks']):
        case=CaseKey(1201,'independent','changed',history,error,engine,role)
        ledger,proposal=fixture(case,tick)
        ledger.open(proposal.receipt,proposal.credits)
        result=ledger.accept(proposal)
        core=ledger.state.core
        ledger.accept_diagnostic(DiagnosticProposal(core,core,ledger.state.diagnostic,DiagnosticHandle(token('new-diagnostic'),tuple(range(9,18)))))
        require(ledger.state.core is core,'Diagnostic replacement changed core object')
        ledger_hash.update(canonical([history,error,engine,role,tick,proposal.receipt.source_index,result])+b'\n')
        receipts.append(result)
    faults=fault_preflight()
    require([f['fault'] for f in faults]==spec['faults'],'Fault roster changed')
    require(len(receipts)==144 and sum(r['symbolic_changed_slots'] for r in receipts)==5184,'Symbolic coverage failed')
    return {'schema':'sara-route-delta-ledger-preflight-result-v1','summary':{'catalog_slots':len(ADDRESSES),
             'symbolic_transactions':144,'symbolic_stamp_changes':5184,'symbolic_unchanged_slots_per_transaction':399,
             'faults_rejected':31,'rejection_preserves_state_identity':True,'diagnostic_replacement_preserves_core_identity':True},
            'fixture_ledger_sha256':ledger_hash.hexdigest(),'fault_injections':faults,
            'status':{'learner_executions':0,'outcome_payload_reads':0,'actual_learning_updates':0,'samples_generated':0,
                      'prediction_scores':0,'new_snapshots':0,'resource_measurements':0,'test_opened':False,
                      'runtime_numeric_deltas_tested':False,'runtime_scalar_equality_tested':False,
                      'original_adaptive_component_passed':False,'execution_authorized':False,
                      'snn_advantage_tested':False,'promotion_authorized':False},
            'pending':['actual key/outcome provenance','numeric signs/magnitudes/no-op deltas',
                       'independent scalar numerical equality','real runtime ownership and independent-task evidence']}
