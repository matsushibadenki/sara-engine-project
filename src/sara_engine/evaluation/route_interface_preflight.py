"""Immutable metadata interfaces and fault preflight; contains no learner state."""
from dataclasses import dataclass, replace, FrozenInstanceError
from functools import lru_cache
from hashlib import sha256
from itertools import product

from sara_engine.evaluation.route_fixed_allocation_contract import active_routes, fixed_readout, canonical, require
from sara_engine.evaluation.route_fixed_control_contract import permutation_indices, SEEDS, ROLES, ENGINES


def integer(value, limit=1024):
    require(type(value) is int and 0<=value<limit,'Invalid bounded integer metadata')


def digest_value(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Invalid opaque digest')


def route_tuple(value):
    require(type(value) is tuple and len(value)<=45 and len(set(value))==len(value),'Mutable/duplicate route addresses')
    for i in value:integer(i,45)


@dataclass(frozen=True,slots=True)
class CaseKey:
    seed:int
    task:str
    condition:str
    history:str
    error_weight:str
    engine:str
    role:str

    def __post_init__(self):
        require(type(self.seed) is int and self.seed in SEEDS,'Unknown seed')
        for value,allowed in ((self.task,('independent','interacting')),(self.condition,('stationary','changed')),
                              (self.history,('cumulative','discounted')),(self.error_weight,('uniform','public_exponential')),
                              (self.engine,ENGINES),(self.role,ROLES)):
            require(type(value) is str and value in allowed,'Unknown case metadata')

    def matching_key(self):
        return (self.seed,self.task,self.condition,self.history,self.error_weight,self.role)


@lru_cache(maxsize=5)
def source_map(seed):
    return tuple(permutation_indices(seed))


@dataclass(frozen=True,slots=True)
class Receipt:
    case:CaseKey
    receipt_id:int
    source_index:int
    active:tuple
    readout:tuple

    def __post_init__(self):
        require(type(self.case) is CaseKey,'Invalid receipt case')
        integer(self.receipt_id);integer(self.source_index)
        route_tuple(self.active);route_tuple(self.readout)


@dataclass(frozen=True,slots=True)
class Delivery:
    case:CaseKey
    receipt_id:int
    source_index:int
    at:int
    purpose:str='training'

    def __post_init__(self):
        require(type(self.case) is CaseKey,'Invalid delivery case')
        integer(self.receipt_id);integer(self.source_index);integer(self.at,1025)
        require(self.purpose in ('training','evaluation') and type(self.purpose) is str,'Invalid delivery purpose')


def make_receipt(case,tick):
    require(type(case) is CaseKey,'Invalid case descriptor')
    integer(tick)
    source=source_map(case.seed)[tick] if case.role=='shuffled_outcome' else tick
    return Receipt(case,tick,source,active_routes(tick),fixed_readout(tick))


class ReceiptGate:
    """Receipt/provenance state machine only; never stores an outcome value."""
    __slots__=('_case','_steps','_pending')

    def __init__(self,case):
        require(type(case) is CaseKey,'Invalid gate descriptor')
        self._case=case;self._steps=0;self._pending=None

    def snapshot(self):
        return (self._steps,self._pending)

    def open(self,receipt):
        require(type(receipt) is Receipt and self._pending is None,'Invalid or duplicate pending receipt')
        require(receipt.case==self._case and receipt.receipt_id==self._steps,'Foreign receipt or nonmonotonic identity')
        require(receipt==make_receipt(self._case,self._steps),'Source or fixed address mismatch')
        self._pending=receipt

    def deliver(self,delivery):
        require(type(delivery) is Delivery and self._pending is not None,'Delivery lacks own pending receipt')
        p=self._pending
        require(delivery.case==self._case,'Foreign delivery case')
        require(delivery.purpose=='training','Evaluation provenance cannot enter training gate')
        require(delivery.receipt_id==p.receipt_id,'Source index is not receipt identity')
        require(delivery.source_index==p.source_index,'Wrong shuffled label provenance')
        require(delivery.at==p.receipt_id+1,'Outcome is not own one-tick delayed delivery')
        self._pending=None;self._steps+=1


@dataclass(frozen=True,slots=True)
class CoreHandle:
    digest:str

    def __post_init__(self):digest_value(self.digest)


@dataclass(frozen=True,slots=True)
class DiagnosticHandle:
    digest:str
    pool:tuple

    def __post_init__(self):
        digest_value(self.digest);route_tuple(self.pool)


@dataclass(frozen=True,slots=True)
class StateBundle:
    core:CoreHandle
    diagnostic:DiagnosticHandle

    def __post_init__(self):
        require(type(self.core) is CoreHandle and type(self.diagnostic) is DiagnosticHandle,'Core/diagnostic namespace mixing')


def replace_diagnostic(bundle,diagnostic):
    require(type(bundle) is StateBundle and type(diagnostic) is DiagnosticHandle,'Diagnostic update has no core-writing parameter')
    result=StateBundle(bundle.core,diagnostic)
    require(result.core is bundle.core,'Diagnostic replacement changed core ownership')
    return result


@dataclass(frozen=True,slots=True)
class ComparisonPacket:
    case:CaseKey
    receipt_ledger:str
    core:str
    primary:str
    shadow:str
    scope:str='metadata_fixture'

    def __post_init__(self):
        require(type(self.case) is CaseKey,'Invalid comparison descriptor')
        for value in (self.receipt_ledger,self.core,self.primary,self.shadow):digest_value(value)
        require(self.scope=='metadata_fixture','Runtime comparison claim is not authorized')


def compare_same_role(left,right):
    require(type(left) is ComparisonPacket and type(right) is ComparisonPacket,'Invalid comparison packet')
    require(left.case.matching_key()==right.case.matching_key(),'Cross-role or cross-condition comparison')
    require({left.case.engine,right.case.engine}==set(ENGINES),'Independent engine pair required')
    for field in ('receipt_ledger','core','primary','shadow'):
        require(getattr(left,field)==getattr(right,field),'Opaque fingerprint mismatch: '+field)
    return {'scope':'metadata_fixture','case':list(left.case.matching_key()),'equal':True,
            'runtime_scalar_equality_tested':False,'original_adaptive_pass':False}


def token(label):
    return sha256(('opaque-interface-fixture:'+label).encode()).hexdigest()


def fixture_pair(case):
    # Shared symbolic fingerprints validate only interface matching, not numerical engines.
    root=canonical(case.matching_key()).decode()
    values=[token(root+':'+field) for field in ('ledger','core','primary','shadow')]
    return (ComparisonPacket(replace(case,engine=ENGINES[0]),*values),
            ComparisonPacket(replace(case,engine=ENGINES[1]),*values))


def fault_preflight():
    case=CaseKey(1201,'independent','changed','cumulative','uniform',ENGINES[0],'shuffled_outcome')
    receipt=make_receipt(case,0)
    require(receipt.source_index!=0,'Registered fault fixture requires distinct source and receipt')
    delivery=Delivery(case,0,receipt.source_index,1)
    core=CoreHandle(token('core'));diagnostic=DiagnosticHandle(token('diagnostic'),tuple(range(9)))
    bundle=StateBundle(core,diagnostic)
    left,right=fixture_pair(case)
    faults=[]

    def rejected(name,operation,pending=False):
        gate=ReceiptGate(case)
        if pending:gate.open(receipt)
        before=gate.snapshot()
        try:operation(gate)
        except (ValueError,TypeError,FrozenInstanceError) as exc:
            require(gate.snapshot()==before and core.digest==token('core'),'Rejected fault changed state')
            faults.append({'fault':name,'rejected':True,'state_unchanged':True,'exception':type(exc).__name__})
        else:raise ValueError('Fault accepted: '+name)

    rejected('source_used_as_receipt_id',lambda g:g.deliver(replace(delivery,receipt_id=receipt.source_index)),True)
    rejected('wrong_source_index',lambda g:g.deliver(replace(delivery,source_index=(receipt.source_index+1)%512)),True)
    rejected('same_tick_outcome',lambda g:g.deliver(replace(delivery,at=0)),True)
    rejected('late_outcome',lambda g:g.deliver(replace(delivery,at=2)),True)
    rejected('boolean_receipt_id',lambda g:g.deliver(replace(delivery,receipt_id=False)),True)
    rejected('duplicate_pending_receipt',lambda g:g.open(receipt),True)
    rejected('outcome_without_pending',lambda g:g.deliver(delivery))
    rejected('evaluation_label_to_training',lambda g:g.deliver(replace(delivery,purpose='evaluation')),True)
    rejected('wrong_active_routes',lambda g:g.open(replace(receipt,active=receipt.active[:-1])))
    rejected('reordered_active_routes',lambda g:g.open(replace(receipt,active=tuple(reversed(receipt.active)))))
    rejected('wrong_fixed_readout',lambda g:g.open(replace(receipt,readout=tuple(reversed(receipt.readout)))))
    for name,field,value in (('foreign_role_envelope','role','candidate'),('foreign_seed_envelope','seed',1213),
                             ('foreign_engine_envelope','engine',ENGINES[1])):
        rejected(name,lambda g,f=field,v=value:g.open(replace(receipt,case=replace(case,**{f:v}))))
    rejected('mutable_active_route_alias',lambda g:g.open(replace(receipt,active=list(receipt.active))))
    rejected('core_as_diagnostic_state',lambda g:replace_diagnostic(bundle,core))
    rejected('mutable_diagnostic_pool_alias',lambda g:DiagnosticHandle(token('diagnostic'),list(range(9))))
    rejected('frozen_core_mutation',lambda g:setattr(core,'digest',token('tampered')))
    for name,field,value in (('comparison_cross_role','role','candidate'),('comparison_cross_seed','seed',1213),
                             ('comparison_cross_task','task','interacting'),('comparison_cross_condition','condition','stationary'),
                             ('comparison_cross_history','history','discounted'),('comparison_cross_error_weight','error_weight','public_exponential'),
                             ('comparison_same_engine','engine',ENGINES[0])):
        rejected(name,lambda g,f=field,v=value:compare_same_role(left,replace(right,case=replace(right.case,**{f:v}))))
    for name,field in (('comparison_core_mismatch','core'),('comparison_primary_mismatch','primary'),
                       ('comparison_shadow_mismatch','shadow'),('comparison_receipt_ledger_mismatch','receipt_ledger')):
        rejected(name,lambda g,f=field:compare_same_role(left,replace(right,**{f:token('mismatch')})))
    rejected('comparison_invalid_digest',lambda g:replace(right,core='not-a-digest'))
    rejected('comparison_runtime_claim',lambda g:replace(right,scope='runtime_scalar_equality'))
    updated=replace_diagnostic(bundle,DiagnosticHandle(token('new-diagnostic'),tuple(range(9,18))))
    require(updated.core is core and bundle.diagnostic is diagnostic,'Diagnostic update leaked to core/original bundle')
    return faults


def analyze(spec):
    require(spec['schema']=='sara-route-interface-preflight-selection-v1','Wrong preflight schema')
    expected_bounds={'metadata_trace_seeds':5,'metadata_trace_roles':3,'receipt_pairs_per_trace':1024,
                     'receipt_pairs_total':15360,'same_role_comparison_pairs':240,'fault_cases':31,
                     'learner_executions':0,'outcome_payload_reads':0,'actual_learning_updates':0,
                     'prediction_scores':0,'result_bytes':262144,'all_artifact_bytes':2097152}
    require(canonical(spec['bounds'])==canonical(expected_bounds),'Finite preflight budget changed')
    traces=[]
    for seed,role in product(SEEDS,ROLES):
        case=CaseKey(seed,'independent','changed','cumulative','uniform',ENGINES[0],role)
        gate=ReceiptGate(case);ledger=sha256()
        for tick in range(1024):
            receipt=make_receipt(case,tick)
            gate.open(receipt)
            gate.deliver(Delivery(case,tick,receipt.source_index,tick+1))
            ledger.update(canonical([tick,receipt.source_index,tick+1,receipt.active,receipt.readout])+b'\n')
        require(gate.snapshot()==(1024,None),'Pending metadata did not finish')
        traces.append({'seed':seed,'role':role,'receipts':1024,'outcomes_metadata':1024,'ledger_sha256':ledger.hexdigest()})
    comparisons=[]
    for seed,task,condition,history,error,role in product(SEEDS,('independent','interacting'),('stationary','changed'),
                                  ('cumulative','discounted'),('uniform','public_exponential'),ROLES):
        comparisons.append(compare_same_role(*fixture_pair(CaseKey(seed,task,condition,history,error,ENGINES[0],role))))
    faults=fault_preflight()
    require([f['fault'] for f in faults]==spec['faults'] and len(faults)==31,'Fault roster changed')
    require(len(comparisons)==240 and len(traces)==15,'Metadata coverage changed')
    return {'schema':'sara-route-interface-preflight-result-v1','traces':traces,'symbolic_comparisons':comparisons,
            'fault_injections':faults,'summary':{'receipt_metadata_pairs':15360,'comparison_fixture_pairs':240,
                'faults_rejected':31,'rejections_preserve_state':True,'diagnostic_replacement_preserves_core_identity':True},
            'status':{'learner_executions':0,'samples_generated':0,'outcome_payload_reads':0,'actual_learning_updates':0,
                'prediction_scores':0,'new_snapshots':0,'resource_measurements':0,'test_opened':False,
                'runtime_scalar_equality_tested':False,'original_adaptive_component_passed':False,
                'execution_authorized':False,'snn_advantage_tested':False,'promotion_authorized':False},
            'pending':['actual independent scalar numerical implementation/equality','real local update ledger and payload validation',
                       'runtime ownership/isolation checks in future learners','independent-task/SNN-specific causal evidence']}
