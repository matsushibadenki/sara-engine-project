"""Hand-built encoding fixtures for logical layout, never candidate/model calls."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from sara_engine.evaluation.route_capture_candidate import CapturePair,PhaseCapture
from sara_engine.evaluation.route_outcome_admission import Input
from sara_engine.evaluation.route_capture_layout import ROUTES,validate,canonical,LayoutError

SELECTION_SHA='3e033e493426068568660e010c0cca5b33d2da04afe8a998ab7889c613eb14ee'


def require(value,message):
    if not value:raise ValueError(message)


def hand_encoding(value):
    """Fixture assembly only; intentionally independent of candidate._primitive."""
    if value is None:return ('none',)
    if type(value) is bool:return ('bool',value)
    if type(value) is int:return ('int',str(value))
    if type(value) is float:return ('float',value.hex())
    if type(value) is str:return ('str',value)
    if type(value) is tuple:return ('tuple',tuple(map(hand_encoding,value)))
    if type(value) is list:return ('list',tuple(map(hand_encoding,value)))
    if type(value) is dict:
        pairs=tuple((hand_encoding(k),hand_encoding(v)) for k,v in value.items())
        return ('dict',tuple(sorted(pairs,key=lambda p:canonical(p[0]))))
    raise ValueError('Unsupported hand fixture value')


def fixture(spec,variant):
    h=spec['hand'];events=tuple(h['events']);active=tuple(h['active'])
    tables=[{} for _ in range(45)];obs=[0]*45;errors=[0.0]*45;weights=[0.0]*45
    for route in active[1:]:
        key=tuple(events[i] for i in ROUTES[route][0])
        masses=[0,0] if route==1 else [1,2]
        tables[route][key]=masses
        if route!=1:obs[route]=3;errors[route]=0.75;weights[route]=3.0
    if variant=='allocated_zero_first_key':tables[0][(events[0],)]=[0,0]
    tables[44][tuple(h['sentinel_key'])]=list(h['sentinel_masses']);obs[44]=16;errors[44]=4.0;weights[44]=16.0
    before={'routes':ROUTES,'tables':tables,'observations':obs,'errors':errors,'weights':weights,'pending':None,
            'clock':h['clock_before'],'steps':h['steps'],'updates':h['updates'],'diagnostic_hand_label':'not-a-real-model'}
    after=deepcopy(before)
    probabilities=(None if variant=='absent_first_key' else 0.5,0.5,0.6,0.6,0.6,0.6,0.6,0.6,0.6)
    after['pending']=(h['receiving_id'],events,active,probabilities,(0.5,0.5,0.5));after['clock']=h['receiving_id']
    packet=Input(h['owner'],h['epoch'],h['receiving_id'],h['source_id'],events)
    def phase(state,revision):
        encoded=hand_encoding(state)
        return PhaseCapture(h['owner'],h['epoch'],revision,encoded,sha256(canonical(encoded)).hexdigest())
    return CapturePair(packet,phase(before,h['pre_revision']),phase(after,h['receipt_revision'])),before,after


def repack(pair,before,after):
    def phase(old,state):
        encoded=hand_encoding(state);return replace(old,state=encoded,digest=sha256(canonical(encoded)).hexdigest())
    return replace(pair,predecessor=phase(pair.predecessor,before),receipt=phase(pair.receipt,after))


def faults(spec):
    rejected=[]
    for name in spec['faults']:
        pair,before,after=fixture(spec,'absent_first_key')
        special=None
        if name=='wrong_digest':pair=replace(pair,receipt=replace(pair.receipt,digest='9'*64))
        elif name=='wrong_owner':pair=replace(pair,receipt=replace(pair.receipt,owner_label='foreign'))
        elif name=='wrong_epoch':pair=replace(pair,receipt=replace(pair.receipt,epoch_label='foreign'))
        elif name=='wrong_revision':pair=replace(pair,receipt=replace(pair.receipt,revision=14))
        elif name=='pre_pending':before['pending']=after['pending']
        elif name in ('wrong_receiving_tick','wrong_events','partial_active','duplicate_active','unbalanced_active','absent_probability','allocated_probability_none'):
            pending=list(after['pending'])
            if name=='wrong_receiving_tick':pending[0]=65
            elif name=='wrong_events':pending[1]=(1,1,0,1,0)
            elif name=='partial_active':pending[2]=pending[2][:-1]
            elif name=='duplicate_active':pending[2]=(0,)*9
            elif name=='unbalanced_active':pending[2]=(0,3,6,9,12,15,18,21,24)
            elif name=='absent_probability':pending[3]=(0.5,)+pending[3][1:]
            else:pending[3]=(pending[3][0],None)+pending[3][2:]
            after['pending']=tuple(pending)
        elif name=='bool_mass':after['tables'][1][(0,)][0]=True
        elif name=='negative_mass':after['tables'][1][(0,)][0]=-1
        elif name=='bool_observation':after['observations'][0]=True
        elif name=='nonfinite_weight':after['weights'][0]=float('inf')
        elif name=='error_exceeds_weight':after['errors'][0]=1.0
        elif name=='missing_field':del after['weights']
        elif name=='wrong_route_catalog':after['routes']=tuple(reversed(ROUTES))
        elif name=='nonbinary_key':after['tables'][0][(2,)]=[0,0]
        elif name=='changed_learning_field':after['tables'][1][(0,)][0]=1
        elif name=='changed_allocation':after['tables'][0][(0,)]=[0,0]
        elif name=='changed_step':after['steps']=65
        elif name=='changed_update':after['updates']=577
        elif name in ('duplicate_encoded_key','noncanonical_integer','unknown_tag'):special=name
        elif name=='byte_budget':pass
        else:raise ValueError('Unknown fault')
        if name not in ('wrong_digest','wrong_owner','wrong_epoch','wrong_revision'):pair=repack(pair,before,after)
        if special:
            node=pair.receipt.state
            if special=='duplicate_encoded_key':node=('dict',node[1]+node[1][:1])
            elif special=='noncanonical_integer':node=('int','01')
            else:node=('unknown',())
            pair=replace(pair,receipt=replace(pair.receipt,state=node,digest=sha256(canonical(node)).hexdigest()))
        try:validate(pair,byte_limit=1 if name=='byte_budget' else 262144)
        except LayoutError:rejected.append(name)
        else:raise ValueError('Undetected layout fault: '+name)
    return rejected


def analyze(spec,sources,reports):
    require(set(sources)==set(spec['sources']) and set(reports)==set(spec['reports']),'Input roster differs')
    require(sum(len(x) for x in (*sources.values(),*reports.values()))<=spec['bounds']['input_bytes'],'Input budget exceeded')
    for pins,inputs in ((spec['sources'],sources),(spec['reports'],reports)):
        for name,raw in inputs.items():require(sha256(raw).hexdigest()==pins[name],'Frozen input changed')
    parent=json.loads(reports['route_capture_candidate_result_v1.json'])
    require(parent['launch_authorized'] is False and parent['real_model_constructed'] is False,'Candidate promoted')
    require(canonical(ROUTES)==canonical(spec['routes']),'Route catalog changed')
    records=[]
    for variant in spec['variants']:
        pair,_,_=fixture(spec,variant);before=repr(pair);result=validate(pair)
        require(repr(pair)==before and result==validate(pair),'Readonly validation differs')
        require(result['fields'][0]==((0,'error_sum',-1),0.0),'First expected field differs')
        masses=dict(result['fields']);require(masses[(44,'mass_left',3)]==7 and masses[(44,'mass_right',3)]==9,'Sentinel key addressing differs')
        require(result['allocations'][0]==(0,0,variant=='allocated_zero_first_key'),'Absence/zero allocation collapsed')
        records.append({'variant':variant,'projection':result})
    require(records[0]['projection']['fields']==records[1]['projection']['fields'],'Absent/zero logical masses differ')
    rejected=faults(spec);require(rejected==spec['faults'] and len(rejected)==28,'Fault roster differs')
    return {'schema':'route-capture-layout-result-v1','hand_pairs':2,'records':records,'faults_rejected':rejected,
            'missing_runtime_evidence':parent['missing_runtime_evidence'],'scope':'Logical encoding/layout validation only; hand states are not captured model histories.',
            'candidate_encoder_executions':0,'learner_executions':0,'calculator_executions':0,'model_storage_executions':0,
            'live_capture_executions':0,'task_scores':0,'acquisitions':0,'resource_measurements':0,
            'actual_model_authority_established':False,'capture_unblocked':False,'launch_authorized':False,
            'snn_advantage_established':False,'conditions_relaxed':False,'test_opened':False}
