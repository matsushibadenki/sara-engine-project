"""Finite hand-record admission checks; no real source, model or arithmetic."""
from dataclasses import asdict,replace,fields
from hashlib import sha256
import json
from sara_engine.evaluation.route_outcome_admission import AdmissionGate,Input,CaptureAck,AdmissionError

SELECTION_SHA = '5b57a8498026188e7c37d2d3796c5a8093c66713aef04ee828c94f9a877271d1'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def require(value,message):
    if not value: raise ValueError(message)


def fixture(spec,role,engine):
    h=spec['hand'];owner=h['owner']+':'+engine+':'+role
    source=h['shuffled_source'] if role=='shuffled_outcome' else h['receiving_id']
    ack=CaptureAck(owner,h['epoch'],h['receiving_id'],source,h['pre_revision'],h['receipt_revision'],h['pre_digest'],h['receipt_digest'],tuple(h['routes']))
    packet=Input(owner,h['epoch'],h['receiving_id'],source,tuple(h['events']))
    gate,producer,collector=AdmissionGate.create(ack)
    return gate,producer,collector,packet,ack


def faults(spec):
    rejected=[]
    for fault in spec['faults']:
        gate,p,c,packet,ack=fixture(spec,'shuffled_outcome',spec['engines'][0])
        operation=None
        if fault=='foreign_producer': operation=lambda:gate.admit(object(),packet)
        elif fault=='foreign_collector':
            gate.admit(p,packet);operation=lambda:gate.acknowledge(object(),ack)
        elif fault=='release_before_ack':
            gate.admit(p,packet);operation=lambda:gate.release(p,65,object())
        elif fault=='ack_before_input':operation=lambda:gate.acknowledge(c,ack)
        elif fault=='copied_input_with_result':
            operation=lambda:gate.admit(p,dict(asdict(packet),outcome=(0,1,0)))
        else:
            gate.admit(p,packet)
            changes={'wrong_owner':('owner','foreign'),'wrong_epoch':('epoch','foreign'),
                     'wrong_receiver':('receiving_id',65),'source_as_receiver':('receiving_id',ack.source_id),
                     'wrong_source':('source_id',ack.source_id+1),'stale_pre_revision':('pre_revision',8),
                     'stale_receipt_revision':('receipt_revision',14),'wrong_pre_digest':('pre_digest','9'*64),
                     'wrong_receipt_digest':('receipt_digest','9'*64),'partial_routes':('routes',ack.routes[:-1]),
                     'duplicate_routes':('routes',(ack.routes[0],)*9),'reordered_routes':('routes',tuple(reversed(ack.routes))),
                     'bool_revision':('pre_revision',True)}
            if fault in changes:
                key,value=changes[fault]
                operation=lambda key=key,value=value:gate.acknowledge(c,replace(ack,**{key:value}))
            elif fault=='duplicate_input':operation=lambda:gate.admit(p,packet)
            else:
                gate.acknowledge(c,ack)
                if fault=='duplicate_ack':operation=lambda:gate.acknowledge(c,ack)
                elif fault=='duplicate_release':
                    gate.release(p,65,(0,1,0));operation=lambda:gate.release(p,65,(0,1,0))
                elif fault=='early_delivery':operation=lambda:gate.release(p,64,(0,1,0))
                elif fault=='late_delivery':operation=lambda:gate.release(p,66,(0,1,0))
                elif fault=='invalid_outcome':operation=lambda:gate.release(p,65,(True,1,0))
        before=gate.phase()
        require(operation is not None,'Unknown fault')
        try:operation()
        except AdmissionError:rejected.append(fault)
        else:raise ValueError('Undetected admission fault: '+fault)
        if fault in ('foreign_producer','foreign_collector'):require(gate.phase()==before,'Foreign capability changed state')
        require(not hasattr(gate,'_outcome'),'Result retained by gate')
    return rejected


def analyze(spec,reports):
    require(set(reports)==set(spec['reports']) and sum(map(len,reports.values()))<=spec['bounds']['input_bytes'],'Input roster/budget differs')
    for name,raw in reports.items():require(sha256(raw).hexdigest()==spec['reports'][name],'Frozen report changed')
    parent=json.loads(reports['route_integration_map_result_v1.json'])
    require(parent['host_attached'] is False and parent['outcome_firewall_implemented'] is False,'Parent promoted')
    require([f.name for f in fields(Input)]==['owner','epoch','receiving_id','source_id','events'],'Input schema contains result')
    records=[]
    for role in spec['roles']:
        for engine in spec['engines']:
            gate,p,c,packet,ack=fixture(spec,role,engine)
            require(gate.input_view() is None,'Initial input differs')
            gate.admit(p,packet)
            require(gate.phase()=='awaiting_ack' and gate.input_view()==packet and not hasattr(packet,'outcome'),'Preack result exposed')
            gate.acknowledge(c,ack)
            delivery=gate.release(p,spec['hand']['delivery_tick'],tuple(spec['hand']['outcome']))
            require(delivery.input==packet and delivery.capture==ack and gate.phase()=='released','Bound delivery differs')
            records.append({'role':role,'engine_namespace':engine,'input':asdict(packet),'capture':asdict(ack),'delivery':asdict(delivery),'phase':'released'})
    return {'schema':'route-outcome-admission-result-v1','hand_cases':len(records),'records':records,
            'faults_rejected':faults(spec),'input_has_result_field':False,'gate_retains_outcome':False,
            'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'real_producer_authority_established':False,'real_outcome_unavailability_established':False,
            'host_attached':False,'capture_unblocked':False,'launch_authorized':False,'snn_advantage_established':False,
            'learner_executions':0,'calculator_executions':0,'model_storage_executions':0,
            'task_scores':0,'acquisitions':0,'resource_measurements':0,'conditions_relaxed':False,'test_opened':False}
