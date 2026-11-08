"""Finite composition checks on control metadata and hand payloads only."""
from dataclasses import asdict,replace
from hashlib import sha256
import json
from threading import Thread
from sara_engine.evaluation.route_owner_host import OwnerHost,HostError
from sara_engine.evaluation.route_outcome_admission import Input,CaptureAck,AdmissionError
from sara_engine.evaluation.route_pause_admission import PauseAdmission

SELECTION_SHA='d5e67545fa71de725dfd4af391e3a652e623a977680f1c4c702342074258dd8f'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def require(value,message):
    if not value:raise ValueError(message)


def fixture(spec,role,engine):
    h=spec['hand'];label=h['owner']+':'+engine+':'+role
    source=h['shuffled_source'] if role=='shuffled_outcome' else h['receiving_id']
    ack=CaptureAck(label,h['epoch'],64,source,0,2,h['pre_digest'],h['receipt_digest'],tuple(h['routes']))
    packet=Input(label,h['epoch'],64,source,tuple(h['events']))
    host,owner=OwnerHost.create(label,h['epoch'])
    return host,owner,ack,packet


def mutate(host,owner,revision):
    with host.mutation(owner,revision):pass


def faults(spec):
    rejected=[]
    for fault in spec['faults']:
        host,owner,ack,packet=fixture(spec,'shuffled_outcome',spec['engines'][0])
        foreign,foreign_owner=OwnerHost.create(ack.owner,ack.epoch)
        controller=None
        try:
            if fault=='wrong_owner_capability':PauseAdmission(host,foreign_owner,ack)
            elif fault=='wrong_epoch_plan':PauseAdmission(host,owner,replace(ack,epoch='foreign'))
            elif fault=='wrong_pre_revision':PauseAdmission(host,owner,replace(ack,pre_revision=2,receipt_revision=4))
            elif fault=='wrong_revision_gap':PauseAdmission(host,owner,replace(ack,receipt_revision=4))
            else:
                controller=PauseAdmission(host,owner,ack);controller.admit(packet)
                if fault in ('copied_pre_ticket','foreign_pre_ticket'):
                    with host.paused(owner,0) as ticket:
                        if fault=='copied_pre_ticket':controller.seal_predecessor(replace(ticket))
                        else:
                            with foreign.paused(foreign_owner,0) as other:controller.seal_predecessor(other)
                else:
                    with host.paused(owner,0) as pre:
                        if fault!='no_preseal':controller.seal_predecessor(pre)
                        if fault=='duplicate_preseal':controller.seal_predecessor(pre)
                    mutate(host,owner,0)
                    if fault=='wrong_receipt_revision':mutate(host,owner,2)
                    revision=host.inspect(owner).revision
                    if fault=='expired_ack_ticket':
                        with host.paused(owner,revision) as expired:pass
                        controller.acknowledge(expired,ack)
                    elif fault=='pre_ticket_as_receipt':controller.acknowledge(pre,ack)
                    else:
                        with host.paused(owner,revision) as receipt:
                            if fault=='copied_receipt_ticket':controller.acknowledge(replace(receipt),ack)
                            elif fault=='foreign_receipt_ticket':
                                mutate(foreign,foreign_owner,0)
                                with foreign.paused(foreign_owner,2) as other:controller.acknowledge(other,ack)
                            elif fault=='release_without_ack':controller.release(receipt,65,(0,1,0))
                            elif fault=='wrong_ack_owner':controller.acknowledge(receipt,replace(ack,owner='foreign'))
                            elif fault=='wrong_ack_source':controller.acknowledge(receipt,replace(ack,source_id=ack.source_id+1))
                            elif fault=='partial_ack_routes':controller.acknowledge(receipt,replace(ack,routes=ack.routes[:-1]))
                            elif fault=='foreign_thread_ack':
                                observed=[]
                                def call():
                                    try:controller.acknowledge(receipt,ack)
                                    except (AdmissionError,HostError) as error:observed.append(type(error).__name__)
                                worker=Thread(target=call,daemon=True);worker.start();worker.join(spec['thread_timeout_seconds'])
                                require(not worker.is_alive() and observed==['HostError'],'Cross-thread acknowledgement differs')
                                rejected.append(fault);continue
                            else:
                                controller.acknowledge(receipt,ack)
                                if fault=='duplicate_ack':controller.acknowledge(receipt,ack)
                                elif fault=='early_delivery':controller.release(receipt,64,(0,1,0))
                                elif fault=='late_delivery':controller.release(receipt,66,(0,1,0))
                                elif fault=='duplicate_release':
                                    controller.release(receipt,65,(0,1,0));controller.release(receipt,65,(0,1,0))
                        if fault=='expired_delivery_ticket':controller.release(receipt,65,(0,1,0))
                        elif fault=='new_pause_after_ack':
                            with host.paused(owner,revision) as new:controller.release(new,65,(0,1,0))
                        elif fault=='poison_after_ack':
                            try:
                                with host.mutation(owner,revision):raise RuntimeError('Hand control-only poison')
                            except RuntimeError:pass
                            controller.release(receipt,65,(0,1,0))
                        elif fault=='close_after_ack':
                            host.close(owner,revision);controller.release(receipt,65,(0,1,0))
            raise ValueError('Undetected composition fault: '+fault)
        except (AdmissionError,HostError):
            rejected.append(fault)
    return rejected


def analyze(spec,reports):
    require(set(reports)==set(spec['reports']) and sum(map(len,reports.values()))<=spec['bounds']['input_bytes'],'Input roster/budget differs')
    for name,raw in reports.items():require(sha256(raw).hexdigest()==spec['reports'][name],'Frozen report changed')
    parent=json.loads(reports['route_outcome_admission_result_v1.json'])
    require(parent['capture_unblocked'] is False and parent['host_attached'] is False,'Reference promoted')
    records=[]
    for role in spec['roles']:
        for engine in spec['engines']:
            host,owner,ack,packet=fixture(spec,role,engine)
            c=PauseAdmission(host,owner,ack);c.admit(packet)
            with host.paused(owner,0) as pre:c.seal_predecessor(pre)
            mutate(host,owner,0)
            with host.paused(owner,2) as receipt:
                c.acknowledge(receipt,ack)
                result=c.release(receipt,65,tuple(spec['hand']['outcome']))
                require(result.capture==ack and result.input==packet,'Bound composition delivery differs')
            records.append({'role':role,'engine_namespace':engine,'delivery':asdict(result),'status':c.status()})
    rejected=faults(spec)
    require(sorted(rejected)==sorted(spec['faults']) and len(rejected)==26,'Fault roster differs')
    return {'schema':'route-pause-admission-result-v1','hand_cases':6,'records':records,'faults_rejected':rejected,
            'same_process_reference_only':True,'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'actual_model_authority_established':False,'all_model_writer_paths_covered':False,
            'real_producer_attached':False,'capture_unblocked':False,'launch_authorized':False,'snn_advantage_established':False,
            'learner_executions':0,'calculator_executions':0,'model_storage_executions':0,
            'live_numeric_capture_executions':0,'task_scores':0,'acquisitions':0,'resource_measurements':0,
            'conditions_relaxed':False,'test_opened':False}
