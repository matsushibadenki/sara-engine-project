"""Hand-pair bridge checks; no live model capture, encoder or numeric producer."""
from dataclasses import asdict,replace
from hashlib import sha256
import json
from threading import Thread
from sara_engine.evaluation.route_owner_host import OwnerHost,HostError
from sara_engine.evaluation.route_outcome_admission import AdmissionError
from sara_engine.evaluation.route_capture_layout import LayoutError
from sara_engine.evaluation.route_capture_layout_preflight import fixture as layout_fixture,repack
from sara_engine.evaluation.route_capture_ack import CaptureAckBridge

SELECTION_SHA='402372e46d3c84392ba8a372601f26ce8de0fb63314f6d00ed0d25b557a76ece'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def require(value,message):
    if not value:raise ValueError(message)


def mutate(host,owner,revision):
    with host.mutation(owner,revision):pass


def fixture(spec,variant):
    pair,before,after=layout_fixture(spec,variant)
    host,owner=OwnerHost.create(pair.input.owner,pair.input.epoch)
    for revision in range(0,pair.predecessor.revision,2):mutate(host,owner,revision)
    return host,owner,CaptureAckBridge(host,owner,pair.input),pair,before,after


def fault_cases(spec):
    rejected=[]
    for fault in spec['faults']:
        host,owner,bridge,pair,before,after=fixture(spec,spec['variants'][0]);pre_rev=pair.predecessor.revision;post_rev=pair.receipt.revision
        foreign,fo=OwnerHost.create(pair.input.owner,pair.input.epoch)
        for revision in range(0,post_rev,2):mutate(foreign,fo,revision)
        try:
            with host.paused(owner,pre_rev) as pre:
                if fault=='copied_pre_ticket':bridge.seal_predecessor(replace(pre),pair.predecessor)
                elif fault=='foreign_pre_ticket':
                    # A foreign receipt-generation ticket still fails host identity first.
                    with foreign.paused(fo,post_rev) as token:bridge.seal_predecessor(token,pair.predecessor)
                elif fault=='wrong_pre_owner':bridge.seal_predecessor(pre,replace(pair.predecessor,owner_label='foreign'))
                elif fault=='wrong_pre_revision':bridge.seal_predecessor(pre,replace(pair.predecessor,revision=pre_rev+2))
                elif fault=='bad_pre_digest':bridge.seal_predecessor(pre,replace(pair.predecessor,digest='9'*64))
                elif fault not in ('no_preseal','expired_pre_ticket'):bridge.seal_predecessor(pre,pair.predecessor)
                if fault=='duplicate_preseal':bridge.seal_predecessor(pre,pair.predecessor)
            if fault=='expired_pre_ticket':bridge.seal_predecessor(pre,pair.predecessor)
            mutate(host,owner,pre_rev)
            if fault=='expired_receipt_ticket':
                with host.paused(owner,post_rev) as token:pass
                bridge.acknowledge(token,pair)
            elif fault=='pre_ticket_as_receipt':bridge.acknowledge(pre,pair)
            else:
                with host.paused(owner,post_rev) as token:
                    if fault=='copied_receipt_ticket':bridge.acknowledge(replace(token),pair)
                    elif fault=='foreign_receipt_ticket':
                        with foreign.paused(fo,post_rev) as other:bridge.acknowledge(other,pair)
                    elif fault=='bad_receipt_digest':bridge.acknowledge(token,replace(pair,receipt=replace(pair.receipt,digest='9'*64)))
                    elif fault=='changed_sealed_pre':
                        before['tables'][44][(1,1)]=[8,9];after['tables'][44][(1,1)]=[8,9]
                        bridge.acknowledge(token,repack(pair,before,after))
                    elif fault=='wrong_input_source':bridge.acknowledge(token,replace(pair,input=replace(pair.input,source_id=8)))
                    elif fault=='partial_routes':
                        pending=list(after['pending']);pending[2]=pending[2][:-1];after['pending']=tuple(pending)
                        bridge.acknowledge(token,repack(pair,before,after))
                    elif fault=='wrong_receipt_revision':bridge.acknowledge(token,replace(pair,receipt=replace(pair.receipt,revision=post_rev+2)))
                    elif fault=='release_before_ack':bridge.release(token,65,(0,1,0))
                    elif fault=='foreign_thread_ack':
                        observed=[]
                        def call():
                            try:bridge.acknowledge(token,pair)
                            except HostError:observed.append('HostError')
                        t=Thread(target=call,daemon=True);t.start();t.join(spec['thread_timeout_seconds'])
                        require(not t.is_alive() and observed==['HostError'],'Cross-thread acknowledgement differs')
                        require(bridge.status()['stopped'],'Cross-thread failure did not stop bridge')
                        rejected.append(fault);continue
                    else:
                        bridge.acknowledge(token,pair)
                        if fault=='duplicate_ack':bridge.acknowledge(token,pair)
                        elif fault=='duplicate_release':bridge.release(token,65,(0,1,0));bridge.release(token,65,(0,1,0))
                        elif fault=='early_delivery':bridge.release(token,64,(0,1,0))
                if fault=='expired_release_ticket':bridge.release(token,65,(0,1,0))
                elif fault=='renewed_release_ticket':
                    with host.paused(owner,post_rev) as new:bridge.release(new,65,(0,1,0))
                elif fault=='poison_after_ack':
                    try:
                        with host.mutation(owner,post_rev):raise RuntimeError('Control-only fault')
                    except RuntimeError:pass
                    bridge.release(token,65,(0,1,0))
                elif fault=='close_after_ack':host.close(owner,post_rev);bridge.release(token,65,(0,1,0))
            raise ValueError('Undetected bridge fault: '+fault)
        except (AdmissionError,HostError,LayoutError):rejected.append(fault)
    return rejected


def analyze(spec,reports):
    require(set(reports)==set(spec['reports']) and sum(map(len,reports.values()))<=spec['bounds']['input_bytes'],'Input roster/budget differs')
    for name,raw in reports.items():require(sha256(raw).hexdigest()==spec['reports'][name],'Frozen report changed')
    parent=json.loads(reports['route_capture_layout_result_v1.json'])
    require(parent['capture_unblocked'] is False and parent['candidate_encoder_executions']==0,'Parent promoted')
    records=[]
    for variant in spec['variants']:
        host,owner,bridge,pair,_,_=fixture(spec,variant)
        with host.paused(owner,pair.predecessor.revision) as token:bridge.seal_predecessor(token,pair.predecessor)
        mutate(host,owner,pair.predecessor.revision)
        with host.paused(owner,pair.receipt.revision) as token:
            ack=bridge.acknowledge(token,pair)
            require(ack.pre_digest==pair.predecessor.digest and ack.receipt_digest==pair.receipt.digest,'Ack not pair-derived')
            require(ack.routes==tuple(spec['hand']['active']),'Ordered routes differ')
            delivery=bridge.release(token,65,(0,1,0))
        records.append({'variant':variant,'derived_ack':asdict(ack),'delivery':asdict(delivery),'status':bridge.status()})
    require(records[0]['derived_ack']['receipt_digest']!=records[1]['derived_ack']['receipt_digest'],'Different allocations collapsed into one digest')
    rejected=fault_cases(spec);require(rejected==spec['faults'] and len(rejected)==26,'Fault roster differs')
    return {'schema':'route-capture-ack-result-v1','hand_pairs':2,'records':records,'faults_rejected':rejected,
            'expected_ack_supplied':False,'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'numeric_state_authenticated':False,'capture_unblocked':False,'launch_authorized':False,'snn_advantage_established':False,
            'learner_executions':0,'calculator_executions':0,'model_storage_executions':0,'candidate_encoder_executions':0,
            'live_numeric_capture_executions':0,'task_scores':0,'acquisitions':0,'resource_measurements':0,
            'conditions_relaxed':False,'test_opened':False}
