"""Finite model-free host scenarios; threads test exclusion, not performance."""
from dataclasses import replace
from hashlib import sha256
import json
from threading import Thread
from sara_engine.evaluation.route_owner_host import OwnerHost, Owner, HostError, HostBusy

SELECTION_SHA = 'd04ddd6f6d84a517ac902986208b22258fd5d6aa553b27682521b7564e1d6aa6'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value:
        raise ValueError(message)


def refused(operation, kind=HostError):
    try:
        operation()
    except kind:
        return
    raise ValueError('Lifecycle fault was not refused')


def enter(context):
    with context:
        pass


def thread_refusal(operation, kind, timeout):
    observed = []
    def run():
        try:
            operation()
        except BaseException as error:
            observed.append(type(error))
        else:
            observed.append(None)
    thread = Thread(target=run, daemon=True)
    thread.start(); thread.join(timeout)
    require(not thread.is_alive() and observed == [kind], 'Cross-thread exclusion differs')


def analyze(spec, reports):
    require(set(reports) == set(spec['reports']) and sum(map(len,reports.values())) <= spec['bounds']['input_bytes'], 'Input roster/budget differs')
    for name, raw in reports.items():
        require(sha256(raw).hexdigest() == spec['reports'][name], 'Frozen report changed')
    parent = json.loads(reports['route_authority_design_result_v1.json'])
    require(parent['current_capture_unblocked'] is False and parent['launch_authorized'] is False, 'Parent promoted')
    done = []; host, owner = OwnerHost.create('hand-owner','hand-epoch')
    require(host.inspect(owner).revision == 0 and host.inspect(owner).phase == 'stable','Initial host differs'); done.append('initial')
    with host.mutation(owner,0) as view:
        require(view.revision == 1 and view.phase == 'mutating','In-progress generation differs')
    require(host.inspect(owner).revision == 2,'Successful stable generation differs'); done.append('successful_mutation')
    with host.paused(owner,2) as ticket:
        require(host.revalidate(owner,ticket) == host.revalidate(owner,ticket) == ticket.view,'Readonly repeat differs')
    require(host.inspect(owner).revision == 2,'Pause changed revision'); done.append('repeated_pause_revalidation')
    poison, po = OwnerHost.create('hand-owner','hand-poison')
    try:
        with poison.mutation(po,0): raise RuntimeError('Registered control-only exception')
    except RuntimeError: pass
    require(poison.inspect(po).phase == 'poisoned' and poison.inspect(po).revision == 1,'Mutation exception not isolated'); done.append('mutation_exception')
    base, bo = OwnerHost.create('hand-owner','hand-baseexception')
    try:
        with base.mutation(bo,0): raise KeyboardInterrupt('Registered control-only BaseException')
    except KeyboardInterrupt: pass
    require(base.inspect(bo).phase == 'poisoned','BaseException escaped poison'); done.append('baseexception_poison')
    try:
        with host.paused(owner,2): raise RuntimeError('Inspection-only failure')
    except RuntimeError: pass
    require(host.inspect(owner).phase == 'stable','Pause failure poisoned nonmutation'); done.append('pause_exception_release')
    closed, co = OwnerHost.create('hand-owner','hand-closed')
    require(closed.close(co,0).phase == 'closed','Close differs'); done.append('close')
    refused(lambda:host.inspect(Owner('hand-owner','hand-epoch'))); done.append('foreign_owner')
    refused(lambda:enter(host.mutation(owner,0))); done.append('stale_revision')
    refused(lambda:enter(host.mutation(owner,True))); done.append('bool_revision')
    other, oo = OwnerHost.create('hand-owner','hand-epoch')
    refused(lambda:host.inspect(oo)); done.append('foreign_host_same_labels')
    with host.mutation(owner,2):
        refused(lambda:enter(host.mutation(owner,2)),HostBusy); done.append('nested_mutation')
        refused(lambda:enter(host.paused(owner,2)),HostBusy); done.append('pause_during_mutation')
        thread_refusal(lambda:enter(host.paused(owner,2)),HostBusy,spec['thread_timeout_seconds']); done.append('thread_exclusion_during_mutation')
    with host.paused(owner,4) as ticket:
        refused(lambda:enter(host.mutation(owner,4)),HostBusy); done.append('mutation_during_pause')
        refused(lambda:host.revalidate(owner,replace(ticket))); done.append('forged_pause_ticket')
        thread_refusal(lambda:enter(host.mutation(owner,4)),HostBusy,spec['thread_timeout_seconds']); done.append('thread_exclusion_during_pause')
        thread_refusal(lambda:host.revalidate(owner,ticket),HostError,spec['thread_timeout_seconds']); done.append('foreign_thread_revalidation')
    refused(lambda:host.revalidate(owner,ticket)); done.append('stale_pause_ticket')
    refused(lambda:enter(closed.paused(co,0))); done.append('closed_access')
    refused(lambda:enter(poison.mutation(po,1))); done.append('poisoned_access')
    require(sorted(done) == sorted(spec['scenarios']) and len(done) == 21,'Scenario roster differs')
    return {'schema':'route-owner-host-result-v1','scenarios_passed':sorted(done),'scenario_count':21,
            'thread_exclusion_checks':3,'same_process_reference_only':True,
            'scope':'Metadata-only lifecycle gate, no model or payload. Capability identity is cooperative, not external authentication.',
            'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'actual_model_authority_established':False,'all_model_writer_paths_covered':False,
            'capture_unblocked':False,'durable_atomicity_established':False,'snn_advantage_established':False,
            'learner_executions':0,'calculator_executions':0,'model_storage_executions':0,
            'live_capture_executions':0,'task_scores':0,'acquisitions':0,'resource_measurements':0,
            'conditions_relaxed':False,'test_opened':False,'launch_authorized':False}
