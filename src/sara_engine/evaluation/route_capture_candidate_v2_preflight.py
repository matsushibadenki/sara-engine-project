"""Static V2 source checks and launch refusal; no model or encoder invocation."""
import ast
from hashlib import sha256
import json
import sys
from unittest.mock import patch
from sara_engine.evaluation.route_capture_candidate import CaptureBlocked
from sara_engine.evaluation.route_capture_candidate_v2 import RouteCaptureCandidateV2

SELECTION_SHA='611802ffecfe5b3a89ae31451b8d900f1403d87a5f55c91601a76137c297d6f2'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def require(value,message):
    if not value:raise ValueError(message)


def call_name(node):return ast.unparse(node.func) if isinstance(node,ast.Call) else ''


def guarded(function):
    first=function.body[0]
    return (isinstance(first,ast.If) and ast.unparse(first.test)=='LAUNCH_AUTHORIZED is not True'
            and len(first.body)==1 and isinstance(first.body[0],ast.Raise))


def source_checks(raw):
    tree=ast.parse(raw.decode('utf-8'));classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    cls=classes['RouteCaptureCandidateV2'];methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    init=methods['__init__'];capture=methods['capture'];reader=methods['_double_read']
    flags=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LAUNCH_AUTHORIZED' for t in n.targets)]
    require(len(flags)==1 and isinstance(flags[0].value,ast.Constant) and flags[0].value.value is False,'Launch flag is not fixed false')
    require(guarded(init),'Constructor guard is not first')
    imports=[n for n in ast.walk(init) if isinstance(n,ast.ImportFrom) and n.module=='sara_engine.evaluation.route_memory_factorial_v4']
    require(len(imports)==1 and imports[0].lineno>init.body[0].end_lineno,'Learner import precedes guard')
    require(guarded(capture),'Capture guard is not first')
    require(not any(isinstance(n,(ast.ImportFrom,ast.Import)) and 'route_memory_factorial_v4' in ast.unparse(n) for n in tree.body),'Top-level learner import')
    require({n for n in methods if not n.startswith('_')}=={'capture'},'Unexpected public interface')
    require([a.arg for a in init.args.args]==['self']
            and [a.arg for a in init.args.kwonlyargs]==['owner_label','epoch_label','seed','history','error_weight']
            and init.args.kwarg is None and init.args.vararg is None,'Constructor accepts model/callback/override')
    require([a.arg for a in capture.args.args]==['self','packet'] and not capture.args.kwonlyargs
            and capture.args.kwarg is None and capture.args.vararg is None,'Capture accepts extra authority/payload')
    contexts=[n for n in ast.walk(capture) if isinstance(n,ast.With)]
    pauses=[n for n in contexts if any(call_name(i.context_expr)=='self._host.paused' for i in n.items)]
    mutations=[n for n in contexts if any(call_name(i.context_expr)=='self._host.mutation' for i in n.items)]
    require(len(pauses)==2 and len(mutations)==1,'Phase contexts differ')
    pauses.sort(key=lambda n:n.lineno);pre,receipt=pauses
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    def one(name):
        found=[n for n in calls if call_name(n)==name]
        require(len(found)==1,'Missing/duplicate call: '+name)
        return found[0]
    bridge=one('CaptureAckBridge');seal=one('self._bridge.seal_predecessor');begin=one('self._model.begin');ack=one('self._bridge.acknowledge')
    require(bridge.lineno<pre.lineno and [ast.unparse(a) for a in bridge.args]==['self._host','self._owner','packet'],'Private bridge construction changed')
    reads=[n for n in ast.walk(reader) if call_name(n)=='self._read']
    require(len(reads)==2,'Second traversal is not independent')
    require(pre.lineno<seal.lineno<=pre.end_lineno
            and [ast.unparse(a) for a in seal.args]==['pre_ticket','predecessor'],'Preseal outside predecessor pause')
    require(mutations[0].lineno<begin.lineno<=mutations[0].end_lineno,'Begin outside mutation gate')
    require(receipt.lineno<ack.lineno<=receipt.end_lineno
            and [ast.unparse(a) for a in ack.args]==['receipt_ticket','pair'],'Ack outside receipt pause')
    checks=[n for n in ast.walk(receipt) if call_name(n)=='self._host.revalidate']
    require(any(n.lineno>ack.lineno and [ast.unparse(a) for a in n.args]==['self._owner','receipt_ticket'] for n in checks),'Missing post-ack revalidation')
    tries=[n for n in ast.walk(capture) if isinstance(n,ast.Try)]
    require(len(tries)==1 and len(tries[0].finalbody)==1 and isinstance(tries[0].finalbody[0],ast.Expr)
            and call_name(tries[0].finalbody[0].value)=='self._close_control','No unconditional control cleanup')
    cleanup=methods['_close_control']
    require(any(call_name(n)=='self._host.inspect' for n in ast.walk(cleanup))
            and any(isinstance(n,ast.If) and ast.unparse(n.test)=="current.phase == 'stable'"
                    and any(call_name(call)=='self._host.close' and [ast.unparse(a) for a in call.args]==['self._owner','current.revision'] for call in ast.walk(n))
                    for n in ast.walk(cleanup)), 'Stable host is not closed by cleanup')
    returns=[n for n in ast.walk(capture) if isinstance(n,ast.Return)]
    require(len(returns)==1 and returns[0].lineno>tries[0].end_lineno
            and ast.unparse(returns[0].value)=='AcknowledgedCapture(pair, ack)','Public return bypasses cleanup or exports control')
    output=classes['AcknowledgedCapture']
    require(any('frozen=True' in ast.unparse(d) and 'slots=True' in ast.unparse(d) for d in output.decorator_list),'Mutable output')
    fields=[(n.target.id,ast.unparse(n.annotation)) for n in output.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)]
    require(fields==[('pair','CapturePair'),('ack','CaptureAck')],'Output exports authority')
    require(not any(isinstance(n,(ast.Yield,ast.YieldFrom)) for n in ast.walk(tree)),'Continuation/ticket lease exported')
    allowed_returns={'_read':['PhaseCapture(view.owner_label, view.epoch_label, view.revision, state, sha256(raw).hexdigest())'],
                     '_double_read':['first'],'capture':['AcknowledgedCapture(pair, ack)']}
    for name,method in methods.items():
        for node in ast.walk(method):
            if isinstance(node,ast.Return):require(ast.unparse(node.value) in allowed_returns.get(name,[]),'Unexpected return/alias export')
    forbidden={'observe','train','evaluate','evaluate_cell','probability','snapshot','state_digest','release','save','load','commit','open','write','write_bytes','write_text'}
    for call in calls:
        terminal=call.func.attr if isinstance(call.func,ast.Attribute) else call.func.id if isinstance(call.func,ast.Name) else ''
        require(terminal not in forbidden,'Forbidden learner/result/storage call: '+terminal)
        for arg in (*call.args,*(k.value for k in call.keywords)):
            if ast.unparse(arg)=='self._model':require(call_name(call)=='vars','Raw model handed to external callable')
    return ['literal_false_launch_flag','constructor_gate_before_lazy_learner_import','capture_gate_first','no_top_level_learner_import',
            'exact_public_interface_no_model_callback_or_override','private_bridge_before_pre_pause','two_pauses_and_two_independent_reads',
            'preseal_under_pre_pause','begin_under_mutation','ack_under_receipt_pause','receipt_revalidated_after_ack',
            'close_before_public_return','frozen_output_exact_fields','no_yield_or_control_object_export',
            'no_learning_result_encoder_or_storage_execution_by_preflight']


def refusal_check():
    before=set(sys.modules)
    module='sara_engine.evaluation.route_capture_candidate_v2.'
    with patch(module+'_primitive',side_effect=AssertionError('Encoder must not run')), \
         patch(module+'OwnerHost.create',side_effect=AssertionError('Host must not be created')):
        try:RouteCaptureCandidateV2(owner_label='untrusted',epoch_label='untrusted',seed=0,history='cumulative',error_weight='uniform')
        except CaptureBlocked:pass
        else:raise ValueError('V2 constructor launched')
        # Exercise capture's guard only, using an uninitialized object with no model.
        blank=object.__new__(RouteCaptureCandidateV2)
        try:blank.capture(object())
        except CaptureBlocked:pass
        else:raise ValueError('V2 capture launched')
    require(set(sys.modules)==before,'Refusal imported runtime modules')


def analyze(spec,sources,reports,candidate):
    require(set(sources)==set(spec['sources']) and len(sources)==10,'Source roster differs')
    require(set(reports)==set(spec['reports']) and len(reports)==2,'Report roster differs')
    require(sum(len(x) for x in (*sources.values(),*reports.values(),candidate))<=spec['bounds']['input_bytes'],'Input budget exceeded')
    for pins,inputs in ((spec['sources'],sources),(spec['reports'],reports)):
        for name,raw in inputs.items():require(sha256(raw).hexdigest()==pins[name],'Frozen source/report changed')
    parent=json.loads(reports['route_capture_ack_result_v2.json'])
    old=json.loads(reports['route_capture_candidate_result_v1.json'])
    require(parent['launch_authorized'] is False and old['real_model_constructed'] is False,'Prior hold promoted')
    checks=source_checks(candidate);refusal_check();checks.append('constructor_refuses_without_model_or_encoder_or_host_creation')
    require(checks==spec['checks'],'Check roster differs')
    return {'schema':'route-capture-candidate-v2-result-v1','candidate_sha256':sha256(candidate).hexdigest(),'checks_passed':checks,
            'decision':'source_candidate_v2_only_launch_false_no_model_encoder_or_storage_execution',
            'limits':'Bounded syntax and refusal checks, not executed capture/bridge integration, numeric encoding, all-writer confinement or authenticated origin.',
            'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'real_model_constructed':False,'host_created':False,'encoder_executions':0,'bridge_executions':0,
            'capture_unblocked':False,'actual_model_authority_established':False,'all_model_writer_paths_covered':False,
            'launch_authorized':False,'snn_advantage_established':False,'learner_executions':0,'calculator_executions':0,
            'model_storage_executions':0,'live_numeric_capture_executions':0,'task_scores':0,'acquisitions':0,
            'resource_measurements':0,'conditions_relaxed':False,'test_opened':False}
