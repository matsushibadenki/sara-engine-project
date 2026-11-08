"""Check dormant candidate source and refusal gate; never run a model/helper."""
import ast
from hashlib import sha256
import json
import sys
from sara_engine.evaluation.route_capture_candidate import RouteCaptureCandidate,CaptureBlocked

SELECTION_SHA='b45240a9c4e290492d7c0e0e93391966511c5873ad3f771d4c81649f6e34b2d3'
CANDIDATE='src/sara_engine/evaluation/route_capture_candidate.py'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def require(value,message):
    if not value:raise ValueError(message)


def source_checks(raw):
    tree=ast.parse(raw.decode()); classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    cls=classes['RouteCaptureCandidate'];methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    init=methods['__init__'];capture=methods['capture'];reader=methods['_double_read']
    flag=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LAUNCH_AUTHORIZED' for t in n.targets))
    require(isinstance(flag.value,ast.Constant) and flag.value.value is False,'Launch gate enabled')
    guard=init.body[0]
    require(isinstance(guard,ast.If) and ast.unparse(guard.test)=='LAUNCH_AUTHORIZED is not True'
            and len(guard.body)==1 and isinstance(guard.body[0],ast.Raise),'Constructor gate not first')
    imports=[n for n in ast.walk(init) if isinstance(n,ast.ImportFrom) and n.module=='sara_engine.evaluation.route_memory_factorial_v4']
    require(len(imports)==1 and imports[0].lineno>guard.end_lineno,'Lazy learner import moved before guard')
    top_imports=[n for n in tree.body if isinstance(n,(ast.ImportFrom,ast.Import))]
    require(all('route_memory_factorial' not in ast.unparse(n) for n in top_imports),'Learner imported at module load')
    params={a.arg for a in (*init.args.args,*init.args.kwonlyargs)}
    require('model' not in params and 'authorized' not in params and init.args.kwarg is None,'Model or launch override argument added')
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    forbidden={'observe','train','evaluate','evaluate_cell','probability','snapshot','state_digest','save','load','commit','open','write','write_bytes','write_text'}
    for call in calls:
        terminal=call.func.attr if isinstance(call.func,ast.Attribute) else call.func.id if isinstance(call.func,ast.Name) else ''
        require(terminal not in forbidden,'Forbidden model/calculator/storage call: '+terminal)
    contexts=[n for n in ast.walk(capture) if isinstance(n,ast.With)]
    mutation=next(n for n in contexts if any('self._host.mutation' in ast.unparse(i.context_expr) for i in n.items))
    begins=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self._model.begin']
    require(len(begins)==1 and mutation.lineno<begins[0].lineno<=mutation.end_lineno,'Begin escaped exclusive owner interval')
    require(sum(any('self._host.paused' in ast.unparse(i.context_expr) for i in n.items) for n in contexts)==2,'Paired phase pauses differ')
    reads=[n for n in ast.walk(reader) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self._read']
    require(len(reads)==2 and reads[0] is not reads[1],'Independent read calls differ')
    for n in ast.walk(tree):
        if isinstance(n,(ast.Return,ast.Yield)) and n.value is not None:
            require(ast.unparse(n.value)!='self._model','Raw model exported')
    for name in ('PhaseCapture','CapturePair'):
        require(any('frozen=True' in ast.unparse(d) and 'slots=True' in ast.unparse(d) for d in classes[name].decorator_list),'Capture record is mutable')
    return ['launch_flag_literal_false','launch_guard_precedes_lazy_learner_import','no_learner_module_import_at_top_level',
            'no_model_parameter','no_forbidden_calls','no_file_writer_calls','begin_within_owner_mutation_context',
            'pre_and_receipt_pauses','two_independent_read_traversals','immutable_capture_fields_only','no_model_return']


def analyze(spec,sources,reports,candidate):
    require(set(sources)==set(spec['sources']) and len(sources)==6,'Source roster differs')
    require(set(reports)==set(spec['reports']),'Report roster differs')
    require(sum(len(x) for x in (*sources.values(),*reports.values(),candidate))<=spec['bounds']['input_bytes'],'Input budget exceeded')
    for pins,inputs in ((spec['sources'],sources),(spec['reports'],reports)):
        for name,raw in inputs.items():require(sha256(raw).hexdigest()==pins[name],'Frozen input changed')
    parent=json.loads(reports['route_pause_admission_result_v1.json'])
    require(parent['capture_unblocked'] is False and parent['launch_authorized'] is False,'Parent promoted')
    checks=source_checks(candidate)
    before=set(sys.modules)
    try:RouteCaptureCandidate(owner_label='not-a-real-owner',epoch_label='not-a-real-epoch',seed=0,history='cumulative',error_weight='uniform')
    except CaptureBlocked:pass
    else:raise ValueError('Candidate constructor launched')
    require(set(sys.modules)==before,'Refusal imported additional runtime modules')
    checks.append('constructor_refuses_without_learner_import')
    require(checks==spec['checks'],'Check roster differs')
    return {'schema':'route-capture-candidate-result-v1','candidate_sha256':sha256(candidate).hexdigest(),'checks_passed':checks,
            'decision':'dormant_source_candidate_only_runtime_gate_remains_false',
            'limits':'Bounded syntax checks and constructor refusal only; numeric copies, begin behavior, complete field encoding and concurrency have not run.',
            'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'capture_unblocked':False,'actual_model_authority_established':False,'all_model_writer_paths_covered':False,
            'real_model_constructed':False,'real_producer_attached':False,'launch_authorized':False,'snn_advantage_established':False,
            'learner_executions':0,'calculator_executions':0,'model_storage_executions':0,'live_numeric_capture_executions':0,
            'task_scores':0,'acquisitions':0,'resource_measurements':0,'conditions_relaxed':False,'test_opened':False}
