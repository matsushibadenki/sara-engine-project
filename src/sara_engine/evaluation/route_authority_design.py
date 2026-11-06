"""Bounded source-only authority connection design; no runtime owner implementation."""
import ast
from hashlib import sha256
import json

SELECTION_SHA = 'fba9fb2d53f59ba9abe4f2de191dcba7428a8c8719091b16a11a28f4a45bc678'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_source(path, raw, dotted):
    text = raw.decode('utf-8'); node = ast.parse(text)
    for part in dotted.split('.'):
        matches = [n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part]
        require(len(matches) == 1, 'Missing/ambiguous source anchor: ' + dotted)
        node = matches[0]
    excerpt = '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])
    return {'path':path, 'symbol':dotted, 'start':node.lineno, 'end':node.end_lineno,
            'source_sha256':sha256(raw).hexdigest(), 'excerpt_sha256':sha256(excerpt.encode()).hexdigest(),
            'calls':sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)}),
            'attributes':sorted({ast.unparse(n) for n in ast.walk(node) if isinstance(n,ast.Attribute)}),
            'stores':sorted({ast.unparse(n) for n in ast.walk(node) if isinstance(n,(ast.Attribute,ast.Subscript)) and isinstance(n.ctx,ast.Store)}),
            'excerpt':excerpt}


def analyze(spec, sources, reports):
    require(set(sources) == set(spec['sources']) and len(sources) == 5, 'Source roster differs')
    require(set(reports) == set(spec['reports']) and len(reports) == 1, 'Report roster differs')
    require(sum(len(x) for x in (*sources.values(),*reports.values())) <= spec['bounds']['input_bytes'], 'Input budget exceeded')
    for pins, inputs in ((spec['sources'],sources),(spec['reports'],reports)):
        for name, raw in inputs.items():
            require(sha256(raw).hexdigest() == pins[name], 'Frozen input changed: ' + name)
    parent = json.loads(reports['route_capture_protocol_result_v1.json'])
    require(parent['launch_authorized'] is False and parent['runtime_feasible_demonstrated'] is False, 'Capture hold promoted')
    require(spec['design']['current_allowed_runtime_calls'] == [], 'Runtime permission changed')
    rows = []
    for module, dotted in spec['anchors']:
        group = 'memory' if module == 'authoritative_sequence' else 'evaluation'
        path = 'src/sara_engine/' + group + '/' + module + '.py'
        rows.append(inspect_source(path,sources[path],dotted))
    require(len(rows) == 10, 'Anchor roster differs')
    by_name = {row['symbol']:row for row in rows}
    require({'self.pending','self.clock'} <= set(by_name['BudgetRouteMemory.begin']['stores']), 'Receipt mutation changed')
    require('FactorialMemory' in by_name['run_case']['calls'] and 'train' in by_name['run_case']['calls'], 'Runner owner construction changed')
    require('row.outcome' in by_name['train']['attributes'] and 'model.begin' in by_name['train']['calls']
            and 'model.observe' in by_name['train']['calls'], 'Input/outcome boundary changed')
    require('self._locked_file' in by_name['AuthoritativeSequenceStore.load']['calls']
            and 'self._write_locked' in by_name['AuthoritativeSequenceStore.advance']['calls'], 'Persistence candidate changed')
    gaps = [
        {'boundary':'single_owner', 'anchor':'run_case', 'status':'new_connection_required',
         'evidence':'Runner constructs model and passes its mutable alias into train. Exclusive host ownership is a possible new seam, not enforced by this source.'},
        {'boundary':'revision_and_pause', 'anchor':'BudgetRouteMemory.begin', 'status':'new_connection_required',
         'evidence':'Begin and observe mutate public fields. The selected methods do not wrap all writers in one host revision/pause gate; steps and clock are not such a generation.'},
        {'boundary':'outcome_firewall', 'anchor':'train', 'status':'new_connection_required',
         'evidence':'One row already exposes row.outcome to the runner before observe; the loop is not an input-only channel. A split input/outcome producer is required.'},
        {'boundary':'watermark_reuse', 'anchor':'AuthoritativeSequenceStore.load', 'status':'not_drop_in',
         'evidence':'Publisher watermark uses file locks and persisted publisher scope, not learner state/receipt revision. Its load is not a no-storage read primitive.'},
        {'boundary':'metadata_commit_reuse', 'anchor':'commit', 'status':'reference_only',
         'evidence':'Hand metadata successor does not own the actual mutable model or supply a stop boundary, process authority or numeric persistence.'},
    ]
    return {'schema':'route-authority-design-result-v1', 'design':spec['design'], 'source_evidence':rows,
            'connection_gaps':gaps, 'decision':'prospective_source_feasible_with_new_host_and_split_input_runner_not_drop_in_or_runtime_ready',
            'design_assessment':'Runner construction and begin/observe boundaries give identifiable insertion seams; feasibility is an engineering interpretation, not executed proof.',
            'current_capture_unblocked':False, 'authority_provider_implemented':False,
            'runtime_owner_authenticated':False, 'revision_pause_proven':False,
            'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'learner_executions':0, 'calculator_executions':0, 'model_storage_executions':0,
            'live_capture_executions':0, 'task_scores':0, 'acquisitions':0, 'resource_measurements':0,
            'new_hand_fixtures':0, 'conditions_relaxed':False, 'test_opened':False,
            'launch_authorized':False, 'snn_advantage_established':False,
            'scope_limit':'Five selected source files; AST extraction is not full writer coverage, capability security, concurrency proof or independent provenance.',
            'next':'Separately preregister a nonlearning single-owner host lifecycle interface and validate ownership/revision/pause/poison transitions without a model, calculator or storage backend; keep live attachment and old runs disabled.'}
