"""Bounded lexical writer/alias inventory; audited modules remain source data."""
import ast
from hashlib import sha256
import json

SELECTION_SHA = '47dbb8582b92496f1a7ca4893e6bcb23c25e77d65c97636fa0885200b597c10c'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value:
        raise ValueError(message)


def names(target):
    return {n.id for n in ast.walk(target) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}


def rooted(node, aliases):
    return sorted({n.id for n in ast.walk(node) if isinstance(n,ast.Name) and n.id in aliases})


def inventory(path, raw, rules, max_passes):
    text = raw.decode('utf-8'); tree = ast.parse(text); records = []
    scopes = []
    for item in tree.body:
        if isinstance(item,ast.FunctionDef): scopes.append((item.name,item))
        elif isinstance(item,ast.ClassDef):
            scopes.extend((item.name+'.'+n.name,n) for n in item.body if isinstance(n,ast.FunctionDef))
    for name, function in scopes:
        nodes = list(ast.walk(function)); aliases = set(rules['roots'])
        for _ in range(max_passes):
            previous = set(aliases)
            for node in nodes:
                if isinstance(node,(ast.Assign,ast.AnnAssign)) and node.value is not None and rooted(node.value,aliases):
                    targets = node.targets if isinstance(node,ast.Assign) else [node.target]
                    for target in targets: aliases.update(names(target))
            if aliases == previous: break
        else:
            raise ValueError('Alias propagation bound exceeded: '+name)
        events = []
        def emit(kind,node,roots):
            events.append({'kind':kind,'line':node.lineno,'expression':ast.unparse(node),'roots':roots})
        for node in nodes:
            if isinstance(node,(ast.Attribute,ast.Subscript)) and isinstance(node.ctx,ast.Store):
                roots = rooted(node,aliases)
                if roots: emit('rooted_store',node,roots)
            if isinstance(node,(ast.Assign,ast.AnnAssign)) and node.value is not None:
                roots = rooted(node.value,aliases)
                if roots: emit('rooted_alias_assignment',node,roots)
            if isinstance(node,ast.Call):
                if isinstance(node.func,ast.Attribute) and node.func.attr in rules['mutators']:
                    roots = rooted(node.func.value,aliases)
                    if roots: emit('rooted_mutator_call',node,roots)
                roots = sorted({root for arg in (*node.args,*(k.value for k in node.keywords)) for root in rooted(arg,aliases)})
                if roots: emit('rooted_argument_escape',node,roots)
                terminal = node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                if terminal in rules['restore_names']: emit('restore_like_call',node,rooted(node,aliases))
            if isinstance(node,ast.Return) and node.value is not None:
                roots = rooted(node.value,aliases)
                if roots: emit('rooted_return',node,roots)
            if isinstance(node,ast.Attribute) and node.attr == 'outcome':
                emit('outcome_attribute',node,rooted(node,aliases))
            if isinstance(node,ast.arg) and node.arg in ('outcome','labels'):
                emit('outcome_parameter',node,[])
        events.sort(key=lambda e:(e['line'],e['kind'],e['expression']))
        excerpt = '\n'.join(text.splitlines()[function.lineno-1:function.end_lineno])
        records.append({'path':path,'symbol':name,'start':function.lineno,'end':function.end_lineno,
                        'source_sha256':sha256(raw).hexdigest(),'excerpt_sha256':sha256(excerpt.encode()).hexdigest(),
                        'possible_alias_names':sorted(aliases-set(rules['roots'])), 'events':events,
                        'coverage_proven':False})
    classes = [{'class':n.name,'bases':[ast.unparse(base) for base in n.bases]} for n in tree.body if isinstance(n,ast.ClassDef)]
    return records,classes


def analyze(spec,sources,reports):
    require(set(sources)==set(spec['sources']) and len(sources)==5,'Source roster differs')
    require(set(reports)==set(spec['reports']) and len(reports)==1,'Report roster differs')
    require(sum(len(x) for x in (*sources.values(),*reports.values()))<=spec['bounds']['input_bytes'],'Input budget exceeded')
    for pins,inputs in ((spec['sources'],sources),(spec['reports'],reports)):
        for name,raw in inputs.items(): require(sha256(raw).hexdigest()==pins[name],'Frozen input changed: '+name)
    parent=json.loads(reports['route_owner_host_result_v1.json'])
    require(parent['capture_unblocked'] is False and parent['all_model_writer_paths_covered'] is False
            and parent['launch_authorized'] is False,'Reference host promoted')
    records=[]; inheritance=[]
    for path,raw in sorted(sources.items()):
        rows,classes=inventory(path,raw,spec['scan'],spec['bounds']['alias_passes'])
        records.extend(rows); inheritance.extend(dict(c,path=path) for c in classes)
    by_name={r['symbol']:r for r in records}
    def has(symbol,kind,needle):
        require(any(e['kind']==kind and needle in e['expression'] for e in by_name[symbol]['events']),
                'Required source boundary changed: '+symbol+':'+needle)
    has('FactorialMemory.observe','rooted_mutator_call','setdefault')
    has('FactorialMemory.observe','rooted_store','count[')
    has('ReactivationMemory.narrow','rooted_mutator_call','self.commits.append')
    has('BudgetRouteMemory.begin','rooted_store','self.pending')
    # Duplicate module-level train/run_case names are addressed with their file below.
    factorial=[r for r in records if r['path'].endswith('/route_memory_factorial_v4.py')]
    require(any(e['kind']=='outcome_attribute' for r in factorial if r['symbol']=='train' for e in r['events']), 'Outcome path missing')
    require(any(e['kind']=='rooted_argument_escape' and 'train(model' in e['expression'] for r in factorial if r['symbol']=='run_case' for e in r['events']), 'Runner alias handoff missing')
    for row in records:
        method=row['symbol'].split('.')[-1]
        if row['path'].endswith('/route_owner_host.py'):
            disposition='control_metadata_only_no_model_attachment'
        elif method == '__init__':
            disposition='construct_privately_inside_authorized_owner_no_alias_escape'
        elif method in ('begin','observe','narrow'):
            disposition='entire_operation_and_inherited_dispatch_under_one_owner_mutation_gate'
        elif method in ('schedule','select','predict','probability','cost','shortlist'):
            disposition='owner_internal_only_not_collector_callable_review_transitive_side_effects'
        elif method.startswith('train') or method == 'run_case':
            disposition='replace_raw_model_handoff_with_private_host_and_input_outcome_firewall'
        elif method in ('snapshot','state_digest','evaluate','evaluate_cell'):
            disposition='diagnostic_evaluator_quarantine_not_raw_capture_or_authority'
        else:
            disposition='review_before_attachment_no_implicit_safe_classification'
        row['integration_disposition']=disposition
    counts={kind:sum(e['kind']==kind for r in records for e in r['events']) for kind in spec['scan']['events']}
    return {'schema':'route-integration-map-result-v1','records':records,'inheritance':inheritance,
            'dispatch_edges_manual_source_review':spec['dispatch_edges'],'event_counts':counts,
            'function_count':len(records),'required_changes':spec['required_changes'],'stop_conditions':spec['stop_conditions'],
            'decision':'new_private_host_runner_and_outcome_firewall_required_before_attachment',
            'alias_limit':'Conservative lexical propagation retains copy/method-result provenance and ignores assignment order; may overreport. Dynamic aliases, imported functions, callbacks and external writers remain unresolved.',
            'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'all_writer_paths_proven':False,'alias_confinement_proven':False,'outcome_firewall_implemented':False,
            'host_attached':False,'capture_unblocked':False,'launch_authorized':False,'snn_advantage_established':False,
            'learner_executions':0,'calculator_executions':0,'model_storage_executions':0,'host_executions':0,
            'task_scores':0,'acquisitions':0,'resource_measurements':0,'conditions_relaxed':False,'test_opened':False}
