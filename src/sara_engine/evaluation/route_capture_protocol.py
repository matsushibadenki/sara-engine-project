"""Parse pinned sources as data to assess a prospective capture protocol."""
import ast
from hashlib import sha256
import json

SELECTION_SHA = '4d3a6befe0a79a4370a191916e21df18670faabd43511c0aed19ff4efb84bb21'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def symbol(tree, dotted):
    node = tree
    for part in dotted.split('.'):
        matches = [n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part]
        require(len(matches) == 1, 'Missing/ambiguous source anchor: ' + dotted)
        node = matches[0]
    return node


def inspect_symbol(path, raw, dotted):
    text = raw.decode('utf-8'); node = symbol(ast.parse(text), dotted)
    excerpt = '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])
    calls = sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)})
    stores = sorted({ast.unparse(n) for n in ast.walk(node)
                     if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, ast.Store)})
    return {'path':path, 'symbol':dotted, 'start':node.lineno, 'end':node.end_lineno,
            'source_sha256':sha256(raw).hexdigest(), 'excerpt_sha256':sha256(excerpt.encode()).hexdigest(),
            'calls_syntactically_present':calls, 'attribute_or_subscript_store_targets':stores,
            'excerpt':excerpt, 'purity_proven':False}


def analyze(spec, sources, reports):
    require(set(sources) == set(spec['sources']) and len(sources) == 5, 'Source roster differs')
    require(set(reports) == set(spec['reports']) and len(reports) == 1, 'Report roster differs')
    require(sum(len(raw) for raw in (*sources.values(), *reports.values())) <= spec['bounds']['input_bytes'], 'Input budget exceeded')
    for pins, inputs in ((spec['sources'], sources), (spec['reports'], reports)):
        for name, raw in inputs.items():
            require(sha256(raw).hexdigest() == pins[name], 'Frozen input changed: ' + name)
    parent = json.loads(reports['route_readiness_decision_result_v1.json'])
    require(parent['launch_authorized'] is False and parent['decision'] == 'hold_runtime_launch_and_scientific_promotion', 'Parent promoted')
    protocol = spec['protocol']
    require(protocol['runtime_calls_allowed'] == [] and protocol['prospective_budget']['runtime_calls'] == 0,
            'Runtime call permission changed')
    rows = [inspect_symbol('src/sara_engine/evaluation/' + module + '.py',
                           sources['src/sara_engine/evaluation/' + module + '.py'], name)
            for module, name in spec['anchors']]
    require(len(rows) == 8, 'Anchor budget differs')
    by_name = {row['symbol']:row for row in rows}
    begin = by_name['BudgetRouteMemory.begin']
    require({'self.pending', 'self.clock'} <= set(begin['attribute_or_subscript_store_targets'])
            and 'self.narrow' in begin['calls_syntactically_present'], 'Receipt writer boundary changed')
    snapshot = by_name['snapshot']
    require('model.probability' in snapshot['calls_syntactically_present']
            and 'model.cost' in snapshot['calls_syntactically_present'], 'Diagnostic calculator boundary changed')
    open_receipt = by_name['open_receipt']
    require('probability' in open_receipt['calls_syntactically_present'], 'Supplied-view calculator boundary changed')
    blockers = [
        {'id':'authority_missing', 'anchors':['state_digest', 'open_receipt'],
         'finding':'state_digest encodes supplied model fields; open_receipt trusts supplied state/view. Neither establishes external owner, monotonic revision, pause or watermark authority.'},
        {'id':'receipt_creation_not_readonly', 'anchors':['BudgetRouteMemory.begin', 'ReactivationMemory.begin'],
         'finding':'Existing begin creates pending state, updates clock and invokes prediction/narrowing. It cannot be called by a read-only collector to manufacture the required receipt.'},
        {'id':'existing_snapshot_exceeds_scope', 'anchors':['snapshot', 'FactorialMemory.probability', 'LocalRouteMemory.probability'],
         'finding':'Existing snapshot recomputes probabilities/scores/ranks and uses true dependency hints; it is not a raw outcome-free receipt copy under the no-calculator contract.'},
        {'id':'independent_readback_not_connected', 'anchors':['state_digest', 'ReactivationMemory.observe'],
         'finding':'These selected boundaries do not supply a paused authoritative live handle or two revision-bound primitive traversals. Observing would update learning and is forbidden.'},
    ]
    return {'schema':'route-capture-protocol-result-v1', 'protocol':protocol, 'source_evidence':rows,
            'blockers':blockers, 'decision':'blocked_on_authority_and_outcome_free_read_boundary_not_runtime_authorized',
            'static_protocol_registered':True, 'runtime_feasible_demonstrated':False,
            'missing_runtime_evidence':[r['category'] for r in parent['evidence_matrix']],
            'scope_limit':'Five selected source files only. AST call/store extraction is not dynamic purity, global absence or authentication proof.',
            'learner_executions':0, 'calculator_executions':0, 'model_storage_executions':0,
            'live_capture_executions':0, 'task_scores':0, 'acquisitions':0, 'resource_measurements':0,
            'new_hand_fixtures':0, 'conditions_relaxed':False, 'test_opened':False,
            'launch_authorized':False, 'snn_advantage_established':False,
            'next':'Hold capture execution. A source-only authority-provider design must identify an actual owner/revision/pause source before any collector or runtime trial; stop if no such existing integration evidence is available.'}
