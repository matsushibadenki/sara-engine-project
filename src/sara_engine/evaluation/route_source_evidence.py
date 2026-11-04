"""Read source as data; never import or execute audited model/calculator modules."""
import ast
from hashlib import sha256
import json

SELECTION_SHA = '45500071b4bb8e9c16067d7313acbe169ae926f21a29eb477c0fe7f27a3fa0a8'
ANCHORS = (
    ('sparse_route_reactivation', 'ReactivationMemory.begin', 'receipt_probability_retained'),
    ('sparse_route_reactivation', 'ReactivationMemory.observe', 'sequential_mutable_updates'),
    ('route_memory_factorial_v4', 'FactorialMemory.observe', 'factorial_sequential_updates'),
    ('route_memory_factorial_v4', 'train', 'in_process_delta_checks'),
    ('route_independent_transition', 'transition', 'hand_fixture_delivery_before_probability'),
    ('route_receipt_probability', 'open_receipt', 'supplied_view_receipt_contract'),
    ('route_receipt_probability', 'resolve', 'frozen_error_reference'),
    ('route_receipt_delta', 'persisted', 'memory_only_serialization'),
    ('route_commit_state_machine', 'commit', 'metadata_only_successor'),
    ('route_commit_state_machine', 'inspect_readonly', 'classification_not_application'),
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value:
        raise ValueError(message)


def symbol(tree, name):
    node = tree
    for part in name.split('.'):
        matches = [n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part]
        require(len(matches) == 1, 'Missing/ambiguous source anchor: ' + name)
        node = matches[0]
    return node


def analyze(spec, sources, reports):
    require(set(sources) == set(spec['sources']) and set(reports) == set(spec['reports']), 'Input roster differs')
    require(len(sources) == 6 and len(reports) == 2, 'Finite roster exceeded')
    require(sum(len(x) for x in (*sources.values(), *reports.values())) <= spec['bounds']['input_bytes'], 'Input budget exceeded')
    for roster, data in ((spec['sources'], sources), (spec['reports'], reports)):
        for name, pin in roster.items():
            require(sha256(data[name]).hexdigest() == pin, 'Frozen input changed: ' + name)
    rows = []
    for module, name, category in ANCHORS:
        path = 'src/sara_engine/evaluation/' + module + '.py'
        text = sources[path].decode('utf-8')
        node = symbol(ast.parse(text), name)
        lines = text.splitlines()
        excerpt = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        rows.append({'path': path, 'symbol': name, 'start': node.lineno, 'end': node.end_lineno,
                     'source_sha256': spec['sources'][path], 'excerpt_sha256': sha256(excerpt.encode()).hexdigest(),
                     'category': category, 'excerpt': excerpt})
    saved = {name: json.loads(raw) for name, raw in reports.items()}
    metadata = saved['route_commit_state_machine_result_v1.json']
    protocol = saved['route_runtime_protocol_result_v1.json']
    require(metadata['model_atomicity_established'] is False and metadata['launch_authorized'] is False, 'Metadata promoted')
    require(protocol['launch_authorized'] is False, 'Protocol promoted')
    require(metadata['missing_runtime_evidence'] == protocol['missing_runtime_evidence'], 'Missing evidence changed')
    return {'schema': 'route-source-evidence-result-v1', 'anchors': rows,
            'source_files': len(sources), 'reports': len(reports),
            'input_bytes': sum(len(x) for x in (*sources.values(), *reports.values())),
            'missing_runtime_evidence': metadata['missing_runtime_evidence'],
            'decision': 'stop_before_runtime_integration',
            'minimum_next_verification': 'Preregister a source-only adapter specification: trusted predecessor and receipt capture boundary, explicit allocation and frozen-error scalar inputs, and state-plus-consumption commit ownership. Do not execute either learner.',
            'scope_limit': 'Six selected files only; no repository-wide absence proof, runtime observation, independent execution or crash test.',
            'learner_executions': 0, 'calculator_executions': 0, 'task_scores': 0,
            'acquisitions': 0, 'resource_measurements': 0, 'conditions_relaxed': False,
            'test_opened': False, 'runtime_atomicity_established': False,
            'independent_runtime_equivalence_established': False, 'snn_advantage_established': False,
            'launch_authorized': False}
