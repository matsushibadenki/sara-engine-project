#!/usr/bin/env python3
"""Execute the registered temporal learning comparison exactly once."""
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import process_time

from sara_engine.evaluation.temporal_boundary_learning import PROTOCOL_SHA256, load_protocol
from sara_engine.evaluation.temporal_boundary_execution import (
    execute_once, gates_for_family, restore_readout, run_arm, run_phase, save_exclusive,
)
from sara_engine.utils.project_paths import model_path, project_path, workspace_path

SOURCES = (
    'src/sara_engine/evaluation/temporal_boundary_learning.py',
    'src/sara_engine/evaluation/temporal_boundary_execution.py',
    'src/sara_engine/evaluation/temporal_capability_boundary.py',
    'src/sara_engine/evaluation/temporal_information.py',
    'src/sara_engine/evaluation/r1_temporal_learning_benchmark.py',
    'src/sara_engine/learning/local_outcome.py',
    'src/sara_engine/utils/project_paths.py',
    'scripts/eval/temporal_boundary_learning.py',
    'tests/test_temporal_boundary_execution.py',
)


def fingerprints():
    return {name: sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in SOURCES}


def preflight():
    protocol = load_protocol()
    audit_path = Path(workspace_path('evaluation', 'temporal_boundary_learning_input_audit_v1.json'))
    audit_raw = audit_path.read_bytes()
    audit = json.loads(audit_raw)
    if not audit['passed'] or audit['protocol_sha256'] != PROTOCOL_SHA256 or audit['learning_executed']:
        raise ValueError('Input preflight not passed')
    if audit['input_sha256'] != '25296622241292401f5c53681b4e91ec0bbb9eb639bb6167093ef85548704809':
        raise ValueError('Input identities changed')
    for name, expected in audit['implementation_sha256'].items():
        if sha256(Path(project_path(name)).read_bytes()).hexdigest() != expected:
            raise ValueError('Audited input implementation changed')
    metadata = {'protocol_sha256': PROTOCOL_SHA256, 'input_audit_sha256': sha256(audit_raw).hexdigest(),
                'source_sha256': fingerprints(), 'python_version': platform.python_version(),
                'learning_execution_ready': True, 'promotion_authorized': False, 'real_data_read': False}
    return protocol, metadata


def experiment(protocol, metadata, checkpoint_path):
    started = process_time()
    runs, states = [], []
    for training_seed, evaluation_seed in zip(protocol['training_seeds'], protocol['evaluation_seeds']):
        for family in protocol['families']:
            arms, weights = {}, {}
            for arm in protocol['arms']:
                arms[arm], weights[arm] = run_arm(protocol, arm, family, training_seed, evaluation_seed)
            gates = gates_for_family(arms, family)
            if not gates['blind_pair_equality']:
                raise ValueError('Blind prediction equality failed')
            runs.append({'training_seed': training_seed, 'evaluation_seed': evaluation_seed,
                         'family': family, 'arms': arms, 'gates': gates})
            states.append({'training_seed': training_seed, 'evaluation_seed': evaluation_seed,
                           'family': family, 'weights': weights})
            print(json.dumps({'completed_seed': training_seed, 'family': family, 'gates': gates}), flush=True)
    checkpoint = {'schema': 'sara-temporal-boundary-research-checkpoint-v1', **metadata,
                  'production_loading_authorized': False, 'states': states}
    checkpoint_hash = save_exclusive(checkpoint_path, checkpoint)
    # Replay is sourced from the serialized file, never from the live models.
    raw = Path(checkpoint_path).read_bytes()
    if sha256(raw).hexdigest() != checkpoint_hash:
        raise ValueError('Checkpoint publication changed')
    saved = json.loads(raw)
    for run, state in zip(runs, saved['states']):
        for arm in protocol['arms']:
            model = restore_readout(protocol, state['weights'][arm])
            replay = run_phase(protocol, model, arm, run['family'], run['evaluation_seed'], training=False)
            expected = run['arms'][arm]['evaluation']
            if any(replay[key] != expected[key] for key in ('prediction_sha256', 'accuracy', 'brier', 'opposite_pair_predictions_equal')):
                raise ValueError('Saved weight replay mismatch')
            run['arms'][arm]['replay'] = replay
        run['gates']['saved_checkpoint_replay'] = True
    if fingerprints() != metadata['source_sha256'] or load_protocol() != protocol:
        raise ValueError('Execution source identity changed')
    return {'schema': 'sara-temporal-boundary-learning-result-v1', **metadata, 'runs': runs,
            'execution_integrity_passed': True,
            'component_learning_passed': all(row['gates']['quality'] and row['gates']['mechanism'] for row in runs),
            'strong_reference_superiority': all(row['gates']['strong_reference_superiority'] for row in runs),
            'checkpoint_path': str(Path(checkpoint_path).relative_to(project_path())),
            'checkpoint_sha256': checkpoint_hash,
            'cpu_process_seconds_complete_experiment': process_time() - started,
            'energy_measured': False,
            'accounting': 'Per-arm CPU includes input generation, encoding, validation, sorting, local readout, feedback, metrics and instrumentation. Counters report source-level visits and sort input items, not CPU instructions. Deep-state bounds sum live state, encoder transients and conservative 2048+512*active scratch; shared objects may be double-counted. Offline SHA shuffle setup is separately reported and capped; its retained schedule is charged to training. Evaluator reports/checkpoint aggregation and interpreter memory are outside the per-arm state cap. Complete experiment CPU includes checkpoint serialization and disk replay; no energy or speed superiority claim.'}


def main():
    protocol, metadata = preflight()
    output = workspace_path('evaluation', 'temporal_boundary_learning_v1.json')
    prefix = workspace_path('evaluation', 'temporal_boundary_learning_v1')
    checkpoint = model_path('research', 'temporal_boundary_learning_v1.json')
    report = execute_once(output, prefix, checkpoint, metadata,
                          lambda: experiment(protocol, metadata, checkpoint))
    print(json.dumps({key: report[key] for key in ('execution_integrity_passed', 'component_learning_passed', 'strong_reference_superiority', 'checkpoint_sha256')}))
    return 0 if report['component_learning_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
