#!/usr/bin/env python3
"""Complete the registered synthetic component-learning research checkpoint."""
from hashlib import sha256
import json
from pathlib import Path
import platform

from r2_electricity_development_v2 import execute_once
from sara_engine.evaluation.shared_observer_learning import run_seed
from sara_engine.utils.project_paths import processed_data_path, workspace_path, project_path, ensure_parent_directory


def main():
    protocol_path = Path(processed_data_path('benchmark_fixtures', 'shared_observer_learning_v1.json'))
    raw = protocol_path.read_bytes()
    expected = '7e999640944764a6b86873e61771fda4fdeb50197142ad0a2c240c43803dfbfa'
    if sha256(raw).hexdigest() != expected:
        raise ValueError('Frozen learning protocol changed')
    protocol = json.loads(raw)
    observer_path = Path(processed_data_path('benchmark_fixtures', 'temporal_information_contract_v1.json'))
    if sha256(observer_path.read_bytes()).hexdigest() != protocol['observer_protocol_sha256']:
        raise ValueError('Observer contract changed')
    sources = ('src/sara_engine/evaluation/shared_observer_learning.py',
               'src/sara_engine/evaluation/temporal_information.py',
               'src/sara_engine/evaluation/electricity_learner.py',
               'src/sara_engine/learning/local_outcome.py',
               'scripts/eval/shared_observer_learning.py')
    metadata = {'protocol_sha256': expected, 'python_version': platform.python_version(),
                'source_sha256': {name: sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in sources},
                'confirmatory': False, 'held_out_access_authorized': False}

    def work():
        runs, states = [], []
        for training_seed, evaluation_seed in zip(protocol['training_seeds'], protocol['evaluation_seeds']):
            run, weights = run_seed(training_seed, evaluation_seed,
                                    protocol['training_pairs_per_seed'], protocol['evaluation_pairs_per_seed'])
            runs.append(run)
            states.append({'training_seed': training_seed, 'weights': weights})
            print(json.dumps({'completed_training_seed': training_seed, 'gates': run['gates']}, sort_keys=True), flush=True)
        integrity = all(row['gates']['deterministic_replay'] and row['gates']['order_blind_pair_control']
                        and row['gates']['backend_only_pair_control'] for row in runs)
        component = integrity and all(row['gates']['component_learning'] for row in runs)
        backend = integrity and all(row['gates']['backend_increment'] for row in runs)
        checkpoint = {'schema': 'sara-shared-observer-research-checkpoint-v1', **metadata,
                      'production_loading_authorized': False, 'states': states}
        checkpoint_bytes = (json.dumps(checkpoint, sort_keys=True, separators=(',', ':')) + '\n').encode()
        checkpoint_path = Path(ensure_parent_directory(project_path('models', 'research', 'shared_observer_learning_v1.json')))
        if integrity:
            with checkpoint_path.open('xb') as stream:
                stream.write(checkpoint_bytes)
                stream.flush()
                import os
                os.fsync(stream.fileno())
        return {'schema': 'sara-shared-observer-learning-result-v1', **metadata, 'runs': runs,
                'execution_integrity_passed': integrity, 'component_learning_passed': component,
                'backend_increment_passed': backend, 'promotion_authorized': False,
                'real_data_read': False, 'energy_measured': False,
                'checkpoint_path': 'models/research/shared_observer_learning_v1.json' if integrity else None,
                'checkpoint_sha256': sha256(checkpoint_bytes).hexdigest() if integrity else None}

    report = execute_once(workspace_path('evaluation', 'shared_observer_learning_v1.json'),
                          workspace_path('evaluation', 'shared_observer_learning_v1'), work, metadata)
    print(json.dumps({key: report[key] for key in ('execution_integrity_passed', 'component_learning_passed', 'backend_increment_passed', 'checkpoint_sha256')}, sort_keys=True))
    return 0 if report['execution_integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
