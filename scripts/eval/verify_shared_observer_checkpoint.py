#!/usr/bin/env python3
"""Verify research artifact identity and reproduce all frozen predictions."""
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.shared_observer_checkpoint import replay_checkpoint
from sara_engine.utils.project_paths import project_path, processed_data_path, workspace_path, ensure_parent_directory


def bounded_json(path):
    path = Path(path)
    if path.stat().st_size > 400000:
        raise ValueError('Research artifact size exceeded')
    return json.loads(path.read_bytes())


def main():
    protocol_path = Path(processed_data_path('benchmark_fixtures', 'shared_observer_learning_v1.json'))
    protocol_hash = sha256(protocol_path.read_bytes()).hexdigest()
    if protocol_hash != '7e999640944764a6b86873e61771fda4fdeb50197142ad0a2c240c43803dfbfa':
        raise ValueError('Frozen learning protocol changed')
    report_path = Path(workspace_path('evaluation', 'shared_observer_learning_v1.json'))
    completed = bounded_json(workspace_path('evaluation', 'shared_observer_learning_v1_completed.json'))
    if sha256(report_path.read_bytes()).hexdigest() != completed['report_sha256']:
        raise ValueError('Completed result digest changed')
    report = bounded_json(report_path)
    checkpoint_path = Path(project_path('models', 'research', 'shared_observer_learning_v1.json'))
    if checkpoint_path.stat().st_size > 400000 or sha256(checkpoint_path.read_bytes()).hexdigest() != report['checkpoint_sha256']:
        raise ValueError('Research checkpoint digest changed')
    checkpoint = bounded_json(checkpoint_path)
    if report['protocol_sha256'] != protocol_hash or checkpoint['protocol_sha256'] != protocol_hash or platform.python_version() != report['python_version']:
        raise ValueError('Research protocol/runtime identity changed')
    sources = ('src/sara_engine/evaluation/shared_observer_learning.py',
               'src/sara_engine/evaluation/temporal_information.py',
               'src/sara_engine/evaluation/electricity_learner.py',
               'src/sara_engine/learning/local_outcome.py',
               'scripts/eval/shared_observer_learning.py')
    if checkpoint['source_sha256'] != report['source_sha256'] or set(report['source_sha256']) != set(sources):
        raise ValueError('Source manifest changed')
    for name in sources:
        if sha256(Path(project_path(name)).read_bytes()).hexdigest() != report['source_sha256'][name]:
            raise ValueError('Research source identity changed')
    verification = replay_checkpoint(checkpoint, report, bounded_json(protocol_path))
    verification['checkpoint_sha256'] = report['checkpoint_sha256']
    output = Path(ensure_parent_directory(workspace_path('evaluation', 'shared_observer_checkpoint_verification_v1.json')))
    with output.open('x') as stream:
        stream.write(json.dumps(verification, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: verification[key] for key in ('passed', 'replayed_predictions', 'checkpoint_sha256')}, sort_keys=True))
    return 0 if verification['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
