#!/usr/bin/env python3
"""Exclusively publish the frozen temporal learning input preflight."""
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.temporal_boundary_learning import audit_inputs, load_protocol
from sara_engine.utils.project_paths import ensure_parent_directory, project_path, workspace_path


def main():
    protocol = load_protocol()
    report = audit_inputs(protocol)
    report['python_version'] = platform.python_version()
    report['implementation_sha256'] = {name: sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in (
        'src/sara_engine/evaluation/temporal_boundary_learning.py',
        'src/sara_engine/evaluation/temporal_capability_boundary.py',
        'src/sara_engine/evaluation/r1_temporal_learning_benchmark.py',
        'scripts/eval/temporal_boundary_learning_audit.py')}
    output = Path(ensure_parent_directory(workspace_path('evaluation', 'temporal_boundary_learning_input_audit_v1.json')))
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
