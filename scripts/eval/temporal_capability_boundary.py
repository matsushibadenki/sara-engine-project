#!/usr/bin/env python3
"""Verify frozen baseline identities and temporal information boundaries."""
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation.temporal_capability_boundary import PROTOCOL_SHA256, audit_boundary
from sara_engine.utils.project_paths import processed_data_path, workspace_path, project_path, ensure_parent_directory


def main():
    protocol_path = Path(processed_data_path('benchmark_fixtures', 'temporal_capability_boundary_v1.json'))
    if sha256(protocol_path.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('Frozen boundary protocol changed')
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol['baseline_sha256'].items():
        if sha256(Path(project_path(name)).read_bytes()).hexdigest() != expected:
            raise ValueError('Research baseline identity changed')
    report = audit_boundary(protocol)
    report['protocol_sha256'] = PROTOCOL_SHA256
    report['baseline_identity_verified'] = True
    report['implementation_sha256'] = sha256(Path(project_path(
        'src/sara_engine/evaluation/temporal_capability_boundary.py')).read_bytes()).hexdigest()
    output = Path(ensure_parent_directory(workspace_path('evaluation', 'temporal_capability_boundary_v1.json')))
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
