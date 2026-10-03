#!/usr/bin/env python3
"""Audit synthetic target identifiability before permitting learning."""
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation.independent_temporal_design import audit_design
from sara_engine.utils.project_paths import processed_data_path, project_path, workspace_path, ensure_parent_directory


def main():
    path = Path(processed_data_path('benchmark_fixtures', 'independent_temporal_design_v1.json'))
    raw = path.read_bytes()
    expected = '59b13be6f44a539d9d9661cd4457a58daf518e08ca0b2673881ab815fd7ec270'
    if sha256(raw).hexdigest() != expected:
        raise ValueError('Frozen design contract changed')
    protocol = json.loads(raw)
    reference = Path(project_path('src/sara_engine/evaluation/electricity_learner.py'))
    if sha256(reference.read_bytes()).hexdigest() != protocol['reference_source_sha256']:
        raise ValueError('Retained reference source changed')
    report = audit_design()
    report['protocol_sha256'] = expected
    report['reference_source_sha256'] = protocol['reference_source_sha256']
    output = Path(ensure_parent_directory(workspace_path('evaluation', 'independent_temporal_design_v1.json')))
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if report['control_integrity_passed'] and report['lif_learning_comparison_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
