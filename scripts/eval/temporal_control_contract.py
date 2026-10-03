#!/usr/bin/env python3
"""Audit synthetic temporal-control contracts without outcomes or scoring."""
from hashlib import sha256
import json
from pathlib import Path
import random

from sara_engine.evaluation.temporal_control_contract import compare_controls
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory, project_path


def main():
    path = Path(processed_data_path('benchmark_fixtures', 'temporal_control_contract_v1.json'))
    protocol_hash = sha256(path.read_bytes()).hexdigest()
    if protocol_hash != 'd4082689b272672428cd030727572efb0e5f865c808e44137fc764963bab40e2':
        raise ValueError('Frozen control contract changed')
    reference = Path(project_path('src/sara_engine/evaluation/electricity_learner.py'))
    reference_hash = sha256(reference.read_bytes()).hexdigest()
    if reference_hash != '8359993c3ace929ff6723ded8b055b092b9378bab6e49eb064ed1d740c452b59':
        raise ValueError('Retained reference implementation changed')
    generator = random.Random(31003)
    traces = [(0,) * 95, tuple(index % 24 for index in range(95)), (0,) * 94 + (1,)]
    traces.extend(tuple(generator.randrange(24) for _ in range(95)) for _ in range(100))
    results = [compare_controls(events) for events in traces]
    report = {
        'schema': 'sara-temporal-control-contract-audit-v1',
        'protocol_sha256': protocol_hash, 'reference_source_sha256': reference_hash,
        'synthetic_seed': 31003, 'traces': len(results),
        'equivalent_route_parity': all(row.reference_routes == row.equivalent_routes for row in results),
        'readout_cardinality_parity': all(len(row.reference_routes) == len(row.recency_routes) for row in results),
        'route_identity_difference_cases': sum(row.reference_routes != row.recency_routes for row in results),
        'zero_comparison_output_cases': sum(not row.reference_routes for row in results),
        'maximum_reference_unit_ticks': max(row.reference_unit_ticks for row in results),
        'maximum_selector_event_visits': max(row.recency_event_visits for row in results),
        'reference_cost_required_for_yoking': True,
        'real_data_read': False, 'outcomes_read': False, 'candidate_scored': False,
        'promotion_authorized': False,
    }
    report['passed'] = report['equivalent_route_parity'] and report['readout_cardinality_parity']
    output = Path(ensure_parent_directory(workspace_path('evaluation', 'temporal_control_contract_v1.json')))
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
