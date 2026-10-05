#!/usr/bin/env python3
"""Verify the shared temporal observer without learning or real data access."""
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation.independent_temporal_design import synthetic_pairs
from sara_engine.evaluation.temporal_information import TemporalInformationObserver
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory


def main():
    protocol = Path(processed_data_path('benchmark_fixtures', 'temporal_information_contract_v1.json'))
    expected = '5dd6daf07b44303d9de2827c43e46f6201394e5f70bbf0e246ed1ba154cf6123'
    if sha256(protocol.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen observer contract changed')
    packets = projections = max_wire = max_channels = max_records = 0
    pairs = synthetic_pairs()
    for pair in pairs:
        snapshots = []
        for events in (pair.mode_zero, pair.mode_one):
            observer = TemporalInformationObserver()
            for event in events:
                observer.observe(event)
            packet = observer.snapshot()
            snapshots.append(packet)
            max_wire = max(max_wire, len(packet.wire_bytes()))
            max_channels = max(max_channels, len(packet.readout_routes()))
            max_records = max(max_records, len(packet.records))
        packets += int(snapshots[0] != snapshots[1])
        projections += int(snapshots[0].readout_routes() != snapshots[1].readout_routes())
    report = {'schema': 'sara-temporal-information-audit-v1', 'protocol_sha256': expected,
              'paired_histories': len(pairs), 'distinct_packet_pairs': packets,
              'distinct_readout_pairs': projections, 'max_wire_bytes': max_wire,
              'max_readout_channels': max_channels, 'max_records': max_records,
              'input_record_visits_per_episode': 95,
              'passed': packets == projections == len(pairs) and max_wire <= 101
                        and max_channels <= 72 and max_records <= 24,
              'real_data_read': False, 'learning_executed': False,
              'spike_specific_advantage_established': False, 'promotion_authorized': False}
    output = Path(ensure_parent_directory(workspace_path('evaluation', 'temporal_information_contract_v1.json')))
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
