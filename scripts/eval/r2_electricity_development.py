#!/usr/bin/env python3
"""Execute the immutable electricity learner comparison once."""
import io
import json
from pathlib import Path
import zipfile

from r2_electricity_load_shape import file_hash, PROTOCOL, PROTOCOL_HASH
from sara_engine.evaluation.electricity_development import iter_episodes, evaluate
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path, workspace_path


def main():
    fixture = Path(processed_data_path('benchmark_fixtures'))
    learner_path = fixture / 'r2_electricity_learner_protocol_v1.json'
    learner_hash = '0093d72c34a72baf49f21220e6cadbef2af0993eb3b0cce366e2e495b7fef89f'
    if file_hash(learner_path) != learner_hash or file_hash(PROTOCOL) != PROTOCOL_HASH:
        raise ValueError('Frozen protocol identity changed')
    learner_protocol = json.loads(learner_path.read_text())
    feasibility_path = fixture / 'r2_electricity_load_feasibility_protocol_v1.json'
    support_path = Path(workspace_path('evaluation', 'r2_electricity_load_support_v1.json'))
    if file_hash(feasibility_path) != learner_protocol['feasibility_protocol_sha256'] or file_hash(support_path) != learner_protocol['support_report_sha256']:
        raise ValueError('Frozen feasibility identity changed')
    feasibility = json.loads(feasibility_path.read_text())
    support = json.loads(support_path.read_text())
    source = json.loads(PROTOCOL.read_text())['source']
    archive = Path(project_path(source['archive_path']))
    if not support['passed'] or file_hash(archive) != feasibility['archive_sha256']:
        raise ValueError('Invalid source or support gate')
    output = Path(ensure_parent_directory(workspace_path('evaluation', 'r2_electricity_development_v1.json')))
    # Keep the reservation on exceptions; never silently repeat a scored run.
    with output.open('x') as result:
        with zipfile.ZipFile(archive) as bundle, bundle.open(source['expected_primary_member']) as member:
            with io.TextIOWrapper(member, encoding='utf-8-sig') as lines:
                next(lines)
                report = evaluate(iter_episodes(lines, feasibility))
        for split in ('training', 'development'):
            expected = [support['splits'][split]['fall_observations'], support['splits'][split]['rise_observations']]
            if report['label_counts_fall_rise'][split] != expected:
                raise ValueError('Eligibility differs from frozen support report')
        report['learner_protocol_sha256'] = learner_hash
        report['archive_sha256'] = feasibility['archive_sha256']
        report['eligibility_matches_support'] = True
        result.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in ('adopted', 'gates', 'reference', 'primary_improvement', 'label_counts_fall_rise')}, sort_keys=True))
    return 0 if report['adopted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
