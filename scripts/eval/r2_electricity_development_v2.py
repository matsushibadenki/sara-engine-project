#!/usr/bin/env python3
"""Run the disclosed signed-score diagnostic with durable attempt receipts."""
import io
import json
import os
from pathlib import Path
import platform
import zipfile

from r2_electricity_load_shape import file_hash, PROTOCOL, PROTOCOL_HASH
from sara_engine.evaluation.electricity_development import iter_episodes, evaluate
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path, workspace_path


PROTOCOL_V2_HASH = '11779a93583c66e510ddb3c639827aaff0d1985306d761d4ed9e52746d746533'


def save_receipt(path, payload):
    """Reserve once and sync; a failure cannot make an attempt reusable."""
    with Path(ensure_parent_directory(str(path))).open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def execute_once(output, receipt_prefix, work, metadata):
    """Persist intent before scoring and preserve failure on any exception."""
    output = Path(output)
    output = Path(ensure_parent_directory(str(output)))
    with output.open('x') as stream:
        save_receipt(str(receipt_prefix) + '_started.json', {**metadata, 'status': 'started', 'attempt_consumed': True})
        try:
            report = work()
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException as error:
            # Persist only the error type, not arbitrary data-bearing messages.
            save_receipt(str(receipt_prefix) + '_failed.json',
                         {**metadata, 'status': 'failed', 'attempt_consumed': True,
                          'error_type': type(error).__name__, 'final_metrics_available': False})
            raise
    save_receipt(str(receipt_prefix) + '_completed.json',
                 {**metadata, 'status': 'completed', 'report_sha256': file_hash(output),
                  'promotion_authorized': False})
    return report


def main():
    fixtures = Path(processed_data_path('benchmark_fixtures'))
    protocol_path = fixtures / 'r2_electricity_learner_protocol_v2.json'
    if file_hash(protocol_path) != PROTOCOL_V2_HASH or file_hash(PROTOCOL) != PROTOCOL_HASH:
        raise ValueError('Frozen protocol changed')
    protocol = json.loads(protocol_path.read_text())
    parent_path = fixtures / 'r2_electricity_learner_protocol_v1.json'
    feasibility_path = fixtures / 'r2_electricity_load_feasibility_protocol_v1.json'
    support_path = Path(workspace_path('evaluation', 'r2_electricity_load_support_v1.json'))
    for path, key in ((parent_path, 'parent_protocol_sha256'),
                      (feasibility_path, 'feasibility_protocol_sha256'),
                      (support_path, 'support_report_sha256')):
        if file_hash(path) != protocol[key]:
            raise ValueError('Frozen dependency changed')
    feasibility = json.loads(feasibility_path.read_text())
    support = json.loads(support_path.read_text())
    source = json.loads(PROTOCOL.read_text())['source']
    archive = Path(project_path(source['archive_path']))
    if not support['passed'] or file_hash(archive) != feasibility['archive_sha256']:
        raise ValueError('Source or support invalid')
    sources = ('src/sara_engine/evaluation/electricity_development.py',
               'src/sara_engine/evaluation/electricity_learner.py',
               'src/sara_engine/evaluation/electricity_access_boundary.py',
               'src/sara_engine/learning/local_outcome.py',
               'scripts/eval/r2_electricity_development_v2.py')
    metadata = {'protocol_sha256': PROTOCOL_V2_HASH, 'python_version': platform.python_version(),
                'source_sha256': {name: file_hash(Path(project_path(name))) for name in sources},
                'confirmatory': False, 'held_out_access_authorized': False}

    def work():
        with zipfile.ZipFile(archive) as bundle, bundle.open(source['expected_primary_member']) as member:
            with io.TextIOWrapper(member, encoding='utf-8-sig') as lines:
                next(lines)
                report = evaluate(iter_episodes(lines, feasibility), scoring_contract='signed_v2')
        for split in ('training', 'development'):
            expected = [support['splits'][split]['fall_observations'], support['splits'][split]['rise_observations']]
            if report['label_counts_fall_rise'][split] != expected:
                raise ValueError('Eligibility differs from support')
        report.update(metadata)
        report['archive_sha256'] = feasibility['archive_sha256']
        report['eligibility_matches_support'] = True
        return report

    report = execute_once(workspace_path('evaluation', 'r2_electricity_development_v2.json'),
                          workspace_path('evaluation', 'r2_electricity_development_v2'), work, metadata)
    print(json.dumps({key: report[key] for key in ('adopted', 'diagnostic_gates_passed', 'gates', 'reference', 'primary_improvement')}, sort_keys=True))
    return 0 if report['diagnostic_gates_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
