#!/usr/bin/env python3
"""Execute the frozen train/development electricity support gate once."""
import io
import json
from pathlib import Path
import zipfile

from r2_electricity_load_shape import file_hash, PROTOCOL, PROTOCOL_HASH
from sara_engine.evaluation.electricity_support import audit_support
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path, workspace_path


def main():
    path = Path(processed_data_path('benchmark_fixtures', 'r2_electricity_load_feasibility_protocol_v1.json'))
    expected = 'a2220cf5d49ac68fd74a71f923b7c43701f190975291d4dc7fd4f2b6ed4e9157'
    if file_hash(path) != expected or file_hash(PROTOCOL) != PROTOCOL_HASH:
        raise ValueError('Frozen protocol identity changed')
    protocol = json.loads(path.read_text())
    source = json.loads(PROTOCOL.read_text())['source']
    archive = Path(project_path(source['archive_path']))
    if file_hash(archive) != protocol['archive_sha256']:
        raise ValueError('Frozen archive identity changed')
    output = Path(ensure_parent_directory(workspace_path('evaluation', 'r2_electricity_load_support_v1.json')))
    # Exclusive reservation prevents an unnoticed second outcome inspection.
    with output.open('x') as result:
        with zipfile.ZipFile(archive) as bundle, bundle.open(source['expected_primary_member']) as member:
            with io.TextIOWrapper(member, encoding='utf-8-sig') as lines:
                next(lines)
                report = audit_support(lines, protocol)
        report['protocol_sha256'] = expected
        report['archive_sha256'] = protocol['archive_sha256']
        result.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
