#!/usr/bin/env python3
"""Independently replay frozen support evidence using canonical JSON keys."""
import json
from pathlib import Path
from sara_engine.evaluation.room_temporal import extract, contrast_audit
from sara_engine.evaluation.motion_temporal import digest
from sara_engine.utils.project_paths import processed_data_path, raw_data_path, project_path


def main():
    path=lambda suffix:Path(processed_data_path('benchmark_fixtures','room_temporal_'+suffix+'_v1.json'))
    registration=json.loads(path('registration').read_text())
    protocol=json.loads(path('protocol').read_text())
    if digest(path('protocol'))!=registration['protocol_sha256']:
        raise ValueError('Protocol changed')
    for p,h in {**registration['code'],**registration['preserved']}.items():
        if digest(project_path(p))!=h:raise ValueError('Pinned evidence changed')
    archive=raw_data_path('r2_room','source.zip')
    if digest(archive)!=protocol['source_sha256']:raise ValueError('Source changed')
    rows,extraction=extract(archive)
    report={'extraction':extraction,'audit':contrast_audit(rows),'test_opened':False}
    if report['audit']['passed']:raise ValueError('Unexpected support pass')
    # JSON object keys are strings; original runner compares integer label keys.
    if json.loads(json.dumps(report))!=json.loads(path('result').read_text()):
        raise ValueError('Canonical replay differs')
    print('Room support replay and preserved evidence hashes match exactly')

if __name__=='__main__':main()
