#!/usr/bin/env python3
"""Check selected documentation gates; no learner, scorer or writer execution."""
import json
from pathlib import Path
from sara_engine.evaluation.roadmap_gate_check import check
from sara_engine.utils.project_paths import processed_data_path, project_path


def main():
    manifest = json.loads(Path(processed_data_path('benchmark_fixtures','roadmap_gate_status_v1.json')).read_bytes())
    limit = manifest['input_byte_limit']
    def read(name):
        with Path(project_path(name)).open('rb') as stream:
            return stream.read(limit + 1)
    evidence = {entry['evidence']:read(entry['evidence']) for entry in manifest['entries']}
    roadmaps = {name:read(name) for name in manifest['roadmaps']}
    errors = check(manifest,evidence,roadmaps)
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(1)
    print('Four selected roadmap gates consistent; runtime authorization remains false')


if __name__ == '__main__':
    main()
