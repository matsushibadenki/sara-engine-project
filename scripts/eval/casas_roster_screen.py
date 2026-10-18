#!/usr/bin/env python3
"""Register, acquire, execute and replay a finite unused-household timestamp screen."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from sara_engine.evaluation.casas_180_source import file_sha256, member_lines
from sara_engine.evaluation.casas_roster_screen import (
    Budget, validate_selection, validate_range, member_range, timestamp_metadata, screen)
from sara_engine.evaluation.casas_timestamp_bound import validate_counterexamples
from sara_engine.utils.project_paths import (
    project_path, raw_data_path, processed_data_path, ensure_parent_directory)


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'casas_roster_screen_' + suffix + '_v1.json'))


def raw(identity, suffix):
    return Path(raw_data_path('r2_casas_roster', identity.lower() + '.' + suffix))


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    with Path(ensure_parent_directory(str(p))).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify():
    registration = read(path('registration'))
    for name, digest in {**registration['code'], **registration['preserved']}.items():
        if file_sha256(project_path(name)) != digest:
            raise ValueError('Pinned evidence changed: ' + name)
    selection = read(path('selection'))
    directory = Path(raw_data_path('r2_casas_180', 'directory.json'))
    if file_sha256(directory) != selection['directory_sha256']:
        raise ValueError('Source directory changed')
    validate_selection(selection, read(directory))
    return selection


def register():
    selection = read(path('selection'))
    validate_selection(selection, read(raw_data_path('r2_casas_180', 'directory.json')))
    prior = read(processed_data_path('benchmark_fixtures', 'casas_timestamp_bound_registration_v1.json'))
    code = {**prior['code'], **{name: file_sha256(project_path(name)) for name in (
        'scripts/eval/casas_roster_screen.py',
        'src/sara_engine/evaluation/casas_roster_screen.py',
        'tests/test_casas_roster_screen.py')}}
    preserved = {str(p.relative_to(project_path())): file_sha256(p)
                 for directory in ('data/processed/benchmark_fixtures', 'models')
                 for p in Path(project_path(directory)).rglob('*')
                 if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.json.gz'))}
    save(path('registration'), {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
         'selection_sha256': file_sha256(path('selection')), 'code': code, 'preserved': preserved,
         'method': 'Unchanged safe minute-bucket injection/interval-union bound; original split policy without payload field counting. Synthetic 8192-mask validation before real timestamp inspection. All capacities charged before work, shared across the frozen roster; capacity/source failures inconclusive. Complete source CRC and size at EOF required for any exclusion. Exact metadata/daily/result replay in a new process; no acquisition retries or roster extension.'})
    print('Finite roster and screening code registered before acquisition and timestamp inspection', flush=True)


def acquire():
    selection = verify()
    save(path('acquisition_started'), {'selection_sha256': file_sha256(path('selection')),
                                     'started_at_utc': datetime.now(timezone.utc).isoformat()})
    budget = Budget(selection['budgets'])
    receipts = []
    for entry in selection['roster']:
        identity = entry['identity']
        requests = []
        record = {'identity': identity, 'entry': entry, 'requests': requests}

        def request(start, end, suffix):
            size = end - start + 1
            if sum(r['reserved_bytes'] for r in requests) + size > selection['budgets']['network_bytes_per_household']:
                raise ValueError('Household network capacity')
            budget.charge('network_bytes', size)
            destination = raw(identity, suffix)
            headers = raw(identity, suffix + '.headers')
            if destination.exists() or headers.exists():
                raise ValueError('Acquisition output already exists; no retry allowed')
            ensure_parent_directory(str(destination))
            reservation = {'start': start, 'end': end, 'reserved_bytes': size}
            requests.append(reservation)
            save(raw(identity, suffix + '.reservation.json'), reservation)
            print('Acquiring ' + identity + ' ' + suffix + ': ' + str(size) + ' reserved bytes', flush=True)
            completed = subprocess.run([
                'curl', '-fL', '--silent', '--show-error', '--proto', '=https', '--proto-redir', '=https',
                '--connect-timeout', '60', '--max-time', '1800', '--max-filesize', str(size),
                '--range', str(start) + '-' + str(end), '-D', str(headers), '-o', str(destination), selection['url']],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            reservation['received_bytes'] = destination.stat().st_size if destination.exists() else 0
            reservation['curl_exit_code'] = completed.returncode
            if completed.returncode:
                raise ValueError('Range acquisition failed; no retry allowed')
            validate_range(headers.read_bytes(), start, end, selection['archive_size'], destination.stat().st_size)
            reservation['range_verified'] = True
            return destination

        try:
            header = request(entry['offset'], entry['offset'] + 1023, 'header')
            start, end = member_range(header.read_bytes(), entry)
            source = request(start, end, 'deflate')
            record.update({'status': 'acquired', 'receipt': {**entry, 'compressed_sha256': file_sha256(source)}})
        except (ValueError, OSError) as error:
            record.update({'status': 'inconclusive', 'error': str(error)})
        receipts.append(record)
        save(path(identity.lower() + '_acquisition'), record)
    verify()
    save(path('acquisition'), {'sources': receipts, 'network_reserved_bytes': budget.used['network_bytes'],
                              'received_bytes': sum(r.get('received_bytes', 0) for s in receipts for r in s['requests'])})


def run(mode):
    selection = verify()
    acquisition = read(path('acquisition'))
    if [r['identity'] for r in acquisition['sources']] != [r['identity'] for r in selection['roster']]:
        raise ValueError('Acquisition roster differs')
    if mode == 'execute':
        save(path('execution'), {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                                'registration_sha256': file_sha256(path('registration')),
                                'acquisition_sha256': file_sha256(path('acquisition'))})
    execution = read(path('execution'))
    if execution['acquisition_sha256'] != file_sha256(path('acquisition')) or execution['registration_sha256'] != file_sha256(path('registration')):
        raise ValueError('Execution provenance changed')
    validation = validate_counterexamples()
    if mode == 'execute':
        save(path('validation'), validation)
    elif validation != read(path('validation')):
        raise ValueError('Counterexample validation replay differs')
    budget = Budget(selection['budgets'])
    report = {'sources': [], 'actual_learning_writes': 0, 'sensor_message_fields_inspected': False,
              'independent_predictive_information_opened': False,
              'metadata_scope': 'All-source dates/times only for original split/clock census; arrival scans skip independent times',
              'target': selection['target'], 'validation_sha256': file_sha256(path('validation')),
              'acquisition_sha256': file_sha256(path('acquisition'))}
    details = {}
    for entry, acquired in zip(selection['roster'], acquisition['sources']):
        identity = entry['identity']
        record = {'identity': identity, 'screen_decision': 'inconclusive', 'timestamp_upper_bound': None}
        try:
            if acquired['status'] != 'acquired':
                raise ValueError('Acquisition incomplete')
            receipt = acquired['receipt']
            if any(receipt[k] != entry[k] for k in entry):
                raise ValueError('Acquired identity differs')
            budget.charge('expanded_bytes_all_passes', entry['size'])
            integrity = {}
            metadata = timestamp_metadata(member_lines(raw(identity, 'deflate'), receipt, integrity),
                                          budget, selection['budgets']['metadata_timestamps_per_household'])
            details[identity] = {'metadata': metadata, 'integrity': integrity}
            budget.charge('expanded_bytes_all_passes', entry['size'])
            result = screen(member_lines(raw(identity, 'deflate'), {**receipt, **integrity}),
                            metadata, budget, selection['budgets'])
            details[identity]['daily'] = result.pop('daily')
            record.update(result)
            record.update({'train_start': metadata['train_start'], 'train_end_exclusive': metadata['development_start'],
                           'valid_recorded_days': metadata['valid_recorded_days'],
                           'source_quality_pass': False, 'input_preflight_authorized': False})
        except (ValueError, OSError) as error:
            record['failure'] = str(error)
        report['sources'].append(record)
        print(identity + ': ' + record['screen_decision'] + ', upper=' + str(record['timestamp_upper_bound']), flush=True)
    report['work'] = dict(budget.used)
    report['network_reserved_bytes'] = acquisition['network_reserved_bytes']
    report['survivors'] = [r['identity'] for r in report['sources'] if r['screen_decision'] == 'cannot_exclude']
    report['inconclusive'] = [r['identity'] for r in report['sources'] if r['screen_decision'] == 'inconclusive']
    report['decision'] = ('separate_preregistration_required' if report['survivors'] else
                          'roster_inconclusive' if report['inconclusive'] else 'stop_roster_no_survivor')
    verify()
    if mode == 'execute':
        save(path('details'), details)
        report['details_sha256'] = file_sha256(path('details'))
        save(path('result'), report)
    else:
        if details != read(path('details')):
            raise ValueError('Metadata, integrity or daily replay differs')
        report['details_sha256'] = file_sha256(path('details'))
        if report != read(path('result')):
            raise ValueError('Fresh-process roster replay differs')
        save(path('replay'), {'exact_match': True, 'preserved_artifacts': len(read(path('registration'))['preserved']),
                              'result_sha256': file_sha256(path('result')), 'details_sha256': file_sha256(path('details'))})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'acquire', 'execute', 'replay'))
    mode = parser.parse_args().mode
    if mode == 'register':
        register()
    elif mode == 'acquire':
        acquire()
    else:
        run(mode)
