"""Preregistered sparse local readout of observed room-motion events."""
from collections import Counter, defaultdict
import csv
from datetime import datetime
from hashlib import sha256
from io import StringIO
import json
import re
from pathlib import Path
from zipfile import ZipFile
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, raw_data_path, project_path

PREFIX = 'motion_temporal'
ARMS = ('current', 'order', 'histogram', 'reversed', 'scalar_order')
CODE = ('src/sara_engine/evaluation/motion_temporal.py', 'scripts/eval/motion_temporal_preflight.py')

def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    with Path(ensure_parent_directory(path)).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

def paths():
    return [processed_data_path('benchmark_fixtures', PREFIX + suffix + '_v1.json') for suffix in ('_protocol', '_registration', '_result')]

def keys(history, time, arm):
    current = (history[-1], time.weekday(), time.hour, time.minute // 15)
    if arm == 'current':
        detail = current
    elif arm == 'histogram':
        detail = (current, tuple(sorted(Counter(history[:-1]).items())))
    else:
        past = history[:-1]
        if arm == 'reversed':
            past = tuple(reversed(past))
        detail = (current, past)
    return repr(('current', current)), repr(('detail', detail))

class LocalReadout:
    """Two changed integer counters per delayed outcome, with bounded routes."""
    def __init__(self):
        self.cells = {}
        self.pending = None
        self.writes = 0
    def predict(self, addresses, cutoff, due):
        if self.pending is not None or due <= cutoff:
            raise ValueError('Invalid pending receipt or due time')
        p = .5
        for address in addresses:
            neg, pos = self.cells.get(address, (0, 0))
            p = (pos + 8 * p) / (neg + pos + 8)
        self.pending = (addresses, due)
        return p
    def deliver(self, due, supplier, training):
        if self.pending is None or due != self.pending[1]:
            raise ValueError('Outcome delivered outside receipt time')
        label = supplier()
        if label not in (0, 1):
            raise ValueError('Nonbinary outcome')
        addresses = self.pending[0]
        new = set(addresses) - self.cells.keys()
        if training and len(self.cells) + len(new) > 100000:
            raise ValueError('Route capacity exceeded')
        if training:
            for address in addresses:
                cell = self.cells.setdefault(address, [0, 0])
                cell[label] += 1
                self.writes += 1
        self.pending = None

def extract(archive):
    """Read timestamps before touching sensor cells; January stays sealed."""
    with ZipFile(archive) as z:
        raw = z.read('DB elder monitoring/database_pos.csv').decode()
    rows = csv.DictReader(StringIO(raw))
    opened, sealed, invalid, excluded = [], 0, 0, 0
    for row in rows:
        stamp = re.sub(r'(\.\d{6})\d+', r'\1', row['datetime'])
        t = datetime.fromisoformat(stamp)
        if t.date().isoformat() >= '2020-01-01':
            sealed += 1
            continue
        if t.date().isoformat() < '2019-11-06':
            excluded += 1
            continue
        try:
            values = tuple(float(row[name]) for name in ('Living room', 'Bedroom', 'Bathroom', 'Kitchen', 'Hallway'))
            if any(v not in (0, 1) for v in values):
                raise ValueError('Nonbinary sensors')
            opened.append((t, tuple(int(v) for v in values)))
        except ValueError:
            invalid += 1
            opened.append((t, None))
    # Duplicate timestamps are discarded completely, independent of their labels.
    counts = Counter(t for t, _ in opened)
    opened = sorted((t, v if counts[t] == 1 else None) for t, v in opened)
    examples = {'train': [], 'development': []}
    for i in range(3, len(opened) - 1):
        window = opened[i-3:i+2]
        times = [t for t, _ in window]
        split = 'train' if times[-1].date().isoformat() < '2019-12-01' else 'development'
        if any(v is None for _, v in window) or any(not 0 < (b-a).total_seconds() <= 1800 for a,b in zip(times,times[1:])):
            continue
        if (times[0].date().isoformat() < '2019-12-01') != (split == 'train'):
            continue
        examples[split].append((times[-2], times[-1], tuple(v for _,v in window[:-1]), window[-1][1][0]))
    return examples, {'sealed_timestamp_only_rows': sealed, 'opened_rows': len(opened), 'invalid_rows': invalid, 'pre_source_excluded_rows': excluded}

def audit(examples):
    groups = defaultdict(lambda: defaultdict(Counter))
    for cutoff, _, history, label in examples['train']:
        current, detail = keys(history, cutoff, 'order')
        groups[current][detail][label] += 1
    contrasts = 0
    for histories in groups.values():
        neg = {h for h,c in histories.items() if c[0] >= 4}
        pos = {h for h,c in histories.items() if c[1] >= 4}
        contrasts += bool(neg and pos and len(neg | pos) > 1)
    support = {s: {'count':len(rows), 'labels': dict(Counter(r[-1] for r in rows)), 'days':len({r[0].date() for r in rows})} for s,rows in examples.items()}
    passed = contrasts >= 20 and all(min(v['labels'].get(0,0),v['labels'].get(1,0)) >= 100 and v['days'] >= 10 for v in support.values())
    return {'support':support, 'contrast_groups':contrasts, 'passed':passed}

def compare(examples):
    reports = {}
    traces = {}
    for arm in ARMS:
        model = LocalReadout()
        for cutoff,due,history,label in examples['train']:
            model.predict(keys(history,cutoff,arm),cutoff,due)
            model.deliver(due, lambda:label, True)
        predictions = []
        for cutoff,due,history,label in examples['development']:
            p = model.predict(keys(history,cutoff,arm),cutoff,due)
            predictions.append(p)
            model.deliver(due, lambda:label, False)
        rows = examples['development']
        brier = sum((p-r[-1])**2 for p,r in zip(predictions,rows))/len(rows)
        ba = sum(sum((p>=.5)==bool(y) for p,(*_,y) in zip(predictions,rows) if y==c)/sum(r[-1]==c for r in rows) for c in (0,1))/2
        reports[arm] = {'brier':brier,'balanced_accuracy':ba,'changed_writes':model.writes,'routes':len(model.cells),'development_writes':0}
        traces[arm] = sha256(json.dumps(predictions).encode()).hexdigest()
    return {'arms':reports,'trace_hashes':traces, 'scalar_equivalent': traces['order']==traces['scalar_order'], 'quality_passed': reports['current']['brier']-reports['order']['brier'] >= .01 and reports['order']['balanced_accuracy']-reports['current']['balanced_accuracy'] >= .02 and reports['reversed']['brier']-reports['order']['brier']>=.005 and reports['histogram']['brier']-reports['order']['brier']>=.005}

def main(mode):
    protocol_path, registration_path, result_path = paths()
    archive = raw_data_path('r2_motion','source.zip')
    if mode == 'register':
        protocol = {'source':'https://archive.ics.uci.edu/dataset/799/single+elder+home+monitoring+gas+and+position','license':'CC BY 4.0','archive_sha256':'9f28c90ee7ea129d7748f4bb3dc73ecbd0bce0c3366b43baa9c927155e41a6d3','task':'Living-room bit at next recorded observation; all four past observations and next outcome separated by positive gaps <=1800 seconds','features':'All five raw binary sensors, weekday and quarter-hour clock; no gas or occupancy claims','train':'2019-11-06 <= timestamp < 2019-12-01','development':'2019-12-01 <= timestamp < 2020-01-01','sealed':'2020-01-01 onward: timestamps only','contrast_gate':'20 current+clock groups with distinct histories; >=4 observations per opposing outcome; each split >=100 per class and >=10 days','arms':ARMS,'learning':'Two integer count changes per training receipt; backoff prior strength 8; 100000 route ceiling; frozen development','quality_gate':'Brier gain .01 and balanced accuracy gain .02 over strong current, Brier gain .005 over histogram and reversed','scope':'Single-home source screening; no SNN advantage or production adoption; scalar_order is exact implementation reference','metadata_exposure':'Header and first five timestamps only, no sensor cells inspected before registration'}
        preserved = {}
        for directory in ('data/processed/benchmark_fixtures','models/appliances'):
            for p in Path(project_path(directory)).glob('*.json'):
                preserved[str(p.relative_to(project_path()))] = digest(p)
        save(protocol_path, protocol)
        save(registration_path, {'protocol_sha256':digest(protocol_path),'code':{p:digest(project_path(p)) for p in CODE},'preserved':preserved})
        print('Registered before numeric inspection')
        return
    registration = json.loads(Path(registration_path).read_text())
    protocol = json.loads(Path(protocol_path).read_text())
    if digest(protocol_path)!=registration['protocol_sha256'] or digest(archive)!=protocol['archive_sha256']:
        raise ValueError('Protocol/source checksum mismatch')
    amendment = Path(processed_data_path('benchmark_fixtures', PREFIX+'_parser_amendment_v1.json'))
    code = json.loads(amendment.read_text())['code'] if amendment.exists() else registration['code']
    for p,h in {**code,**registration['preserved']}.items():
        if digest(project_path(p))!=h:
            raise ValueError('Pinned evidence changed: '+p)
    if mode == 'execute':
        marker = processed_data_path('benchmark_fixtures',PREFIX+('_repaired_execution_v1.json' if amendment.exists() else '_execution_v1.json'))
        save(marker, {'started':True})
    examples, extraction = extract(archive)
    report = {'extraction':extraction,'audit':audit(examples),'test_opened':False}
    if report['audit']['passed']:
        report['comparison'] = compare(examples)
    if mode == 'execute':
        save(result_path,report)
    elif report != json.loads(Path(result_path).read_text()):
        raise ValueError('Replay differs')
    print(json.dumps(report,indent=2))
