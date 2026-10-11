"""Prospective HH101 notification-state comparison with observed annotations."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from heapq import heappush, heappop
from io import TextIOWrapper
import json
import re
from hashlib import sha256
from zipfile import ZipFile
from sara_engine.evaluation.delayed_sparse_counts import DelayedSparseCounts

ARMS = ('current', 'current_fine', 'histogram', 'transition', 'joint',
        'constant_gap', 'shuffled_outcome', 'scalar_joint')
TRAIN_END = datetime(2012, 8, 20)
TEST_START = datetime(2012, 9, 4)
START = datetime(2012, 7, 22)


@dataclass(frozen=True)
class Example:
    cutoff: datetime
    due: datetime
    history: tuple
    gaps: tuple
    label: int


def addresses(example, arm):
    h, t = example.history, example.cutoff
    current = (h[-1], t.weekday(), t.hour, t.minute // 15)
    if arm == 'current':
        detail = current
    elif arm == 'current_fine':
        detail = (h[-1], t.weekday(), t.hour, t.minute // 5)
    elif arm == 'histogram':
        detail = (current, tuple(sorted(Counter(h[:-1]).items())))
    elif arm == 'transition':
        detail = (current, h[-2], h[-5])
    else:
        gaps = (2, 2) if arm == 'constant_gap' else (example.gaps[-1], example.gaps[-4])
        detail = (current, h[-2], h[-5], gaps)
    return ('global', repr(('current', current)), repr(('detail', detail)))


def gap_bin(seconds):
    return 0 if seconds < 1 else 1 if seconds < 10 else 2 if seconds < 60 else 3


def extract_lines(lines):
    """Only date/time fields are accessed in the sealed interval."""
    history = deque(maxlen=8)
    pending = deque()
    active = {}
    examples = {'train': [], 'development': []}
    audit = Counter()
    last_cutoff = None
    previous = None
    # Split the first four fields without interpreting the remaining annotation.
    for line in lines:
        fields = line.rstrip('\r\n').split(',', 4)
        if len(fields) < 2:
            raise ValueError('Missing date/time fields')
        t = datetime.strptime(fields[0], '%Y-%m-%d')
        if t >= TEST_START:
            audit['sealed_timestamp_only_rows'] += 1
            continue
        if t < START:
            audit['excluded_first_day_rows'] += 1
            continue
        stamp = fields[0] + ' ' + fields[1]
        t = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S.%f' if '.' in fields[1] else '%Y-%m-%d %H:%M:%S')
        audit['opened_rows'] += 1
        if previous is not None and t < previous:
            raise ValueError('Source clock regressed')
        previous = t
        if len(fields) == 4:
            fields.append('')
        if len(fields) != 5:
            raise ValueError('Unknown annotation record schema')
        for name, edge in re.findall(r'([A-Za-z_]+)="(begin|end)"', fields[4]):
            if edge == 'begin':
                active[name] = t
            else:
                active.pop(name, None)
        active = {name: begin for name, begin in active.items()
                  if (t - begin).total_seconds() <= 43200}
        if fields[3] not in ('ON', 'OFF', 'OPEN', 'CLOSE', 'CLOSED'):
            audit['nonbinary_rows'] += 1
            continue
        event = (fields[2], fields[3])
        # First binary event at least five minutes later is the outcome event.
        while pending and pending[0][0] + timedelta(seconds=300) <= t:
            cutoff, old_history, gaps = pending.popleft()
            if (t - cutoff).total_seconds() > 600:
                audit['late_outcomes'] += 1
                continue
            if (cutoff < TRAIN_END) != (t < TRAIN_END):
                audit['cross_split_outcomes'] += 1
                continue
            if not active:
                audit['unannotated_outcomes'] += 1
                continue
            y = int(any(name.lower() in ('sleep', 'sleeping') for name in active))
            split = 'train' if cutoff < TRAIN_END else 'development'
            examples[split].append(Example(cutoff, t, old_history, gaps, y))
        history.append((t, event))
        if last_cutoff is not None and (t-last_cutoff).total_seconds() < 60:
            continue
        last_cutoff = t
        if len(history) != 8 or (history[0][0] < TRAIN_END) != (t < TRAIN_END):
            continue
        intervals = [(b[0]-a[0]).total_seconds() for a,b in zip(history, list(history)[1:])]
        if any(not 0 <= d <= 300 for d in intervals):
            audit['interrupted_histories'] += 1
            continue
        pending.append((t, tuple(v for _,v in history), tuple(gap_bin(d) for d in intervals)))
        if len(pending) > 16:
            raise ValueError('Extraction pending bound exceeded')
    audit['unresolved_outcomes'] = len(pending)
    return examples, dict(audit)


def extract(archive):
    with ZipFile(archive) as z:
        info = z.getinfo('labeled/hh101.csv')
        if info.file_size > 100_000_000:
            raise ValueError('Source member exceeds bounded acquisition size')
        with z.open(info) as stream:
            return extract_lines(TextIOWrapper(stream, encoding='utf-8'))


def support(examples):
    summary = {}
    for split, rows in examples.items():
        labels = Counter(e.label for e in rows)
        class_days = {str(c):len({e.cutoff.date() for e in rows if e.label==c}) for c in (0,1)}
        summary[split] = {'count':len(rows), 'labels':{str(c):labels[c] for c in (0,1)},
                          'days':len({e.cutoff.date() for e in rows}), 'class_days':class_days}
    groups = defaultdict(lambda:defaultdict(Counter))
    for e in examples['train']:
        key = addresses(e, 'joint')
        groups[key[1]][key[2]][e.label] += 1
    contrasts = 0
    for branches in groups.values():
        neg = {k for k,v in branches.items() if v[0]>=4}
        pos = {k for k,v in branches.items() if v[1]>=4}
        contrasts += bool(neg and pos and len(neg | pos)>1)
    train, dev = summary['train'], summary['development']
    passed = (min(train['labels'].values()) >= 500 and min(dev['labels'].values()) >= 200
              and min(train['class_days'].values()) >= 20 and min(dev['class_days'].values()) >= 10
              and contrasts >= 10)
    return {'splits':summary, 'contrast_groups':contrasts, 'passed':passed}


def phase(model, rows, arm, *, training):
    queue = []
    predictions = []
    # Deliberately mismatched labels are a training-only negative control.
    labels = [e.label for e in rows]
    if training and arm == 'shuffled_outcome':
        offset = len(labels)//2
        labels = labels[offset:] + labels[:offset]
    def flush(at):
        while queue and queue[0][0] <= at:
            due, _, receipt, label = heappop(queue)
            model.deliver(receipt, due, lambda:label, training=training)
    for i, e in enumerate(rows):
        flush(e.cutoff)
        receipt = model.predict(addresses(e, arm), e.cutoff, e.due)
        predictions.append(receipt.probability)
        heappush(queue, (e.due, i, receipt, labels[i]))
    if queue:
        flush(max(v[0] for v in queue))
    return predictions


def metrics(rows, predictions):
    by_day = defaultdict(list)
    for e,p in zip(rows,predictions):
        by_day[e.cutoff.date().isoformat()].append((e.label,p))
    brier = sum((p-e.label)**2 for e,p in zip(rows,predictions))/len(rows)
    balanced = sum(sum((p>=.5)==bool(e.label) for e,p in zip(rows,predictions) if e.label==c)
                   /sum(e.label==c for e in rows) for c in (0,1))/2
    daily = {d:sum((p-y)**2 for y,p in values)/len(values) for d,values in by_day.items()}
    return {'brier':brier, 'daily_brier':sum(daily.values())/len(daily),
            'balanced_accuracy':balanced, 'days':daily}


def compare(examples):
    reports, checkpoints, traces = {}, {}, {}
    for arm in ARMS:
        model = DelayedSparseCounts()
        phase(model, examples['train'], arm, training=True)
        cp = model.checkpoint()
        restored = DelayedSparseCounts.restore(json.loads(json.dumps(cp)))
        before = model.changed_writes
        predictions = phase(model, examples['development'], arm, training=False)
        replay = phase(restored, examples['development'], arm, training=False)
        if predictions != replay or model.changed_writes != before:
            raise ValueError('Saved-state replay or frozen evaluation differs')
        if before != 3 * len(examples['train']):
            raise ValueError('Actual local writes not matched')
        reports[arm] = {**metrics(examples['development'], predictions),
                        'changed_writes':before, 'development_writes':0,
                        'routes':len(model.cells), 'peak_pending':model.peak_pending,
                        'saved_state_replay':True}
        checkpoints[arm] = cp
        traces[arm] = sha256(json.dumps(predictions).encode()).hexdigest()
    best_brier = min(reports[a]['daily_brier'] for a in ('current','current_fine','histogram','transition'))
    best_balanced = max(reports[a]['balanced_accuracy'] for a in ('current','current_fine','histogram','transition'))
    joint = reports['joint']
    gates = {'brier_gain':best_brier-joint['daily_brier'] >= .01,
             'balanced_gain':joint['balanced_accuracy']-best_balanced >= .02,
             'gap_necessity':reports['constant_gap']['daily_brier']-joint['daily_brier'] >= .005,
             'paired_outcome':reports['shuffled_outcome']['daily_brier']-joint['daily_brier'] >= .01,
             'scalar_equivalence':traces['joint']==traces['scalar_joint']}
    return {'arms':reports,'trace_sha256':traces,'gates':gates,
            'quality_passed':all(gates.values()),'test_opened':False}, checkpoints
