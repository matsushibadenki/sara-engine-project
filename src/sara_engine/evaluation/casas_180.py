"""Fresh long-household task with unchanged delayed comparison conditions."""
from collections import Counter, deque
from datetime import timedelta
from sara_engine.evaluation.casas_180_source import member_lines, timestamp
from sara_engine.evaluation.casas_longitudinal import Example, gap_bin, support, addresses


def extract_lines(lines, metadata):
    """Discard sealed fields before sensor/message or time interpretation."""
    start = metadata['train_start']
    development = metadata['development_start']
    sealed = metadata['test_start']
    excluded = set(metadata['clock_regression_days'])
    history, pending = deque(maxlen=8), deque()
    examples = {'train': [], 'development': []}
    audit = Counter()
    previous = None
    last_cutoff = None
    for line in lines:
        # Only date is consulted for sealed or metadata-invalid days.
        prefix = line.split(',', 1) if ',' in line else line.split(None, 1)
        if len(prefix) != 2:
            raise ValueError('Missing date field')
        day, remainder = prefix
        if day >= sealed:
            audit['sealed_date_only_rows'] += 1
            continue
        if day < start or day in excluded:
            audit['metadata_excluded_rows'] += 1
            continue
        fields = remainder.split(',', 3) if ',' in line else remainder.split(None, 3)
        if len(fields) < 3:
            audit['non_notification_rows'] += 1
            continue
        t = timestamp(day, fields[0])
        if previous is not None and t < previous:
            raise ValueError('Qualified source clock regressed')
        previous = t
        audit['opened_rows'] += 1
        if fields[2] not in ('ON', 'OFF', 'OPEN', 'CLOSE', 'CLOSED'):
            audit['nonbinary_rows'] += 1
            continue
        event = (fields[1], fields[2])
        while pending and pending[0][0] + timedelta(seconds=300) <= t:
            cutoff, past, gaps = pending.popleft()
            if (t-cutoff).total_seconds() > 600:
                audit['late_outcomes'] += 1
                continue
            if (cutoff.date().isoformat() < development) != (day < development):
                audit['cross_split_outcomes'] += 1
                continue
            y = int(event[0].lower().startswith('kitchen') and event[1] == 'ON')
            split = 'train' if cutoff.date().isoformat() < development else 'development'
            examples[split].append(Example(cutoff, t, past, gaps, y))
        history.append((t, event))
        if last_cutoff is not None and (t-last_cutoff).total_seconds() < 60:
            continue
        last_cutoff = t
        if len(history) != 8 or (history[0][0].date().isoformat() < development) != (day < development):
            continue
        intervals = [(b[0]-a[0]).total_seconds() for a,b in zip(history, list(history)[1:])]
        if any(not 0 <= d <= 300 for d in intervals):
            audit['interrupted_histories'] += 1
            continue
        pending.append((t, tuple(v for _,v in history), tuple(gap_bin(d) for d in intervals)))
        if len(pending) > 16:
            raise ValueError('Pending extraction bound exceeded')
    audit['unresolved_outcomes'] = len(pending)
    return examples, dict(audit)


def extract(source, receipt, metadata):
    return extract_lines(member_lines(source, receipt), metadata)


def qualification(examples, metadata):
    """Retain every original class/day/contrast threshold, add recurrence audit."""
    report = support(examples)
    raw = Counter(e.history[-1] for e in examples['train'])
    rich = Counter(addresses(e, 'current')[1] for e in examples['train'])
    n = len(examples['train'])
    recurrence = {'raw_states':len(raw), 'repeated_raw_states':sum(v>=2 for v in raw.values()),
                  'raw_repeat_row_fraction':sum(v for v in raw.values() if v>=2)/n if n else 0,
                  'rich_current_contexts':len(rich), 'rich_contexts_with_at_least_eight':sum(v>=8 for v in rich.values())}
    report['recurrence'] = recurrence
    report['long_source'] = metadata['valid_recorded_days'] >= 180
    report['passed'] = (report['passed'] and report['long_source']
                        and recurrence['raw_repeat_row_fraction'] >= .9
                        and recurrence['rich_contexts_with_at_least_eight'] >= 10)
    return report
