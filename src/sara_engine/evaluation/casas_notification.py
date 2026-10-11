"""Fresh HH102 observed kitchen-notification task; annotations stay unused."""
from collections import Counter, deque
from datetime import datetime, timedelta
from io import TextIOWrapper
from zipfile import ZipFile
from sara_engine.evaluation.casas_longitudinal import Example, gap_bin

START = datetime(2011, 6, 15)
TRAIN_END = datetime(2011, 7, 15)
TEST_START = datetime(2011, 7, 30)


def extract_lines(lines):
    history, pending = deque(maxlen=8), deque()
    examples = {'train': [], 'development': []}
    audit = Counter()
    previous = None
    last_cutoff = None
    for line in lines:
        fields = line.rstrip('\r\n').split(',', 4)
        if len(fields) < 2:
            raise ValueError('Missing date/time fields')
        day = datetime.strptime(fields[0], '%Y-%m-%d')
        if day >= TEST_START:
            audit['sealed_timestamp_only_rows'] += 1
            continue
        if day < START:
            audit['excluded_rows'] += 1
            continue
        stamp = fields[0] + ' ' + fields[1]
        t = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S.%f' if '.' in fields[1] else '%Y-%m-%d %H:%M:%S')
        if previous is not None and t < previous:
            raise ValueError('Source clock regressed')
        previous = t
        if len(fields) < 4:
            audit['non_notification_rows'] += 1
            continue
        audit['opened_rows'] += 1
        if fields[3] not in ('ON', 'OFF', 'OPEN', 'CLOSE', 'CLOSED'):
            audit['nonbinary_rows'] += 1
            continue
        event = (fields[2], fields[3])
        while pending and pending[0][0]+timedelta(seconds=300) <= t:
            cutoff, old_history, gaps = pending.popleft()
            if (t-cutoff).total_seconds() > 600:
                audit['late_outcomes'] += 1
                continue
            if (cutoff < TRAIN_END) != (t < TRAIN_END):
                audit['cross_split_outcomes'] += 1
                continue
            label = int(event[0].lower().startswith('kitchen') and event[1]=='ON')
            split = 'train' if cutoff < TRAIN_END else 'development'
            examples[split].append(Example(cutoff,t,old_history,gaps,label))
        history.append((t,event))
        if last_cutoff is not None and (t-last_cutoff).total_seconds() < 60:
            continue
        last_cutoff = t
        if len(history)!=8 or (history[0][0]<TRAIN_END)!=(t<TRAIN_END):
            continue
        intervals = [(b[0]-a[0]).total_seconds() for a,b in zip(history,list(history)[1:])]
        if any(not 0<=d<=300 for d in intervals):
            audit['interrupted_histories'] += 1
            continue
        pending.append((t,tuple(v for _,v in history),tuple(gap_bin(d) for d in intervals)))
        if len(pending)>16:
            raise ValueError('Pending extraction capacity exceeded')
    audit['unresolved_outcomes'] = len(pending)
    return examples,dict(audit)


def extract(archive):
    with ZipFile(archive) as z:
        info = z.getinfo('labeled/hh102.csv')
        if info.file_size>100_000_000:
            raise ValueError('Source size bound exceeded')
        with z.open(info) as stream:
            return extract_lines(TextIOWrapper(stream,encoding='utf-8'))
