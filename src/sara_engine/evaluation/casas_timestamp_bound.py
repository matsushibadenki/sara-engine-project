"""Safe timestamp-only arrival bounds for any binary subsequence of raw rows."""
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from sara_engine.evaluation.casas_180_source import timestamp

TARGET = 70212
OBSERVATION_BUDGET = 263295
WRITE_BUDGET = 789885


@dataclass(frozen=True)
class BoundLimits:
    opened_timestamps: int = 5000000
    pending_buckets: int = 16


@dataclass
class Bucket:
    minute: datetime
    first: datetime
    last: datetime


class TimestampBound:
    """A minute contains at most one cutoff at the unchanged >=60s spacing."""
    def __init__(self, limits=BoundLimits()):
        self.limits = limits
        self.previous = None
        self.run = 0
        self.pending = deque()
        self.counts = Counter()
        self.daily = {}
        self.terminal_stop = None

    def feed(self, at):
        if self.previous is not None and at < self.previous:
            raise ValueError('Training timestamp clock regressed')
        day = at.date().isoformat()
        row = self.daily.setdefault(day,Counter())
        row['timestamps'] += 1
        self.counts['timestamps'] += 1
        # Each bucket's possible-result windows overlap: span<60s, width300s.
        while self.pending and self.pending[0].first+timedelta(seconds=300) <= at:
            bucket = self.pending.popleft()
            status = 'eligible_buckets' if at <= bucket.last+timedelta(seconds=600) else 'expired_buckets'
            self.counts[status] += 1
            self.daily[bucket.first.date().isoformat()][status] += 1
            self.counts['queue_pops'] += 1
        self.run = 1 if self.previous is None or at-self.previous > timedelta(seconds=300) else min(8,self.run+1)
        self.previous = at
        if self.run < 8:
            return
        self.counts['potential_cutoff_rows'] += 1
        row['potential_cutoff_rows'] += 1
        minute = at.replace(second=0,microsecond=0)
        if self.pending and self.pending[-1].minute == minute:
            self.pending[-1].last = at
        else:
            if len(self.pending) >= self.limits.pending_buckets:
                self.terminal_stop = {'reason':'pending_bucket_capacity','at':at.isoformat()}
                return
            self.pending.append(Bucket(minute,at,at))
            self.counts['candidate_buckets'] += 1
            self.counts['queue_pushes'] += 1
            row['candidate_buckets'] += 1
            self.counts['peak_pending_buckets'] = max(self.counts['peak_pending_buckets'],len(self.pending))

    def finish(self):
        unresolved = len(self.pending)
        daily = {day:dict(counts) for day,counts in sorted(self.daily.items())}
        for bucket in self.pending:
            day = bucket.first.date().isoformat()
            daily[day]['unresolved_buckets'] = daily[day].get('unresolved_buckets',0)+1
        if self.counts['candidate_buckets'] != self.counts['eligible_buckets']+self.counts['expired_buckets']+unresolved:
            raise ValueError('Bucket conservation failed')
        return {'work':dict(self.counts),'unresolved_buckets':unresolved,
                'daily':daily,
                'terminal_stop':self.terminal_stop}


def bound_lines(lines, metadata, limits=BoundLimits()):
    """Read only date/time prefixes; complete-stream failure is inconclusive."""
    if any(type(v) is not int or v < 1 for v in vars(limits).values()):
        raise ValueError('Positive integer bound limits required')
    audit = Counter()
    bound = TimestampBound(limits)
    start,end = metadata['train_start'],metadata['development_start']
    excluded = set(metadata['clock_regression_days'])
    for line in lines:
        audit['streamed_rows'] += 1
        # Choose delimiter at the ISO-date boundary, never from event payload.
        comma = len(line)>10 and line[10]==','
        prefix = line.split(',',1) if comma else line.split(None,1)
        if not prefix:
            raise ValueError('Missing date prefix')
        day = prefix[0]
        if day >= end:
            audit['sealed_date_only_rows'] += 1
            continue
        if day < start or day in excluded:
            audit['metadata_excluded_rows'] += 1
            continue
        if bound.terminal_stop is not None:
            audit['uninspected_training_rows'] += 1
            continue
        if audit['training_timestamp_rows'] >= limits.opened_timestamps:
            bound.terminal_stop = {'reason':'timestamp_row_capacity','at':day}
            audit['uninspected_training_rows'] += 1
            continue
        if len(prefix) != 2:
            raise ValueError('Missing training timestamp')
        remainder = prefix[1]
        time_parts = remainder.split(',',1) if comma else remainder.split(None,1)
        if not time_parts or not time_parts[0]:
            raise ValueError('Missing training timestamp')
        time = time_parts[0]
        at = timestamp(day,time)
        audit['training_timestamp_rows'] += 1
        bound.feed(at)
    result = bound.finish()
    complete = result['terminal_stop'] is None
    upper = result['work'].get('eligible_buckets',0) if complete else None
    return {**result,'audit':dict(audit),'timestamp_upper_bound':upper,
            'budget_capped_upper_bound':min(upper,OBSERVATION_BUDGET,WRITE_BUDGET//3) if complete else None,
            'screen_decision':('exclude_below_target' if upper < TARGET else 'cannot_exclude') if complete else 'inconclusive',
            'target':TARGET,'scope':'Any binary subsequence of complete valid training timestamps; count upper bound only',
            'sensor_message_fields_inspected':False,'outcome_labels_computed':False,
            'actual_learning_writes':0,'original_development_opened':False,'test_opened':False}


def synthetic_lines(seconds, mask=None):
    origin = datetime(2011,6,15)
    if mask is None:
        mask = [1]*len(seconds)
    return [f'{(origin+timedelta(seconds=s)).date()} {(origin+timedelta(seconds=s)).time()} Sensor '+('ON' if binary else '17')
            for s,binary in zip(seconds,mask)]


def validate_counterexamples():
    """Exhaust all binary masks in three frozen traces using the original extractor."""
    from sara_engine.evaluation.casas_arrival_preflight import measure
    meta = {'train_start':'2011-06-15','development_start':'2011-06-16',
            'clock_regression_days':[],'valid_recorded_days':180,'train_recorded_days':1}
    witness = [0]*8+[60,119,700]
    naive = measure(synthetic_lines(witness),meta,[])['qualified_outcomes']
    mask = [1]*len(witness)
    mask[8] = 0
    filtered = measure(synthetic_lines(witness,mask),meta,[])['qualified_outcomes']
    upper = bound_lines(synthetic_lines(witness),meta)['timestamp_upper_bound']
    if not naive < filtered <= upper:
        raise ValueError('Frozen all-raw greedy counterexample failed')
    traces = (witness,[0,1,2,3,4,5,6,7,60,359,360,719],[0]*8+[300,600,900])
    cases = 0
    maxima = []
    for times in traces:
        ceiling = bound_lines(synthetic_lines(times),meta)['timestamp_upper_bound']
        maximum = 0
        for flags in product((0,1),repeat=len(times)):
            count = measure(synthetic_lines(times,flags),meta,[])['qualified_outcomes']
            if count > ceiling:
                raise ValueError('Timestamp bound undercounts a binary subsequence')
            maximum = max(maximum,count)
            cases += 1
        maxima.append({'rows':len(times),'binary_masks':2**len(times),'maximum_actual_arrivals':maximum,'timestamp_upper_bound':ceiling})
    return {'all_raw_greedy_arrivals':naive,'filtered_arrivals':filtered,'safe_upper_bound':upper,
            'mask_cases_checked':cases,'undercounts':0,'traces':maxima,'passed':True}
