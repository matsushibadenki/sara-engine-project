"""Bounded label-free arrival and address-growth audit of fresh training inputs."""
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from sara_engine.evaluation.casas_180_source import timestamp
from sara_engine.evaluation.casas_longitudinal import Example, gap_bin
from sara_engine.evaluation.casas_pooling import routes
from sara_engine.evaluation.casas_vocabulary_budget import universe_bound

ARMS = ('pool_evidence','joint_evidence')
ARRIVAL_TARGET = 70212


@dataclass(frozen=True)
class Limits:
    opened_rows: int = 5000000
    qualified_outcomes: int = 263295
    hypothetical_writes: int = 789885
    routes: int = 100000
    vocabulary: int = 100000
    pending: int = 16


def measure(lines, metadata, reference, limits=Limits()):
    """Only training input fields are parsed; unknown binary events remain eligible."""
    if any(type(v) is not int or v < 1 for v in vars(limits).values()):
        raise ValueError('Positive integer audit limits required')
    reference = {tuple(v) for v in reference}
    start, end = metadata['train_start'], metadata['development_start']
    excluded = set(metadata['clock_regression_days'])
    history, pending = deque(maxlen=8), deque()
    previous = last_cutoff = None
    audit, raw_current, rich_current, stream = Counter(), Counter(), Counter(), Counter()
    state = {arm:set() for arm in ARMS}
    retained, growth = set(), {}
    qualified = peak_pending = novel_types = 0
    delay_total = 0.0
    delay_min = delay_max = None
    terminal = None

    def stop(reason, at):
        nonlocal terminal
        terminal = {'reason':reason,'at':at.isoformat() if hasattr(at,'isoformat') else at,
                    'accepted_qualified_outcomes':qualified}

    def snapshot(day):
        row = growth[day]
        row.update({'cumulative_qualified':qualified,'cumulative_stream_event_types':len(stream),
                    'cumulative_new_reference_types':novel_types,
                    'cumulative_retained_event_types':len(retained),
                    'cumulative_rich_contexts':len(rich_current),
                    'cumulative_hypothetical_writes_per_arm':3*qualified,
                    **{arm+'_routes':len(keys) for arm,keys in state.items()}})

    def admit(cutoff, past, gaps, due):
        nonlocal qualified, delay_total, delay_min, delay_max
        if qualified >= limits.qualified_outcomes or 3*(qualified+1) > limits.hypothetical_writes:
            stop('qualified_or_write_budget',due)
            return False
        # A dummy label is needed only for the existing address API; no label is read.
        example = Example(cutoff,due,past,gaps,0)
        keys = {arm:routes(example,arm) for arm in ARMS}
        if any(len(state[arm])+sum(key not in state[arm] for key in addresses) > limits.routes
               for arm,addresses in keys.items()):
            audit['uncommitted_qualified_outcomes'] += 1
            stop('route_capacity',due)
            return False
        # Commit both audit address sets atomically; these are not learned counters.
        for arm,addresses in keys.items():
            state[arm].update(addresses)
        raw_current[past[-1]] += 1
        rich_current[keys['pool_evidence'][1]] += 1
        retained.update((past[-1],past[-2],past[-5]))
        qualified += 1
        growth[due.date().isoformat()]['qualified_arrivals'] += 1
        delay = (due-cutoff).total_seconds()
        delay_total += delay
        delay_min = delay if delay_min is None else min(delay_min,delay)
        delay_max = delay if delay_max is None else max(delay_max,delay)
        if qualified == limits.qualified_outcomes or 3*(qualified+1) > limits.hypothetical_writes:
            stop('qualified_or_write_budget',due)
        return True

    for line in lines:
        prefix = line.split(',',1) if ',' in line else line.split(None,1)
        if len(prefix) != 2:
            raise ValueError('Missing date field')
        day,remainder = prefix
        if day >= end:
            audit['sealed_date_only_rows'] += 1
            continue
        if day < start or day in excluded:
            audit['metadata_excluded_rows'] += 1
            continue
        if terminal is not None:
            audit['uninspected_training_rows'] += 1
            continue
        if audit['opened_rows'] >= limits.opened_rows:
            stop('opened_row_budget',day)
            audit['uninspected_training_rows'] += 1
            continue
        fields = remainder.split(',',3) if ',' in line else remainder.split(None,3)
        if len(fields) < 3:
            raise ValueError('Training input schema differs')
        at = timestamp(day,fields[0])
        if previous is not None and at < previous:
            raise ValueError('Qualified training clock regressed')
        previous = at
        audit['opened_rows'] += 1
        row = growth.setdefault(day,{'opened_rows':0,'binary_events':0,'qualified_arrivals':0})
        row['opened_rows'] += 1
        if fields[2] not in ('ON','OFF','OPEN','CLOSE','CLOSED'):
            audit['nonbinary_rows'] += 1
            snapshot(day)
            continue
        event = (fields[1],fields[2])
        if event not in stream and len(stream) >= limits.vocabulary:
            stop('audit_vocabulary_capacity',at)
            snapshot(day)
            continue
        if event not in stream and event not in reference:
            novel_types += 1
        stream[event] += 1
        audit['binary_events'] += 1
        audit['new_reference_event_occurrences'] += int(event not in reference)
        row['binary_events'] += 1
        while pending and pending[0][0]+timedelta(seconds=300) <= at:
            cutoff,past,gaps = pending.popleft()
            if (at-cutoff).total_seconds() > 600:
                audit['late_outcomes'] += 1
                continue
            if not admit(cutoff,past,gaps,at) or terminal is not None:
                break
        if terminal is not None:
            snapshot(day)
            continue
        history.append((at,event))
        if last_cutoff is not None and (at-last_cutoff).total_seconds() < 60:
            snapshot(day)
            continue
        last_cutoff = at
        if len(history) == 8:
            intervals = [(b[0]-a[0]).total_seconds() for a,b in zip(history,list(history)[1:])]
            if any(not 0 <= interval <= 300 for interval in intervals):
                audit['interrupted_histories'] += 1
            else:
                if len(pending) >= limits.pending:
                    stop('pending_capacity',at)
                else:
                    pending.append((at,tuple(v for _,v in history),tuple(gap_bin(v) for v in intervals)))
                    peak_pending = max(peak_pending,len(pending))
        snapshot(day)
    audit['unresolved_outcomes'] = len(pending)
    recurrence = {'raw_current_types':len(raw_current),
                  'raw_repeat_row_fraction':sum(v for v in raw_current.values() if v>=2)/qualified if qualified else 0,
                  'rich_current_contexts':len(rich_current),
                  'rich_contexts_support8':sum(v>=8 for v in rich_current.values())}
    gates = {'long_source':metadata['valid_recorded_days']>=180,
             'whole_training_window':terminal is None,
             'arrival_target':qualified>=ARRIVAL_TARGET,
             'raw_recurrence':recurrence['raw_repeat_row_fraction']>=.9,
             'rich_recurrence':recurrence['rich_contexts_support8']>=10,
             'observed_route_budget':all(len(keys)<=limits.routes for keys in state.values()),
             'matched_hypothetical_budget':3*qualified<=limits.hypothetical_writes}
    calendar_days = (datetime.fromisoformat(end)-datetime.fromisoformat(start)).days
    return {'audit':dict(audit),'terminal_stop':terminal,'qualified_outcomes':qualified,
            'hypothetical_readout_writes_per_arm':3*qualified,'actual_learning_writes_per_arm':{a:0 for a in ARMS},
            'routes':{a:len(keys) for a,keys in state.items()},'peak_pending':peak_pending,
            'recurrence':recurrence,'stream_event_types':len(stream),
            'new_reference_event_types':novel_types,
            'retained_event_types':len(retained),
            'retained_vocabulary_envelopes':{a:universe_bound(len(retained),a) if retained else None for a in ARMS},
            'arrival_per_recorded_training_day':qualified/metadata['train_recorded_days'],
            'arrival_per_calendar_day':qualified/calendar_days,
            'qualified_per_binary_event':qualified/audit['binary_events'] if audit['binary_events'] else 0,
            'arrival_dates':sum(v['qualified_arrivals']>0 for v in growth.values()),
            'delay_seconds':{'minimum':delay_min,'maximum':delay_max,'mean':delay_total/qualified if qualified else None},
            'growth':growth,'gates':gates,'preflight_passed':all(gates.values()),
            'task_class_quality_gates_assessed':False,'independent_validation_passed':False,
            'outcome_labels_computed':False,'original_development_opened':False,'test_opened':False,
            'inventory':{'stream_events':[{'event':e,'occurrences':count,'new_to_reference':e not in reference} for e,count in sorted(stream.items())],
                         'retained_events':sorted(retained)}}
