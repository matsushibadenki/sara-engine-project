"""Bounded, forward-only support audit; never exports load features."""
from collections import deque
from datetime import timedelta

from sara_engine.evaluation.electricity_access_boundary import parse_support_row


def audit_support(lines, protocol, *, client_count=370):
    history = protocol['history_observations_including_cutoff']
    delay = protocol['outcome_delay_intervals']
    capacity = history + delay
    step = timedelta(minutes=protocol['interval_minutes'])
    excluded = set(protocol['excluded_transition_dates'])
    splits = {'training': set(protocol['training_years']),
              'development': set(protocol['development_years'])}
    counts = {name: [[0, 0, 0] for _ in range(client_count)] for name in splits}
    active = {name: set() for name in splits}
    activation = [None] * client_count
    window = deque(maxlen=capacity)
    previous = None
    parsed = sealed = peak = 0
    for line in lines:
        timestamp, values = parse_support_row(line, client_count=client_count)
        if previous is not None and timestamp - previous != step:
            raise ValueError('Non-contiguous source timestamps')
        previous = timestamp
        if values is None:
            sealed += 1
            window.clear()
            continue
        parsed += client_count
        split = next(name for name, years in splits.items() if timestamp.year in years)
        for index, value in enumerate(values):
            if value is not None and value > 0:
                active[split].add(index)
                if activation[index] is None:
                    activation[index] = timestamp
        window.append((timestamp, values, split))
        peak = max(peak, len(window))
        if len(window) != capacity:
            continue
        cutoff = window[history - 1]
        if cutoff[0].strftime('%H:%M:%S') != protocol['cutoff_local_time']:
            continue
        if any(row[2] != split or row[0].date().isoformat() in excluded for row in window):
            continue
        for index in range(client_count):
            if activation[index] is None or activation[index] > window[0][0]:
                continue
            if any(row[1][index] is None for row in window):
                continue
            start, end = cutoff[1][index], values[index]
            direction = 0 if end > start else 1 if end < start else 2
            counts[split][index][direction] += 1
    minimum = protocol['minimum_observations_per_class_per_qualifying_client']
    summary = {}
    for name, clients in counts.items():
        qualifying = sum(rise >= minimum and fall >= minimum for rise, fall, _ in clients)
        summary[name] = {
            'active_clients': len(active[name]), 'qualifying_clients': qualifying,
            'rise_observations': sum(row[0] for row in clients),
            'fall_observations': sum(row[1] for row in clients),
            'excluded_ties': sum(row[2] for row in clients),
            'passed': len(active[name]) >= protocol['minimum_active_clients_per_split']
                      and qualifying >= protocol['minimum_clients_with_both_classes_per_split'],
        }
    return {'schema': 'sara-r2-electricity-support-v1',
            'passed': all(row['passed'] for row in summary.values()), 'splits': summary,
            'numeric_cells_parsed': parsed, 'opaque_rows': sealed,
            'peak_buffer_rows': peak, 'buffer_capacity_rows': capacity,
            'held_out_numeric_cells_parsed': 0, 'candidate_scored': False,
            'features_materialized': False, 'support_is_independent_sample_size': False}
