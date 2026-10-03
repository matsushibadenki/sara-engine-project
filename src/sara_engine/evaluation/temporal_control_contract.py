"""Input-only temporal controls, isolated from consumed electricity scoring."""
from dataclasses import dataclass
from itertools import islice


@dataclass(frozen=True, slots=True)
class ControlTrace:
    reference_routes: tuple[int, ...]
    equivalent_routes: tuple[int, ...]
    recency_routes: tuple[int, ...]
    reference_unit_ticks: int
    equivalent_gate_ticks: int
    recency_event_visits: int
    ranked_routes: int


def _bounded_events(events):
    values = tuple(islice(iter(events), 96))
    if len(values) != 95:
        raise ValueError('Exactly 95 comparison events required')
    if any(type(route) is not int or not 0 <= route < 24 for route in values):
        raise ValueError('Comparison route outside contract')
    return values


def equivalent_gate(events):
    """Scalar accumulators re-express, rather than ablate, LIF arithmetic."""
    events = _bounded_events(events)
    sums = [0.0] * 24
    cooldown = [0] * 24
    admitted = set()
    for route in events:
        for index in range(24):
            if cooldown[index] > 0:
                cooldown[index] -= 1
                sums[index] = 0.0
            else:
                sums[index] = sums[index] * 0.95 + (0.6 if route == index else 0.0)
                if sums[index] >= 1.0:
                    admitted.add(index)
                    sums[index] = 0.0
                    cooldown[index] = 2
    return tuple(sorted(admitted)), 2280


def recency_selector(events, count):
    """Select a reference-yoked number of routes using history alone."""
    events = _bounded_events(events)
    if type(count) is not int or not 0 <= count <= 24:
        raise ValueError('Invalid bounded output count')
    last_seen = {route: index for index, route in enumerate(events)}
    if count > len(last_seen):
        raise ValueError('Output count exceeds observed routes')
    ranked = sorted(last_seen, key=lambda route: (-last_seen[route], route))
    return tuple(sorted(ranked[:count])), len(events), len(last_seen)


def compare_controls(events):
    """Produce no predictions; reference count is disclosed input-only yoking."""
    events = _bounded_events(events)
    # Import the unchanged retained implementation only as a reference.
    from sara_engine.evaluation.electricity_learner import HistoryRoutes, unit_activity
    active, reference_ticks = unit_activity(HistoryRoutes(events, (24, 31, 43)), 'stateful_lif')
    reference = tuple(route for route, _ in active if route < 24)
    equivalent, equivalent_ticks = equivalent_gate(events)
    recency, visits, ranked = recency_selector(events, len(reference))
    if equivalent != reference or len(recency) != len(reference):
        raise ValueError('Temporal control contract violated')
    return ControlTrace(reference, equivalent, recency, reference_ticks,
                        equivalent_ticks, visits, ranked)
