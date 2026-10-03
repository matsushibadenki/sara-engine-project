"""Independent scalar temporal controls and a label-blind signature gate."""
from dataclasses import dataclass
from itertools import islice
import random


@dataclass(frozen=True, slots=True)
class TemporalPair:
    mode_zero: tuple[int, ...]
    mode_one: tuple[int, ...]


def bounded_events(events):
    result = tuple(islice(iter(events), 96))
    if len(result) != 95:
        raise ValueError('Exactly 95 events required')
    if any(type(route) is not int or not 0 <= route < 24 for route in result):
        raise ValueError('Invalid comparison route')
    return result


def independent_signature(events, control):
    """No reference computation or outcome is available at this boundary."""
    if control not in ('leaky_analog', 'first_visit', 'histogram'):
        raise ValueError('Unknown independent control')
    events = bounded_events(events)
    if control == 'leaky_analog':
        state = [0.0] * 24
        seen = set()
        for route in events:
            seen.add(route)
            for index in range(24):
                state[index] = 0.95 * state[index] + (0.05 if index == route else 0.0)
        return tuple((route, state[route]) for route in sorted(seen))
    state = {}
    for index, route in enumerate(events):
        if control == 'first_visit':
            state.setdefault(route, index)
        else:
            state[route] = state.get(route, 0) + 1
    return tuple((route, 1.0 / (1 + value) if control == 'first_visit' else value / 95)
                 for route, value in sorted(state.items()))


def synthetic_pairs():
    generator = random.Random(6200410)
    blocks = [(2,) * 91, tuple(2 + index % 22 for index in range(91))]
    blocks.extend(tuple(generator.randrange(2, 24) for _ in range(91)) for _ in range(100))
    suffixes = [block + (0, 1) for block in blocks]
    return tuple(TemporalPair((0, 1) + suffix, (1, 0) + suffix) for suffix in suffixes)


def audit_design():
    # Import only for the separate reference branch, never independent controls.
    from sara_engine.evaluation.electricity_learner import HistoryRoutes, unit_activity
    names = ('leaky_analog', 'first_visit', 'histogram', 'retained_lif')
    collisions = dict.fromkeys(names, 0)
    for pair in synthetic_pairs():
        for name in names:
            if name == 'retained_lif':
                signatures = [tuple((route, strength) for route, strength in
                                    unit_activity(HistoryRoutes(events, (24, 31, 43)), 'stateful_lif')[0]
                                    if route < 24)
                              for events in (pair.mode_zero, pair.mode_one)]
            else:
                signatures = [independent_signature(events, name) for events in (pair.mode_zero, pair.mode_one)]
            collisions[name] += int(signatures[0] == signatures[1])
    pairs = len(synthetic_pairs())
    return {'schema': 'sara-independent-temporal-design-audit-v1',
            'pairs': pairs, 'target_counts': {'0': pairs, '1': pairs},
            'opposite_target_signature_collisions': collisions,
            'control_integrity_passed': collisions['first_visit'] == collisions['leaky_analog'] == 0
                                        and collisions['histogram'] == pairs,
            'lif_learning_comparison_ready': collisions['retained_lif'] == 0,
            'candidate_scored': False, 'real_data_read': False,
            'learning_executed': False, 'promotion_authorized': False}
