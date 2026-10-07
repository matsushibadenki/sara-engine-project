"""Input-only bounded temporal references; no outcomes or learning interface."""
from itertools import combinations

from sara_engine.evaluation.temporal_information import TemporalInformationObserver

PROTOCOL_SHA256 = '7574d2d1ea3ddec03af52229b7db247d53728e10bca425ea518a10017eb2c789'


class BoundedSequenceObserver:
    """Retain at most 95 route/gap pairs; reject overflow before mutation."""

    def __init__(self):
        self._events = []

    def observe(self, route, gap):
        if type(route) is not int or not 0 <= route < 24:
            raise ValueError('Route outside contract')
        if type(gap) is not int or not 0 <= gap <= 255:
            raise ValueError('Gap outside contract')
        if (not self._events and gap != 0) or (self._events and gap == 0):
            raise ValueError('Invalid relative timestamp')
        if len(self._events) >= 95:
            raise ValueError('Event capacity exceeded')
        self._events.append((route, gap))

    def snapshot(self):
        return tuple(self._events)

    def wire_bytes(self):
        return b'TS1' + bytes((len(self._events),)) + bytes(
            value for event in self._events for value in event)


def signatures(events):
    ordinal = TemporalInformationObserver()
    sequence = BoundedSequenceObserver()
    for route, gap in events:
        ordinal.observe(route)
        sequence.observe(route, gap)
    info = ordinal.snapshot()
    return {'histogram': tuple((r, n) for r, _, _, n in info.records),
            'first_last_count': info.records,
            'full_order': tuple(r for r, _ in sequence.snapshot()),
            'full_order_gap': sequence.snapshot()}, len(sequence.wire_bytes())


def audit_boundary(protocol):
    """Exhaust every route pair; labels remain outside observer inputs."""
    results = {}
    max_wire = max_events = visits = 0
    for family in protocol['families']:
        distinct = {name: 0 for name in protocol['controls']}
        pairs = nuisance = 0
        for routes in combinations(range(24), 2):
            observed = []
            histories = []
            for template, gaps in zip(protocol['templates'][family], protocol['gap_templates'][family]):
                if len(template) != len(gaps):
                    raise ValueError('Template cardinality changed')
                events = tuple((routes[index], gap) for index, gap in zip(template, gaps))
                sig, size = signatures(events)
                observed.append(sig)
                histories.append(events)
                max_wire = max(max_wire, size)
                max_events = max(max_events, len(events))
                visits += len(events)
            pairs += 1
            nuisance += int(observed[0]['histogram'] == observed[1]['histogram']
                            and sum(g for _, g in histories[0]) == sum(g for _, g in histories[1]))
            for name in distinct:
                distinct[name] += int(observed[0][name] != observed[1][name])
        expected = protocol['expected_distinct'][family]
        passed = nuisance == pairs and all(
            count == (pairs if name in expected else 0) for name, count in distinct.items())
        results[family] = {'pairs': pairs, 'nuisance_matched_pairs': nuisance,
                           'distinct_pairs': distinct, 'passed': passed}
    return {'schema': 'sara-temporal-capability-boundary-audit-v1',
            'families': results, 'max_wire_bytes': max_wire, 'max_events': max_events,
            'input_visits': visits, 'wire_capacity_bytes': 194,
            'accounting': 'Wire size excludes Python object overhead; every reference pays its own observation and projection cost.',
            'passed': all(r['passed'] for r in results.values()) and max_wire <= 194 and max_events <= 95,
            'learning_executed': False, 'real_data_read': False,
            'spike_specific_advantage_established': False, 'promotion_authorized': False}
