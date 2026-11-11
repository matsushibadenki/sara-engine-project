"""Compare a scalar replay to the frozen retained-route captures."""
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from random import Random

from scripts.eval import route_retained_receipt as retained
from sara_engine.evaluation.route_capture_candidate import CapturePair, PhaseCapture
from sara_engine.evaluation.route_capture_layout import canonical, validate
from sara_engine.utils.project_paths import processed_data_path


def replay(spec):
    """Yield scalar state immediately before each outcome, using selection inputs only."""
    routes = [(channels, output) for width in (1, 2)
              for channels in combinations(range(5), width) for output in range(3)]
    groups = [[i for i, (_, target) in enumerate(routes) if target == output]
              for output in range(3)]
    rng = Random(spec['model']['seed'] + 300000)
    for group in groups:
        rng.shuffle(group)
    counts = {}
    statistics = [(0, 0.0, 0.0)] * len(routes)
    events = spec['input']['events']
    for tick in range(spec['budget']['begin']):
        fields = []
        flags = []
        for route, (channels, _) in enumerate(routes):
            for index, key in enumerate(product((0, 1), repeat=len(channels))):
                address = (route, key)
                flags.append((route, index, address in counts))
                left, right = counts.get(address, (0, 0))
                fields.extend([((route, 'mass_left', index), left),
                               ((route, 'mass_right', index), right)])
            fields.extend(((route, name, -1), value) for name, value in
                          zip(('observations', 'error_sum', 'weight_sum'), statistics[route]))

        credits = []
        for group in groups:
            for offset in range(3):
                route = group[(3 * tick + offset) % len(group)]
                key = tuple(events[channel] for channel in routes[route][0])
                mass = counts.get((route, key))
                probability = None if mass is None else (mass[1] + 1) / (sum(mass) + 2)
                credits.append((route, key, mass is not None, probability))
        yield {'fields': sorted(fields), 'allocations': flags, 'receipt_credits': credits}

        if tick < spec['budget']['observe']:
            for route, key, _, probability in credits:
                outcome = spec['outcome'][routes[route][1]]
                mass = list(counts.get((route, key), (0, 0)))
                mass[outcome] += 1
                counts[(route, key)] = tuple(mass)
                observations, error, weight = statistics[route]
                estimate = 0.5 if probability is None else probability
                statistics[route] = observations + 1, error + (estimate - outcome) ** 2, weight + 1.0


def compare():
    selection_path = Path(processed_data_path('benchmark_fixtures', 'route_scalar_comparison_selection_v1.json'))
    selection = json.loads(selection_path.read_bytes())
    retained.verify_sources(selection)
    spec = retained.read('selection')
    retained.require(spec['model']['history'] == 'cumulative' and spec['model']['error_weight'] == 'uniform',
                     'Scalar replay requires cumulative/uniform inputs')
    result = retained.read('result')
    retained.verify_result(spec, result)
    for tick, (packet, record, expected) in enumerate(
        zip(retained.packets(spec), result['records'], replay(spec), strict=True)
    ):
        saved = record['pair']
        phases = [PhaseCapture(**dict(saved[name], state=retained.tuple_tree(saved[name]['state'])))
                  for name in ('predecessor', 'receipt')]
        actual = validate(CapturePair(packet, *phases))
        for name, values in expected.items():
            retained.require(canonical(actual[name]) == canonical(values),
                             f'Scalar comparison differs at tick {tick}: {name}')
    print('Scalar replay matched all six saved receipts')


if __name__ == '__main__':
    compare()
