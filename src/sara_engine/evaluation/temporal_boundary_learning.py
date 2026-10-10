"""Frozen input-only preflight for matched temporal boundary learning."""
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation.temporal_capability_boundary import BoundedSequenceObserver
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.utils.project_paths import processed_data_path, project_path

PROTOCOL_SHA256 = '390ab21e3310e4d9253dc8cc5d1b9925a2f445c3fd50a575bf5d9018dcea7aa9'
ARMS = ('histogram', 'first_last_count', 'ordinal', 'gap', 'joint')


def load_protocol():
    raw = Path(processed_data_path('benchmark_fixtures', 'temporal_boundary_learning_v1.json')).read_bytes()
    if sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('Frozen learning protocol changed')
    protocol = json.loads(raw)
    for name, expected in protocol['baseline_sha256'].items():
        if sha256(Path(project_path(name)).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen baseline changed')
    return protocol


def paired_inputs(protocol, split, seed, family):
    """Yield paired histories; targets and identities never enter the encoder."""
    if split not in ('training', 'evaluation') or family not in protocol['families']:
        raise ValueError('Unknown input partition')
    if seed not in protocol[split + '_seeds']:
        raise ValueError('Unregistered seed')
    for index in range(protocol['pairs_per_family'][split]):
        key = sha256(f'{seed}:{family}:{index}'.encode()).digest()
        routes = sorted(protocol['route_pools'][split], key=lambda r: sha256(key + b':' + str(r).encode()).digest())[:4]
        repetitions = protocol['suffix_repetitions'][split][key[0] % 2]
        suffix = ((routes[2], 1), (routes[3], 1)) * repetitions
        pair = tuple(tuple((routes[r], gap) for r, gap in zip(template, gaps)) + suffix
                     for template, gaps in zip(protocol['templates'][family], protocol['gaps'][family]))
        yield f'{split}:{seed}:{family}:{index}', tuple((mode, pair[mode]) for mode in ((1, 0) if key[1] % 2 else (0, 1)))


def encode(events, arm):
    """Expose only bounded history features; no identity or outcome argument."""
    if arm not in (*ARMS, 'joint_shuffled'):
        raise ValueError('Unknown arm')
    observer = BoundedSequenceObserver()
    for route, gap in events:
        observer.observe(route, gap)
    snapshot = observer.snapshot()
    if not snapshot:
        raise ValueError('Empty history')
    ranks, records, active = {}, {}, []
    for position, (route, gap) in enumerate(snapshot):
        rank = ranks.setdefault(route, len(ranks))
        if rank not in records:
            records[rank] = [position, position, 0]
        records[rank][1] = position
        records[rank][2] += 1
        if arm in ('ordinal', 'joint', 'joint_shuffled'):
            active.append((24 * position + rank, 1.0))
        if arm in ('gap', 'joint', 'joint_shuffled'):
            active.append((2280 + 256 * position + gap, 1.0))
    if arm in ('histogram', 'first_last_count'):
        for rank, (first, last, count) in records.items():
            if arm == 'first_last_count':
                active.extend(((26600 + 3 * rank, 1 / (first + 1)),
                               (26601 + 3 * rank, (last + 1) / 95)))
            active.append((26602 + 3 * rank, count / 95))
    active.append((26672, 1.0))
    features = tuple(sorted(active))
    costs = {'observation_visits': len(snapshot),
             'projection_visits': len(snapshot) + (3 * len(records) if arm == 'first_last_count' else len(records) if arm == 'histogram' else 0),
             'active_channels': len(features), 'wire_bytes': len(observer.wire_bytes()),
             'transient_bytes': _deep_size((observer.__dict__, snapshot, ranks, records, active, features))}
    return features, costs


def audit_inputs(protocol):
    """Check paired identifiability and disjoint raw histories without learners."""
    seeds = protocol['training_seeds'] + protocol['evaluation_seeds']
    if len(seeds) != len(set(seeds)):
        raise ValueError('Seed overlap')
    if set(protocol['route_pools']['training']) & set(protocol['route_pools']['evaluation']):
        raise ValueError('Route pool overlap')
    digest = sha256()
    families = {}
    peaks = dict.fromkeys(('observation_visits', 'projection_visits', 'active_channels', 'wire_bytes', 'transient_bytes'), 0)
    for split in ('training', 'evaluation'):
        for family in protocol['families']:
            count = 0
            for seed in protocol[split + '_seeds']:
                for identity, pair in paired_inputs(protocol, split, seed, family):
                    histories = dict(pair)
                    left, right = histories[0], histories[1]
                    if sorted(r for r, _ in left) != sorted(r for r, _ in right) or sum(g for _, g in left) != sum(g for _, g in right):
                        raise ValueError('Paired nuisance mismatch')
                    for arm in ARMS:
                        a, costs_a = encode(left, arm)
                        b, costs_b = encode(right, arm)
                        if (a != b) != (arm in protocol['expected_distinct'][family]):
                            raise ValueError('Unexpected readout information boundary')
                        for costs in (costs_a, costs_b):
                            for name, value in costs.items():
                                peaks[name] = max(peaks[name], value)
                        if max(channel for channel, _ in (*a, *b)) > protocol['budgets']['max_channel_id']:
                            raise ValueError('Readout vocabulary exceeded')
                    digest.update(json.dumps([identity, pair], separators=(',', ':')).encode() + b'\n')
                    count += 1
            families[f'{split}:{family}'] = {'pairs': count, 'passed': True}
    bounds = {'observation_visits': 'max_observation_visits', 'projection_visits': 'max_projection_visits',
              'active_channels': 'max_active_channels', 'wire_bytes': 'max_wire_bytes', 'transient_bytes': 'max_state_bytes_per_arm'}
    if any(peaks[name] > protocol['budgets'][budget] for name, budget in bounds.items()):
        raise ValueError('Input resource budget exceeded')
    return {'schema': 'sara-temporal-boundary-learning-input-audit-v1', 'passed': True,
            'protocol_sha256': PROTOCOL_SHA256, 'input_sha256': digest.hexdigest(),
            'families': families, 'peaks': peaks, 'raw_history_split_disjoint': True,
            'split_disjointness_basis': 'Disjoint route pools; seeds and suffix lengths also disjoint',
            'learning_executed': False, 'real_data_read': False, 'promotion_authorized': False,
            'accounting_limit': 'Input transient state only; full learner, receipt, update and replay accounting required before scoring.'}
