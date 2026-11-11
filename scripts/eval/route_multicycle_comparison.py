"""Compare scalar and model reactivation, pool replacement and cooldown."""
import argparse
from hashlib import sha256
from itertools import combinations, product
from random import Random
import json
from pathlib import Path
import platform

from scripts.eval.route_first_receipt import digest, require, verify_sources
from sara_engine.evaluation.route_capture_ack import CaptureAckBridge
from sara_engine.evaluation.route_capture_candidate import CapturePair, PhaseCapture, _primitive
from sara_engine.evaluation.route_capture_layout import canonical, validate
from sara_engine.evaluation.route_outcome_admission import Input
from sara_engine.evaluation.route_owner_host import OwnerHost
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path

SELECTION_SHA = 'c8d88f53774c43565b6db0a3a1c9588f2d356cb73626c81d8b4a35437ef0e1a8'
LIMIT = 262144


def artifact(kind):
    return Path(processed_data_path('benchmark_fixtures', f'route_multicycle_comparison_{kind}_v1.json'))


def read(kind):
    with artifact(kind).open('rb') as stream:
        raw = stream.read(LIMIT + 1)
    require(len(raw) <= LIMIT, 'Artifact exceeds byte limit')
    return json.loads(raw)


def save(kind, value):
    raw = canonical(value) + b'\n'
    require(len(raw) <= LIMIT, 'Artifact exceeds byte limit')
    with Path(ensure_parent_directory(str(artifact(kind)))).open('xb') as stream:
        stream.write(raw)


def packets(spec):
    label = 'multicycle-comparison:' + SELECTION_SHA[:16]
    return tuple(Input(label, 'primary', tick, tick, tuple(events))
                 for tick, events in enumerate([spec['events']] * spec['receipts']))


def outcome_at(spec, tick):
    if spec['outcome_switch'] <= tick < spec['second_outcome_switch']:
        return spec['switched_outcome']
    return spec['outcome']


def checkpoint(spec, tick, scheduler):
    return (tick in spec['expected']['checkpoints'] or tick in scheduler['starts']
            or any(tick in (commit['tick'], commit['tick'] + spec['expected']['cooldown_ticks'] - 1,
                                  commit['tick'] + spec['expected']['cooldown_ticks'])
                   for commit in scheduler['commits']))


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
    events = spec['events']
    pool = tuple(range(45))
    detector = 0.25
    explore_until = None
    cooldown_until = 180
    starts = []
    commits = []
    prune_tick = spec['expected']['first_pruning_tick']
    for tick in range(spec['receipts']):
        scale = 1 << (tick // 64)
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

        def cost(route):
            _, error, weight = statistics[route]
            return (error / weight if weight else 0.25) + 0.005 * len(routes[route][0]), route

        def read_route(route):
            key = tuple(events[channel] for channel in routes[route][0])
            mass = counts.get((route, key))
            probability = None if mass is None else (mass[1] + scale) / (sum(mass) + 2 * scale)
            return route, key, mass is not None, probability

        if tick == prune_tick or tick == explore_until:
            previous_pool = pool
            pool = tuple(sorted(route for group in groups for route in sorted(group, key=cost)[:3]))
            if tick == explore_until:
                commits.append({'tick': tick, 'added': sorted(set(pool) - set(previous_pool)),
                                'removed': sorted(set(previous_pool) - set(pool))})
                explore_until = None
                cooldown_until = tick + 64
                detector = 0.25
        if tick >= prune_tick and explore_until is None and tick >= cooldown_until and detector > 0.20:
            starts.append(tick)
            explore_until = tick + 180
        if tick >= prune_tick and explore_until is None:
            active = pool
        else:
            active = tuple(group[(3 * tick + offset) % len(group)]
                           for group in groups for offset in range(3))
        credits = [read_route(route) for route in active]
        prediction = []
        for output in range(3):
            eligible = [route for route in pool
                        if routes[route][1] == output and statistics[route][0] >= 16]
            prediction.append(read_route(min(eligible, key=cost))[3] if eligible else None)
        yield {'fields': sorted(fields), 'allocations': flags, 'receipt_credits': credits,
               'scheduler': {'pool': pool, 'detector': detector, 'starts': list(starts), 'commits': list(commits),
                             'explore_until': explore_until, 'cooldown_until': cooldown_until,
                             'pruned': tick >= prune_tick},
               'prediction': prediction}

        if tick < spec['deliveries']:
            loss = sum(((0.5 if p is None else p) - y) ** 2
                       for p, y in zip(prediction, outcome_at(spec, tick))) / 3
            detector += (loss - detector) / 16
            for route, key, _, probability in credits:
                outcome = outcome_at(spec, tick)[routes[route][1]]
                mass = list(counts.get((route, key), (0, 0)))
                mass[outcome] += scale
                counts[(route, key)] = tuple(mass)
                observations, error, weight = statistics[route]
                estimate = 0.5 if probability is None else probability
                statistics[route] = observations + 1, error + scale * (estimate - outcome) ** 2, weight + float(scale)


def capture(spec):
    from sara_engine.evaluation.route_memory_factorial_v4 import FactorialMemory

    inputs = packets(spec)
    host, owner = OwnerHost.create(inputs[0].owner, inputs[0].epoch)
    model = FactorialMemory(**spec['model'])
    checkpoints = {}
    transcript = sha256()
    delivered = 0

    def copy_phase(ticket):
        view = host.revalidate(owner, ticket)
        copies = []
        for _ in range(2):
            state = _primitive(vars(model))
            copies.append(PhaseCapture(view.owner_label, view.epoch_label, view.revision,
                                       state, sha256(canonical(state)).hexdigest()))
        require(copies[0] == copies[1] and host.revalidate(owner, ticket) == view,
                'Readback disagreement')
        return copies[0], [item.digest for item in copies]

    try:
        for packet, expected in zip(inputs, replay(spec), strict=True):
            tick = packet.receiving_id
            revision = 4 * tick
            bridge = CaptureAckBridge(host, owner, packet)
            with host.paused(owner, revision) as ticket:
                pre, _ = copy_phase(ticket)
                bridge.seal_predecessor(ticket, pre)
            with host.mutation(owner, revision):
                model.begin(packet.events, packet.receiving_id)
            with host.paused(owner, revision + 2) as ticket:
                post, _ = copy_phase(ticket)
                pair = CapturePair(packet, pre, post)
                bridge.acknowledge(ticket, pair)
                projection = validate(pair)
                actual = {name: projection[name] for name in ('fields', 'allocations', 'receipt_credits')}
                actual['scheduler'] = {'pool': model.pool, 'detector': model.detector,
                                       'starts': list(model.starts), 'commits': list(model.commits),
                                       'explore_until': model.explore_until, 'cooldown_until': model.cooldown_until,
                                       'pruned': model.pruned}
                actual['prediction'] = model.pending[4]
                require(canonical(actual) == canonical(expected), f'Scalar mismatch at tick {tick}')
                transcript.update(canonical(actual) + b'\n')
                if checkpoint(spec, tick, actual['scheduler']):
                    checkpoints[str(tick)] = actual
                require(model.updates == tick * 9, 'Actual update counter differs')
                if packet.receiving_id < spec['deliveries']:
                    delivery = bridge.release(ticket, packet.receiving_id + 1, tuple(outcome_at(spec, tick)))
                    delivered += 1
                host.revalidate(owner, ticket)
            if packet.receiving_id < spec['deliveries']:
                with host.mutation(owner, revision + 2):
                    model.observe(delivery.outcome, delivery.tick)
    finally:
        view = host.inspect(owner)
        if view.phase == 'stable':
            host.close(owner, view.revision)
    return {'registration_sha256': digest(artifact('registration')),
            'checkpoints': checkpoints, 'transcript_sha256': transcript.hexdigest(),
            'updates': model.updates, 'deliveries': delivered, 'final_host_phase': host.inspect(owner).phase}


def verify_result(spec, result):
    require(result['registration_sha256'] == digest(artifact('registration'))
            and result['updates'] == spec['expected']['updates']
            and result['deliveries'] == spec['deliveries']
            and result['final_host_phase'] == 'closed', 'Run metadata differs')
    checkpoints = {}
    transcript = sha256()
    for tick, projection in enumerate(replay(spec)):
        transcript.update(canonical(projection) + b'\n')
        if checkpoint(spec, tick, projection['scheduler']):
            checkpoints[str(tick)] = projection
    require(canonical(result['checkpoints']) == canonical(checkpoints)
            and result['transcript_sha256'] == transcript.hexdigest(), 'Saved scalar evidence differs')
    allocated = sum(flag for _, _, flag in checkpoints[str(spec['receipts'] - 1)]['allocations'])
    require(allocated == spec['expected']['allocated_keys'], 'Allocated key count differs')
    final = checkpoints[str(spec['receipts'] - 1)]['scheduler']
    require(len(final['commits']) >= spec['expected']['minimum_commits'], 'No reactivation commit')
    for start, commit in zip(final['starts'], final['commits']):
        require(commit['tick'] == start + spec['expected']['explore_ticks']
                and str(commit['tick'] + spec['expected']['cooldown_ticks']) in checkpoints,
                'Exploration or cooldown boundary missing')
    expiry = final['commits'][0]['tick'] + spec['expected']['cooldown_ticks']
    before_expiry = checkpoints[str(expiry - 1)]['scheduler']
    require(before_expiry['detector'] > 0.20 and before_expiry['explore_until'] is None
            and before_expiry['cooldown_until'] == expiry and final['starts'][1] == expiry,
            'Cooldown inhibition or release not observed')
    return {'starts': final['starts'], 'commits': final['commits'], 'matched_receipts': spec['receipts'], 'actual_updates': result['updates'],
            'allocated_keys': allocated}


def main(mode):
    require(digest(artifact('selection')) == SELECTION_SHA, 'Selection differs')
    spec = read('selection')
    verify_sources(spec)
    registration = {'selection_sha256': SELECTION_SHA, 'python': platform.python_version(),
                    'runner_sha256': digest(project_path('scripts/eval/route_multicycle_comparison.py')),
                    'paths_sha256': digest(project_path('src/sara_engine/utils/project_paths.py'))}
    if mode == 'register':
        save('registration', registration)
        return
    require(read('registration') == registration, 'Registration differs')
    if mode == 'verify':
        require(artifact('execution').exists() and not artifact('failure').exists(), 'Successful attempt required')
        print(json.dumps(verify_result(spec, read('result'))))
        return
    save('execution', {'registration_sha256': digest(artifact('registration'))})
    try:
        result = capture(spec)
        summary = verify_result(spec, json.loads(canonical(result)))
        save('result', result)
    except BaseException as error:
        save('failure', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'verify'))
    main(parser.parse_args().mode)
