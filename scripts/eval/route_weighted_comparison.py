"""Match weighted model receipts against a scalar replay through tick 69."""
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

SELECTION_SHA = 'b91f2a757763300e5bcc013aa95165ef8fb0ae9e606f1f5a81037208246607c1'
LIMIT = 262144


def artifact(kind):
    return Path(processed_data_path('benchmark_fixtures', f'route_weighted_comparison_{kind}_v1.json'))


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
    label = 'weighted-comparison:' + SELECTION_SHA[:16]
    return tuple(Input(label, 'primary', tick, tick, tuple(events))
                 for tick, events in enumerate([spec['events']] * spec['receipts']))


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

        credits = []
        for group in groups:
            for offset in range(3):
                route = group[(3 * tick + offset) % len(group)]
                key = tuple(events[channel] for channel in routes[route][0])
                mass = counts.get((route, key))
                probability = None if mass is None else (mass[1] + scale) / (sum(mass) + 2 * scale)
                credits.append((route, key, mass is not None, probability))
        yield {'fields': sorted(fields), 'allocations': flags, 'receipt_credits': credits}

        if tick < spec['deliveries']:
            for route, key, _, probability in credits:
                outcome = spec['outcome'][routes[route][1]]
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
                actual = {name: projection[name] for name in expected}
                require(canonical(actual) == canonical(expected), f'Scalar mismatch at tick {tick}')
                transcript.update(canonical(actual) + b'\n')
                if tick in spec['expected']['checkpoints']:
                    checkpoints[str(tick)] = actual
                require(model.updates == tick * 9, 'Actual update counter differs')
                if packet.receiving_id < spec['deliveries']:
                    delivery = bridge.release(ticket, packet.receiving_id + 1, tuple(spec['outcome']))
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
        if tick in spec['expected']['checkpoints']:
            checkpoints[str(tick)] = projection
    require(canonical(result['checkpoints']) == canonical(checkpoints)
            and result['transcript_sha256'] == transcript.hexdigest(), 'Saved scalar evidence differs')
    allocated = sum(flag for _, _, flag in checkpoints[str(spec['receipts'] - 1)]['allocations'])
    require(allocated == spec['expected']['allocated_keys'], 'Allocated key count differs')
    return {'matched_receipts': spec['receipts'], 'actual_updates': result['updates'],
            'allocated_keys': allocated}


def main(mode):
    require(digest(artifact('selection')) == SELECTION_SHA, 'Selection differs')
    spec = read('selection')
    verify_sources(spec)
    registration = {'selection_sha256': SELECTION_SHA, 'python': platform.python_version(),
                    'runner_sha256': digest(project_path('scripts/eval/route_weighted_comparison.py')),
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
