"""Two receipts and one delayed outcome, with saved-state verification."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform

from scripts.eval.route_first_receipt import digest, require, tuple_tree, verify_sources
from sara_engine.evaluation.route_capture_ack import CaptureAckBridge
from sara_engine.evaluation.route_capture_candidate import CapturePair, PhaseCapture, _primitive
from sara_engine.evaluation.route_capture_layout import canonical, decode, validate, ROUTES
from sara_engine.evaluation.route_outcome_admission import CaptureAck, Input
from sara_engine.evaluation.route_owner_host import OwnerHost
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path

SELECTION_SHA = '0f82102d59f942081ba3d9f784cbfffdb35aebfa26696f32aff7d47ec07cd463'
LIMIT = 262144


def artifact(kind):
    return Path(processed_data_path('benchmark_fixtures', f'route_second_receipt_{kind}_v1.json'))


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
    label = 'second-receipt:' + SELECTION_SHA[:16]
    return tuple(Input(label, 'primary', tick, tick, tuple(events))
                 for tick, events in enumerate((spec['input']['events'], spec['second_events'])))


def capture(spec):
    from sara_engine.evaluation.route_memory_factorial_v4 import FactorialMemory

    inputs = packets(spec)
    host, owner = OwnerHost.create(inputs[0].owner, inputs[0].epoch)
    model = FactorialMemory(**spec['model'])
    records = []
    delivery = None

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
        for packet, revision in zip(inputs, (0, 4)):
            bridge = CaptureAckBridge(host, owner, packet)
            with host.paused(owner, revision) as ticket:
                pre, pre_hashes = copy_phase(ticket)
                bridge.seal_predecessor(ticket, pre)
            with host.mutation(owner, revision):
                model.begin(packet.events, packet.receiving_id)
            with host.paused(owner, revision + 2) as ticket:
                post, post_hashes = copy_phase(ticket)
                pair = CapturePair(packet, pre, post)
                ack = bridge.acknowledge(ticket, pair)
                if packet.receiving_id == 0:
                    delivery = bridge.release(ticket, 1, tuple(spec['outcome']))
                host.revalidate(owner, ticket)
            records.append({'pair': asdict(pair), 'ack': asdict(ack),
                            'readback_digests': [pre_hashes, post_hashes]})
            if packet.receiving_id == 0:
                with host.mutation(owner, revision + 2):
                    model.observe(delivery.outcome, delivery.tick)
    finally:
        view = host.inspect(owner)
        if view.phase == 'stable':
            host.close(owner, view.revision)
    return {'registration_sha256': digest(artifact('registration')), 'records': records,
            'delivery': asdict(delivery), 'final_host_phase': host.inspect(owner).phase}


def verify_result(spec, result):
    require(result['registration_sha256'] == digest(artifact('registration'))
            and result['final_host_phase'] == 'closed' and len(result['records']) == 2,
            'Result provenance or lifecycle differs')
    projections = []
    states = []
    for packet, revision, record in zip(packets(spec), (0, 4), result['records']):
        saved = record['pair']
        require(canonical(saved['input']) == canonical(asdict(packet)), 'Input differs')
        pair = CapturePair(packet, *(PhaseCapture(**dict(saved[phase], state=tuple_tree(saved[phase]['state'])))
                                     for phase in ('predecessor', 'receipt')))
        projection = validate(pair)
        projections.append(projection)
        states.append(decode(pair.predecessor.state))
        require(pair.predecessor.revision == revision and pair.receipt.revision == revision + 2
                and record['readback_digests'] == [[pair.predecessor.digest] * 2, [pair.receipt.digest] * 2],
                'Phase revision or readbacks differ')
        ack = CaptureAck(packet.owner, packet.epoch, packet.receiving_id, packet.source_id,
                         revision, revision + 2, pair.predecessor.digest, pair.receipt.digest,
                         tuple(credit[0] for credit in projection['receipt_credits']))
        require(canonical(record['ack']) == canonical(asdict(ack)), 'Ack differs')

    first, second = projections
    initial = dict(first['fields'])
    require(all(value == 0 for value in initial.values())
            and not any(allocated for _, _, allocated in first['allocations']), 'Expected cold predecessor')
    require(canonical(result['delivery']) == canonical({
        'input': asdict(packets(spec)[0]), 'capture': result['records'][0]['ack'],
        'tick': 1, 'outcome': spec['outcome'],
    }), 'Delivery differs')
    expected = dict(initial)
    allocated_keys = set()
    for route, key, allocated, probability in first['receipt_credits']:
        require(not allocated and probability is None, 'First receipt differs')
        rank = int(''.join(map(str, key)), 2)
        side = spec['outcome'][ROUTES[route][1]]
        allocated_keys.add((route, rank))
        expected[(route, 'mass_right' if side else 'mass_left', rank)] = 1
        expected[(route, 'observations', -1)] = 1
        expected[(route, 'error_sum', -1)] = 0.25
        expected[(route, 'weight_sum', -1)] = 1.0
    require(canonical(second['fields']) == canonical(tuple(sorted(expected.items()))), 'Actual update differs')
    require({(r, k) for r, k, allocated in second['allocations'] if allocated} == allocated_keys,
            'Allocated keys differ')
    changed = sum(value != initial[address] for address, value in second['fields'])
    require(changed == spec['expect']['changed_fields'] and len(allocated_keys) == spec['expect']['allocations']
            and states[0]['updates'] == 0 and states[1]['updates'] == spec['expect']['updated_routes'],
            'Actual update count differs')
    for route, key, allocated, probability in second['receipt_credits']:
        rank = int(''.join(map(str, key)), 2)
        present = (route, rank) in allocated_keys
        expected_probability = (1 + spec['outcome'][ROUTES[route][1]]) / 3 if present else None
        require(allocated == present and probability == expected_probability, 'Second receipt probability differs')
    return {'changed_learning_fields': changed, 'allocated_keys': len(allocated_keys),
            'populated_second_credits': sum(credit[2] for credit in second['receipt_credits'])}


def main(mode):
    require(digest(artifact('selection')) == SELECTION_SHA, 'Selection differs')
    spec = read('selection')
    verify_sources(spec)
    registration = {'selection_sha256': SELECTION_SHA, 'python': platform.python_version(),
                    'runner_sha256': digest(project_path('scripts/eval/route_second_receipt.py')),
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
