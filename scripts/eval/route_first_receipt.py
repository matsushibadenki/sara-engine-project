"""One-shot first-receipt integration smoke with a read-only saved verifier."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.route_capture_ack import CaptureAckBridge
from sara_engine.evaluation.route_capture_candidate import CapturePair, PhaseCapture, _primitive
from sara_engine.evaluation.route_capture_layout import canonical, validate
from sara_engine.evaluation.route_outcome_admission import CaptureAck, Input
from sara_engine.evaluation.route_owner_host import OwnerHost
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path

SELECTION_SHA = 'af3211f8f2d7ed77de1663fcb494b17f9cb041c8b31d142326c1fa1d28a00c6d'
LIMIT = 262144
SCRIPT = 'scripts/eval/route_first_receipt.py'


def artifact(kind):
    return Path(processed_data_path('benchmark_fixtures', f'route_first_receipt_{kind}_v1.json'))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def verify_sources(selection):
    for name, expected in (selection['sources'] | selection['preserved']).items():
        require(digest(project_path(name)) == expected, 'Pinned file differs: ' + name)
    require(not Path(processed_data_path('benchmark_fixtures', 'sparse_route_reactivation_replay_v1.json')).exists(),
            'Historical failed replay status changed')


def packet_for(selection):
    packet = selection['input']
    return Input('first-receipt:' + SELECTION_SHA[:16], 'primary',
                 packet['receiving_id'], packet['source_id'], tuple(packet['events']))


def capture_once(selection):
    # The runner reaches this import only after source checks and exclusive reservation.
    from sara_engine.evaluation.route_memory_factorial_v4 import FactorialMemory

    packet = packet_for(selection)
    host, owner = OwnerHost.create(packet.owner, packet.epoch)
    model = FactorialMemory(**selection['model'])
    bridge = CaptureAckBridge(host, owner, packet)
    readback_digests = []

    def read_phase(ticket):
        view = host.revalidate(owner, ticket)
        copies = []
        for _ in range(2):
            state = _primitive(vars(model))
            raw = canonical(state)
            require(len(raw) <= LIMIT, 'Phase exceeds byte limit')
            copies.append(PhaseCapture(view.owner_label, view.epoch_label, view.revision,
                                       state, sha256(raw).hexdigest()))
        require(copies[0] == copies[1] and view == host.revalidate(owner, ticket),
                'Independent readbacks differ')
        readback_digests.append([copy.digest for copy in copies])
        return copies[0]

    try:
        with host.paused(owner, 0) as ticket:
            predecessor = read_phase(ticket)
            bridge.seal_predecessor(ticket, predecessor)
        with host.mutation(owner, 0):
            model.begin(packet.events, packet.receiving_id)
        with host.paused(owner, 2) as ticket:
            receipt = read_phase(ticket)
            pair = CapturePair(packet, predecessor, receipt)
            ack = bridge.acknowledge(ticket, pair)
            host.revalidate(owner, ticket)
    finally:
        view = host.inspect(owner)
        if view.phase == 'stable':
            host.close(owner, view.revision)

    return {
        'schema': 'route-first-receipt-result-v1',
        'selection_sha256': SELECTION_SHA,
        'registration_sha256': digest(artifact('registration')),
        'preserved_hashes': len(selection['preserved']),
        'pair': asdict(pair),
        'ack': asdict(ack),
        'readback_digests': readback_digests,
        'final_host_phase': host.inspect(owner).phase,
        'calls': dict(selection['budget'], artifact_bytes=LIMIT),
    }


def tuple_tree(value):
    if type(value) is list:
        return tuple(tuple_tree(child) for child in value)
    return value


def verify_result(selection, result):
    require(result['schema'] == 'route-first-receipt-result-v1', 'Result schema differs')
    require(result['selection_sha256'] == SELECTION_SHA
            and result['registration_sha256'] == digest(artifact('registration'))
            and result['preserved_hashes'] == len(selection['preserved']), 'Result provenance differs')
    require(canonical(result['calls']) == canonical(selection['budget']), 'Call budget differs')
    records = result['pair']
    packet = Input(**dict(records['input'], events=tuple(records['input']['events'])))
    require(packet == packet_for(selection), 'Saved input differs')
    pair = CapturePair(packet, *(PhaseCapture(**dict(records[name], state=tuple_tree(records[name]['state'])))
                                 for name in ('predecessor', 'receipt')))
    projection = validate(pair)
    require(all(value == 0 for _, value in projection['fields'])
            and not any(allocated for _, _, allocated in projection['allocations']), 'Nonzero learning state')
    ack = CaptureAck(**dict(result['ack'], routes=tuple(result['ack']['routes'])))
    expected = CaptureAck(packet.owner, packet.epoch, packet.receiving_id, packet.source_id,
                          0, 2, pair.predecessor.digest, pair.receipt.digest,
                          tuple(credit[0] for credit in projection['receipt_credits']))
    require(ack == expected and pair.predecessor.revision == 0 and pair.receipt.revision == 2,
            'Acknowledgement binding differs')
    require(result['readback_digests'] == [[pair.predecessor.digest] * 2, [pair.receipt.digest] * 2]
            and result['final_host_phase'] == 'closed', 'Readback or cleanup evidence differs')


def main(mode):
    require(digest(artifact('selection')) == SELECTION_SHA, 'Selection differs')
    selection = read('selection')
    verify_sources(selection)
    registration = {'selection_sha256': SELECTION_SHA, 'python': platform.python_version(),
                    'runner_sha256': digest(project_path(SCRIPT)),
                    'paths_sha256': digest(project_path('src/sara_engine/utils/project_paths.py'))}
    if mode == 'register':
        save('registration', registration)
        return
    require(read('registration') == registration, 'Registration differs')
    if mode == 'verify':
        require(artifact('execution').exists() and not artifact('failure').exists(), 'Successful attempt required')
        verify_result(selection, read('result'))
        print('Saved first receipt verified')
        return
    save('execution', {'registration_sha256': digest(artifact('registration'))})
    try:
        result = capture_once(selection)
        # Verify through the same JSON representation used by the saved verifier.
        verify_result(selection, json.loads(canonical(result)))
        save('result', result)
    except BaseException as error:
        save('failure', {'type': type(error).__name__, 'message': str(error)})
        raise
    print('First receipt captured and acknowledged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'verify'))
    main(parser.parse_args().mode)
