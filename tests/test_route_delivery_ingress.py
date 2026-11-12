"""Exercise a serialized delivery from separate producer processes."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from sara_engine.evaluation.route_delivery_origin import RouteOriginKeys, input_tag, outcome_tag
from sara_engine.evaluation.route_durable_commit import RouteReceiptStore
from sara_engine.utils.project_paths import ensure_output_directory, project_path, workspace_path
from test_route_durable_commit import TEST_KEYS, staged_receipt


PRODUCER = """
import hmac
import json
import os
import sys

operation = sys.argv[1]
data = json.load(sys.stdin)
domain = b'sara-route-input-v1\\0' if operation == 'stage' else b'sara-route-outcome-v1\\0'
payload = data[0] if operation == 'stage' else data
message = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
tag = hmac.digest(bytes.fromhex(os.environ['PRODUCER_KEY']), domain + message, 'sha256')
print(json.dumps({'operation': operation, 'payload': payload, 'tag': tag.hex()}))
"""


class RouteDeliveryIngressTests(unittest.TestCase):
    def test_separate_producers_and_ingress_after_rotation(self):
        with TemporaryDirectory(dir=ensure_output_directory(workspace_path('route_delivery_ingress'))) as directory:
            database = Path(directory) / 'stream.sqlite'
            pair, first = staged_receipt()
            store = RouteReceiptStore(database, origin_keys=TEST_KEYS)
            store.stage(pair, first.capture, input_tag=input_tag(
                TEST_KEYS.input_key, pair.input, pair.predecessor.revision, pair.predecessor.digest))
            self.assertTrue(store.apply(first, outcome_tag=outcome_tag(TEST_KEYS.outcome_key, first)))
            keys = RouteOriginKeys(b'u' * 32, b'v' * 32)
            store = store.rotate_origin_keys(keys)
            checkpoint = store.read_committed()
            packet = replace(first.input, receiving_id=1, source_id=80)

            def produce(operation, data, key):
                child = subprocess.run([sys.executable, '-c', PRODUCER, operation],
                                       input=json.dumps(data), text=True, capture_output=True,
                                       env={'PRODUCER_KEY': key.hex()},
                                       check=True, timeout=15)
                return json.loads(child.stdout)

            def send(message, *, succeeds=True):
                child = subprocess.run(
                    [sys.executable, project_path('scripts/eval/route_delivery_ingress.py'), str(database)],
                    input=json.dumps(message), text=True, capture_output=True, timeout=15,
                    env=dict(os.environ, PYTHONPATH=project_path('src'),
                             SARA_ROUTE_INPUT_KEY_HEX=keys.input_key.hex(),
                             SARA_ROUTE_OUTCOME_KEY_HEX=keys.outcome_key.hex()),
                )
                self.assertEqual(child.returncode, 0 if succeeds else 1, child.stderr)
                return json.loads(child.stdout) if succeeds else None

            data = [asdict(packet), checkpoint.revision, checkpoint.digest]
            send(produce('stage', data, TEST_KEYS.input_key), succeeds=False)
            ack = send(produce('stage', data, keys.input_key))['ack']
            self.assertEqual((ack['source_id'], ack['pre_revision']), (80, checkpoint.revision))
            delivery = dict(input=asdict(packet), capture=ack, tick=2, outcome=[1, 0, 1])
            message = produce('apply', delivery, keys.outcome_key)
            tampered = json.loads(json.dumps(message))
            tampered['payload']['input']['source_id'] += 1
            send(tampered, succeeds=False)
            self.assertEqual(store.read_committed(), checkpoint)
            self.assertEqual(send(message), {'committed': True})
            self.assertEqual(send(message), {'committed': False})
            self.assertEqual(store.read_committed().revision, checkpoint.revision + 4)
            self.assertEqual(store.read()[0]['updates'], 18)
