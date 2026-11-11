"""Verify SQLite recovery using the saved cold receipt and real model updates."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sara_engine.evaluation.route_capture_candidate import CapturePair, PhaseCapture
from sara_engine.evaluation.route_capture_layout import ROUTES, project
from sara_engine.evaluation.route_delivery_origin import RouteOriginKeys, input_tag, outcome_tag
from sara_engine.evaluation.route_durable_commit import RouteReceiptStore, _tuple_tree
from sara_engine.evaluation.route_outcome_admission import AdmissionGate, CaptureAck, Delivery, Input
from sara_engine.utils.project_paths import ensure_output_directory, processed_data_path, workspace_path


TEST_KEYS = RouteOriginKeys(b'i' * 32, b'o' * 32)


def staged_receipt():
    path = processed_data_path('benchmark_fixtures', 'route_first_receipt_result_v1.json')
    saved = json.loads(Path(path).read_bytes())
    pair = saved['pair']
    packet = Input(**dict(pair['input'], events=tuple(pair['input']['events'])))
    phases = [PhaseCapture(**dict(pair[name], state=_tuple_tree(pair[name]['state'])))
              for name in ('predecessor', 'receipt')]
    ack = CaptureAck(**dict(saved['ack'], routes=tuple(saved['ack']['routes'])))
    gate, producer, collector = AdmissionGate.create(ack)
    gate.admit(producer, packet)
    gate.acknowledge(collector, ack)
    delivery = gate.release(producer, 1, (1, 0, 1))
    return CapturePair(packet, *phases), delivery


class DurableCommitTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=ensure_output_directory(workspace_path('route_durable_commit')))
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'receipt.sqlite'
        self.store = RouteReceiptStore(self.path)
        pair, self.delivery = staged_receipt()
        self.store.stage(pair, self.delivery.capture)
        self.pre, consumed = self.store.read()
        self.assertFalse(consumed)

    def assert_committed(self):
        after, consumed = RouteReceiptStore(self.path).read()
        self.assertTrue(consumed)
        self.assertEqual(after['updates'], 9)
        self.assertEqual(after['steps'], 1)
        self.assertIsNone(after['pending'])
        initial = dict(project(self.pre).fields)
        expected = dict(initial)
        allocated = set()
        for route in self.delivery.capture.routes:
            channels, target = ROUTES[route]
            rank = int(''.join(str(self.delivery.input.events[i]) for i in channels), 2)
            side = 'mass_right' if self.delivery.outcome[target] else 'mass_left'
            expected[(route, side, rank)] = 1
            expected[(route, 'observations', -1)] = 1
            expected[(route, 'error_sum', -1)] = 0.25
            expected[(route, 'weight_sum', -1)] = 1.0
            allocated.add((route, rank))
        projection = project(after)
        self.assertEqual(dict(projection.fields), expected)
        self.assertEqual(sum(initial[key] != value for key, value in expected.items()), 36)
        self.assertEqual({(route, rank) for route, rank, flag in projection.allocations if flag}, allocated)
        return after

    def crash(self, stage, delivery=None, *, authenticated=False):
        code = """
import os
import sys
from unittest.mock import patch
import json
from sara_engine.evaluation.route_outcome_admission import CaptureAck, Delivery, Input
from sara_engine.evaluation.route_durable_commit import RouteReceiptStore

def crash_at(stage):
    if stage == sys.argv[2]:
        os._exit(73)

with patch('sara_engine.evaluation.route_durable_commit._interrupt', crash_at):
    raw = json.loads(sys.argv[3])
    packet = Input(**dict(raw['input'], events=tuple(raw['input']['events'])))
    ack = CaptureAck(**dict(raw['capture'], routes=tuple(raw['capture']['routes'])))
    delivery = Delivery(packet, ack, raw['tick'], tuple(raw['outcome']))
    if sys.argv[4] == '1':
        from test_route_durable_commit import TEST_KEYS, outcome_tag
        RouteReceiptStore(sys.argv[1], origin_keys=TEST_KEYS).apply(
            delivery, outcome_tag=outcome_tag(TEST_KEYS.outcome_key, delivery))
    else:
        RouteReceiptStore(sys.argv[1]).apply(delivery)
"""
        child = subprocess.run([sys.executable, '-c', code, str(self.path), stage,
                                json.dumps(asdict(delivery or self.delivery)), str(int(authenticated))],
                               env=dict(os.environ, PYTHONPATH='src:tests:.'),
                               capture_output=True, text=True, timeout=15)
        self.assertEqual(child.returncode, 73, child.stderr)

    def test_concurrent_duplicate_and_conflicting_outcome(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(self.store.apply, (self.delivery, self.delivery)))
        self.assertCountEqual(results, [True, False])
        committed = self.assert_committed()
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            self.store.apply(replace(self.delivery, outcome=(0, 0, 1)))
        self.assertEqual(self.store.read(), (committed, True))

    def test_crash_after_state_write(self):
        self.crash('after_state')
        self.assertEqual(RouteReceiptStore(self.path).read(), (self.pre, False))
        self.assertTrue(self.store.apply(self.delivery))
        self.assert_committed()

    def test_crash_after_both_writes_before_commit(self):
        self.crash('before_commit')
        self.assertEqual(RouteReceiptStore(self.path).read(), (self.pre, False))
        self.assertTrue(self.store.apply(self.delivery))
        self.assert_committed()

    def test_crash_after_commit_before_reply(self):
        self.crash('after_commit')
        committed = self.assert_committed()
        self.assertFalse(RouteReceiptStore(self.path).apply(self.delivery))
        self.assertEqual(self.store.read(), (committed, True))

    def next_delivery(self, receiving_id):
        packet = replace(self.delivery.input, receiving_id=receiving_id, source_id=receiving_id)
        ack = self.store.stage_next(packet)
        gate, producer, collector = AdmissionGate.create(ack)
        gate.admit(producer, packet)
        gate.acknowledge(collector, ack)
        return gate.release(producer, receiving_id + 1, self.delivery.outcome)

    def test_sequential_resume_with_old_duplicate_and_future_delivery(self):
        self.assertTrue(self.store.apply(self.delivery))
        first = self.store.read_committed()
        second = self.next_delivery(1)
        self.assertEqual(second.capture.pre_revision, first.revision)
        self.assertEqual(second.capture.pre_digest, first.digest)
        self.assertEqual(second.capture.receipt_revision, first.revision + 2)
        self.assertEqual(self.store.read_committed(), first)
        self.assertFalse(self.store.read()[1])
        self.assertFalse(self.store.apply(self.delivery))
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            self.store.apply(replace(self.delivery, outcome=(0, 0, 1)))
        future = replace(second, input=replace(second.input, receiving_id=2), tick=3)
        with self.assertRaisesRegex(ValueError, 'out of order'):
            self.store.apply(future)
        self.crash('before_commit', second)
        self.store = RouteReceiptStore(self.path)
        self.assertEqual(self.store.read_committed(), first)
        self.assertFalse(self.store.read()[1])
        self.crash('after_commit', second)
        self.store = RouteReceiptStore(self.path)
        committed = self.store.read_committed()
        self.assertEqual(committed.revision, first.revision + 4)
        self.assertFalse(self.store.apply(second))
        self.assertFalse(self.store.apply(self.delivery))
        self.assertEqual(self.store.read()[0]['updates'], 18)
        third = self.next_delivery(2)
        self.assertEqual(third.capture.pre_digest, committed.digest)
        self.assertTrue(self.store.apply(third))
        self.assertEqual(self.store.read()[0]['updates'], 27)

    def test_stage_refuses_revision_gap_and_pending_overwrite(self):
        self.assertTrue(self.store.apply(self.delivery))
        packet = replace(self.delivery.input, receiving_id=1, source_id=1)
        with patch.object(self.store, 'stage') as stage:
            self.store.stage_next(packet)
        pair, ack = stage.call_args.args
        shifted = replace(pair,
                          predecessor=replace(pair.predecessor, revision=pair.predecessor.revision + 2),
                          receipt=replace(pair.receipt, revision=pair.receipt.revision + 2))
        with self.assertRaisesRegex(ValueError, 'does not continue'):
            self.store.stage(shifted, replace(ack, pre_revision=ack.pre_revision + 2,
                                              receipt_revision=ack.receipt_revision + 2))
        self.store.stage(pair, ack)
        with self.assertRaisesRegex(ValueError, 'still pending'):
            self.store.stage(pair, ack)

    def test_legacy_committed_database_continues(self):
        self.assertTrue(self.store.apply(self.delivery))
        committed = self.store.read_committed()
        # The original schema stored the final state in receipt and had no checkpoint table.
        with sqlite3.connect(self.path) as connection:
            connection.execute('DROP TABLE checkpoint')
        self.assertEqual(self.store.read_committed(), committed)
        second = self.next_delivery(1)
        self.assertEqual(self.store.read_committed(), committed)
        self.assertTrue(self.store.apply(second))
        self.assertEqual(self.store.read_committed().revision, committed.revision + 4)

    def test_external_origin_survives_recovery_and_binds_source_revision(self):
        self.path = Path(self.temp.name) / 'authenticated.sqlite'
        self.store = RouteReceiptStore(self.path, origin_keys=TEST_KEYS)
        pair, first = staged_receipt()
        tag = input_tag(TEST_KEYS.input_key, pair.input, pair.predecessor.revision, pair.predecessor.digest)
        with self.assertRaisesRegex(ValueError, 'origin verification'):
            self.store.stage(pair, first.capture)
        self.store.stage(pair, first.capture, input_tag=tag)
        self.assertTrue(self.store.apply(first, outcome_tag=outcome_tag(TEST_KEYS.outcome_key, first)))
        pre = self.store.read_committed()
        packet = replace(first.input, receiving_id=1, source_id=77)
        with self.assertRaisesRegex(ValueError, 'origin verification'):
            self.store.stage_next(packet, input_tag=input_tag(TEST_KEYS.input_key, packet,
                                                              pre.revision + 2, pre.digest))
        ack = self.store.stage_next(packet, input_tag=input_tag(TEST_KEYS.input_key, packet,
                                                               pre.revision, pre.digest))
        second = Delivery(packet, ack, 2, first.outcome)
        proof = outcome_tag(TEST_KEYS.outcome_key, second)
        for changed in (replace(second, input=replace(packet, source_id=78)),
                        replace(second, input=replace(packet, receiving_id=2), tick=3),
                        replace(second, capture=replace(ack, receipt_revision=ack.receipt_revision + 2))):
            with self.assertRaisesRegex(ValueError, 'origin verification'):
                self.store.apply(changed, outcome_tag=proof)
            with self.assertRaisesRegex(ValueError, 'does not bind'):
                self.store.apply(changed, outcome_tag=outcome_tag(TEST_KEYS.outcome_key, changed))
        self.assertEqual(self.store.read_committed(), pre)
        self.crash('before_commit', second, authenticated=True)
        self.store = RouteReceiptStore(self.path, origin_keys=TEST_KEYS)
        self.assertEqual(self.store.read_committed(), pre)
        with self.assertRaisesRegex(ValueError, 'origin policy'):
            RouteReceiptStore(self.path).apply(second)
        with self.assertRaisesRegex(ValueError, 'origin policy'):
            RouteReceiptStore(self.path, origin_keys=RouteOriginKeys(b'x' * 32, b'y' * 32)).apply(second,
                                                                                               outcome_tag=proof)
        with self.assertRaisesRegex(ValueError, 'origin verification'):
            self.store.apply(second, outcome_tag=outcome_tag(TEST_KEYS.input_key, second))
        self.crash('after_commit', second, authenticated=True)
        self.store = RouteReceiptStore(self.path, origin_keys=TEST_KEYS)
        self.assertFalse(self.store.apply(second, outcome_tag=proof))
        with self.assertRaisesRegex(ValueError, 'origin verification'):
            self.store.apply(second)
        self.assertEqual(self.store.read_committed().revision, pre.revision + 4)
        self.assertEqual(self.store.read()[0]['updates'], 18)

    def test_wrong_tick_leaves_pending_state(self):
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            self.store.apply(replace(self.delivery, tick=2))
        self.assertEqual(self.store.read(), (self.pre, False))
