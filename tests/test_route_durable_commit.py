"""Verify SQLite recovery using the saved cold receipt and real model updates."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from sara_engine.evaluation.route_capture_candidate import CapturePair, PhaseCapture
from sara_engine.evaluation.route_capture_layout import ROUTES, project
from sara_engine.evaluation.route_durable_commit import RouteReceiptStore, _tuple_tree
from sara_engine.evaluation.route_outcome_admission import AdmissionGate, CaptureAck, Input
from sara_engine.utils.project_paths import ensure_output_directory, processed_data_path, workspace_path


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

    def crash(self, stage):
        code = """
import os
import sys
from unittest.mock import patch
from test_route_durable_commit import staged_receipt
from sara_engine.evaluation.route_durable_commit import RouteReceiptStore

def crash_at(stage):
    if stage == sys.argv[2]:
        os._exit(73)

with patch('sara_engine.evaluation.route_durable_commit._interrupt', crash_at):
    RouteReceiptStore(sys.argv[1]).apply(staged_receipt()[1])
"""
        child = subprocess.run([sys.executable, '-c', code, str(self.path), stage],
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

    def test_wrong_tick_leaves_pending_state(self):
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            self.store.apply(replace(self.delivery, tick=2))
        self.assertEqual(self.store.read(), (self.pre, False))
