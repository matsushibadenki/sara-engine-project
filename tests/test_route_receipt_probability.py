"""Pure receipt-contract tests; never update a model."""
from dataclasses import FrozenInstanceError, replace
import inspect
import json
from pathlib import Path
import unittest
from sara_engine.evaluation.route_receipt_probability import open_receipt, resolve, probability
from sara_engine.evaluation.route_receipt_probability_preflight import analyze, fixture, LOCAL, PLAN

BASE = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures'


class ReceiptProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((BASE / 'route_receipt_probability_selection_v1.json').read_bytes())
        self.inputs = {name: (BASE / name).read_bytes() for name in self.spec['inputs']}
        self.rows = json.loads(self.inputs[LOCAL])['local_rows']
        self.plans = json.loads(self.inputs[PLAN])['plans']

    def test_all_records_branches_and_faults(self):
        result = analyze(self.spec, self.inputs)
        self.assertEqual(result['receipt_snapshots'], 33)
        self.assertEqual(result['delivery_branches'], 66)
        self.assertEqual(result['frozen_branch_pairs_equal'], 33)
        self.assertEqual(result['faults_detected'], self.spec['faults'])

    def test_capture_api_excludes_outcome(self):
        self.assertEqual(list(inspect.signature(open_receipt).parameters),
                         ['state', 'receiving_id', 'source_id', 'view'])

    def test_absent_and_allocated_zero_are_distinct(self):
        absent = fixture(self.rows[9], 'candidate', self.plans)
        zero = fixture(self.rows[10], 'candidate', self.plans)
        self.assertIsNone(absent[1].pending.probability)
        self.assertEqual(zero[1].pending.probability.hex(), (0.5).hex())
        a = resolve(absent[1], absent[2], absent[4])[1]
        z = resolve(zero[1], zero[2], zero[4])[1]
        self.assertTrue(a['fallback_used']); self.assertFalse(z['fallback_used'])
        self.assertEqual(a['error_increment'], z['error_increment'])

    def test_changed_current_state_never_replaces_snapshot(self):
        _, pending, event, old, changed = fixture(self.rows[1], 'candidate', self.plans)
        self.assertIsNone(probability(changed, event.receiving_id))
        result = resolve(pending, event, changed)[1]
        self.assertEqual(result['probability_at_receipt'], 0.75)
        self.assertEqual(result['error_increment'], 0.5625)
        self.assertEqual(pending.pending.view, old)

    def test_side_is_revealed_only_at_delivery(self):
        left = fixture(self.rows[1], 'candidate', self.plans)
        right = fixture(self.rows[2], 'candidate', self.plans)
        self.assertEqual(left[1].pending, right[1].pending)
        self.assertEqual(resolve(left[1], left[2], left[4])[1]['error_increment'], 0.5625)
        self.assertEqual(resolve(right[1], right[2], right[4])[1]['error_increment'], 0.0625)

    def test_receipt_clock_across_bucket_boundary(self):
        _, pending, event, _, changed = fixture(self.rows[7], 'shuffled_outcome', self.plans)
        self.assertEqual(event.at, 1024)
        result = resolve(pending, event, changed)[1]
        self.assertEqual(result['error_weight'], 32768.0)
        self.assertEqual(result['error_increment'], 18432.0)

    def test_snapshot_is_immutable(self):
        _, pending, _, _, _ = fixture(self.rows[5], 'candidate', self.plans)
        with self.assertRaises(FrozenInstanceError): pending.pending.probability = 0.5
        with self.assertRaises(FrozenInstanceError): pending.pending.view.allocated = False

    def test_duplicates_refused_on_returned_state(self):
        _, pending, event, _, current = fixture(self.rows[5], 'candidate', self.plans)
        closed, _ = resolve(pending, event, current)
        with self.assertRaises(ValueError): resolve(closed, event, current)

    def test_shuffled_source_is_not_clock(self):
        _, pending, event, _, current = fixture(self.rows[5], 'shuffled_outcome', self.plans)
        self.assertNotEqual(event.receiving_id, event.source_id)
        self.assertEqual(resolve(pending, event, current)[1]['error_increment'], 1.125)

    def test_rejected_early_delivery_leaves_pending_unchanged(self):
        _, pending, event, _, current = fixture(self.rows[5], 'candidate', self.plans)
        before = repr(pending)
        with self.assertRaises(ValueError): resolve(pending, replace(event, at=event.receiving_id), current)
        self.assertEqual(repr(pending), before)

    def test_input_binding_refused(self):
        changed = dict(self.inputs); changed[LOCAL] += b' '
        with self.assertRaises(ValueError): analyze(self.spec, changed)

    def test_no_learning_or_runtime_promotion(self):
        result = analyze(self.spec, self.inputs)
        for key in ('actual_learning_updates', 'learner_executions', 'task_scores', 'acquisitions', 'resource_measurements'):
            self.assertEqual(result[key], 0)
        for key in ('unallocated_learning_implemented', 'runtime_provenance_authenticated', 'runtime_equivalence_established',
                    'snn_advantage_established', 'promotion_authorized', 'conditions_relaxed', 'test_opened'):
            self.assertIs(result[key], False)


if __name__ == '__main__': unittest.main()
