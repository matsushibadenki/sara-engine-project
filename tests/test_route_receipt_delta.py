"""Nonlearning receipt-to-serialized-delta checks on fixed literal views."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from sara_engine.evaluation.route_receipt_delta import validate, persisted
from sara_engine.evaluation.route_receipt_delta_preflight import analyze, fixture, changed_field, LOCAL, NUMERIC, PLAN

BASE = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures'


class ReceiptDeltaTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((BASE / 'route_receipt_delta_selection_v1.json').read_bytes())
        self.inputs = {name: (BASE / name).read_bytes() for name in self.spec['inputs']}
        self.local = json.loads(self.inputs[LOCAL])['local_rows']
        self.manual = json.loads(self.inputs[NUMERIC])['manual_fixture']
        self.plans = json.loads(self.inputs[PLAN])['plans']

    def fixture(self, index=5, branch='changed_view', role='shuffled_outcome'):
        return fixture(self.local, self.manual, self.plans, self.spec['goldens'][index], role, branch)

    def test_all_registered_transactions_and_faults(self):
        result = analyze(self.spec, self.inputs)
        self.assertEqual(result['totals'], self.spec['gates'])
        self.assertEqual(len(result['records']), 66)
        self.assertEqual(result['faults_detected'], self.spec['faults'])

    def test_allocation_targets_are_not_actual_learning_changes(self):
        for index, branch, expected in ((1, 'stable_view', 0), (1, 'changed_view', 1),
                                         (9, 'stable_view', 1), (9, 'changed_view', 0)):
            p, b = self.fixture(index, branch)
            result = validate(p, b)
            self.assertEqual(result['requested_allocation'], 1)
            self.assertEqual(result['serialized_changed_allocation'], expected)
            self.assertEqual(result['requested_learning'], 36)
            self.assertEqual(result['serialized_changed_learning'], 36)
            self.assertFalse(result['actual_runtime_learning_writes_observed'])

    def test_serialized_representation_and_fresh_inputs(self):
        p, b = self.fixture()
        before = repr((p, b))
        copied = persisted(p.after)
        self.assertIsNot(copied, p.after)
        self.assertEqual(copied.encoded(), p.after.encoded())
        validate(p, b)
        self.assertEqual(repr((p, b)), before)

    def test_frozen_witness_is_independent_of_candidate_after(self):
        p, b = self.fixture()
        r, k = b.allocation_target
        j = next(i for i, s in enumerate(b.pending) if s.pending.view.route == r)
        claims = list(p.error_claims); claims[j] = 0.5
        wrong = replace(p, after=changed_field(p.after, (r, 'error_sum', -1), 8.5), error_claims=tuple(claims))
        with patch('sara_engine.evaluation.route_receipt_delta.validate_numeric') as numeric:
            with self.assertRaisesRegex(ValueError, 'frozen receipt error'): validate(wrong, b)
            numeric.assert_not_called()

    def test_delivery_state_changes_do_not_change_error_witness(self):
        p, b = self.fixture(1, 'stable_view')
        q, c = self.fixture(1, 'changed_view')
        left, right = validate(p, b), validate(q, c)
        self.assertEqual(left['frozen_error_witnesses'], right['frozen_error_witnesses'])
        self.assertNotEqual(p.before.encoded(), q.before.encoded())

    def test_absent_and_allocated_zero_remain_distinct(self):
        _, absent = self.fixture(9, 'stable_view')
        _, zero = self.fixture(10, 'stable_view')
        r, k = absent.allocation_target
        a = next(s.pending for s in absent.pending if s.pending.view.route == r)
        z = next(s.pending for s in zero.pending if s.pending.view.route == r)
        self.assertIsNone(a.probability)
        self.assertEqual(z.probability, 0.5)
        self.assertFalse(a.view.allocated); self.assertTrue(z.view.allocated)

    def test_learning_noop_refused_even_with_request(self):
        p, b = self.fixture()
        r, k = b.allocation_target
        wrong = replace(p, after=changed_field(p.after, (r, 'error_sum', -1), 8.0))
        with self.assertRaises(ValueError): validate(wrong, b)

    def test_metadata_changes_cannot_fill_learning_budget(self):
        p, b = self.fixture()
        with self.assertRaises(ValueError): validate(replace(p, changed_learning_claim=37), b)
        with self.assertRaises(ValueError): validate(replace(p, changed_allocation_claim=True), b)

    def test_receipt_clock_at_late_boundary(self):
        p, b = self.fixture(7)
        result = validate(p, b)
        self.assertIn(18432.0, result['frozen_error_witnesses'])
        self.assertEqual(b.deliveries[0].at, 1024)

    def test_requested_address_types_are_strict(self):
        p, b = self.fixture()
        index = next(i for i, a in enumerate(p.learning_targets) if a[2] == 0)
        first = p.learning_targets[index]
        # Key 0 exists in the fixed schedule; bool must not compare equal to it.
        targets = p.learning_targets[:index] + ((first[0], first[1], False),) + p.learning_targets[index + 1:]
        with self.assertRaises(ValueError): validate(replace(p, learning_targets=targets), b)

    def test_input_binding(self):
        bad = dict(self.inputs); bad[LOCAL] += b' '
        with self.assertRaises(ValueError): analyze(self.spec, bad)

    def test_no_model_update_or_promotion(self):
        result = analyze(self.spec, self.inputs)
        for key in ('actual_learning_updates', 'learner_executions', 'task_scores', 'acquisitions', 'resource_measurements'):
            self.assertEqual(result[key], 0)
        for key in ('unallocated_key_learning_implemented', 'runtime_provenance_authenticated', 'runtime_equivalence_established',
                    'snn_advantage_established', 'promotion_authorized', 'test_opened', 'conditions_relaxed'):
            self.assertIs(result[key], False)


if __name__ == '__main__': unittest.main()
