"""Hand-metadata lifecycle tests; no model or storage execution."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from sara_engine.evaluation.route_commit_state_machine import prepare, commit, classify, inspect_readonly, fingerprint
from sara_engine.evaluation.route_commit_state_machine_preflight import analyze, fixture

BASE = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures'


class CommitMetadataTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((BASE / 'route_commit_state_machine_selection_v1.json').read_bytes())
        self.inputs = {name: (BASE / name).read_bytes() for name in self.spec['inputs']}
        self.pre, self.pending, self.event, self.expected = fixture(self.spec, 'shuffled_outcome', 'local_sparse_reference')

    def ticket(self): return prepare(self.pre, self.pending, self.event, self.pending.routes, 10)

    def test_full_roster_masks_and_faults(self):
        result = analyze(self.spec, self.inputs)
        self.assertEqual(result['baseline_metadata_commits'], 6)
        self.assertEqual(result['readonly_checks'], 24)
        self.assertEqual(result['recovery_classifications'], 3126)
        self.assertEqual(result['faults_detected'], self.spec['faults'])

    def test_prepare_is_readonly_and_repeatable(self):
        before = fingerprint(self.pre)
        self.assertEqual(self.ticket(), self.ticket())
        self.assertEqual(fingerprint(self.pre), before)
        self.assertNotIn(64, self.pre.consumed)

    def test_full_successor_matches_literal_and_duplicate_is_refused(self):
        ticket = self.ticket(); after = commit(self.pre, self.pending, ticket, 10)
        self.assertEqual(after, self.expected)
        self.assertEqual(after.route_mask, 511)
        self.assertEqual(after.consumed, frozenset((64,)))
        self.assertNotIn(64, self.pre.consumed)
        with self.assertRaises(ValueError): commit(after, self.pending, ticket, 11)
        with self.assertRaises(ValueError): prepare(after, self.pending, self.event, self.pending.routes, 11)

    def test_same_revision_changed_core_or_metadata_refused(self):
        for field in ('core', 'allocation', 'diagnostic'):
            altered = replace(self.pre, **{field: '9' * 64})
            with self.assertRaises(ValueError): commit(altered, self.pending, self.ticket(), 10)

    def test_all_proper_subsets_rejected(self):
        for mask in range(511):
            self.assertEqual(classify(replace(self.expected, route_mask=mask), self.pending, 11), 'reject_inconsistent')
        self.assertEqual(classify(self.expected, self.pending, 11), 'successor_consumed')

    def test_recovery_rejects_split_state_and_consumption(self):
        state_only = replace(self.expected, consumed=frozenset(), last_receipt=None)
        consumed_only = replace(self.pre, consumed=frozenset((64,)), last_receipt=64, route_mask=511)
        self.assertEqual(classify(state_only, self.pending, 10), 'reject_inconsistent')
        self.assertEqual(classify(consumed_only, self.pending, 10), 'reject_inconsistent')

    def test_watermark_detects_known_rollback(self):
        self.assertEqual(classify(self.pre, self.pending, 10), 'predecessor_unconsumed')
        self.assertEqual(classify(self.pre, self.pending, 11), 'reject_stale')
        with self.assertRaises(ValueError): commit(self.pre, self.pending, self.ticket(), 11)

    def test_inspection_never_calls_commit(self):
        with patch('sara_engine.evaluation.route_commit_state_machine.commit', side_effect=AssertionError('Inspection committed')):
            before = fingerprint(self.expected)
            a = inspect_readonly(self.expected, self.pending, 11)
            b = inspect_readonly(self.expected, self.pending, 11)
            self.assertEqual(a, b)
            self.assertFalse(a['application_authorized'])
            self.assertEqual(fingerprint(self.expected), before)

    def test_ticket_and_state_immutable(self):
        with self.assertRaises(FrozenInstanceError): self.pre.revision = 9
        with self.assertRaises(FrozenInstanceError): self.ticket().proposed_routes = ()

    def test_ticket_contains_delayed_delivery_metadata(self):
        ticket = self.ticket()
        self.assertEqual(ticket.delivery.at, ticket.pending.receiving_id + 1)
        self.assertNotEqual(ticket.delivery.receiving_id, ticket.delivery.source_id)
        with self.assertRaises(ValueError): replace(ticket, delivery=replace(self.event, at=64))

    def test_saved_input_binding(self):
        bad = dict(self.inputs); bad[next(iter(bad))] += b' '
        with self.assertRaises(ValueError): analyze(self.spec, bad)

    def test_no_runtime_atomicity_or_learning_claim(self):
        result = analyze(self.spec, self.inputs)
        for key in ('learner_executions', 'numeric_fixture_executions', 'actual_learning_updates', 'task_scores',
                    'acquisitions', 'resource_measurements', 'storage_crash_tests'):
            self.assertEqual(result[key], 0)
        for key in ('model_atomicity_established', 'durable_exactly_once_established', 'independent_runtime_equivalence_established',
                    'snn_advantage_established', 'promotion_authorized', 'launch_authorized', 'conditions_relaxed', 'test_opened'):
            self.assertIs(result[key], False)


if __name__ == '__main__': unittest.main()
