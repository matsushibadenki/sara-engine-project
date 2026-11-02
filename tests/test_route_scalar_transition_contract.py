"""No learner or transition execution in contract tests."""
import json
from pathlib import Path
import unittest
from sara_engine.evaluation.route_scalar_transition_contract import analyze, canonical, faults

BASE = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures'


class TransitionContractTests(unittest.TestCase):
    def setUp(self):
        self.raw = (BASE / 'route_scalar_transition_contract_selection_v1.json').read_bytes()
        self.inputs = {name: (BASE / name).read_bytes() for name in json.loads(self.raw)['inputs']}

    def test_copies_all_literal_expectations(self):
        result = analyze(self.raw, self.inputs)
        original = json.loads(self.inputs['route_numeric_delta_preflight_selection_v1.json'])['manual_fixture']
        self.assertEqual(canonical(result['literal_rows']), canonical(original['rows']))
        self.assertEqual(canonical(result['literal_before']), canonical(original['before']))
        self.assertEqual(len(result['literal_rows']), 12)

    def test_scope_has_no_runtime_promotion(self):
        result = analyze(self.raw, self.inputs)
        for key in ('new_transition_executions', 'learner_executions', 'task_scores', 'acquisitions', 'resource_measurements'):
            self.assertEqual(result[key], 0)
        for key in ('runtime_equivalence_established', 'promotion_authorized', 'test_opened', 'conditions_relaxed'):
            self.assertIs(result[key], False)
        self.assertEqual(result['evidence']['independent_transition_implementation'], 'missing')

    def test_required_boundaries(self):
        requirements = analyze(self.raw, self.inputs)['requirements']
        self.assertEqual(set(requirements), {'arithmetic', 'writes', 'identity', 'delay', 'phases', 'comparison', 'types', 'scope'})
        self.assertIn('36', requirements['writes'])
        self.assertIn('399', requirements['writes'])
        self.assertIn('same-role', requirements['comparison'])

    def test_mutation_faults_and_input_preservation(self):
        before = dict(self.inputs)
        self.assertEqual(len(faults(self.raw, self.inputs)), 5)
        self.assertEqual(self.inputs, before)

    def test_input_roster_rejected(self):
        with self.assertRaises(ValueError): analyze(self.raw, {})

    def test_deterministic_fresh_output(self):
        first = analyze(self.raw, self.inputs)
        second = analyze(self.raw, self.inputs)
        self.assertEqual(canonical(first), canonical(second))
        first['literal_rows'][0][3] = 100
        self.assertEqual(second['literal_rows'][0][3], 3)


if __name__ == '__main__': unittest.main()
