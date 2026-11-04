"""Saved-report/static-protocol checks; no learner or numeric fixture imported."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from sara_engine.evaluation.route_runtime_protocol import analyze, validate_plan, CONTROL

BASE = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures'


class RuntimeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((BASE / 'route_runtime_protocol_selection_v1.json').read_bytes())
        self.inputs = {name: (BASE / name).read_bytes() for name in self.spec['inputs']}
        self.control = json.loads(self.inputs[CONTROL])

    def test_six_schemas_and_all_faults(self):
        result = analyze(self.spec, self.inputs)
        self.assertEqual(result['static_checks']['case_descriptors'], 6)
        self.assertEqual(result['static_checks']['stage_bindings'], 42)
        self.assertEqual(result['static_checks']['same_role_pairs'], 3)
        self.assertEqual(result['faults_detected'], self.spec['faults'])

    def test_real_runtime_evidence_remains_missing(self):
        result = analyze(self.spec, self.inputs)
        self.assertEqual(len(result['missing_runtime_evidence']), 7)
        for key in ('runtime_provenance_authenticated', 'atomicity_tested', 'durable_write_evidence_observed', 'independent_runtime_execution_tested'):
            self.assertIs(result['static_checks'][key], False)
        self.assertFalse(result['launch_authorized'])

    def test_cross_role_and_same_engine_refused(self):
        for index in (0, 3):
            plan = deepcopy(self.spec['protocol']); plan['pairs'][0]['right'] = index
            with self.assertRaises(ValueError): validate_plan(self.spec, plan, self.control)

    def test_zero_runtime_and_numeric_execution(self):
        result = analyze(self.spec, self.inputs)
        for key in ('new_learner_executions', 'new_fixture_calculations', 'task_scores', 'acquisitions', 'resource_measurements'):
            self.assertEqual(result[key], 0)
        self.assertFalse(result['test_opened']); self.assertFalse(result['conditions_relaxed'])

    def test_endpoint_scopes_preserved(self):
        result = analyze(self.spec, self.inputs)
        self.assertEqual(result['endpoint_policy_preserved'], self.control['endpoint_policy'])
        self.assertEqual(result['endpoint_policy_preserved']['gain_over_permanent']['fixed_readout'], 'structural_zero')

    def test_input_bytes_and_unexpected_files_refused(self):
        changed = dict(self.inputs); changed[CONTROL] += b' '
        with self.assertRaises(ValueError): analyze(self.spec, changed)
        changed = dict(self.inputs); changed['extra'] = b'{}'
        with self.assertRaises(ValueError): analyze(self.spec, changed)

    def test_protocol_remains_unmodified(self):
        original = deepcopy(self.spec)
        result = analyze(self.spec, self.inputs)
        self.assertEqual(original, self.spec)
        result['protocol']['requirements']['atomic_nine_credit_commit'] = False
        self.assertTrue(self.spec['protocol']['requirements']['atomic_nine_credit_commit'])

    def test_readback_and_atomic_consumption_are_mandatory(self):
        for name in ('durable_readback', 'consume_with_state_revision', 'atomic_nine_credit_commit'):
            plan = deepcopy(self.spec['protocol']); plan['requirements'][name] = False
            with self.assertRaises(ValueError): validate_plan(self.spec, plan, self.control)


if __name__ == '__main__': unittest.main()
