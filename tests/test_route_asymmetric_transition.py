"""Fixed allocated asymmetric snapshot tests; no scientific learner runs."""
import inspect
import json
from pathlib import Path
import unittest
import sys
from unittest.mock import patch
from sara_engine.evaluation.route_asymmetric_transition import analyze, case, compare, mutants, source_guard
from sara_engine.evaluation.route_independent_transition import transition

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/processed/benchmark_fixtures'


class AsymmetricTransitionTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((BASE / 'route_asymmetric_transition_selection_v1.json').read_bytes())
        self.inputs = {name: (BASE / name).read_bytes() for name in self.spec['inputs']}
        self.source = (ROOT / self.spec['kernel']['file']).read_text()
        self.local = json.loads(self.inputs['route_error_binding_preflight_selection_v1.json'])['local_rows']
        self.manual = json.loads(self.inputs['route_numeric_delta_preflight_selection_v1.json'])['manual_fixture']
        self.plans = json.loads(self.inputs['route_independent_transition_selection_v1.json'])['plans']

    def test_fixed_baselines_and_all_mutants(self):
        result = analyze(self.spec, self.source, self.inputs)
        self.assertEqual(result['baseline_transitions'], 30)
        self.assertEqual(result['changed_fixture_fields'], 1080)
        self.assertEqual(result['unchanged_fixture_fields'], 11970)
        constant = result['mutants'][0]
        self.assertEqual(len(constant['rejected']), 24)
        self.assertEqual(len(constant['matched']), 6)
        self.assertEqual(result['faults_detected'], self.spec['faults'])
        self.assertTrue(all(m['rejected'] for m in result['mutants']))

    def test_left_and_right_asymmetric_goldens(self):
        for golden in self.spec['goldens'][1:5]:
            args, expected, _ = case(self.spec, self.local, self.manual, self.plans, golden, 'candidate')
            compare(transition(*args)[0], expected, args[1])

    def test_public_discount_and_weight_at_late_tick(self):
        for golden in self.spec['goldens'][7:9]:
            args, expected, target = case(self.spec, self.local, self.manual, self.plans, golden, 'shuffled_outcome')
            after = transition(*args)[0]
            compare(after, expected, args[1])
            self.assertEqual(dict(after)[(target[0], 'weight_sum', -1)], 32784.0)
            self.assertNotEqual(args[2][1], args[2][2])

    def test_allocated_zero_kept_separate_from_absent(self):
        args, expected, _ = case(self.spec, self.local, self.manual, self.plans, self.spec['goldens'][-1], 'candidate')
        compare(transition(*args)[0], expected, args[1])
        with self.assertRaisesRegex(ValueError, 'Unallocated'):
            case(self.spec, self.local, self.manual, self.plans, [9, 2, 8.25, 17.0], 'candidate')

    def test_kernel_never_receives_goldens(self):
        self.assertEqual(list(inspect.signature(transition).parameters),
                         ['owner', 'before', 'pending', 'delivery', 'diagnostic', 'consumed'])
        self.assertNotIn('golden', inspect.getsource(transition))

    def test_frozen_source_and_input_bindings(self):
        with self.assertRaises(ValueError): source_guard(self.spec, self.source + '\n')
        changed = dict(self.inputs); changed[next(iter(changed))] += b' '
        with self.assertRaises(ValueError): analyze(self.spec, self.source, changed)

    def test_old_calculator_module_rejected(self):
        with patch.dict(sys.modules, {'sara_engine.evaluation.route_numeric_delta_preflight': object()}):
            with self.assertRaises(ValueError): source_guard(self.spec, self.source)

    def test_mutants_are_fresh_copies(self):
        original_source = self.source
        copies = mutants(self.source)
        self.assertEqual(len(copies), 6)
        self.assertTrue(all(fn is not transition for _, fn, _ in copies))
        self.assertEqual(self.source, original_source)

    def test_no_runtime_claims(self):
        result = analyze(self.spec, self.source, self.inputs)
        for key in ('actual_learning_updates', 'learner_executions', 'task_scores', 'acquisitions', 'resource_measurements'):
            self.assertEqual(result[key], 0)
        for key in ('unallocated_transitions_tested', 'runtime_equivalence_established',
                    'runtime_write_provenance_authenticated', 'snn_advantage_established', 'promotion_authorized', 'test_opened'):
            self.assertIs(result[key], False)


if __name__ == '__main__': unittest.main()
