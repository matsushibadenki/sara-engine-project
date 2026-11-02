"""Independent transition tests on fixed snapshots; never run a learner."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest
from sara_engine.evaluation.route_independent_transition import transition
from sara_engine.evaluation.route_independent_transition_preflight import analyze, compare, fixture, independence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/processed/benchmark_fixtures'


class IndependentTransitionTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((BASE / 'route_independent_transition_selection_v1.json').read_bytes())
        self.raw = (BASE / self.spec['input']['file']).read_bytes()
        self.manual = json.loads(self.raw)['manual_fixture']
        self.source = (ROOT / 'src/sara_engine/evaluation/route_independent_transition.py').read_text()

    def test_all_registered_records_and_faults(self):
        result = analyze(self.spec, self.source, self.raw)
        self.assertEqual(result['transitions'], 36)
        self.assertEqual(result['changed_fixture_fields'], 1296)
        self.assertEqual(result['unchanged_fixture_fields'], 14364)
        self.assertEqual(result['faults_detected'], self.spec['faults'])
        self.assertFalse(result['runtime_equivalence_established'])
        self.assertFalse(result['test_opened'])
        self.assertEqual(result['actual_learning_updates'], 0)

    def test_api_has_no_expected_values(self):
        self.assertEqual(list(inspect.signature(transition).parameters),
                         ['owner', 'before', 'pending', 'delivery', 'diagnostic', 'consumed'])

    def test_output_does_not_alias_input(self):
        args, expected = fixture(self.spec, self.manual, self.manual['rows'][0], 'candidate')
        signature = repr(args)
        first = transition(*args); second = transition(*args)
        self.assertEqual(first, second)
        self.assertIsNot(first[0], args[1])
        self.assertIsNot(first[0], second[0])
        self.assertEqual(repr(args), signature)
        compare(first[0], expected, args[1])

    def test_duplicate_delivery_uses_returned_ledger(self):
        args, _ = fixture(self.spec, self.manual, self.manual['rows'][0], 'candidate')
        after, diagnostic, consumed = transition(*args)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            transition(args[0], after, args[2], args[3], diagnostic, consumed)

    def test_shuffled_address_is_not_receipt_clock(self):
        args, _ = fixture(self.spec, self.manual, self.manual['rows'][7], 'shuffled_outcome')
        self.assertNotEqual(args[2][1], args[2][2])
        transition(*args)
        self.assertEqual(args[3][3], args[2][1] + 1)
        self.assertEqual(args[2][1] // 512, args[2][2] // 512)

    def test_roles_are_not_interchangeable(self):
        args, _ = fixture(self.spec, self.manual, self.manual['rows'][0], 'candidate')
        wrong = (args[0][:-1] + ('permanent',),) + args[3][1:]
        with self.assertRaises(ValueError): transition(*args[:3], wrong, *args[4:])

    def test_literal_changed_expected_is_detected(self):
        args, expected = fixture(self.spec, self.manual, self.manual['rows'][0], 'candidate')
        after = transition(*args)[0]
        bad = list(expected); address, value = bad[0]; bad[0] = (address, value + 1.0)
        with self.assertRaises(ValueError): compare(after, tuple(bad), args[1])

    def test_input_bytes_are_bound(self):
        with self.assertRaises(ValueError): analyze(self.spec, self.source, self.raw + b' ')

    def test_no_primary_helper_or_dynamic_import(self):
        independence(self.source)
        with self.assertRaises(ValueError): independence('from sara_engine.evaluation.route_numeric_delta_preflight import fixture\n' + self.source)
        with self.assertRaises(ValueError): independence(self.source + "\neval('1')\n")

    def test_unmodified_manual_rows(self):
        old = deepcopy(self.manual)
        analyze(self.spec, self.source, self.raw)
        self.assertEqual(self.manual, old)


if __name__ == '__main__': unittest.main()
