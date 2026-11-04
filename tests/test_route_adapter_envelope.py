"""Static hand bindings, never a live model/calculator/storage test."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_adapter_envelope import analyze, canonical, mutations, validate
from sara_engine.utils.project_paths import processed_data_path


class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        base = Path(processed_data_path('benchmark_fixtures'))
        self.spec = json.loads((base / 'route_adapter_envelope_selection_v1.json').read_bytes())
        self.reports = {n:(base/n).read_bytes() for n in self.spec['reports']}
        self.case = deepcopy(self.spec['cases'][-1])

    def test_all_cases_and_no_runtime_import(self):
        before = set(sys.modules)
        result = analyze(self.spec, self.reports)
        self.assertEqual(set(sys.modules), before)
        self.assertEqual(result['hand_cases'], 6)
        self.assertEqual(len(result['faults_rejected']), 28)
        self.assertFalse(result['launch_authorized'])
        self.assertEqual(result['calculator_executions'], 0)

    def test_every_fault_without_mutation(self):
        before = canonical(self.case)
        for name, changed in mutations(self.case):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate(changed['envelope'], changed['authority'])
        self.assertEqual(canonical(self.case), before)

    def test_receipt_and_delivery_differ_and_remain_valid(self):
        a = self.case['authority']
        self.assertNotEqual(a['receipt_revision'], a['delivery_revision'])
        credit = self.case['envelope']['scalar']['credits'][0]
        self.assertFalse(credit['receipt_allocated'])
        self.assertTrue(credit['delivery_allocated'])
        self.assertIsNone(credit['receipt_probability'])
        self.assertEqual(credit['delivery_probability'], 0.75)
        self.assertTrue(validate(self.case['envelope'], a)['binding_valid'])

    def test_bool_allocation_not_integer(self):
        self.case['envelope']['scalar']['credits'][0]['receipt_allocated'] = 0
        with self.assertRaises(ValueError): validate(self.case['envelope'], self.case['authority'])

    def test_unknown_record_field_refused(self):
        self.case['envelope']['commit']['retry'] = True
        with self.assertRaises(ValueError): validate(self.case['envelope'], self.case['authority'])

    def test_duplicate_consumption_reference_refused(self):
        self.case['authority']['consumed_before'] = [64]
        with self.assertRaises(ValueError): validate(self.case['envelope'], self.case['authority'])

    def test_authority_is_supplied_not_authenticated(self):
        # A consistently relabeled hand reference can pass; no real provenance claim.
        self.case['authority']['ref'] = 'fabricated-consistent'
        for record in self.case['envelope'].values(): record['authority_ref'] = 'fabricated-consistent'
        result = validate(self.case['envelope'], self.case['authority'])
        self.assertFalse(result['authority_authenticated'])
        self.assertFalse(result['atomicity_established'])

    def test_readonly_repeat(self):
        before = canonical(self.case)
        a = validate(self.case['envelope'], self.case['authority'])
        self.assertEqual(a, validate(self.case['envelope'], self.case['authority']))
        self.assertEqual(canonical(self.case), before)
        self.assertFalse(a['application_authorized'])

    def test_report_drift_and_budget(self):
        changed = dict(self.reports); changed[next(iter(changed))] += b'\n'
        with self.assertRaises(ValueError): analyze(self.spec, changed)
        self.spec['bounds']['input_bytes'] = 1
        with self.assertRaises(ValueError): analyze(self.spec, self.reports)
