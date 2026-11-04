"""Static contract drift tests; no model, calculator or writer import."""
import json
from pathlib import Path
import sys
import unittest
from copy import deepcopy
from sara_engine.evaluation.route_adapter_spec import analyze, canonical, mutations, validate
from sara_engine.utils.project_paths import processed_data_path


class AdapterSpecTests(unittest.TestCase):
    def setUp(self):
        base = Path(processed_data_path('benchmark_fixtures'))
        self.spec = json.loads((base / 'route_adapter_spec_selection_v1.json').read_bytes())
        self.contract = self.spec['contract']
        self.reports = {n: (base / n).read_bytes() for n in self.spec['reports']}

    def test_baseline_and_no_runtime_import(self):
        before = set(sys.modules)
        result = analyze(self.spec, self.reports)
        self.assertEqual(set(sys.modules), before)
        self.assertEqual(result['same_role_pairs'], 3)
        self.assertFalse(result['launch_authorized'])
        self.assertEqual(result['learner_executions'], 0)

    def test_every_registered_drift_rejected_without_mutation(self):
        before = canonical(self.contract)
        names = []
        for name, changed in mutations(self.contract):
            names.append(name)
            with self.subTest(name=name), self.assertRaises(ValueError): validate(changed, self.contract)
        self.assertEqual(names, self.spec['faults'])
        self.assertEqual(before, canonical(self.contract))

    def test_numeric_false_is_not_boolean(self):
        changed = deepcopy(self.contract); changed['gates']['live_launch'] = 0
        with self.assertRaises(ValueError): validate(changed, self.contract)

    def test_stage_order_is_bound(self):
        changed = deepcopy(self.contract); changed['stages'].reverse()
        with self.assertRaises(ValueError): validate(changed, self.contract)

    def test_report_drift_refused(self):
        self.reports[next(iter(self.reports))] += b'\n'
        with self.assertRaises(ValueError): analyze(self.spec, self.reports)

    def test_report_roster_refused(self):
        with self.assertRaises(ValueError): analyze(self.spec, {})

    def test_input_bound_refused(self):
        self.spec['bounds']['input_bytes'] = 1
        with self.assertRaises(ValueError): analyze(self.spec, self.reports)
