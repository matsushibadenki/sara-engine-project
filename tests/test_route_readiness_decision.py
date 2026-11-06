"""Finite decision validation using only frozen saved reports."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_readiness_decision import analyze, canonical
from sara_engine.utils.project_paths import processed_data_path


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        base = Path(processed_data_path('benchmark_fixtures'))
        self.spec = json.loads((base/'route_readiness_decision_selection_v1.json').read_bytes())
        self.reports = {n:(base/n).read_bytes() for n in self.spec['reports']}

    def test_decision_is_finite_and_does_not_import_runtime(self):
        before = set(sys.modules)
        result = analyze(self.spec, self.reports)
        self.assertEqual(set(sys.modules), before)
        self.assertEqual(len(result['evidence_matrix']), 7)
        self.assertTrue(all(len(row['evidence']) == 3 for row in result['evidence_matrix']))
        self.assertFalse(result['launch_authorized'])
        self.assertEqual(result['new_hand_fixtures'], 0)
        self.assertEqual(result['minimum_next_boundary']['mode'], 'prospective_protocol_only_not_executed')

    def test_repeat_is_readonly(self):
        before = canonical(self.spec), dict(self.reports)
        self.assertEqual(analyze(self.spec, self.reports), analyze(self.spec, self.reports))
        self.assertEqual(before, (canonical(self.spec), self.reports))

    def test_report_hash_and_roster_refused(self):
        changed = dict(self.reports); changed[next(iter(changed))] += b'\n'
        with self.assertRaises(ValueError): analyze(self.spec, changed)
        with self.assertRaises(ValueError): analyze(self.spec, {})

    def test_budget_refused(self):
        self.spec['bounds']['input_bytes'] = 1
        with self.assertRaises(ValueError): analyze(self.spec, self.reports)

    def test_promoted_report_conflicts_even_with_rebound_test_hash(self):
        # Unit-only negative input, not new scientific evidence or an artifact.
        name = 'route_adapter_envelope_result_v1.json'
        for field in ('launch_authorized','runtime_provenance_established','snn_advantage_established'):
            with self.subTest(field=field):
                obj = json.loads(self.reports[name]); obj[field] = True
                changed = dict(self.reports); changed[name] = canonical(obj)
                self.spec['reports'][name] = sha256(changed[name]).hexdigest()
                with self.assertRaises(ValueError): analyze(self.spec, changed)

    def test_removed_missing_category_conflicts(self):
        name = 'route_adapter_envelope_result_v1.json'
        obj = json.loads(self.reports[name]); obj['missing_runtime_evidence'].pop()
        self.reports[name] = canonical(obj)
        self.spec['reports'][name] = sha256(self.reports[name]).hexdigest()
        with self.assertRaises(ValueError): analyze(self.spec, self.reports)
