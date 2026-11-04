"""Source-only audit checks; audited modules are never imported."""
import ast
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_source_evidence import analyze, symbol
from sara_engine.utils.project_paths import project_path, processed_data_path


class SourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(processed_data_path('benchmark_fixtures'))
        self.spec = json.loads((self.base / 'route_source_evidence_selection_v1.json').read_bytes())
        self.sources = {n: Path(project_path(n)).read_bytes() for n in self.spec['sources']}
        self.reports = {n: (self.base / n).read_bytes() for n in self.spec['reports']}

    def test_scope_and_no_runtime(self):
        before = set(sys.modules)
        result = analyze(self.spec, self.sources, self.reports)
        self.assertEqual(len(result['anchors']), 10)
        self.assertEqual(set(sys.modules), before)
        self.assertFalse(result['launch_authorized'])
        self.assertFalse(result['snn_advantage_established'])
        self.assertEqual(result['learner_executions'], 0)

    def test_changed_source_refused(self):
        self.sources[next(iter(self.sources))] += b'\n'
        with self.assertRaises(ValueError): analyze(self.spec, self.sources, self.reports)

    def test_changed_report_refused(self):
        self.reports[next(iter(self.reports))] += b'\n'
        with self.assertRaises(ValueError): analyze(self.spec, self.sources, self.reports)

    def test_unregistered_source_refused(self):
        self.sources['extra.py'] = b''
        with self.assertRaises(ValueError): analyze(self.spec, self.sources, self.reports)

    def test_missing_ambiguous_anchor_refused(self):
        for source in ('pass', 'def x(): pass\ndef x(): pass'):
            with self.assertRaises(ValueError): symbol(ast.parse(source), 'x')

    def test_budget_refused(self):
        self.spec['bounds']['input_bytes'] = 1
        with self.assertRaises(ValueError): analyze(self.spec, self.sources, self.reports)
