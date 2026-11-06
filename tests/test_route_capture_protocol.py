"""Static feasibility checks without importing audited models/helpers."""
import ast
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_capture_protocol import analyze, canonical, inspect_symbol, symbol
from sara_engine.utils.project_paths import processed_data_path, project_path


class CaptureProtocolTests(unittest.TestCase):
    def setUp(self):
        base = Path(processed_data_path('benchmark_fixtures'))
        self.spec = json.loads((base/'route_capture_protocol_selection_v1.json').read_bytes())
        self.sources = {n:Path(project_path(n)).read_bytes() for n in self.spec['sources']}
        self.reports = {n:(base/n).read_bytes() for n in self.spec['reports']}

    def test_blocked_without_runtime_imports(self):
        before = set(sys.modules)
        result = analyze(self.spec, self.sources, self.reports)
        self.assertEqual(set(sys.modules), before)
        self.assertEqual(len(result['source_evidence']), 8)
        self.assertEqual(len(result['blockers']), 4)
        self.assertFalse(result['runtime_feasible_demonstrated'])
        self.assertFalse(result['launch_authorized'])
        self.assertEqual(result['protocol']['runtime_calls_allowed'], [])
        self.assertEqual(result['live_capture_executions'], 0)

    def test_repeat_is_readonly(self):
        before = canonical(self.spec), dict(self.sources), dict(self.reports)
        self.assertEqual(analyze(self.spec,self.sources,self.reports), analyze(self.spec,self.sources,self.reports))
        self.assertEqual(before, (canonical(self.spec),self.sources,self.reports))

    def test_source_and_report_drift(self):
        for inputs in (self.sources, self.reports):
            name = next(iter(inputs)); raw = inputs[name]; inputs[name] += b'\n'
            with self.assertRaises(ValueError): analyze(self.spec,self.sources,self.reports)
            inputs[name] = raw

    def test_roster_and_budget(self):
        with self.assertRaises(ValueError): analyze(self.spec,{},self.reports)
        self.spec['bounds']['input_bytes'] = 1
        with self.assertRaises(ValueError): analyze(self.spec,self.sources,self.reports)

    def test_no_runtime_permission(self):
        self.spec['protocol']['runtime_calls_allowed'] = ['begin']
        with self.assertRaises(ValueError): analyze(self.spec,self.sources,self.reports)

    def test_named_symbol_is_required(self):
        for source in ('pass','def x(): pass\ndef x(): pass'):
            with self.assertRaises(ValueError): symbol(ast.parse(source),'x')

    def test_known_writer_detected_as_syntax_not_purity(self):
        path = 'src/sara_engine/evaluation/sparse_route_budget.py'
        row = inspect_symbol(path,self.sources[path],'BudgetRouteMemory.begin')
        self.assertIn('self.pending',row['attribute_or_subscript_store_targets'])
        self.assertIn('self.narrow',row['calls_syntactically_present'])
        self.assertFalse(row['purity_proven'])
