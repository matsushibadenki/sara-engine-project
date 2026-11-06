"""Validate source-only connection assessment without executing a model/provider."""
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_authority_design import analyze, canonical, inspect_source
from sara_engine.utils.project_paths import processed_data_path, project_path


class AuthorityDesignTests(unittest.TestCase):
    def setUp(self):
        base = Path(processed_data_path('benchmark_fixtures'))
        self.spec = json.loads((base/'route_authority_design_selection_v1.json').read_bytes())
        self.sources = {n:Path(project_path(n)).read_bytes() for n in self.spec['sources']}
        self.reports = {n:(base/n).read_bytes() for n in self.spec['reports']}

    def test_no_runtime_import_or_unblocking(self):
        before = set(sys.modules)
        result = analyze(self.spec,self.sources,self.reports)
        self.assertEqual(set(sys.modules),before)
        self.assertEqual(len(result['source_evidence']),10)
        self.assertEqual(len(result['connection_gaps']),5)
        self.assertFalse(result['current_capture_unblocked'])
        self.assertFalse(result['authority_provider_implemented'])
        self.assertFalse(result['launch_authorized'])
        self.assertEqual(result['live_capture_executions'],0)

    def test_readonly_repeat(self):
        before = canonical(self.spec),dict(self.sources),dict(self.reports)
        self.assertEqual(analyze(self.spec,self.sources,self.reports),analyze(self.spec,self.sources,self.reports))
        self.assertEqual(before,(canonical(self.spec),self.sources,self.reports))

    def test_changed_source_or_report_refused(self):
        for inputs in (self.sources,self.reports):
            name = next(iter(inputs)); raw = inputs[name]; inputs[name] += b'\n'
            with self.assertRaises(ValueError): analyze(self.spec,self.sources,self.reports)
            inputs[name] = raw

    def test_roster_and_budget(self):
        with self.assertRaises(ValueError): analyze(self.spec,{},self.reports)
        self.spec['bounds']['input_bytes'] = 1
        with self.assertRaises(ValueError): analyze(self.spec,self.sources,self.reports)

    def test_runtime_permission_refused(self):
        self.spec['design']['current_allowed_runtime_calls'] = ['load']
        with self.assertRaises(ValueError): analyze(self.spec,self.sources,self.reports)

    def test_selected_storage_read_is_not_raw_read(self):
        path = 'src/sara_engine/memory/authoritative_sequence.py'
        row = inspect_source(path,self.sources[path],'AuthoritativeSequenceStore.load')
        self.assertIn('self._locked_file',row['calls'])

    def test_existing_runner_has_outcome_and_mutation_calls(self):
        path = 'src/sara_engine/evaluation/route_memory_factorial_v4.py'
        row = inspect_source(path,self.sources[path],'train')
        self.assertIn('row.outcome',row['attributes'])
        self.assertIn('model.begin',row['calls'])
        self.assertIn('model.observe',row['calls'])
