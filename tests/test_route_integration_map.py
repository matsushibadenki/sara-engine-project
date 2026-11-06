"""Source inventory checks without model or host imports/execution."""
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_integration_map import analyze,canonical,inventory
from sara_engine.utils.project_paths import processed_data_path,project_path


class IntegrationMapTests(unittest.TestCase):
    def setUp(self):
        base=Path(processed_data_path('benchmark_fixtures'))
        self.spec=json.loads((base/'route_integration_map_selection_v1.json').read_bytes())
        self.sources={n:Path(project_path(n)).read_bytes() for n in self.spec['sources']}
        self.reports={n:(base/n).read_bytes() for n in self.spec['reports']}

    def test_frozen_inventory_without_runtime_import(self):
        before=set(sys.modules); result=analyze(self.spec,self.sources,self.reports)
        self.assertEqual(set(sys.modules),before)
        self.assertGreater(result['event_counts']['rooted_store'],0)
        self.assertGreater(result['event_counts']['rooted_argument_escape'],0)
        self.assertFalse(result['all_writer_paths_proven'])
        self.assertFalse(result['host_attached'])
        self.assertEqual(result['host_executions'],0)

    def test_local_count_alias_is_not_missed(self):
        path='src/sara_engine/evaluation/route_memory_factorial_v4.py'
        rows,_=inventory(path,self.sources[path],self.spec['scan'],16)
        row=next(r for r in rows if r['symbol']=='FactorialMemory.observe')
        self.assertIn('count',row['possible_alias_names'])
        self.assertTrue(any(e['kind']=='rooted_store' and e['expression'].startswith('count[') for e in row['events']))

    def test_all_class_methods_including_host_metadata_are_listed(self):
        result=analyze(self.spec,self.sources,self.reports)
        names={r['symbol'] for r in result['records']}
        self.assertTrue({'OwnerHost.mutation','OwnerHost.paused','FactorialMemory.observe','ReactivationMemory.narrow'}<=names)

    def test_return_and_copy_are_conservative_not_proven_escapes(self):
        path='src/sara_engine/evaluation/route_memory_factorial_v4.py'
        rows,_=inventory(path,self.sources[path],self.spec['scan'],16)
        train=next(r for r in rows if r['symbol']=='train')
        self.assertTrue(any('deepcopy(model.tables)' in e['expression'] for e in train['events']))
        self.assertFalse(train['coverage_proven'])

    def test_repeat_nonmutation(self):
        before=canonical(self.spec),dict(self.sources),dict(self.reports)
        self.assertEqual(analyze(self.spec,self.sources,self.reports),analyze(self.spec,self.sources,self.reports))
        self.assertEqual(before,(canonical(self.spec),self.sources,self.reports))

    def test_drift_roster_budget(self):
        for inputs in (self.sources,self.reports):
            key=next(iter(inputs));raw=inputs[key];inputs[key]+=b'\n'
            with self.assertRaises(ValueError):analyze(self.spec,self.sources,self.reports)
            inputs[key]=raw
        with self.assertRaises(ValueError):analyze(self.spec,{},self.reports)
        self.spec['bounds']['input_bytes']=1
        with self.assertRaises(ValueError):analyze(self.spec,self.sources,self.reports)
