"""Documentation gate regressions, without science/model execution."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from sara_engine.evaluation.roadmap_gate_check import check
from sara_engine.utils.project_paths import processed_data_path, project_path


class RoadmapGateTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(Path(processed_data_path('benchmark_fixtures','roadmap_gate_status_v1.json')).read_bytes())
        self.evidence = {e['evidence']:Path(project_path(e['evidence'])).read_bytes() for e in self.manifest['entries']}
        self.roadmaps = {n:Path(project_path(n)).read_bytes() for n in self.manifest['roadmaps']}

    def test_current_gates_pass_without_mutation(self):
        before = deepcopy((self.manifest,self.evidence,self.roadmaps))
        self.assertEqual(check(self.manifest,self.evidence,self.roadmaps),[])
        self.assertEqual((self.manifest,self.evidence,self.roadmaps),before)

    def test_stale_consumed_next_is_rejected(self):
        name = next(iter(self.roadmaps))
        self.roadmaps[name] += ('\n- 🟠 [Next] '+self.manifest['forbidden_next_phrases'][0]).encode()
        self.assertTrue(any('Consumed work' in e for e in check(self.manifest,self.evidence,self.roadmaps)))

    def test_pending_cannot_silently_become_next(self):
        name = next(iter(self.roadmaps))
        self.roadmaps[name] = self.roadmaps[name].replace(b'- \xe2\xad\x95\xef\xb8\x8f [Pending] New task',b'- [Next] New task')
        self.assertTrue(check(self.manifest,self.evidence,self.roadmaps))

    def test_missing_or_duplicate_marker(self):
        name = next(iter(self.roadmaps))
        marker = b'<!-- gate:cartpole-fresh-task -->'
        for replacement in (b'',marker+marker):
            changed = dict(self.roadmaps); changed[name] = changed[name].replace(marker,replacement)
            # Two occurrences on a single row must also be rejected.
            self.assertTrue(check(self.manifest,self.evidence,changed))

    def test_evidence_change_requires_review(self):
        name = next(iter(self.evidence)); self.evidence[name] += b'\n'
        self.assertTrue(any('Evidence changed' in e for e in check(self.manifest,self.evidence,self.roadmaps)))

    def test_authorization_never_granted(self):
        self.manifest['runtime_authorized'] = True
        self.assertTrue(check(self.manifest,self.evidence,self.roadmaps))

    def test_budget_and_roster(self):
        self.assertTrue(check(self.manifest,{},self.roadmaps))
        self.manifest['input_byte_limit'] = 1
        self.assertTrue(check(self.manifest,self.evidence,self.roadmaps))
