"""Static candidate checks and refusal only; never construct a real model."""
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_capture_candidate import RouteCaptureCandidate,CaptureBlocked
from sara_engine.evaluation.route_capture_candidate_preflight import source_checks
from sara_engine.utils.project_paths import project_path


class CaptureCandidateTests(unittest.TestCase):
    def setUp(self):
        self.raw=Path(project_path('src/sara_engine/evaluation/route_capture_candidate.py')).read_bytes()

    def test_static_contract(self):
        self.assertEqual(len(source_checks(self.raw)),11)

    def test_launch_refuses_before_runtime_import(self):
        before=set(sys.modules)
        with self.assertRaises(CaptureBlocked):
            RouteCaptureCandidate(owner_label='hand',epoch_label='hand',seed=0,history='cumulative',error_weight='uniform')
        self.assertEqual(set(sys.modules),before)
        self.assertNotIn('sara_engine.evaluation.route_memory_factorial_v4',sys.modules)

    def test_enabled_flag_is_rejected(self):
        changed=self.raw.replace(b'LAUNCH_AUTHORIZED = False',b'LAUNCH_AUTHORIZED = True')
        with self.assertRaises(ValueError):source_checks(changed)

    def test_import_before_guard_is_rejected(self):
        changed=b'from sara_engine.evaluation.route_memory_factorial_v4 import FactorialMemory\n'+self.raw
        with self.assertRaises(ValueError):source_checks(changed)

    def test_learner_or_storage_call_is_rejected(self):
        for call in (b'self._model.observe(0, 1)',b'open("forbidden", "w")'):
            changed=self.raw.replace(b'self._model.begin(packet.events,packet.receiving_id)',call)
            with self.assertRaises(ValueError):source_checks(changed)

    def test_missing_independent_read_is_rejected(self):
        changed=self.raw.replace(b'second=self._read(before)',b'second=first')
        with self.assertRaises(ValueError):source_checks(changed)

    def test_raw_model_return_is_rejected(self):
        changed=self.raw.replace(b'return first',b'return self._model')
        with self.assertRaises(ValueError):source_checks(changed)

    def test_model_argument_is_rejected(self):
        changed=self.raw.replace(b'seed,history,error_weight):',b'seed,history,error_weight,model=None):')
        with self.assertRaises(ValueError):source_checks(changed)
