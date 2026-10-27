"""Read-only verification of retained factorial evidence and consumed phases."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/processed/benchmark_fixtures'
SPEC = importlib.util.spec_from_file_location('factorial_runner', ROOT / 'scripts/eval/route_memory_factorial_v4.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RetainedFactorialTests(unittest.TestCase):
    def test_exact_replay_and_all_budget_invariants(self):
        runner.verify()
        primary = runner.path('primary').read_bytes()
        self.assertEqual(primary, runner.path('replication').read_bytes())
        proof = runner.read('result')
        self.assertEqual(sha256(primary).hexdigest(), proof['primary_sha256'])
        report = json.loads(primary)
        self.assertEqual(len(report['cases']),20)
        snapshots=0
        for case in report['cases']:
            self.assertTrue(case['historical_baseline_equal'])
            self.assertEqual(len(case['cells']),4)
            for cell in case['cells'].values():
                stats=cell['training']
                self.assertEqual(stats['route_updates'],9216)
                self.assertEqual(stats['actual_learning_field_changes'],36864)
                self.assertTrue(stats['frozen']['scalar_reference_equal'])
                self.assertTrue(stats['frozen']['packet_permutation_equal'])
                self.assertTrue(stats['frozen_replay_equal'])
                self.assertEqual([s['boundary'] for s in cell['snapshots']],[180,512,768,1024])
                snapshots+=len(cell['snapshots'])
                for snapshot in cell['snapshots']:
                    self.assertTrue(snapshot['read_only'])
                    self.assertEqual(len(snapshot['routes']),45)
        self.assertEqual(snapshots,320)
        self.assertLess(len(primary),8388608)
        self.assertLess(sum(p.stat().st_size for p in BASE.glob('route_memory_factorial_v4_*_v1.json')),33554432)

    def test_consumed_phases_refuse_before_learner(self):
        before={p.name:sha256(p.read_bytes()).hexdigest() for p in BASE.glob('route_memory_factorial_v4_*_v1.json')}
        for phase in ('primary','replication'):
            with patch.object(sys,'argv',['factorial',phase]), patch.object(runner,'run_case',side_effect=AssertionError('Learner must never run')):
                with self.assertRaises(FileExistsError):runner.main()
        after={p.name:sha256(p.read_bytes()).hexdigest() for p in BASE.glob('route_memory_factorial_v4_*_v1.json')}
        self.assertEqual(before,after)

    def test_read_only_proof_verification(self):
        with patch.object(sys,'argv',['factorial','verify']), patch.object(runner,'run_case',side_effect=AssertionError('No learner in verification')):
            runner.main()


if __name__ == '__main__':unittest.main()
