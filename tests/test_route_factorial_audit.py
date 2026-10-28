"""Pure fixed-report checks, including fail-closed corruptions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_factorial_audit import analyze,validate_snapshot,pair_at_boundary,boundary_diagnostic,canonical

BASE=Path(__file__).resolve().parents[1]/'data/processed/benchmark_fixtures'


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((BASE/'route_memory_factorial_v4_primary_v1.json').read_text())

    def test_no_learner_import(self):
        for name in ('sparse_route_reactivation','route_memory_factorial_v4','sparse_factor_routing'):
            self.assertNotIn('sara_engine.evaluation.'+name,sys.modules)

    def test_full_coverage_and_input_immutability(self):
        before=canonical(self.report)
        result=analyze(self.report)
        self.assertEqual(result['summary']['paired_boundary_comparisons'],480)
        self.assertEqual(result['summary']['intervals'],240)
        self.assertEqual(before,canonical(self.report))

    def test_identical_pair_has_no_difference(self):
        s=next(iter(self.report['cases'][0]['cells'].values()))['snapshots'][0]
        self.assertTrue(all(not v for v in pair_at_boundary(s,s)['differences'].values()))

    def test_warmup_boundary_is_before_narrowing(self):
        s=next(iter(self.report['cases'][0]['cells'].values()))['snapshots'][0]
        self.assertEqual(sum(r['in_pool'] for r in s['routes']),45)
        validate_snapshot(s,'cumulative',180)
        self.assertEqual(len(boundary_diagnostic(s)['eligible_readout_routes']),3)

    def test_rank_corruption_rejected(self):
        s=deepcopy(next(iter(self.report['cases'][0]['cells'].values()))['snapshots'][0])
        s['routes'][0]['rank']=99
        with self.assertRaisesRegex(ValueError,'rank'):validate_snapshot(s,'cumulative',180)

    def test_mass_and_probability_corruption_rejected(self):
        s=next(iter(self.report['cases'][0]['cells'].values()))['snapshots'][0]
        for field,value in [('probability',-1),('effective_mass',[999,999])]:
            changed=deepcopy(s);changed['routes'][0]['keys'][0][field]=value
            with self.assertRaises(ValueError):validate_snapshot(changed,'cumulative',180)

    def test_observation_and_identity_corruption_rejected(self):
        s=deepcopy(next(iter(self.report['cases'][0]['cells'].values()))['snapshots'][0])
        s['routes'][0]['observations']+=1
        with self.assertRaisesRegex(ValueError,'conservation'):validate_snapshot(s,'cumulative',180)
        r=deepcopy(self.report);r['cases'][0]['seed']=0
        with self.assertRaisesRegex(ValueError,'identities'):analyze(r)

    def test_missing_recovery_not_imputed(self):
        r=deepcopy(self.report)
        case=next(c for c in r['cases'] if c['condition']=='changed')
        self.assertIsNone(case['cells']['cumulative__uniform']['training']['recovery_ticks'])
        case['cells']['cumulative__uniform']['training']['recovery_ticks']=0
        with self.assertRaisesRegex(ValueError,'Recovery'):analyze(r)


if __name__=='__main__':unittest.main()
