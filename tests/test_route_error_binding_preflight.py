"""Read-only mathematical witness binding on hand-authored records."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import unittest
from sara_engine.evaluation.route_error_binding_preflight import (
    OUTPUT_SIDES,LookupCredit,local_check,bind_numeric,fixture,fault_preflight,analyze,
)
from sara_engine.evaluation.route_numeric_delta_preflight import MANUAL_ROWS,fixture as old_fixture,validate_numeric
from sara_engine.evaluation.route_fixed_allocation_contract import ROUTES,canonical

BASE=Path(__file__).resolve().parents[1]/'data/processed/benchmark_fixtures'


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.spec=json.loads((BASE/'route_error_binding_preflight_selection_v1.json').read_text())

    def test_no_learner_import(self):
        for name in ('sparse_route_reactivation','route_memory_factorial_v4','sparse_factor_routing'):
            self.assertNotIn('sara_engine.evaluation.'+name,sys.modules)

    def test_all_hand_records_and_integrated_fixtures(self):
        result=analyze(self.spec)
        self.assertEqual(result['summary']['local_records'],11)
        self.assertEqual(result['summary']['route_error_bindings'],648)
        self.assertEqual(result['summary']['numeric_fixture_transactions'],72)
        self.assertEqual(result['status']['actual_learning_updates'],0)
        self.assertFalse(result['status']['runtime_provenance_authenticated'])

    def test_asymmetric_class_and_weight_math(self):
        self.assertEqual(local_check(0,'cumulative','uniform',True,0,2,'mass_left',.75,.5625)['prior'],1)
        self.assertEqual(local_check(64,'discounted','public_exponential',True,0,4,'mass_right',.75,.125)['prior'],2)
        self.assertEqual(local_check(1023,'discounted','public_exponential',True,0,65536,'mass_left',.75,18432.0)['prior'],32768)

    def test_absent_and_allocated_zero_distinct(self):
        absent=local_check(64,'discounted','uniform',False,0,0,'mass_left',None,.25)
        seen=local_check(64,'discounted','uniform',True,0,0,'mass_left',.5,.25)
        self.assertIsNone(absent['probability'])
        self.assertTrue(absent['residual_fallback'])
        self.assertEqual(seen['probability'],.5)
        self.assertFalse(seen['residual_fallback'])

    def test_new_fixture_credits_consistent_per_output(self):
        p,b,w,records,flags,sides=fixture(MANUAL_ROWS[0],'local_sparse_reference','shuffled_outcome')
        for c in p.credits:self.assertEqual(c.mass_side,sides[ROUTES[c.route][1]])
        self.assertEqual(bind_numeric(p,b,w,records,flags,sides)['route_bindings'],9)

    def test_old_numeric_fixture_not_relabelled_as_joint_delivery(self):
        p,b,w=old_fixture(MANUAL_ROWS[0],'local_sparse_reference','shuffled_outcome')
        self.assertEqual(validate_numeric(p,b,w)['active_field_checks'],36)
        records=tuple(LookupCredit(c,True,.5,d) for c,d in zip(w.credits,w.error_increments))
        with self.assertRaisesRegex(ValueError,'Same-output'):
            bind_numeric(p,b,w,records,(True,)*9,OUTPUT_SIDES)

    def test_invalid_probability_types_rejected(self):
        args=fixture(MANUAL_ROWS[0],'local_sparse_reference','candidate');r=args[3][0]
        for value in (True,1,float('nan'),float('inf'),-.1,1.1):
            with self.assertRaises(ValueError):replace(r,probability=value)

    def test_all_registered_faults_and_input_immutability(self):
        before=canonical(self.spec);faults=fault_preflight();analyze(self.spec)
        self.assertEqual(before,canonical(self.spec))
        self.assertEqual([f['fault'] for f in faults],self.spec['faults'])
        self.assertEqual(len(faults),20)
        self.assertTrue(all(f['rejected'] and f['inputs_unchanged'] for f in faults))

    def test_scope_change_rejected(self):
        for section,key,value in [('bounds','transactions',1),('integrated_fixtures','output_sides',['mass_right']*3)]:
            spec=json.loads(canonical(self.spec));spec[section][key]=value
            with self.assertRaises(ValueError):analyze(spec)


if __name__=='__main__':unittest.main()
