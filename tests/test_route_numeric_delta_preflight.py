"""Literal numeric snapshots; no learner, data stream or actual update calls."""
from dataclasses import replace,FrozenInstanceError
from math import nextafter,inf
from pathlib import Path
import json
import sys
import unittest
from sara_engine.evaluation.route_numeric_delta_preflight import (
    NumericView,NumericWitness,MANUAL_ROWS,LIMIT,fixture,altered,validate_numeric,fault_preflight,analyze,
)
from sara_engine.evaluation.route_fixed_allocation_contract import canonical

BASE=Path(__file__).resolve().parents[1]/'data/processed/benchmark_fixtures'


class NumericDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((BASE/'route_numeric_delta_preflight_selection_v1.json').read_text())

    def example(self):return fixture(MANUAL_ROWS[0],'local_sparse_reference','shuffled_outcome')

    def test_no_learner_import(self):
        for name in ('sparse_route_reactivation','route_memory_factorial_v4','sparse_factor_routing'):
            self.assertNotIn('sara_engine.evaluation.'+name,sys.modules)

    def test_hand_authored_expected_rows_and_all_contexts(self):
        self.assertEqual(canonical(self.spec['manual_fixture']['rows']),canonical(MANUAL_ROWS))
        result=analyze(self.spec)
        self.assertEqual(result['summary']['numeric_fixture_transactions'],72)
        self.assertEqual(result['summary']['active_field_checks'],2592)
        self.assertEqual(result['summary']['unchanged_field_checks'],28728)
        self.assertEqual(result['status']['actual_learning_updates'],0)
        self.assertFalse(result['status']['runtime_scalar_equality_tested'])

    def test_exact_integer_and_float_types(self):
        p,_,_=self.example();c=p.credits[0]
        for address,value in [((c.route,'observations',-1),True),((c.route,'observations',-1),8.0),
                              ((c.route,c.mass_side,c.key_slot),2.0),((c.route,'error_sum',-1),8),
                              ((c.route,'weight_sum',-1),'16.0')]:
            with self.assertRaises(ValueError):altered(p.after,{address:value})

    def test_nonfinite_negative_and_negative_zero_refused(self):
        p,_,_=self.example();a=(p.credits[0].route,'error_sum',-1)
        for value in (float('nan'),inf,-inf,-1.0,-0.0):
            with self.assertRaises(ValueError):altered(p.after,{a:value})

    def test_one_ulp_active_error_mismatch_refused(self):
        p,b,w=self.example();a=(p.credits[0].route,'error_sum',-1)
        wrong=nextafter(dict(p.after.fields)[a],inf)
        with self.assertRaises(ValueError):validate_numeric(replace(p,after=altered(p.after,{a:wrong})),b,w)

    def test_one_ulp_inactive_error_write_refused(self):
        p,b,w=self.example();r=next(i for i in range(45) if i not in p.receipt.active);a=(r,'error_sum',-1)
        with self.assertRaises(ValueError):validate_numeric(replace(p,after=altered(p.after,{a:nextafter(8.0,inf)})),b,w)

    def test_positive_addition_absorption_is_not_a_change(self):
        p,b,w=self.example();a=(p.credits[0].route,'error_sum',-1)
        tiny=float.fromhex('0x0.0000000000001p-1022')
        self.assertGreater(tiny,0)
        self.assertEqual(8.0+tiny,8.0)
        with self.assertRaisesRegex(ValueError,'no-op'):
            validate_numeric(replace(p,after=altered(p.after,{a:8.0})),b,replace(w,error_increments=(tiny,)+w.error_increments[1:]))

    def test_numeric_upper_bounds(self):
        p,_,_=self.example();r=p.credits[0].route
        altered(p.before,{(r,'observations',-1):205,(r,'weight_sum',-1):float(LIMIT),(r,'error_sum',-1):float(LIMIT)})
        with self.assertRaises(ValueError):altered(p.before,{(r,'observations',-1):206})
        with self.assertRaises(ValueError):altered(p.before,{(r,'weight_sum',-1):float(LIMIT+1)})

    def test_trusted_before_and_inputs_are_immutable(self):
        p,b,w=self.example();signature=(p.before.typed_signature(),p.after.typed_signature())
        validate_numeric(p,b,w)
        self.assertEqual(signature,(p.before.typed_signature(),p.after.typed_signature()))
        with self.assertRaises(FrozenInstanceError):p.before.fields=()
        c=p.credits[0]
        with self.assertRaisesRegex(ValueError,'Stale'):
            validate_numeric(p,altered(b,{(c.route,c.mass_side,c.key_slot):3}),w)

    def test_all_40_faults_rejected(self):
        faults=fault_preflight()
        self.assertEqual([f['fault'] for f in faults],self.spec['faults'])
        self.assertEqual(len(faults),40)
        self.assertTrue(all(f['rejected'] and f['inputs_unchanged'] for f in faults))

    def test_error_witness_is_independent_positive_and_bounded(self):
        p,b,w=self.example()
        for value in (0.0,-1.0,float('nan'),True,1):
            with self.assertRaises(ValueError):replace(w,error_increments=(value,)+w.error_increments[1:])
        with self.assertRaises(ValueError):validate_numeric(p,b,replace(w,error_increments=(2.0,)+w.error_increments[1:]))

    def test_manual_conditions_and_rows_not_tuned(self):
        before=canonical(self.spec);analyze(self.spec)
        self.assertEqual(before,canonical(self.spec))
        for section,key,value in [('bounds','transactions',1),('manual_fixture','roles',['candidate']*3),
                                  ('manual_fixture','seed',1213)]:
            changed=json.loads(canonical(self.spec));changed[section][key]=value
            with self.assertRaises(ValueError):analyze(changed)
        changed=json.loads(canonical(self.spec));changed['manual_fixture']['rows'][0][3]=4
        with self.assertRaises(ValueError):analyze(changed)


if __name__=='__main__':unittest.main()
