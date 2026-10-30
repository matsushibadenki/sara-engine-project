"""Independent literal arithmetic and separation from all primary helpers."""
import inspect
import json
from math import nextafter,inf
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from sara_engine.evaluation import route_independent_scalar as kernel
from sara_engine.evaluation.route_independent_scalar_preflight import (
    calculate,analyze,compare_literal,audit_kernel_source,assert_independent_modules,
    validate_rows,fault_preflight,canonical,
)

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/processed/benchmark_fixtures'


class IndependentScalarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((BASE/'route_independent_scalar_preflight_selection_v1.json').read_text())
        cls.rows=json.loads((BASE/'route_error_binding_preflight_selection_v1.json').read_text())['local_rows']
        cls.source=(ROOT/'src/sara_engine/evaluation/route_independent_scalar.py').read_text()

    def test_all_eleven_literal_goldens_exact(self):
        with patch('sara_engine.evaluation.route_independent_scalar_preflight.calculate',wraps=calculate) as calls:
            result=analyze(self.spec,self.rows,self.source)
        self.assertEqual(calls.call_count,11)
        self.assertEqual(result['summary']['records'],11)
        self.assertEqual(result['summary']['exact_matches'],11)
        self.assertEqual(result['summary']['value_comparisons'],22)
        self.assertTrue(result['summary']['all_exact'])

    def test_expected_values_not_kernel_parameters(self):
        self.assertEqual(list(inspect.signature(calculate).parameters),['tick','history','error_weight','allocated','n0','n1','side'])
        with self.assertRaises(TypeError):calculate(*self.rows[0][:7],expected_probability=.5)

    def test_no_primary_helper_modules_loaded(self):
        assert_independent_modules()
        with patch.dict(sys.modules,{'sara_engine.evaluation.route_error_binding_preflight':object()}):
            with self.assertRaises(ValueError):assert_independent_modules()

    def test_kernel_source_is_math_only_and_stateless(self):
        result=audit_kernel_source(self.source)
        self.assertTrue(result['source_only_math_import'])
        self.assertTrue(result['no_global_state_or_defaults'])

    def test_stateless_repeated_calls_fresh_results_and_no_globals_written(self):
        before=dict(vars(kernel));row=list(self.rows[0]);a=calculate(*row[:7]);a['probability']=.9
        b=calculate(*row[:7])
        self.assertIsNot(a,b)
        self.assertEqual(b['probability'],.5)
        self.assertEqual(before,dict(vars(kernel)))
        self.assertEqual(row,self.rows[0])

    def test_absent_and_allocated_zero_are_distinct(self):
        absent=calculate(*self.rows[9][:7]);seen=calculate(*self.rows[10][:7])
        self.assertIsNone(absent['probability']);self.assertTrue(absent['residual_fallback'])
        self.assertEqual(seen['probability'],.5);self.assertFalse(seen['residual_fallback'])
        self.assertEqual(absent['error_increment'],seen['error_increment'])

    def test_asymmetric_class_and_public_weight_goldens(self):
        for i in (1,2,3,4,5,6,7,8):
            row=self.rows[i]
            self.assertTrue(compare_literal(calculate(*row[:7]),row[7],row[8])['exact_match'])

    def test_ulp_and_type_mismatch_never_tolerated(self):
        a=calculate(*self.rows[0][:7])
        self.assertFalse(compare_literal(a,nextafter(.5,inf),.25)['exact_match'])
        self.assertFalse(compare_literal(a,.5,nextafter(.25,inf))['exact_match'])
        for p,e in ((1,.25),(.5,1),(True,.25),(.5,float('nan'))):
            with self.assertRaises(ValueError):compare_literal(a,p,e)

    def test_source_import_state_dynamic_and_default_injections_rejected(self):
        for source in ('from sara_engine.evaluation.route_error_binding_preflight import local_check\n'+self.source,
                       'CACHE=[]\n'+self.source,
                       self.source.replace('    public_scale =',"    eval('1')\n    public_scale =",1),
                       self.source.replace('n1, side):',"n1, side='mass_left'):")):
            with self.assertRaises(ValueError):audit_kernel_source(source)

    def test_frozen_rows_and_all_registered_faults(self):
        before=canonical(self.rows);faults=fault_preflight(self.rows,self.source)
        self.assertEqual([f['fault'] for f in faults],self.spec['faults'])
        self.assertTrue(all(f['detected'] for f in faults))
        self.assertEqual(before,canonical(self.rows))
        with self.assertRaises(ValueError):validate_rows(list(reversed(self.rows)))

    def test_invalid_scalar_input_refused(self):
        for index,value in ((0,True),(4,True),(4,-1),(3,1),(6,'unknown')):
            values=list(self.rows[0][:7]);values[index]=value
            with self.assertRaises(ValueError):calculate(*values)

    def test_fixture_computation_never_promoted_to_runtime_or_training(self):
        result=analyze(self.spec,self.rows,self.source)
        for key in ('new_learner_executions','actual_learning_updates','task_prediction_scores','new_acquisitions','new_resource_measurements'):
            self.assertEqual(result['status'][key],0)
        for key in ('runtime_learner_equivalence_tested','runtime_causal_provenance_authenticated',
                    'snn_advantage_established','original_adaptive_component_passed','training_authorized'):
            self.assertFalse(result['status'][key])


if __name__=='__main__':unittest.main()
