"""Hand encoding/schema validation; no real model or candidate encoder run."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_capture_layout import validate,decode,project,LayoutError,canonical
from sara_engine.evaluation.route_capture_layout_preflight import analyze,fixture,repack
from sara_engine.utils.project_paths import processed_data_path,project_path


class CaptureLayoutTests(unittest.TestCase):
    def setUp(self):
        base=Path(processed_data_path('benchmark_fixtures'))
        self.spec=json.loads((base/'route_capture_layout_selection_v1.json').read_bytes())
        self.sources={n:Path(project_path(n)).read_bytes() for n in self.spec['sources']}
        self.reports={n:(base/n).read_bytes() for n in self.spec['reports']}

    def test_registered_pairs_faults_and_no_learner_import(self):
        before=set(sys.modules);result=analyze(self.spec,self.sources,self.reports)
        self.assertEqual(set(sys.modules),before)
        self.assertEqual(result['hand_pairs'],2)
        self.assertEqual(len(result['faults_rejected']),28)
        self.assertFalse(result['capture_unblocked'])
        self.assertEqual(result['candidate_encoder_executions'],0)

    def test_absent_and_zero_have_equal_fields_but_distinct_flags(self):
        absent=validate(fixture(self.spec,'absent_first_key')[0])
        zero=validate(fixture(self.spec,'allocated_zero_first_key')[0])
        self.assertEqual(absent['fields'],zero['fields'])
        self.assertEqual(len(absent['fields']),435);self.assertEqual(len(absent['allocations']),150)
        self.assertEqual(sum(a!=b for a,b in zip(absent['allocations'],zero['allocations'])),1)
        self.assertEqual(absent['receipt_credits'][0],(0,(0,),False,None))
        self.assertEqual(zero['receipt_credits'][0],(0,(0,),True,0.5))

    def test_inactive_width_two_key_address_is_retained(self):
        result=validate(fixture(self.spec,'absent_first_key')[0]);fields=dict(result['fields'])
        self.assertEqual(fields[(44,'mass_left',3)],7)
        self.assertEqual(fields[(44,'mass_right',3)],9)
        self.assertIn((44,3,True),result['allocations'])

    def test_bool_route_catalog_not_equal_to_integer_catalog(self):
        pair,before,after=fixture(self.spec,'absent_first_key')
        routes=list(after['routes']);routes[0]=((False,),0);after['routes']=tuple(routes)
        with self.assertRaises(LayoutError):validate(repack(pair,before,after))

    def test_signed_zero_change_not_hidden_by_float_equality(self):
        pair,before,after=fixture(self.spec,'absent_first_key');after['errors'][0]=-0.0
        with self.assertRaises(LayoutError):validate(repack(pair,before,after))

    def test_probability_not_recomputed_or_scored(self):
        pair,before,after=fixture(self.spec,'absent_first_key')
        pending=list(after['pending']);p=list(pending[3]);p[2]=0.9;pending[3]=tuple(p);after['pending']=tuple(pending)
        result=validate(repack(pair,before,after))
        self.assertEqual(result['receipt_credits'][2][-1],0.9)
        self.assertFalse(result['probabilities_recomputed'])
        self.assertFalse(result['authority_authenticated'])

    def test_decoder_rejects_unknown_depth_and_duplicate_keys(self):
        with self.assertRaises(LayoutError):decode(('unknown',()))
        with self.assertRaises(LayoutError):decode(('tuple',(('none',),)),depth_limit=0)
        with self.assertRaises(LayoutError):decode(('dict',((('str','x'),('none',)),(('str','x'),('none',)))))

    def test_repeat_is_readonly(self):
        pair,_,_=fixture(self.spec,'absent_first_key');before=repr(pair)
        self.assertEqual(validate(pair),validate(pair));self.assertEqual(repr(pair),before)

    def test_frozen_input_drift(self):
        key=next(iter(self.sources));self.sources[key]+=b'\n'
        with self.assertRaises(ValueError):analyze(self.spec,self.sources,self.reports)
