"""Plain encoder contract only; no candidate/model/host/learning execution."""
from pathlib import Path
import json
import sys
import unittest
from sara_engine.evaluation.route_encoder_check import analyze,bounded_encode,EncoderBoundError
from sara_engine.evaluation.route_capture_layout import decode
from sara_engine.utils.project_paths import processed_data_path,project_path


class EncoderCheckTests(unittest.TestCase):
    def setUp(self):
        base=Path(processed_data_path('benchmark_fixtures'))
        self.spec=json.loads((base/'route_encoder_check_selection_v1.json').read_bytes())
        self.sources={n:Path(project_path(n)).read_bytes() for n in self.spec['sources']}
        self.reports={n:(base/n).read_bytes() for n in self.spec['reports']}

    def test_registered_plain_encoder_comparison(self):
        before=set(sys.modules);result=analyze(self.spec,self.sources,self.reports)
        self.assertEqual(set(sys.modules),before)
        self.assertEqual(result['baseline_encoder_calls'],8)
        self.assertEqual(result['negative_encoder_calls'],1)
        self.assertEqual(len(result['pair_projection_checks']),4)
        self.assertFalse(result['capture_unblocked'])
        self.assertFalse(result['real_model_constructed'])

    def test_exact_literal_type_tags(self):
        encoded,_=bounded_encode((None,False,0,0.0,-0.0,[],()))
        self.assertEqual(encoded,('tuple',(('none',),('bool',False),('int','0'),('float','0x0.0p+0'),
                                         ('float','-0x0.0p+0'),('list',()),('tuple',()))))
        self.assertEqual(type(decode(encoded)[1]),bool)
        self.assertEqual(type(decode(encoded)[2]),int)

    def test_same_dict_values_different_insertion_order(self):
        a=bounded_encode({(0,1):[0,0],(1,0):None})
        b=bounded_encode({(1,0):None,(0,1):[0,0]})
        self.assertEqual(a,b)

    def test_repeated_alias_is_not_a_cycle_or_retained_mutable_state(self):
        item=[1,2];value=[item,item];encoded,_=bounded_encode(value)
        item.append(3)
        self.assertEqual(decode(encoded),[[1,2],[1,2]])

    def test_hostile_object_is_not_inspected_or_called(self):
        class Hostile:
            def __iter__(self):raise AssertionError('Must not run')
            def __repr__(self):raise AssertionError('Must not run')
        with self.assertRaises(EncoderBoundError):bounded_encode(Hostile())

    def test_cycle_and_bounds_prevent_encoder_call(self):
        counter=[0];value=[];value.append(value)
        with self.assertRaises(EncoderBoundError):bounded_encode(value,counter=counter)
        with self.assertRaises(EncoderBoundError):bounded_encode([[[None]]],depth_limit=1,counter=counter)
        self.assertEqual(counter[0],0)

    def test_source_report_and_budget_drift(self):
        for inputs in (self.sources,self.reports):
            name=next(iter(inputs));raw=inputs[name];inputs[name]+=b'\n'
            with self.assertRaises(EncoderBoundError):analyze(self.spec,self.sources,self.reports)
            inputs[name]=raw
        self.spec['bounds']['input_bytes']=1
        with self.assertRaises(EncoderBoundError):analyze(self.spec,self.sources,self.reports)
