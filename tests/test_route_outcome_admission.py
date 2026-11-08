"""Model-free admission protocol tests; no live producer or learner."""
from dataclasses import FrozenInstanceError,fields,replace
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_outcome_admission import AdmissionError,Input
from sara_engine.evaluation.route_outcome_admission_preflight import analyze,fixture
from sara_engine.utils.project_paths import processed_data_path


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        base=Path(processed_data_path('benchmark_fixtures'))
        self.spec=json.loads((base/'route_outcome_admission_selection_v1.json').read_bytes())
        self.reports={n:(base/n).read_bytes() for n in self.spec['reports']}
        self.g,self.p,self.c,self.packet,self.ack=fixture(self.spec,'shuffled_outcome',self.spec['engines'][0])

    def test_registered_cases_without_runtime_import(self):
        before=set(sys.modules);result=analyze(self.spec,self.reports)
        self.assertEqual(set(sys.modules),before)
        self.assertEqual(result['hand_cases'],6)
        self.assertEqual(len(result['faults_rejected']),24)
        self.assertFalse(result['capture_unblocked'])
        self.assertFalse(result['real_outcome_unavailability_established'])

    def test_input_schema_cannot_retain_result(self):
        self.assertNotIn('outcome',[f.name for f in fields(Input)])
        with self.assertRaises(TypeError):Input('o','e',1,1,(0,0,0,0,0),outcome=(0,1,0))
        self.g.admit(self.p,self.packet)
        with self.assertRaises(FrozenInstanceError):self.packet.source_id=7
        self.assertFalse(hasattr(self.g.input_view(),'outcome'))

    def test_preack_release_does_not_inspect_payload_and_stops(self):
        class Trap:
            def __iter__(self):raise AssertionError('Payload was inspected')
        self.g.admit(self.p,self.packet)
        with self.assertRaises(AdmissionError):self.g.release(self.p,65,Trap())
        self.assertEqual(self.g.phase(),'stopped')
        with self.assertRaises(AdmissionError):self.g.acknowledge(self.c,self.ack)
        self.assertFalse(hasattr(self.g,'_outcome'))

    def test_bad_bound_ack_stops_without_retry(self):
        self.g.admit(self.p,self.packet)
        with self.assertRaises(AdmissionError):self.g.acknowledge(self.c,replace(self.ack,epoch='wrong'))
        self.assertEqual(self.g.phase(),'stopped')
        with self.assertRaises(AdmissionError):self.g.acknowledge(self.c,self.ack)

    def test_foreign_capability_does_not_poison(self):
        with self.assertRaises(AdmissionError):self.g.admit(object(),self.packet)
        self.assertEqual(self.g.phase(),'empty')
        self.g.admit(self.p,self.packet)
        with self.assertRaises(AdmissionError):self.g.acknowledge(object(),self.ack)
        self.assertEqual(self.g.phase(),'awaiting_ack')

    def test_exact_bound_delivery_and_no_replay(self):
        self.g.admit(self.p,self.packet);self.g.acknowledge(self.c,self.ack)
        d=self.g.release(self.p,65,(0,1,0))
        self.assertEqual(d.capture,self.ack)
        self.assertEqual(d.input.source_id,self.packet.source_id)
        with self.assertRaises(AdmissionError):self.g.release(self.p,65,(0,1,0))
        self.assertEqual(self.g.phase(),'released')

    def test_strict_types_and_complete_routes(self):
        for change in ({'pre_revision':True},{'routes':self.ack.routes[:-1]},{'routes':(self.ack.routes[0],)*9}):
            with self.assertRaises(AdmissionError):replace(self.ack,**change)

    def test_producer_and_collector_cannot_exchange_capabilities(self):
        with self.assertRaises(AdmissionError):self.g.admit(self.c,self.packet)
        self.g.admit(self.p,self.packet)
        with self.assertRaises(AdmissionError):self.g.acknowledge(self.p,self.ack)

    def test_report_drift_budget(self):
        n=next(iter(self.reports));self.reports[n]+=b'\n'
        with self.assertRaises(ValueError):analyze(self.spec,self.reports)
        self.spec['bounds']['input_bytes']=1
        with self.assertRaises(ValueError):analyze(self.spec,self.reports)
