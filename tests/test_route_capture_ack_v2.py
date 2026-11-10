"""Capture-derived acknowledgement, using only hand numeric records/control host."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_capture_ack import CaptureAckBridge
from sara_engine.evaluation.route_capture_ack_preflight import analyze,fixture,mutate
from sara_engine.evaluation.route_outcome_admission import AdmissionError
from sara_engine.evaluation.route_owner_host import HostError
from sara_engine.evaluation.route_pause_admission import PauseAdmission
from sara_engine.evaluation.route_capture_layout_preflight import repack
from sara_engine.utils.project_paths import processed_data_path


class CaptureAckTests(unittest.TestCase):
    def setUp(self):
        base=Path(processed_data_path('benchmark_fixtures'))
        self.spec=json.loads((base/'route_capture_ack_selection_v2.json').read_bytes())
        self.reports={n:(base/n).read_bytes() for n in self.spec['reports']}
        self.h,self.o,self.b,self.pair,self.before,self.after=fixture(self.spec,self.spec['variants'][0])

    def seal(self):
        with self.h.paused(self.o,10) as token:self.b.seal_predecessor(token,self.pair.predecessor)
        mutate(self.h,self.o,10)

    def test_registered_pairs_faults_no_learner_import(self):
        before=set(sys.modules);result=analyze(self.spec,self.reports)
        self.assertEqual(set(sys.modules),before)
        self.assertEqual(result['hand_pairs'],2);self.assertEqual(len(result['faults_rejected']),26)
        self.assertFalse(result['expected_ack_supplied'])
        self.assertFalse(result['numeric_state_authenticated'])

    def test_derived_values_and_fixed_plan_incompatibility(self):
        self.seal()
        with self.h.paused(self.o,12) as token:
            ack=self.b.acknowledge(token,self.pair)
            self.assertEqual(ack.pre_digest,self.pair.predecessor.digest)
            self.assertEqual(ack.receipt_digest,self.pair.receipt.digest)
            self.assertEqual(ack.routes,tuple(self.spec['hand']['active']))
            self.assertEqual(self.b.release(token,65,(0,1,0)).capture,ack)
        # The old constructor requires its pre-phase host and a fixed expected ack.
        with self.assertRaises(AdmissionError):PauseAdmission(self.h,self.o,ack)

    def test_valid_pair_cannot_replace_privately_sealed_predecessor(self):
        self.seal();self.before['tables'][44][(1,1)]=[8,9];self.after['tables'][44][(1,1)]=[8,9]
        pair=repack(self.pair,self.before,self.after)
        with self.h.paused(self.o,12) as token:
            with self.assertRaises(AdmissionError):self.b.acknowledge(token,pair)
        self.assertTrue(self.b.status()['stopped'])

    def test_no_manual_expected_ack_argument(self):
        with self.assertRaises(TypeError):CaptureAckBridge(self.h,self.o,self.pair.input,expected_ack=object())

    def test_metadata_copy_not_pause_capability(self):
        self.seal()
        with self.h.paused(self.o,12) as token:
            with self.assertRaises(HostError):self.b.acknowledge(replace(token),self.pair)
        self.assertTrue(self.b.status()['stopped'])

    def test_expired_delivery_stops_before_payload_inspection(self):
        self.seal()
        with self.h.paused(self.o,12) as token:self.b.acknowledge(token,self.pair)
        with self.assertRaises(HostError):self.b.release(token,65,object())
        self.assertTrue(self.b.status()['stopped'])
        self.assertEqual(self.b.status()['admission_phase'],'ready')

    def test_ack_copy_is_not_authority(self):
        self.seal()
        with self.h.paused(self.o,12) as token:
            ack=self.b.acknowledge(token,self.pair)
            with self.assertRaises(AdmissionError):self.b.release(ack,65,(0,1,0))

    def test_input_source_binding_not_receiving_clock(self):
        self.seal()
        with self.h.paused(self.o,12) as token:
            wrong=replace(self.pair,input=replace(self.pair.input,source_id=8))
            with self.assertRaises(AdmissionError):self.b.acknowledge(token,wrong)

    def test_report_drift_budget(self):
        n=next(iter(self.reports));self.reports[n]+=b'\n'
        with self.assertRaises(ValueError):analyze(self.spec,self.reports)
        self.spec['bounds']['input_bytes']=1
        with self.assertRaises(ValueError):analyze(self.spec,self.reports)
