"""Model-free lifecycle/admission composition, no numerical learner or source."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_owner_host import HostError,OwnerHost
from sara_engine.evaluation.route_outcome_admission import AdmissionError
from sara_engine.evaluation.route_pause_admission import PauseAdmission
from sara_engine.evaluation.route_pause_admission_preflight import analyze,fixture,mutate
from sara_engine.utils.project_paths import processed_data_path


class PauseAdmissionTests(unittest.TestCase):
    def setUp(self):
        base=Path(processed_data_path('benchmark_fixtures'))
        self.spec=json.loads((base/'route_pause_admission_selection_v1.json').read_bytes())
        self.reports={n:(base/n).read_bytes() for n in self.spec['reports']}
        self.h,self.o,self.ack,self.packet=fixture(self.spec,'shuffled_outcome',self.spec['engines'][0])
        self.c=PauseAdmission(self.h,self.o,self.ack);self.c.admit(self.packet)

    def preseal(self):
        with self.h.paused(self.o,0) as pre:self.c.seal_predecessor(pre)
        mutate(self.h,self.o,0)

    def test_registered_cases_and_no_numeric_runtime_import(self):
        before=set(sys.modules);result=analyze(self.spec,self.reports)
        self.assertEqual(set(sys.modules),before)
        self.assertEqual(result['hand_cases'],6)
        self.assertEqual(len(result['faults_rejected']),26)
        self.assertFalse(result['capture_unblocked'])
        self.assertFalse(result['actual_model_authority_established'])

    def test_equal_metadata_is_not_live_ticket(self):
        self.preseal()
        with self.h.paused(self.o,2) as ticket:
            with self.assertRaises(HostError):self.c.acknowledge(replace(ticket),self.ack)
        self.assertTrue(self.c.status()['stopped'])

    def test_same_revision_new_pause_cannot_release(self):
        self.preseal()
        with self.h.paused(self.o,2) as ticket:self.c.acknowledge(ticket,self.ack)
        with self.h.paused(self.o,2) as new:
            with self.assertRaises(AdmissionError):self.c.release(new,65,(0,1,0))
        self.assertTrue(self.c.status()['stopped'])

    def test_expired_pause_stops_before_outcome_inspection(self):
        self.preseal()
        with self.h.paused(self.o,2) as ticket:self.c.acknowledge(ticket,self.ack)
        with self.assertRaises(HostError):self.c.release(ticket,65,object())
        self.assertEqual(self.c.status()['admission_phase'],'ready')
        self.assertTrue(self.c.status()['stopped'])
        with self.h.paused(self.o,2) as new:
            with self.assertRaises(AdmissionError):self.c.release(new,65,(0,1,0))

    def test_active_ack_and_release_do_not_change_host_generation(self):
        self.preseal()
        with self.h.paused(self.o,2) as ticket:
            before=self.h.revalidate(self.o,ticket)
            self.c.acknowledge(ticket,self.ack)
            d=self.c.release(ticket,65,(0,1,0))
            self.assertEqual(before,self.h.revalidate(self.o,ticket))
        self.assertEqual(d.capture,self.ack)
        self.assertEqual(self.c.status()['admission_phase'],'released')
        self.assertFalse(self.c.status()['application_authorized'])

    def test_wrong_host_same_label_not_authoritative(self):
        self.preseal()
        other,owner=OwnerHost.create(self.ack.owner,self.ack.epoch);mutate(other,owner,0)
        with other.paused(owner,2) as ticket:
            with self.assertRaises(HostError):self.c.acknowledge(ticket,self.ack)

    def test_missing_preseal_refused(self):
        mutate(self.h,self.o,0)
        with self.h.paused(self.o,2) as ticket:
            with self.assertRaises(AdmissionError):self.c.acknowledge(ticket,self.ack)
        self.assertTrue(self.c.status()['stopped'])

    def test_report_drift_budget(self):
        name=next(iter(self.reports));self.reports[name]+=b'\n'
        with self.assertRaises(ValueError):analyze(self.spec,self.reports)
        self.spec['bounds']['input_bytes']=1
        with self.assertRaises(ValueError):analyze(self.spec,self.reports)
