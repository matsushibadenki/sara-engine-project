"""Static source mutations and launch refusal; no model/encoder/body execution."""
from pathlib import Path
import unittest
from sara_engine.evaluation.route_capture_candidate_v2_preflight import source_checks,refusal_check
from sara_engine.utils.project_paths import project_path


class CaptureCandidateV2Tests(unittest.TestCase):
    def setUp(self):
        self.raw=Path(project_path('src/sara_engine/evaluation/route_capture_candidate_v2.py')).read_bytes()

    def rejected(self,old,new):
        self.assertIn(old,self.raw)
        with self.assertRaises(ValueError):source_checks(self.raw.replace(old,new))

    def test_static_contract(self):
        self.assertEqual(len(source_checks(self.raw)),15)

    def test_launch_and_capture_refusal_with_encoder_host_traps(self):
        refusal_check()

    def test_enabled_flag_rejected(self):
        self.rejected(b'LAUNCH_AUTHORIZED = False',b'LAUNCH_AUTHORIZED = True')

    def test_eager_learner_import_rejected(self):
        with self.assertRaises(ValueError):
            source_checks(b'from sara_engine.evaluation.route_memory_factorial_v4 import FactorialMemory\n'+self.raw)

    def test_model_or_callback_constructor_argument_rejected(self):
        for parameter in (b'model=None',b'callback=None',b'permit=None'):
            self.rejected(b'seed,history,error_weight):',b'seed,history,error_weight,'+parameter+b'):')

    def test_host_return_and_control_output_field_rejected(self):
        self.rejected(b'return AcknowledgedCapture(pair,ack)',b'return self._host')
        self.rejected(b'    ack: CaptureAck',b'    ack: CaptureAck\n    owner: object')

    def test_yield_instead_of_finalized_output_rejected(self):
        self.rejected(b'return AcknowledgedCapture(pair,ack)',b'yield AcknowledgedCapture(pair,ack)')

    def test_wrong_preseal_token_rejected(self):
        self.rejected(b'seal_predecessor(pre_ticket,predecessor)',b'seal_predecessor(view,predecessor)')

    def test_wrong_ack_token_rejected(self):
        self.rejected(b'acknowledge(receipt_ticket,pair)',b'acknowledge(pre_ticket,pair)')

    def test_duplicate_read_alias_rejected(self):
        self.rejected(b'second=self._read(before)',b'second=first')

    def test_result_delivery_and_learning_calls_rejected(self):
        self.rejected(b'self._model.begin(packet.events,packet.receiving_id)',b'self._model.observe(packet.events,packet.receiving_id)')
        self.rejected(b'ack=self._bridge.acknowledge(receipt_ticket,pair)',b'ack=self._bridge.release(receipt_ticket,65,(0,1,0))')

    def test_raw_model_transfer_and_storage_call_rejected(self):
        self.rejected(b'state=_primitive(vars(self._model))',b'state=_primitive(self._model)')
        self.rejected(b'state=_primitive(vars(self._model))',b'state=open("blocked","w")')

    def test_cleanup_cannot_be_skipped(self):
        self.rejected(b'            self._close_control()',b'            pass')
        self.rejected(b"if current.phase=='stable':self._host.close(self._owner,current.revision)",b"if current.phase=='stable':pass")
