"""Model-free host ownership, generation and exclusion tests."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_owner_host import OwnerHost, Owner, HostError, HostBusy
from sara_engine.evaluation.route_owner_host_preflight import analyze, enter, thread_refusal
from sara_engine.utils.project_paths import processed_data_path


class OwnerHostTests(unittest.TestCase):
    def setUp(self):
        self.host,self.owner = OwnerHost.create('test-owner','test-epoch')

    def test_registered_scenarios_without_runtime_import(self):
        base = Path(processed_data_path('benchmark_fixtures'))
        spec = json.loads((base/'route_owner_host_selection_v1.json').read_bytes())
        reports = {n:(base/n).read_bytes() for n in spec['reports']}
        before = set(sys.modules)
        result = analyze(spec,reports)
        self.assertEqual(set(sys.modules),before)
        self.assertEqual(result['scenario_count'],21)
        self.assertEqual(result['thread_exclusion_checks'],3)
        self.assertFalse(result['capture_unblocked'])
        self.assertFalse(result['all_model_writer_paths_covered'])

    def test_strict_revision_and_owner_without_state_change(self):
        for revision in (True,0.0,-1,2):
            with self.subTest(revision=revision), self.assertRaises(HostError):
                enter(self.host.mutation(self.owner,revision))
        with self.assertRaises(HostError): self.host.inspect(Owner('test-owner','test-epoch'))
        self.assertEqual(self.host.inspect(self.owner).revision,0)

    def test_snapshot_and_ticket_immutable(self):
        with self.host.paused(self.owner,0) as ticket:
            with self.assertRaises(FrozenInstanceError): ticket.view.revision = 9
            with self.assertRaises(HostError): self.host.revalidate(self.owner,replace(ticket))
        with self.assertRaises(HostError): self.host.revalidate(self.owner,ticket)

    def test_mutation_exception_poison_is_terminal_and_gate_released(self):
        with self.assertRaises(RuntimeError):
            with self.host.mutation(self.owner,0): raise RuntimeError('test')
        view = self.host.inspect(self.owner)
        self.assertEqual((view.phase,view.revision),('poisoned',1))
        for operation in (lambda:enter(self.host.mutation(self.owner,1)),lambda:enter(self.host.paused(self.owner,1)),lambda:self.host.close(self.owner,1)):
            with self.assertRaises(HostError): operation()

    def test_pause_exception_does_not_advance_or_poison(self):
        with self.assertRaises(RuntimeError):
            with self.host.paused(self.owner,0): raise RuntimeError('test')
        self.assertEqual(self.host.inspect(self.owner).phase,'stable')
        self.assertEqual(self.host.inspect(self.owner).revision,0)

    def test_exclusion_is_between_different_threads(self):
        with self.host.paused(self.owner,0) as ticket:
            thread_refusal(lambda:enter(self.host.mutation(self.owner,0)),HostBusy,5)
            thread_refusal(lambda:self.host.revalidate(self.owner,ticket),HostError,5)
        with self.host.mutation(self.owner,0): pass
        self.assertEqual(self.host.inspect(self.owner).revision,2)

    def test_close_is_terminal(self):
        self.host.close(self.owner,0)
        with self.assertRaises(HostError): enter(self.host.paused(self.owner,0))
        with self.assertRaises(HostError): enter(self.host.mutation(self.owner,0))
        self.assertEqual(self.host.inspect(self.owner).phase,'closed')

    def test_model_callback_and_invalid_labels_not_accepted(self):
        with self.assertRaises(TypeError): OwnerHost.create('owner','epoch',model=object())
        for label in ('',' spaced ',3):
            with self.assertRaises(HostError): OwnerHost.create(label,'epoch')

    def test_same_labels_do_not_transfer_host_capability(self):
        other,owner = OwnerHost.create('test-owner','test-epoch')
        with self.assertRaises(HostError): other.inspect(self.owner)
        self.assertEqual(other.inspect(owner).revision,0)
