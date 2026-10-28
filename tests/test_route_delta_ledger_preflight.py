"""Opaque stamp deltas and ownership tests; no actual learner values."""
from dataclasses import replace,FrozenInstanceError
from pathlib import Path
import json
import sys
import unittest
from sara_engine.evaluation.route_delta_ledger_preflight import (
    ADDRESSES,CoreView,Credit,DeltaProposal,DiagnosticProposal,DeltaLedger,changed_addresses,
    altered,fixture,fault_preflight,analyze,
)
from sara_engine.evaluation.route_interface_preflight import CaseKey,DiagnosticHandle,token
from sara_engine.evaluation.route_fixed_allocation_contract import canonical

BASE=Path(__file__).resolve().parents[1]/'data/processed/benchmark_fixtures'


class DeltaLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((BASE/'route_delta_ledger_preflight_selection_v1.json').read_text())
        cls.case=CaseKey(1201,'independent','changed','cumulative','uniform','local_sparse_reference','shuffled_outcome')

    def test_no_learner_import(self):
        for name in ('sparse_route_reactivation','route_memory_factorial_v4','sparse_factor_routing'):
            self.assertNotIn('sara_engine.evaluation.'+name,sys.modules)

    def test_complete_catalog(self):
        self.assertEqual(len(ADDRESSES),435)
        self.assertEqual(sum(a[1].startswith('mass_') for a in ADDRESSES),300)
        self.assertEqual(len(set(ADDRESSES)),435)

    def test_accepts_only_exact_36_changes(self):
        ledger,p=fixture(self.case,0);ledger.open(p.receipt,p.credits)
        result=ledger.accept(p)
        self.assertEqual(result['symbolic_changed_slots'],36)
        self.assertEqual(result['symbolic_unchanged_slots'],399)
        self.assertEqual(result['actual_learning_updates'],0)
        self.assertFalse(result['runtime_numeric_delta_tested'])
        self.assertIs(ledger.state.core,p.after)
        self.assertEqual(ledger.state.tick,1)
        self.assertIsNone(ledger.state.pending)

    def test_inactive_write_rejected_atomically(self):
        ledger,p=fixture(self.case,0);ledger.open(p.receipt,p.credits);before=ledger.state
        inactive=next(a for a in ADDRESSES if a not in changed_addresses(p.credits))
        bad=replace(p,after=altered(p.after,{inactive:token('bad')}))
        with self.assertRaises(ValueError):ledger.accept(bad)
        self.assertIs(ledger.state,before)
        ledger.accept(p)

    def test_no_op_active_field_rejected_atomically(self):
        ledger,p=fixture(self.case,0);ledger.open(p.receipt,p.credits);before=ledger.state
        address=next(iter(changed_addresses(p.credits)))
        bad=replace(p,after=altered(p.after,{address:dict(p.before.fields)[address]}))
        with self.assertRaises(ValueError):ledger.accept(bad)
        self.assertIs(ledger.state,before)

    def test_core_and_diagnostic_namespaces(self):
        ledger,p=fixture(self.case,0);core=ledger.state.core
        new=DiagnosticHandle(token('new-diag'),tuple(range(9,18)))
        ledger.accept_diagnostic(DiagnosticProposal(core,core,ledger.state.diagnostic,new))
        self.assertIs(ledger.state.core,core)
        self.assertIs(ledger.state.diagnostic,new)
        before=ledger.state
        with self.assertRaises(ValueError):ledger.accept_diagnostic(DiagnosticProposal(core,p.after,new,new))
        self.assertIs(ledger.state,before)

    def test_pinned_key_and_mass_credit(self):
        ledger,p=fixture(self.case,0);ledger.open(p.receipt,p.credits);before=ledger.state
        for changed in (replace(p.credits[0],key_slot=1),replace(p.credits[0],mass_side='mass_right')):
            with self.assertRaises(ValueError):ledger.accept(replace(p,credits=(changed,)+p.credits[1:]))
            self.assertIs(ledger.state,before)

    def test_views_are_immutable_and_numeric_payloads_refused(self):
        _,p=fixture(self.case,0)
        with self.assertRaises(FrozenInstanceError):p.before.fields=()
        with self.assertRaises(ValueError):replace(p.before,fields=list(p.before.fields))
        with self.assertRaises(ValueError):altered(p.after,{ADDRESSES[0]:1})
        a=ADDRESSES[0];bad_address=(False,a[1],a[2])
        with self.assertRaises(ValueError):replace(p.before,fields=((bad_address,p.before.fields[0][1]),)+p.before.fields[1:])

    def test_all_registered_faults_rejected_without_state_change(self):
        faults=fault_preflight()
        self.assertEqual([f['fault'] for f in faults],self.spec['faults'])
        self.assertEqual(len(faults),31)
        self.assertTrue(all(f['rejected'] and f['adapter_state_unchanged'] for f in faults))

    def test_all_boundary_fixtures_and_preserved_input(self):
        before=canonical(self.spec);result=analyze(self.spec)
        self.assertEqual(before,canonical(self.spec))
        self.assertEqual(result['summary']['symbolic_transactions'],144)
        self.assertEqual(result['summary']['symbolic_stamp_changes'],5184)
        self.assertEqual(result['status']['actual_learning_updates'],0)
        self.assertFalse(result['status']['runtime_scalar_equality_tested'])

    def test_bounds_and_fixture_tuning_rejected(self):
        for section,key,value in [('bounds','actual_learning_updates',1),('fixtures','ticks',[0,64]),('fixtures','transactions',1)]:
            spec=json.loads(canonical(self.spec));spec[section][key]=value
            with self.assertRaises(ValueError):analyze(spec)


if __name__=='__main__':unittest.main()
