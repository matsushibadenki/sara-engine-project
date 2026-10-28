"""Interface/fault tests over addresses and opaque digests, never model payloads."""
from dataclasses import replace,FrozenInstanceError,fields
from pathlib import Path
import json
import sys
import unittest
from sara_engine.evaluation.route_interface_preflight import (
    CaseKey,Receipt,Delivery,ReceiptGate,CoreHandle,DiagnosticHandle,StateBundle,ComparisonPacket,
    make_receipt,replace_diagnostic,compare_same_role,fixture_pair,token,source_map,fault_preflight,analyze,
)
from sara_engine.evaluation.route_fixed_allocation_contract import canonical

BASE=Path(__file__).resolve().parents[1]/'data/processed/benchmark_fixtures'


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((BASE/'route_interface_preflight_selection_v1.json').read_text())
        cls.case=CaseKey(1201,'independent','changed','cumulative','uniform','local_sparse_reference','shuffled_outcome')

    def test_no_learner_or_payload_interface(self):
        for name in ('sparse_route_reactivation','route_memory_factorial_v4','sparse_factor_routing'):
            self.assertNotIn('sara_engine.evaluation.'+name,sys.modules)
        self.assertEqual({f.name for f in fields(Delivery)},{'case','receipt_id','source_index','at','purpose'})

    def test_source_not_receipt_and_one_tick_delivery(self):
        g=ReceiptGate(self.case);p=make_receipt(self.case,0)
        self.assertNotEqual(p.source_index,p.receipt_id)
        g.open(p);before=g.snapshot()
        with self.assertRaises(ValueError):g.deliver(Delivery(self.case,p.source_index,p.source_index,1))
        self.assertEqual(g.snapshot(),before)
        g.deliver(Delivery(self.case,0,p.source_index,1))
        self.assertEqual(g.snapshot(),(1,None))

    def test_all_role_source_rules(self):
        for role in ('candidate','permanent','shuffled_outcome'):
            case=replace(self.case,role=role)
            for tick in (0,511,512,1023):
                p=make_receipt(case,tick)
                self.assertEqual(p.source_index,source_map(case.seed)[tick] if role=='shuffled_outcome' else tick)

    def test_evaluation_and_foreign_delivery_fail_before_mutation(self):
        p=make_receipt(self.case,0);g=ReceiptGate(self.case);g.open(p)
        before=g.snapshot()
        invalid=[Delivery(self.case,0,p.source_index,1,'evaluation'),
                 Delivery(replace(self.case,role='candidate'),0,p.source_index,1),
                 Delivery(replace(self.case,engine='independent_event_driven_scalar'),0,p.source_index,1)]
        for d in invalid:
            with self.assertRaises(ValueError):g.deliver(d)
            self.assertEqual(g.snapshot(),before)

    def test_failed_address_change_atomic(self):
        g=ReceiptGate(self.case);p=make_receipt(self.case,0)
        for bad in (replace(p,active=p.active[:-1]),replace(p,readout=tuple(reversed(p.readout)))):
            with self.assertRaises(ValueError):g.open(bad)
            self.assertEqual(g.snapshot(),(0,None))

    def test_immutable_state_and_diagnostic_replacement(self):
        core=CoreHandle(token('test-core'));diagnostic=DiagnosticHandle(token('test-diag'),tuple(range(9)))
        bundle=StateBundle(core,diagnostic)
        updated=replace_diagnostic(bundle,DiagnosticHandle(token('new'),(9,10,11)))
        self.assertIs(updated.core,core)
        self.assertIs(bundle.diagnostic,diagnostic)
        with self.assertRaises(FrozenInstanceError):core.digest=token('bad')
        with self.assertRaises(ValueError):replace_diagnostic(bundle,core)
        with self.assertRaises(ValueError):DiagnosticHandle(token('bad'),[0,1,2])

    def test_same_role_fixture_only_comparison(self):
        left,right=fixture_pair(self.case)
        result=compare_same_role(left,right)
        self.assertTrue(result['equal'])
        self.assertEqual(result['scope'],'metadata_fixture')
        self.assertFalse(result['runtime_scalar_equality_tested'])
        self.assertFalse(result['original_adaptive_pass'])
        with self.assertRaises(ValueError):compare_same_role(left,replace(right,case=replace(right.case,role='candidate')))
        with self.assertRaises(ValueError):replace(right,scope='runtime_scalar_equality')

    def test_all_registered_faults_rejected_with_state_unchanged(self):
        faults=fault_preflight()
        self.assertEqual([f['fault'] for f in faults],self.spec['faults'])
        self.assertEqual(len(faults),31)
        self.assertTrue(all(f['rejected'] and f['state_unchanged'] for f in faults))

    def test_full_metadata_coverage_without_learning(self):
        before=canonical(self.spec);result=analyze(self.spec)
        self.assertEqual(before,canonical(self.spec))
        self.assertEqual(result['summary']['receipt_metadata_pairs'],15360)
        self.assertEqual(result['summary']['comparison_fixture_pairs'],240)
        self.assertEqual(result['status']['actual_learning_updates'],0)
        self.assertFalse(result['status']['execution_authorized'])

    def test_corrupt_fault_roster_and_budget_rejected(self):
        bad=json.loads(canonical(self.spec));bad['bounds']['learner_executions']=1
        with self.assertRaises(ValueError):analyze(bad)
        bad=json.loads(canonical(self.spec));bad['faults']=bad['faults'][:-1]
        with self.assertRaises(ValueError):analyze(bad)


if __name__=='__main__':unittest.main()
