"""Two-phase ordering and immutable witness faults; no actual outcomes."""
from dataclasses import fields,replace
from pathlib import Path
import json
import sys
import unittest
from sara_engine.evaluation.route_two_phase_preflight import (
    PreLookup,PreOutcome,DelayedClass,TwoPhaseGate,frame_for,fixture,
    MANUAL_ROWS,altered,fault_preflight,analyze,
)
from sara_engine.evaluation.route_fixed_allocation_contract import canonical

BASE=Path(__file__).resolve().parents[1]/'data/processed/benchmark_fixtures'


class TwoPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.spec=json.loads((BASE/'route_two_phase_preflight_selection_v1.json').read_text())

    def example(self):
        args=fixture(MANUAL_ROWS[0],'local_sparse_reference','shuffled_outcome')
        pre,result=frame_for(args)
        return args,pre,result,TwoPhaseGate(pre.receipt.case)

    def test_no_learner_import(self):
        for name in ('sparse_route_reactivation','route_memory_factorial_v4','sparse_factor_routing'):
            self.assertNotIn('sara_engine.evaluation.'+name,sys.modules)

    def test_pre_fields_exclude_class_and_error(self):
        self.assertEqual([f.name for f in fields(PreOutcome)],['receipt','before','lookups'])
        self.assertEqual([f.name for f in fields(PreLookup)],['route','key_slot','allocated','probability'])
        self.assertEqual([f.name for f in fields(DelayedClass)],['delivery','output_sides'])

    def test_success_consumes_exact_own_seal_once(self):
        args,pre,result,gate=self.example();p,b,w,*_=args
        cap=gate.seal(pre);reply=gate.finish(cap,result,p,w)
        self.assertEqual(reply['route_bindings'],9)
        self.assertEqual(reply['actual_learning_updates'],0)
        self.assertFalse(reply['runtime_causal_provenance_authenticated'])
        self.assertIsNone(gate.state.pending)
        before=gate.state
        with self.assertRaises(ValueError):gate.finish(cap,result,p,w)
        self.assertIs(gate.state,before)

    def test_result_cannot_precede_seal(self):
        args,pre,result,gate=self.example();p,b,w,*_=args;before=gate.state
        with self.assertRaises(ValueError):gate.finish(None,result,p,w)
        self.assertIs(gate.state,before)

    def test_copied_and_foreign_capabilities_rejected(self):
        args,pre,result,gate=self.example();p,b,w,*_=args;cap=gate.seal(pre)
        other=TwoPhaseGate(pre.receipt.case);other_cap=other.seal(pre)
        before=gate.state
        for invalid in (replace(cap),other_cap):
            with self.assertRaises(ValueError):gate.finish(invalid,result,p,w)
            self.assertIs(gate.state,before)
        gate.finish(cap,result,p,w)

    def test_replacement_failure_does_not_consume_pending(self):
        args,pre,result,gate=self.example();p,b,w,*_=args;cap=gate.seal(pre);before=gate.state
        first=pre.lookups[0];bad=altered(b,{(first.route,'mass_left',first.key_slot):3})
        with self.assertRaises(ValueError):gate.finish(cap,result,replace(p,before=bad),w)
        self.assertIs(gate.state,before)
        gate.finish(cap,result,p,w)

    def test_invalid_prelookup_cannot_open_pending(self):
        args,pre,result,gate=self.example();before=gate.state
        bad=replace(pre,lookups=(replace(pre.lookups[0],probability=.75),)+pre.lookups[1:])
        with self.assertRaises(ValueError):gate.seal(bad)
        self.assertIs(gate.state,before)

    def test_absent_and_allocated_zero_preseal(self):
        args,pre,result,gate=self.example();first=pre.lookups[0];r=first.route;k=first.key_slot
        before=altered(pre.before,{(r,'mass_left',k):0,(r,'mass_right',k):0})
        for allocated,p in ((False,None),(True,.5)):
            packet=replace(pre,before=before,lookups=(replace(first,allocated=allocated,probability=p),)+pre.lookups[1:])
            TwoPhaseGate(pre.receipt.case).seal(packet)

    def test_all_registered_faults(self):
        faults=fault_preflight()
        self.assertEqual([f['fault'] for f in faults],self.spec['faults'])
        self.assertEqual(len(faults),26)
        self.assertTrue(all(f['rejected'] and f['state_unchanged'] for f in faults))

    def test_complete_finite_coverage_and_input_preservation(self):
        before=canonical(self.spec);result=analyze(self.spec)
        self.assertEqual(canonical(self.spec),before)
        self.assertEqual(result['summary']['finished_fixture_transactions'],72)
        self.assertEqual(result['summary']['route_bindings'],648)
        self.assertEqual(result['summary']['pre_only_records'],11)
        self.assertEqual(result['status']['actual_learning_updates'],0)
        self.assertFalse(result['status']['execution_authorized'])

    def test_budget_change_rejected(self):
        spec=json.loads(canonical(self.spec));spec['bounds']['actual_learning_updates']=1
        with self.assertRaises(ValueError):analyze(spec)


if __name__=='__main__':unittest.main()
