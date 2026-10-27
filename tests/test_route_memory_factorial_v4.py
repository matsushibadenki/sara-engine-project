"""Development-only checks for the separately registered local factorial model."""
from copy import deepcopy
import unittest
from sara_engine.evaluation.route_memory_factorial_v4 import (
    FactorialMemory, CELLS, public_weight, snapshot, train, recovery_pass, evaluate_cell,
)
from sara_engine.evaluation.sparse_route_reactivation import (
    ReactivationMemory, rows, train as original_train, state_digest,
)
from sara_engine.evaluation.sparse_factor_routing import ROUTES


class FactorialTests(unittest.TestCase):
    def test_factor_validation(self):
        with self.assertRaises(ValueError):
            FactorialMemory(81001, 'reset', 'uniform')

    def test_public_time_boundaries(self):
        self.assertEqual([public_weight(t) for t in (0,63,64,127,128,1023)], [1,1,2,2,4,32768])

    def test_count_prior_and_absence(self):
        for history, error in CELLS:
            model = FactorialMemory(81001, history, error)
            events = (0,)*5
            self.assertIsNone(model.probability(0, events))
            key = tuple(events[j] for j in ROUTES[0][0])
            model.tables[0][key] = [3,1]
            model.steps = 64
            expected = 3/8 if history == 'discounted' else 2/6
            self.assertEqual(model.probability(0, events), expected)
            model.steps = 1024
            p = model.probability(0,events)
            model.steps = 1023
            self.assertEqual(p,model.probability(0,events))

    def test_delay_and_rejected_receipts_are_atomic(self):
        model = FactorialMemory(81001, 'discounted','uniform')
        model.begin((0,)*5,0)
        before = state_digest(model)
        for bad in (0,2,True):
            with self.assertRaises(ValueError):model.observe((0,)*3,bad)
            self.assertEqual(before,state_digest(model))
        with self.assertRaises(ValueError):model.begin((0,)*5,1)
        self.assertEqual(before,state_digest(model))
        model.observe((0,)*3,1)
        self.assertEqual(model.updates,9)

    def test_count_and_error_factors_are_separable(self):
        models = [FactorialMemory(81001,h,e) for h,e in CELLS]
        # Identical synthetic receipt state at a public weight boundary.
        for m in models:
            m.steps=64
            m.clock=63
            m.begin((0,)*5,64)
            m.observe((1,)*3,65)
        self.assertEqual(models[0].tables,models[1].tables)
        self.assertEqual(models[2].tables,models[3].tables)
        self.assertEqual(models[0].errors,models[2].errors)
        self.assertEqual(models[1].errors,models[3].errors)
        self.assertEqual(sum(models[1].weights),18)
        self.assertEqual(sum(models[0].weights),9)
        for m in models:
            self.assertEqual(m.updates,9)
            self.assertEqual(sum(m.observations),9)

    def test_snapshot_is_read_only_and_exhaustive(self):
        m=FactorialMemory(81001,'discounted','uniform')
        m.begin((0,)*5,0);m.observe((0,)*3,1)
        before=state_digest(m)
        report=snapshot(m,(0,1,2))
        self.assertEqual(before,state_digest(m))
        self.assertEqual(len(report['routes']),45)
        self.assertEqual(sum(len(r['keys']) for r in report['routes']),150)
        self.assertEqual(sum(r['true_dependency'] for r in report['routes']),3)
        for slot in range(3):
            self.assertEqual(sorted(r['rank'] for r in report['routes'] if ROUTES[r['route']][1]==slot),list(range(1,16)))

    def test_no_capture_and_capture_numerically_identical(self):
        samples=rows(81002,'interacting',True)
        for h,e in CELLS:
            a=FactorialMemory(81002,h,e);b=deepcopy(a)
            stats,snapshots=train(a,samples,(0,1,2),True)
            other,empty=train(b,samples,(0,1,2),False)
            self.assertEqual(stats,other)
            self.assertEqual(state_digest(a),state_digest(b))
            self.assertEqual(len(snapshots),4)
            self.assertEqual(empty,[])
            self.assertEqual(stats['route_updates'],9216)
            self.assertEqual(stats['actual_learning_field_changes'],36864)
            frozen=rows(81002,'interacting',True,256,True)
            self.assertTrue(evaluate_cell(a,frozen)['scalar_reference_equal'])

    def test_original_baseline_development_numerical_identity(self):
        samples=rows(81001,'independent',True)
        old=ReactivationMemory('error_triggered',81001)
        new=FactorialMemory(81001,'cumulative','public_exponential')
        stats,_=train(new,samples,(0,1,2))
        self.assertEqual(stats,original_train(old,samples,[r.outcome for r in samples]))
        for k,v in old.__dict__.items():self.assertEqual(v,new.__dict__[k])

    def test_missing_recovery_is_failure(self):
        self.assertFalse(recovery_pass(None))
        self.assertTrue(recovery_pass(256))
        self.assertFalse(recovery_pass(320))


if __name__ == '__main__':unittest.main()
