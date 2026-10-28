"""Static address plans and receipt metadata, without learner execution."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import unittest
from sara_engine.evaluation.route_fixed_allocation_contract import (
    ROUTES,active_routes,fixed_readout,validate_trace,validate_contract,analyze,canonical,
)

BASE=Path(__file__).resolve().parents[1]/'data/processed/benchmark_fixtures'


class StaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.contract=json.loads((BASE/'route_fixed_allocation_contract_selection_v1.json').read_text())

    def test_no_learner_import_or_execution_hook(self):
        for name in ('sparse_route_reactivation','sparse_factor_routing','route_memory_factorial_v4'):
            self.assertNotIn('sara_engine.evaluation.'+name,sys.modules)
        self.assertEqual(list(inspect.signature(active_routes).parameters),['tick'])

    def test_finite_plan_and_no_learning_claim(self):
        result=analyze(self.contract)
        self.assertEqual(result['address_plan']['planned_route_updates'],9216)
        self.assertEqual(result['address_plan']['planned_changed_learning_fields'],36864)
        counts=result['address_plan']['per_route_receipts']
        self.assertEqual(counts.count(205),36)
        self.assertEqual(counts.count(204),9)
        self.assertEqual(result['status']['actual_learning_updates'],0)
        self.assertFalse(result['status']['execution_authorized'])
        self.assertFalse(result['finite_non_no_op_bound']['runtime_changes_measured'])

    def test_exhaustive_schedule_and_readout_balance(self):
        for tick in range(1024):
            active=active_routes(tick)
            self.assertEqual(len(set(active)),9)
            self.assertEqual([sum(ROUTES[i][1]==s for i in active) for s in range(3)],[3,3,3])
            self.assertEqual([ROUTES[i][1] for i in fixed_readout(tick)],[0,1,2])

    def test_no_oracle_schedule_inputs(self):
        for tick in (-1,1024,True,1.0):
            with self.assertRaises(ValueError):active_routes(tick)
        with self.assertRaises(TypeError):active_routes(1,label=0)

    def test_all_cells_and_engines_share_address_plan(self):
        plans=[tuple(active_routes(t) for t in range(1024)) for _ in range(8)]
        self.assertTrue(all(p==plans[0] for p in plans))

    def test_one_tick_delay_and_no_idle_dispatch(self):
        report=validate_trace([('receipt',0,0),('outcome',1,0),('receipt',1,1),('outcome',2,1)],2)
        self.assertEqual(report['event_dispatches'],4)
        self.assertEqual(report['idle_dispatches'],0)
        self.assertEqual(report['max_pending'],1)

    def test_invalid_delayed_metadata_rejected(self):
        invalid=[ [('receipt',0,0),('outcome',0,0)], [('receipt',0,0),('outcome',2,0)],
                 [('receipt',0,0),('outcome',1,1)], [('outcome',1,0),('receipt',0,0)],
                 [('receipt',0,0),('receipt',1,1)], [('receipt',True,0),('outcome',1,0)],
                 [('receipt',0,0),('unknown',1,0)] ]
        for events in invalid:
            with self.assertRaises(ValueError):validate_trace(events,1)
        with self.assertRaises(ValueError):validate_trace([('receipt',0,0)],1)

    def test_condition_and_gate_relaxation_rejected(self):
        mutations=[('scientific_conditions','change_at',500),('scientific_conditions','minimum_route_observations',8),
                   ('allocation','labels_or_predictions_allowed',True),('allocation','shared_input_drop_allowed',True),
                   ('budgets','actual_route_updates_per_future_cell',9200),('budgets','current_learner_executions',1),
                   ('delayed_outcomes','max_pending',2),('separation','primary_predictions_to_shadow',True)]
        for section,key,value in mutations:
            c=deepcopy(self.contract);c[section][key]=value
            with self.assertRaises(ValueError):validate_contract(c)
        c=deepcopy(self.contract);c['unchanged_gates']['changed']['recovery_max_ticks']=320
        with self.assertRaises(ValueError):validate_contract(c)

    def test_scalar_control_required(self):
        c=deepcopy(self.contract);c['engines']=['local_sparse_reference']
        with self.assertRaises(ValueError):validate_contract(c)

    def test_static_validation_preserves_contract(self):
        before=canonical(self.contract);analyze(self.contract)
        self.assertEqual(before,canonical(self.contract))

    def test_shared_input_catalog_and_support_boundary(self):
        self.assertEqual(sum(2**len(inputs) for inputs,_ in ROUTES),150)
        visits=[0]*45
        for tick in range(80):
            for i in active_routes(tick):visits[i]+=1
        self.assertEqual(set(visits),{16})
        self.assertEqual(analyze(self.contract)['address_plan']['first_all_routes_support_16_tick'],80)


if __name__=='__main__':unittest.main()
