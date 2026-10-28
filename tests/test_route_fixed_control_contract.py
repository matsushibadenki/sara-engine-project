"""Metadata-only shuffled maps and non-promoting gate applicability."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import unittest
from sara_engine.evaluation.route_fixed_control_contract import (
    SEEDS,PRIMARY_STATUS,CHANNELS,permutation_indices,validate_permutation,
    endpoint_record,validate_endpoint_record,validate_contract,analyze,
)
from sara_engine.evaluation.route_fixed_allocation_contract import canonical

BASE=Path(__file__).resolve().parents[1]/'data/processed/benchmark_fixtures'


class FixedControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((BASE/'route_fixed_control_contract_selection_v1.json').read_text())
        cls.parent=json.loads((BASE/'route_fixed_allocation_contract_selection_v1.json').read_text())
        cls.saved=json.loads((BASE/'route_fixed_allocation_contract_result_v1.json').read_text())

    def test_no_learner_import(self):
        for name in ('sparse_route_reactivation','route_memory_factorial_v4','sparse_factor_routing'):
            self.assertNotIn('sara_engine.evaluation.'+name,sys.modules)

    def test_all_maps_preserve_phases_and_global_rng(self):
        before=random.getstate()
        for seed in SEEDS:
            indices=permutation_indices(seed)
            validate_permutation(indices)
            self.assertEqual(indices,permutation_indices(seed))
            self.assertEqual(sorted(indices[:512]),list(range(512)))
            self.assertEqual(sorted(indices[512:]),list(range(512,1024)))
        self.assertEqual(before,random.getstate())

    def test_fixed_points_are_not_rejected(self):
        self.assertEqual(validate_permutation(list(range(1024)))['fixed_points'],1024)

    def test_bad_maps_rejected(self):
        duplicate=list(range(1024));duplicate[0]=1
        crossing=list(range(1024));crossing[0],crossing[512]=crossing[512],crossing[0]
        boolean=list(range(1024));boolean[0]=False
        for indices in (duplicate,crossing,boolean,list(range(1023))):
            with self.assertRaises(ValueError):validate_permutation(indices)
        with self.assertRaises(ValueError):permutation_indices(True)
        with self.assertRaises(ValueError):permutation_indices(81001)

    def test_complete_control_and_endpoint_scope(self):
        result=analyze(self.spec,self.parent,self.saved)
        self.assertEqual(len(result['planned_cells_per_case']),24)
        self.assertEqual(len(result['endpoint_matrix']),30)
        self.assertEqual(len({c['training_address_sha256'] for c in result['planned_cells_per_case']}),1)
        self.assertEqual(result['prospective_budget']['models_per_phase'],480)
        self.assertEqual(result['prospective_budget']['actual_current_model_runs'],0)
        self.assertFalse(result['status']['original_adaptive_component_passed'])

    def test_all_channels_unscored_and_non_promoting(self):
        for gate in PRIMARY_STATUS:
            for channel in CHANNELS:
                record=endpoint_record(gate,channel)
                self.assertTrue(validate_endpoint_record(record))
                self.assertIsNone(record['value'])
                self.assertFalse(record['original_adaptive_pass'])
                if channel=='original_adaptive_system':self.assertEqual(record['status'],'untested')

    def test_promotion_metric_and_cross_channel_substitution_rejected(self):
        for field,value in [('original_adaptive_pass',True),('value',1.0),('status','passed'),('roles',['candidate','shuffled_outcome'])]:
            record=endpoint_record('gain_over_permanent','fixed_readout');record[field]=value
            with self.assertRaises(ValueError):validate_endpoint_record(record)
        record=endpoint_record('gain_over_permanent','fixed_readout');record['channel']='shadow_selection'
        with self.assertRaises(ValueError):validate_endpoint_record(record)

    def test_primary_zero_gain_is_not_acceptance(self):
        result=analyze(self.spec,self.parent,self.saved)
        identities=result['structural_identifiability']
        self.assertEqual(identities['primary_gain_over_permanent'],0)
        self.assertLess(identities['primary_gain_over_permanent'],identities['original_gain_requirement'])
        self.assertTrue(identities['original_gain_cannot_be_established_by_primary'])
        self.assertFalse(result['status']['original_adaptive_gates_evaluated'])

    def test_no_parent_condition_or_gate_tuning(self):
        for section,key,value in [('conditions','change_at',500),('budgets','future_models_per_phase',160),
                                  ('permanent','primary_candidate_identity',False),('shuffle','seed_offset',0)]:
            spec=deepcopy(self.spec);spec[section][key]=value
            with self.assertRaises(ValueError):validate_contract(spec,self.parent)
        spec=deepcopy(self.spec);spec['original_gates']['changed']['recovery_max_ticks']=320
        with self.assertRaises(ValueError):validate_contract(spec,self.parent)

    def test_missing_controls_and_wrong_comparators_rejected(self):
        spec=deepcopy(self.spec);spec['roles'].remove('permanent')
        with self.assertRaises(ValueError):validate_contract(spec,self.parent)
        spec=deepcopy(self.spec);spec['comparators']['gain_over_permanent']=['candidate','shuffled_outcome']
        with self.assertRaises(ValueError):validate_contract(spec,self.parent)
        spec=deepcopy(self.spec);spec['endpoint_policy']['tail_accuracy']['original_adaptive_system']='passed'
        with self.assertRaises(ValueError):validate_contract(spec,self.parent)

    def test_preserved_parent_proof_and_input_immutability(self):
        before=canonical([self.spec,self.parent,self.saved]);analyze(self.spec,self.parent,self.saved)
        self.assertEqual(before,canonical([self.spec,self.parent,self.saved]))
        saved=deepcopy(self.saved);saved['address_plan']['planned_route_updates']=0
        with self.assertRaises(ValueError):analyze(self.spec,self.parent,saved)


if __name__=='__main__':unittest.main()
