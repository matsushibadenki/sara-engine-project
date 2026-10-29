"""Saved-report-only evidence scopes, missing facts and no promotion."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import sys
import unittest
from sara_engine.evaluation.route_readiness_audit import INPUTS,analyze,canonical

BASE=Path(__file__).resolve().parents[1]/'data/processed/benchmark_fixtures'


class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((BASE/'route_readiness_audit_selection_v1.json').read_text())
        cls.reports={name:json.loads((BASE/name).read_text()) for name in INPUTS}
        cls.hashes={name:sha256((BASE/name).read_bytes()).hexdigest() for name in INPUTS}

    def test_imports_no_models_calculators_or_fixture_executors(self):
        for name in ('sparse_route_reactivation','route_memory_factorial_v4','route_two_phase_preflight',
                     'route_numeric_delta_preflight','route_error_binding_preflight','sparse_factor_routing'):
            self.assertNotIn('sara_engine.evaluation.'+name,sys.modules)

    def test_evidence_scope_and_minimum_next_check(self):
        result=analyze(self.spec,self.reports,self.hashes)
        self.assertEqual(result['selected_next_check']['id'],'independent_stateless_scalar_fixture_check')
        self.assertFalse(result['selected_next_check']['execution_performed'])
        self.assertTrue(result['selected_next_check']['requires_new_registration'])
        self.assertEqual(sum(c['selected'] for c in result['alternatives']),1)
        for key in ('runtime_provenance','independent_route_scalar_computation','snn_specific_causal_evidence'):
            self.assertIsNone(result['evidence_categories'][key]['established'])
        self.assertIsNone(result['evidence_categories']['independent_task']['qualified'])
        self.assertTrue(all(result['holds'].values()))

    def test_unknowns_and_historical_scalar_scopes_retained(self):
        result=analyze(self.spec,self.reports,self.hashes)
        rows=result['evidence_categories']['independent_route_scalar_computation']['historical_other_component_equivalence']
        self.assertTrue(any(r['scalar_equivalence'] is True for r in rows))
        self.assertTrue(any(r['scalar_equivalence'] is None for r in rows))
        self.assertTrue(any(r['pointer']=='/numeric_trace_equality' and r['value'] is None for r in result['retained_failures_and_unknowns']))

    def test_fixture_to_runtime_promotion_rejected(self):
        for name,key in [('route_two_phase_preflight_result_v1.json','runtime_causal_provenance_authenticated'),
                         ('route_interface_preflight_result_v1.json','runtime_scalar_equality_tested')]:
            reports=deepcopy(self.reports);reports[name]['status'][key]=True
            with self.assertRaises(ValueError):analyze(self.spec,reports,self.hashes)

    def test_original_failure_or_scientific_negative_reclassification_rejected(self):
        reports=deepcopy(self.reports);reports['sparse_route_reactivation_failure_v1.json']['numeric_trace_equality']=True
        with self.assertRaises(ValueError):analyze(self.spec,reports,self.hashes)
        reports=deepcopy(self.reports);reports['route_memory_factorial_v4_primary_v1.json']['summary']['cells']['discounted__public_exponential']['changed_recovery_within_256_passes']=10
        with self.assertRaises(ValueError):analyze(self.spec,reports,self.hashes)

    def test_missing_required_scope_never_defaults_to_success(self):
        reports=deepcopy(self.reports);del reports['route_two_phase_preflight_result_v1.json']['status']['runtime_causal_provenance_authenticated']
        with self.assertRaises(ValueError):analyze(self.spec,reports,self.hashes)
        reports=deepcopy(self.reports);reports.pop(INPUTS[0])
        with self.assertRaises(ValueError):analyze(self.spec,reports,self.hashes)

    def test_source_mechanism_resource_holds_not_promoted(self):
        for change in ('task','mechanism','resource'):
            reports=deepcopy(self.reports);r=reports['r2_mechanism_decision_result_v1.json']
            if change=='task':r['independent_task']['selected_source']='invented-ready-source'
            elif change=='mechanism':next(m for m in r['mechanism_candidates'] if m['id']=='branch_local_delayed_eligibility')['experiment_authorized']=True
            else:r['new_resource_measurement_authorized']=True
            with self.assertRaises(ValueError):analyze(self.spec,reports,self.hashes)

    def test_unchanged_gate_and_control_channel(self):
        reports=deepcopy(self.reports);reports['route_fixed_control_contract_selection_v1.json']['original_gates']['changed']['recovery_max_ticks']=320
        with self.assertRaises(ValueError):analyze(self.spec,reports,self.hashes)
        reports=deepcopy(self.reports);reports['route_fixed_control_contract_result_v1.json']['endpoint_matrix'][0]['original_adaptive_pass']=True
        with self.assertRaises(ValueError):analyze(self.spec,reports,self.hashes)

    def test_numeric_proof_and_case_roster_bound(self):
        hashes=dict(self.hashes);hashes['route_memory_factorial_v4_primary_v1.json']='0'*64
        with self.assertRaises(ValueError):analyze(self.spec,self.reports,hashes)
        reports=deepcopy(self.reports);reports['route_memory_factorial_v4_primary_v1.json']['cases'][0]['seed']=0
        with self.assertRaises(ValueError):analyze(self.spec,reports,self.hashes)

    def test_read_only_inputs_and_zero_new_work(self):
        before=canonical([self.spec,self.reports,self.hashes]);result=analyze(self.spec,self.reports,self.hashes)
        self.assertEqual(before,canonical([self.spec,self.reports,self.hashes]))
        for key in ('new_learner_executions','new_prediction_scores','new_acquisitions','new_measurements','fixture_executions','stateless_scalar_executions'):
            self.assertEqual(result['status'][key],0)
        self.assertFalse(result['status']['training_authorized'])


if __name__=='__main__':unittest.main()
