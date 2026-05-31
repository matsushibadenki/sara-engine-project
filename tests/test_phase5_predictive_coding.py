import importlib.util
import os


def _load_script(script_name: str):
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", script_name)
    )
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase5_predictive_coding_benchmark_passes_required_metrics() -> None:
    module = _load_script("phase5_predictive_coding_benchmark.py")

    report = module.run_phase5_predictive_coding_benchmark()

    assert report["suite_name"] == "Phase5PredictiveCodingBenchmark"
    assert report["passed"] is True
    assert report["overall_score"] == 1.0
    assert all(report["threshold_results"].values())
    assert report["details"]["primary_transition"]["trace_complete"] is True
    assert report["details"]["multi_step_trace"]["trace_complete"] is True
    assert report["metrics"]["multi_step_latent_chain_integrity"] == 1.0
    assert report["metrics"]["long_horizon_error_correction_convergence"] == 1.0
    assert report["metrics"]["horizon_bucket_stability"] == 1.0
    assert report["metrics"]["macro_action_effectiveness"] == 1.0
    assert report["metrics"]["subgoal_decomposition_integrity"] == 1.0
    assert report["metrics"]["depth_selective_routing_integrity"] == 1.0
    assert report["metrics"]["micro_es_policy_refinement_integrity"] == 1.0
    assert report["metrics"]["manifold_transition_locality"] == 1.0
    assert report["metrics"]["manifold_rollout_stability"] == 1.0
    assert report["metrics"]["causal_route_sparsity"] == 1.0
    assert report["metrics"]["withheld_trajectory_recall"] == 1.0
    assert report["metrics"]["manifold_trajectory_case_coverage"] == 1.0
    assert report["metrics"]["manifold_average_case_recall"] == 1.0
    assert report["metrics"]["manifold_scan_budget_integrity"] == 1.0
    assert report["metrics"]["manifold_indexed_candidate_integrity"] == 1.0
    assert report["metrics"]["manifold_index_scan_reduction"] == 1.0
    assert report["metrics"]["manifold_candidate_miss_guard"] == 1.0
    assert report["details"]["branch_comparison"]["separable"] is True
    assert report["details"]["horizon_buckets"]["short"]["success_ratio"] == 1.0
    assert report["details"]["horizon_buckets"]["medium"]["success_ratio"] == 1.0
    assert report["details"]["horizon_buckets"]["long"]["success_ratio"] == 1.0
    assert report["details"]["macro_step_reduction"] >= 2.0
    assert report["details"]["macro_cost_reduction"] >= 0.30
    assert report["details"]["subgoal_coverage_ratio"] == 1.0
    assert report["details"]["depth_route_avg_selected_ratio"] <= 0.80
    assert report["details"]["depth_route_max_weight_sum_deviation"] <= 0.05
    micro_es = report["details"]["micro_es_refinement"]
    assert micro_es["strategy"] == "energy_aware_micro_es_low_rank_rank1"
    assert micro_es["low_rank_trace_complete"] is True
    assert micro_es["fitness_improvement"] > 0.05
    assert micro_es["event_cost_reduction"] >= 0.04
    assert micro_es["population_event_cost_proxy"] <= micro_es["event_budget"]
    manifold = report["details"]["manifold_transition_memory"]
    assert manifold["strategy"] == "local_manifold_transition_memory_observed_only"
    assert manifold["observed_only"] is True
    assert manifold["nearest_trajectories"][0]["overlap"] >= 0.95
    assert manifold["withheld_trajectory_recall_ratio"] >= 0.80
    assert len(manifold["causal_edges_used"]) <= manifold["causal_route_budget"]
    assert manifold["trajectory_case_count"] == 3
    assert manifold["trajectory_top_match_ratio"] == 1.0
    assert manifold["average_case_recall"] >= 0.80
    assert manifold["max_scanned_trajectory_count"] <= manifold["scan_budget"]
    assert manifold["indexed_candidate_case_ratio"] == 1.0
    assert manifold["indexed_scan_reduction_ratio"] > 0.0
    assert manifold["candidate_miss_count"] == 0
    assert manifold["candidate_miss"] is False
    assert all(case["top_match"] for case in manifold["case_results"])
    assert all(case["sparse_route_ok"] for case in manifold["case_results"])
    assert all(case["scan_budget_ok"] for case in manifold["case_results"])
    assert all(case["indexed_candidate_ok"] for case in manifold["case_results"])


def test_phase5_entry_gate_accepts_valid_report_and_rejects_metric_drop() -> None:
    benchmark = _load_script("phase5_predictive_coding_benchmark.py")
    gate = _load_script("phase5_entry_gate.py")
    report = benchmark.run_phase5_predictive_coding_benchmark()

    assert gate.validate_phase5_entry(report) == []

    broken = dict(report)
    broken["metrics"] = dict(report["metrics"])
    broken["threshold_results"] = dict(report["threshold_results"])
    broken["metrics"]["correction_event_coverage"] = 0.0
    broken["threshold_results"]["correction_event_coverage"] = False

    errors = gate.validate_phase5_entry(broken)

    assert any("correction_event_coverage" in error for error in errors)

    broken_horizon = dict(report)
    broken_horizon["metrics"] = dict(report["metrics"])
    broken_horizon["threshold_results"] = dict(report["threshold_results"])
    broken_horizon["metrics"]["horizon_bucket_stability"] = 0.0
    broken_horizon["threshold_results"]["horizon_bucket_stability"] = False
    horizon_errors = gate.validate_phase5_entry(broken_horizon)
    assert any("horizon_bucket_stability" in error for error in horizon_errors)


def test_phase5_entry_gate_report_exposes_structured_checks() -> None:
    benchmark = _load_script("phase5_predictive_coding_benchmark.py")
    gate = _load_script("phase5_entry_gate.py")
    report = benchmark.run_phase5_predictive_coding_benchmark()

    gate_report = gate.build_phase5_entry_gate_report(report)
    summary = gate.format_phase5_entry_gate_summary(gate_report)

    assert gate_report["suite_name"] == "Phase5EntryGate"
    assert gate_report["passed"] is True
    assert gate_report["failed_checks"] == []
    assert gate_report["checks"]["metric.latent_transition_alignment"]["passed"] is True
    assert gate_report["checks"]["counterfactual_branch_separable"]["passed"] is True
    assert gate_report["checks"]["multi_step_trace_complete"]["passed"] is True
    assert gate_report["checks"]["metric.multi_step_latent_chain_integrity"]["passed"] is True
    assert "- gate_status: PASS" in summary
    assert "- metric.correction_event_coverage: PASS" in summary


def test_phase5_entry_gate_report_marks_failed_metric_checks() -> None:
    benchmark = _load_script("phase5_predictive_coding_benchmark.py")
    gate = _load_script("phase5_entry_gate.py")
    report = benchmark.run_phase5_predictive_coding_benchmark()
    broken = dict(report)
    broken["metrics"] = dict(report["metrics"])
    broken["threshold_results"] = dict(report["threshold_results"])
    broken["metrics"]["counterfactual_transition_separation"] = 0.0
    broken["threshold_results"]["counterfactual_transition_separation"] = False

    gate_report = gate.build_phase5_entry_gate_report(broken)
    summary = gate.format_phase5_entry_gate_summary(gate_report)

    assert gate_report["passed"] is False
    assert "metric.counterfactual_transition_separation" in gate_report["failed_checks"]
    assert "threshold.counterfactual_transition_separation" in gate_report["failed_checks"]
    assert "- gate_status: FAIL" in summary
