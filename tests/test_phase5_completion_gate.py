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


def _phase4_pass_report() -> dict:
    return {
        "evaluator_name": "Phase4ScaleContinualBenchmark",
        "passed": True,
        "metrics": {
            "structural_plasticity_stability": 1.0,
            "hippocampal_transfer_integrity": 1.0,
            "scale_out_retention_integrity": 1.0,
            "continual_drift_recovery_integrity": 1.0,
        },
        "threshold_results": {
            "structural_plasticity_stability": True,
            "hippocampal_transfer_integrity": True,
            "scale_out_retention_integrity": True,
            "continual_drift_recovery_integrity": True,
        },
    }


def _sparse_diffusion_pass_report(passed: bool = True) -> dict:
    value = 1.0 if passed else 0.0
    return {
        "suite_name": "SparseDiffusionBlockReadiness",
        "passed": bool(passed),
        "overall_score": value,
        "block_count": 3,
        "metrics": {
            "sparse_diffusion_partition_integrity": value,
            "sparse_diffusion_independent_block_integrity": value,
            "sparse_diffusion_denoise_accuracy": value,
            "sparse_diffusion_event_cost_advantage": 2.25 if passed else 1.0,
            "sparse_diffusion_block_ablation_integrity": value,
            "sparse_diffusion_single_pass_recurrent_integrity": value,
            "sparse_diffusion_policy_compatibility": value,
        },
    }


def test_phase5_completion_gate_accepts_valid_reports() -> None:
    benchmark = _load_script("phase5_predictive_coding_benchmark.py")
    entry_gate = _load_script("phase5_entry_gate.py")
    completion_gate = _load_script("phase5_completion_gate.py")

    phase4 = _phase4_pass_report()
    phase5 = benchmark.run_phase5_predictive_coding_benchmark()
    phase5_entry = entry_gate.build_phase5_entry_gate_report(phase5)
    sparse_diffusion = _sparse_diffusion_pass_report()

    errors = completion_gate.validate_phase5_completion(phase4, phase5, phase5_entry, sparse_diffusion)
    gate_report = completion_gate.build_phase5_completion_gate_report(phase4, phase5, phase5_entry, sparse_diffusion)
    summary = completion_gate.format_phase5_completion_gate_summary(gate_report)

    assert errors == []
    assert gate_report["suite_name"] == "Phase5CompletionGate"
    assert gate_report["passed"] is True
    assert gate_report["failed_checks"] == []
    assert gate_report["checks"]["phase4_prerequisite_passed"]["passed"] is True
    assert gate_report["checks"]["phase5_entry_gate_passed"]["passed"] is True
    assert gate_report["checks"]["sparse_diffusion_block_readiness_passed"]["passed"] is True
    assert gate_report["checks"]["sparse_diffusion.sparse_diffusion_denoise_accuracy"]["passed"] is True
    assert gate_report["checks"]["multi_step_trace_complete"]["passed"] is True
    assert "- gate_status: PASS" in summary
    assert "- macro_step_reduction_value: 3.000 required_min=2.000" in summary
    assert "- macro_cost_reduction_value: 0.420 required_min=0.300" in summary
    assert "- subgoal_coverage_ratio_value: 1.000 required_min=1.000" in summary
    assert "- micro_es_fitness_improvement_value: 0.249 required_gt=0.050" in summary
    assert "- micro_es_event_cost_reduction_value: 0.090 required_min=0.040" in summary
    assert "- micro_es_population_event_budget_value: 0.160 event_budget=0.250" in summary


def test_phase5_completion_gate_rejects_phase4_or_entry_failures() -> None:
    benchmark = _load_script("phase5_predictive_coding_benchmark.py")
    entry_gate = _load_script("phase5_entry_gate.py")
    completion_gate = _load_script("phase5_completion_gate.py")

    phase4 = _phase4_pass_report()
    phase5 = benchmark.run_phase5_predictive_coding_benchmark()
    phase5_entry = entry_gate.build_phase5_entry_gate_report(phase5)
    sparse_diffusion = _sparse_diffusion_pass_report()

    phase4["passed"] = False
    phase5_entry["passed"] = False
    phase5_entry["failed_checks"] = ["metric.correction_event_coverage"]

    errors = completion_gate.validate_phase5_completion(phase4, phase5, phase5_entry, sparse_diffusion)

    assert any("Phase 4 completion prerequisite is not passed." in error for error in errors)
    assert any("Phase 5 entry gate is not passed." in error for error in errors)
    assert any("metric.correction_event_coverage" in error for error in errors)


def test_phase5_completion_gate_rejects_primary_alignment_drop() -> None:
    benchmark = _load_script("phase5_predictive_coding_benchmark.py")
    entry_gate = _load_script("phase5_entry_gate.py")
    completion_gate = _load_script("phase5_completion_gate.py")

    phase4 = _phase4_pass_report()
    phase5 = benchmark.run_phase5_predictive_coding_benchmark()
    phase5_entry = entry_gate.build_phase5_entry_gate_report(phase5)
    sparse_diffusion = _sparse_diffusion_pass_report()

    broken = dict(phase5)
    broken["details"] = dict(phase5["details"])
    broken_primary = dict(phase5["details"]["primary_transition"])
    broken_primary["alignment_ratio"] = 0.5
    broken["details"]["primary_transition"] = broken_primary

    errors = completion_gate.validate_phase5_completion(phase4, broken, phase5_entry, sparse_diffusion)
    gate_report = completion_gate.build_phase5_completion_gate_report(phase4, broken, phase5_entry, sparse_diffusion)

    assert any("primary alignment ratio is below completion threshold" in error for error in errors)
    assert gate_report["passed"] is False
    assert "primary_alignment_ratio" in gate_report["failed_checks"]


def test_phase5_completion_gate_rejects_macro_and_subgoal_detail_regressions() -> None:
    benchmark = _load_script("phase5_predictive_coding_benchmark.py")
    entry_gate = _load_script("phase5_entry_gate.py")
    completion_gate = _load_script("phase5_completion_gate.py")

    phase4 = _phase4_pass_report()
    phase5 = benchmark.run_phase5_predictive_coding_benchmark()
    phase5_entry = entry_gate.build_phase5_entry_gate_report(phase5)
    sparse_diffusion = _sparse_diffusion_pass_report()

    broken = dict(phase5)
    broken["details"] = dict(phase5["details"])
    broken["details"]["macro_step_reduction"] = 1.0
    broken["details"]["macro_cost_reduction"] = 0.10
    broken["details"]["subgoal_coverage_ratio"] = 0.5

    errors = completion_gate.validate_phase5_completion(phase4, broken, phase5_entry, sparse_diffusion)
    gate_report = completion_gate.build_phase5_completion_gate_report(phase4, broken, phase5_entry, sparse_diffusion)

    assert any("macro step reduction is below completion threshold" in error for error in errors)
    assert any("macro cost reduction is below completion threshold" in error for error in errors)
    assert any("subgoal coverage ratio is below completion threshold" in error for error in errors)
    assert gate_report["passed"] is False
    assert "macro_step_reduction" in gate_report["failed_checks"]
    assert "macro_cost_reduction" in gate_report["failed_checks"]
    assert "subgoal_coverage_ratio" in gate_report["failed_checks"]


def test_phase5_completion_gate_rejects_micro_es_detail_regressions() -> None:
    benchmark = _load_script("phase5_predictive_coding_benchmark.py")
    entry_gate = _load_script("phase5_entry_gate.py")
    completion_gate = _load_script("phase5_completion_gate.py")

    phase4 = _phase4_pass_report()
    phase5 = benchmark.run_phase5_predictive_coding_benchmark()
    phase5_entry = entry_gate.build_phase5_entry_gate_report(phase5)
    sparse_diffusion = _sparse_diffusion_pass_report()

    broken = dict(phase5)
    broken["details"] = dict(phase5["details"])
    broken_micro_es = dict(phase5["details"]["micro_es_refinement"])
    broken_micro_es["low_rank_trace_complete"] = False
    broken_micro_es["fitness_improvement"] = 0.01
    broken_micro_es["event_cost_reduction"] = 0.01
    broken_micro_es["population_event_cost_proxy"] = 0.40
    broken_micro_es["event_budget"] = 0.25
    broken["details"]["micro_es_refinement"] = broken_micro_es

    errors = completion_gate.validate_phase5_completion(phase4, broken, phase5_entry, sparse_diffusion)
    gate_report = completion_gate.build_phase5_completion_gate_report(phase4, broken, phase5_entry, sparse_diffusion)

    assert any("micro-ES low-rank refinement trace is incomplete" in error for error in errors)
    assert any("micro-ES fitness improvement is below completion threshold" in error for error in errors)
    assert any("micro-ES event cost reduction is below completion threshold" in error for error in errors)
    assert any("micro-ES population event cost exceeds its event budget" in error for error in errors)
    assert gate_report["passed"] is False
    assert "micro_es_low_rank_trace_complete" in gate_report["failed_checks"]
    assert "micro_es_fitness_improvement" in gate_report["failed_checks"]
    assert "micro_es_event_cost_reduction" in gate_report["failed_checks"]
    assert "micro_es_population_event_budget" in gate_report["failed_checks"]


def test_phase5_completion_gate_rejects_sparse_diffusion_regression() -> None:
    benchmark = _load_script("phase5_predictive_coding_benchmark.py")
    entry_gate = _load_script("phase5_entry_gate.py")
    completion_gate = _load_script("phase5_completion_gate.py")

    phase4 = _phase4_pass_report()
    phase5 = benchmark.run_phase5_predictive_coding_benchmark()
    phase5_entry = entry_gate.build_phase5_entry_gate_report(phase5)
    sparse_diffusion = _sparse_diffusion_pass_report(False)

    errors = completion_gate.validate_phase5_completion(phase4, phase5, phase5_entry, sparse_diffusion)
    gate_report = completion_gate.build_phase5_completion_gate_report(phase4, phase5, phase5_entry, sparse_diffusion)

    assert any("Sparse diffusion block readiness did not pass." in error for error in errors)
    assert any("sparse_diffusion_denoise_accuracy" in error for error in errors)
    assert gate_report["passed"] is False
    assert "sparse_diffusion_block_readiness_passed" in gate_report["failed_checks"]
    assert "sparse_diffusion.sparse_diffusion_denoise_accuracy" in gate_report["failed_checks"]
