import importlib.util
import json
import os
import uuid


def _load_script():
    module_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "research_automation_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location("research_automation_benchmark_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase3_report(
    passed=True,
    linear_value=1.0,
    linear_regressions=None,
    neuromorphic_regression_count=0,
    architecture_value=1.0,
    architecture_regressions=None,
    stage_e_observed_acceptance_failures=None,
):
    linear_regressions = list(linear_regressions or [])
    architecture_regressions = list(architecture_regressions or [])
    stage_e_observed_acceptance_failures = list(stage_e_observed_acceptance_failures or [])
    return {
        "suite_name": "Phase3AccuracySuite",
        "passed": bool(passed),
        "component_reports": {
            "cognitive_runtime": {
                "passed": bool(passed),
                "overall_score": 1.0 if passed else 0.0,
                "metrics": {
                    "predictive_spike_entropy_reduction_observed": float(linear_value),
                    "phase_binding_coincidence_integrity_observed": float(linear_value),
                    "forward_only_local_update_stability_observed": float(linear_value),
                    "lejepa_linear_identifiability_proxy_observed": float(linear_value),
                    "lejepa_latent_whitening_health_observed": float(linear_value),
                    "lejepa_factor_disentanglement_observed": float(linear_value),
                    "lejepa_latent_planning_consistency_observed": float(linear_value),
                    "lejepa_positive_pair_alignment_observed": float(linear_value),
                    "micro_turn_event_budget_observed": float(architecture_value),
                    "foreground_background_context_handoff_observed": float(architecture_value),
                    "interrupt_recovery_trace_observed": float(architecture_value),
                    "simultaneous_stream_route_integrity_observed": float(architecture_value),
                    "time_aligned_backchannel_policy_observed": float(architecture_value),
                    "phase_assigned_submodel_route_observed": float(architecture_value),
                    "uncertainty_bucket_specialization_observed": float(architecture_value),
                    "denoising_correction_trace_integrity_observed": float(architecture_value),
                    "block_independent_local_update_budget_observed": float(architecture_value),
                    "plastic_submodel_registry_integrity_observed": float(architecture_value),
                    "dynamic_submodel_route_integrity_observed": float(architecture_value),
                    "submodel_relearning_trace_integrity_observed": float(architecture_value),
                    "interpretable_submodel_concept_trace_observed": float(architecture_value),
                    "runtime_submodel_route_action_grounding_observed": float(architecture_value),
                    "runtime_submodel_counterfactual_route_separation_observed": float(architecture_value),
                    "runtime_submodel_concept_trace_observed": float(architecture_value),
                    "submodel_intervention_trace_integrity_observed": float(architecture_value),
                    "submodel_ablation_effect_observed": float(architecture_value),
                    "submodel_reactivation_recovery_observed": float(architecture_value),
                    "submodel_credit_assignment_trace_integrity_observed": float(architecture_value),
                    "submodel_credit_selectivity_observed": float(architecture_value),
                    "submodel_credit_state_budget_observed": float(architecture_value),
                    "runtime_submodel_local_credit_assignment_observed": float(architecture_value),
                    "runtime_submodel_feedback_trace_observed": float(architecture_value),
                    "submodel_structural_adaptation_trace_integrity_observed": float(architecture_value),
                    "submodel_structural_growth_bounded_observed": float(architecture_value),
                    "submodel_structural_pruning_observed": float(architecture_value),
                    "submodel_scientific_hypothesis_trace_integrity_observed": float(architecture_value),
                    "submodel_counterexample_revision_observed": float(architecture_value),
                    "submodel_scientific_model_budget_observed": float(architecture_value),
                    "submodel_hypothesis_bank_integrity_observed": float(architecture_value),
                    "submodel_open_ended_selection_observed": float(architecture_value),
                    "submodel_hypothesis_bank_budget_observed": float(architecture_value),
                },
            },
            "energy_efficiency": {
                "passed": bool(passed),
                "overall_score": 1.0 if passed else 0.0,
                "metrics": {},
                "neuromorphic_profile_trend": {
                    "has_previous": True,
                    "regression_count": int(neuromorphic_regression_count),
                    "policy_change_count": 0,
                    "regressions": [
                        {
                            "profile": "lava_loihi2",
                            "kind": "compatibility",
                            "check": "online_update_policy",
                        }
                    ] if neuromorphic_regression_count else [],
                    "missing_profiles": [],
                    "policy_changes": [],
                },
            },
        },
        "linear_snn_fusion_observed_trend": {
            "has_previous": True,
            "regression_count": len(linear_regressions),
            "regressions": linear_regressions,
            "release_gate_blocking": False,
        },
        "stage_e_architecture_integration_observed_trend": {
            "has_previous": True,
            "regression_count": len(architecture_regressions),
            "regressions": architecture_regressions,
            "release_gate_blocking": False,
        },
        "stage_e_readiness": {
            "observed_acceptance_candidate_count": 49,
            "observed_acceptance_candidate_ready_count": 49 - len(stage_e_observed_acceptance_failures),
            "observed_acceptance_candidates_ready": not stage_e_observed_acceptance_failures,
            "observed_acceptance_candidate_failure_count": len(stage_e_observed_acceptance_failures),
            "observed_acceptance_candidate_failures": stage_e_observed_acceptance_failures,
        },
    }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _managed_output_paths(module):
    base = module.workspace_path("evaluation", "test_research_automation", uuid.uuid4().hex)
    os.makedirs(base, exist_ok=True)
    return {
        "report": os.path.join(base, "research_review_report.json"),
        "suggestion": os.path.join(base, "roadmap_patch_suggestion.json"),
        "journal": os.path.join(base, "research_journal.jsonl"),
    }


def test_research_automation_report_marks_stable_inputs(tmp_path):
    module = _load_script()
    phase3_path = tmp_path / "phase3.json"
    release_path = tmp_path / "release.json"
    operational_path = tmp_path / "operational.json"
    output_paths = _managed_output_paths(module)

    _write_json(phase3_path, _phase3_report(passed=True))
    _write_json(release_path, {"passed": True})
    _write_json(operational_path, {"passed": True})

    report = module.run_research_automation_benchmark(
        phase3_report_path=str(phase3_path),
        release_soak_report_path=str(release_path),
        operational_report_path=str(operational_path),
        report_path=output_paths["report"],
        roadmap_patch_suggestion_path=output_paths["suggestion"],
        journal_path=output_paths["journal"],
        generated_at=123.0,
    )

    assert report["passed"] is True
    assert report["review_score"] == 1.0
    assert report["signals"]["linear_snn_fusion"]["ready"] is True
    assert report["signals"]["stage_e_architecture_integration"]["ready"] is True
    assert report["signals"]["neuromorphic_profile"]["ready"] is True
    assert report["signals"]["sara_policy_alignment"]["ready"] is True
    assert report["review_dimensions"]["sara_policy_no_backprop_alignment"]["status"] == "pass"
    assert report["review_dimensions"]["sara_policy_sparse_event_alignment"]["status"] == "pass"
    assert report["review_dimensions"]["sara_policy_local_learning_alignment"]["status"] == "pass"
    assert report["review_dimensions"]["sara_policy_interpretability_trace_coverage"]["status"] == "pass"
    assert report["review_dimensions"]["sara_policy_submodel_integration_impact"]["status"] == "pass"
    assert report["experiment_planner"]["negative_results"] == []
    graph = report["experiment_planner"]["bounded_experiment_graph"]
    assert graph["schema"] == "sara-bounded-experiment-graph-v1"
    assert graph["bounded"] is True
    assert graph["node_count"] == 4
    assert graph["stage_counts"] == {"promotion_candidate": 4}
    status_summary = report["experiment_planner"]["experiment_status_summary"]
    assert status_summary["adoption_candidate_count"] == 4
    assert status_summary["regressing_item_count"] == 0
    assert status_summary["falsified_item_count"] == 0
    assert status_summary["human_review_pending_count"] == 0
    priority_plan = report["experiment_planner"]["experiment_priority_plan"]
    assert priority_plan["action_count"] == 1
    assert priority_plan["top_priority_source"] == "experiment_adoption_candidate_review"
    assert priority_plan["actions"][0]["category"] == "adoption_candidate"
    promotion_plan = report["experiment_planner"]["experiment_promotion_target_plan"]
    assert promotion_plan["candidate_count"] == 4
    assert promotion_plan["review_action_count"] == 4
    assert promotion_plan["target_stage_counts"] == {
        "neuromorphic_profile": 1,
        "research_policy": 1,
        "stage_e": 2,
    }
    assert {
        item["promotion_path"]
        for item in promotion_plan["targets"]
    } == {
        "neuromorphic_adapter_policy_review",
        "sara_policy_guard_review",
        "stage_e_architecture_acceptance_review",
        "stage_e_sparse_runtime_acceptance_review",
    }
    assert os.path.exists(output_paths["report"])
    assert os.path.exists(output_paths["suggestion"])

    with open(output_paths["suggestion"], "r", encoding="utf-8") as handle:
        suggestion = json.load(handle)
    assert suggestion["apply_automatically"] is False
    assert suggestion["requires_human_approval"] is True

    compact = module.compact_research_review_report(report)
    assert compact["passed"] is True
    assert compact["release_gate_blocking"] is False
    assert compact["requires_human_approval"] is True
    assert compact["bounded_experiment_graph_node_count"] == 4
    assert compact["experiment_adoption_candidate_count"] == 4
    assert compact["experiment_regressing_item_count"] == 0
    assert compact["experiment_falsified_item_count"] == 0
    assert compact["experiment_human_review_pending_count"] == 0
    assert compact["experiment_priority_action_count"] == 1
    assert compact["experiment_top_priority_source"] == "experiment_adoption_candidate_review"
    assert compact["experiment_top_priority_category"] == "adoption_candidate"
    assert compact["experiment_promotion_target_candidate_count"] == 4
    assert compact["experiment_promotion_target_review_action_count"] == 4
    assert compact["experiment_promotion_target_stage_counts"] == {
        "neuromorphic_profile": 1,
        "research_policy": 1,
        "stage_e": 2,
    }


def test_research_automation_report_tracks_regressions_and_journal(tmp_path):
    module = _load_script()
    phase3_path = tmp_path / "phase3.json"
    release_path = tmp_path / "release.json"
    operational_path = tmp_path / "operational.json"
    output_paths = _managed_output_paths(module)

    _write_json(
        phase3_path,
        _phase3_report(
            passed=False,
            linear_value=0.5,
            linear_regressions=[
                {
                    "metric": "predictive_spike_entropy_reduction_observed",
                    "previous": 1.0,
                    "current": 0.5,
                    "delta": -0.5,
                }
            ],
            neuromorphic_regression_count=1,
        ),
    )
    _write_json(release_path, {"passed": False})
    _write_json(operational_path, {"passed": False})

    report = module.run_research_automation_benchmark(
        phase3_report_path=str(phase3_path),
        release_soak_report_path=str(release_path),
        operational_report_path=str(operational_path),
        report_path=output_paths["report"],
        roadmap_patch_suggestion_path=output_paths["suggestion"],
        journal_path=output_paths["journal"],
        append_journal=True,
        generated_at=456.0,
    )

    assert report["passed"] is False
    assert report["signals"]["linear_snn_fusion"]["ready"] is False
    assert report["signals"]["stage_e_architecture_integration"]["ready"] is True
    assert report["signals"]["linear_snn_fusion"]["regression_count"] == 1
    assert report["signals"]["neuromorphic_profile"]["regression_count"] == 1
    assert report["artifacts"]["journal_appended"] is True
    assert len(report["experiment_planner"]["negative_results"]) == 8
    assert report["signals"]["sara_policy_alignment"]["ready"] is False
    assert report["review_dimensions"]["sara_policy_no_backprop_alignment"]["status"] == "needs_review"
    assert report["review_dimensions"]["sara_policy_sparse_event_alignment"]["status"] == "needs_review"
    graph = report["experiment_planner"]["bounded_experiment_graph"]
    assert graph["node_count"] >= 10
    assert graph["stage_counts"]["negative_result_review"] == 8
    assert graph["stage_counts"]["template_based_probe"] >= 2
    assert any(edge["kind"] == "proposes_targeted_probe" for edge in graph["edges"])
    assert report["experiment_planner"]["experiment_status_summary"]["regressing_item_count"] >= 2
    assert report["experiment_planner"]["experiment_status_summary"]["falsified_item_count"] == 8
    assert report["experiment_planner"]["experiment_priority_plan"]["top_priority_source"] == (
        "experiment_regression_remeasure"
    )
    assert report["experiment_planner"]["experiment_priority_plan"]["actions"][0]["priority"] == "high"

    with open(output_paths["suggestion"], "r", encoding="utf-8") as handle:
        suggestion = json.load(handle)
    assert any("linear_snn_fusion_metric_recovery" in item for item in suggestion["suggestions"])
    assert any("release_gate_safety_review" in item for item in suggestion["suggestions"])

    with open(output_paths["journal"], "r", encoding="utf-8") as handle:
        journal_lines = handle.read().strip().splitlines()
    assert len(journal_lines) == 1
    journal_entry = json.loads(journal_lines[0])
    assert journal_entry["passed"] is False
    assert journal_entry["negative_results"]
    assert journal_entry["bounded_experiment_graph"]["node_count"] == graph["node_count"]
    assert journal_entry["seen_count"] == 1
    assert "dedupe_key" in journal_entry


def test_research_automation_tracks_stage_e_architecture_integration_failures(tmp_path):
    module = _load_script()
    phase3_path = tmp_path / "phase3.json"
    release_path = tmp_path / "release.json"
    operational_path = tmp_path / "operational.json"
    output_paths = _managed_output_paths(module)

    _write_json(phase3_path, _phase3_report(passed=True, architecture_value=0.5))
    _write_json(release_path, {"passed": True})
    _write_json(operational_path, {"passed": True})

    report = module.run_research_automation_benchmark(
        phase3_report_path=str(phase3_path),
        release_soak_report_path=str(release_path),
        operational_report_path=str(operational_path),
        report_path=output_paths["report"],
        roadmap_patch_suggestion_path=output_paths["suggestion"],
        journal_path=output_paths["journal"],
        generated_at=789.0,
    )

    assert report["signals"]["linear_snn_fusion"]["ready"] is True
    assert report["signals"]["stage_e_architecture_integration"]["ready"] is False
    assert report["review_dimensions"]["stage_e_architecture_integration"]["status"] == "needs_review"
    assert len(report["signals"]["stage_e_architecture_integration"]["failures"]) == 9
    assert any(
        item["id"] == "stage_e_architecture_integration_metric_recovery"
        for item in report["experiment_planner"]["next_hypotheses"]
    )
    assert set(report["signals"]["stage_e_architecture_integration"]["required_metrics"]).issubset({
        item["metric"] for item in report["experiment_planner"]["negative_results"]
    })
    assert report["signals"]["sara_policy_alignment"]["ready"] is False
    assert report["review_dimensions"]["sara_policy_submodel_integration_impact"]["status"] == "needs_review"

    with open(output_paths["suggestion"], "r", encoding="utf-8") as handle:
        suggestion = json.load(handle)
    assert any("stage_e_architecture_integration_metric_recovery" in item for item in suggestion["suggestions"])


def test_research_automation_tracks_stage_e_architecture_integration_trend_regressions(tmp_path):
    module = _load_script()
    phase3_path = tmp_path / "phase3.json"
    release_path = tmp_path / "release.json"
    operational_path = tmp_path / "operational.json"
    output_paths = _managed_output_paths(module)

    _write_json(
        phase3_path,
        _phase3_report(
            passed=True,
            architecture_value=1.0,
            architecture_regressions=[
                {
                    "metric": "foreground_background_context_handoff_observed",
                    "previous": 1.0,
                    "current": 0.75,
                    "delta": -0.25,
                }
            ],
        ),
    )
    _write_json(release_path, {"passed": True})
    _write_json(operational_path, {"passed": True})

    report = module.run_research_automation_benchmark(
        phase3_report_path=str(phase3_path),
        release_soak_report_path=str(release_path),
        operational_report_path=str(operational_path),
        report_path=output_paths["report"],
        roadmap_patch_suggestion_path=output_paths["suggestion"],
        journal_path=output_paths["journal"],
        generated_at=790.0,
    )

    assert report["signals"]["stage_e_architecture_integration"]["ready"] is True
    assert report["signals"]["stage_e_architecture_integration"]["regression_count"] == 1
    assert any(
        item["id"] == "stage_e_architecture_integration_observed_regression"
        for item in report["experiment_planner"]["regression_watchlist"]
    )


def test_research_automation_tracks_stage_e_observed_acceptance_candidate_failures(tmp_path):
    module = _load_script()
    phase3_path = tmp_path / "phase3.json"
    release_path = tmp_path / "release.json"
    operational_path = tmp_path / "operational.json"
    output_paths = _managed_output_paths(module)
    failure = {
        "check": "metric.micro_turn_event_budget_observed",
        "metric": "micro_turn_event_budget_observed",
        "value": 0.0,
        "threshold": 1.0,
    }

    _write_json(
        phase3_path,
        _phase3_report(
            passed=True,
            stage_e_observed_acceptance_failures=[failure],
        ),
    )
    _write_json(release_path, {"passed": True})
    _write_json(operational_path, {"passed": True})

    report = module.run_research_automation_benchmark(
        phase3_report_path=str(phase3_path),
        release_soak_report_path=str(release_path),
        operational_report_path=str(operational_path),
        report_path=output_paths["report"],
        roadmap_patch_suggestion_path=output_paths["suggestion"],
        journal_path=output_paths["journal"],
        generated_at=791.0,
    )

    signal = report["signals"]["stage_e_observed_acceptance_candidates"]
    assert signal["available"] is True
    assert signal["ready"] is False
    assert signal["failure_count"] == 1
    assert report["review_dimensions"]["stage_e_observed_acceptance_candidates"]["status"] == "needs_review"
    assert any(
        item["id"] == "stage_e_observed_acceptance_candidate_repair"
        for item in report["experiment_planner"]["next_hypotheses"]
    )
    assert any(
        item.get("metric") == "micro_turn_event_budget_observed"
        for item in report["experiment_planner"]["negative_results"]
    )
    assert report["experiment_planner"]["experiment_priority_plan"]["top_priority_source"] == (
        "experiment_regression_remeasure"
    )
    assert any(
        node["id"] == "stage_e_observed_acceptance_candidate_repair"
        for node in report["experiment_planner"]["bounded_experiment_graph"]["nodes"]
    )


def test_research_journal_suppresses_duplicate_recent_entries(tmp_path):
    module = _load_script()
    phase3_path = tmp_path / "phase3.json"
    release_path = tmp_path / "release.json"
    operational_path = tmp_path / "operational.json"
    output_paths = _managed_output_paths(module)

    _write_json(phase3_path, _phase3_report(passed=False, linear_value=0.5))
    _write_json(release_path, {"passed": False})
    _write_json(operational_path, {"passed": False})

    first = module.run_research_automation_benchmark(
        phase3_report_path=str(phase3_path),
        release_soak_report_path=str(release_path),
        operational_report_path=str(operational_path),
        report_path=output_paths["report"],
        roadmap_patch_suggestion_path=output_paths["suggestion"],
        journal_path=output_paths["journal"],
        append_journal=True,
        generated_at=1000.0,
        journal_dedupe_window_seconds=3600.0,
    )
    second = module.run_research_automation_benchmark(
        phase3_report_path=str(phase3_path),
        release_soak_report_path=str(release_path),
        operational_report_path=str(operational_path),
        report_path=output_paths["report"],
        roadmap_patch_suggestion_path=output_paths["suggestion"],
        journal_path=output_paths["journal"],
        append_journal=True,
        generated_at=1200.0,
        journal_dedupe_window_seconds=3600.0,
    )

    assert first["artifacts"]["journal_appended"] is True
    assert second["artifacts"]["journal_appended"] is False
    assert second["artifacts"]["journal_duplicate_suppressed"] is True
    assert second["artifacts"]["journal_entry_count"] == 1

    with open(output_paths["journal"], "r", encoding="utf-8") as handle:
        journal_lines = handle.read().strip().splitlines()
    assert len(journal_lines) == 1
    journal_entry = json.loads(journal_lines[0])
    assert journal_entry["seen_count"] == 2
    assert journal_entry["last_seen_at"] == 1200.0


def test_research_journal_prunes_by_age_and_limit(tmp_path):
    module = _load_script()
    journal_path = module.workspace_path("evaluation", "test_research_automation", uuid.uuid4().hex, "journal.jsonl")
    old_report = module.build_research_review_report(
        phase3_report=_phase3_report(passed=False, linear_value=0.1),
        release_soak_report={"passed": False},
        operational_report={"passed": False},
        input_snapshots=[],
        generated_at=100.0,
    )
    new_report = module.build_research_review_report(
        phase3_report=_phase3_report(passed=True, linear_value=1.0),
        release_soak_report={"passed": True},
        operational_report={"passed": True},
        input_snapshots=[],
        generated_at=1000.0,
    )

    first = module.append_research_journal_entry(
        journal_path,
        old_report,
        max_age_seconds=10_000.0,
    )
    second = module.append_research_journal_entry(
        journal_path,
        new_report,
        max_entries=1,
        max_age_seconds=500.0,
    )

    assert first["entry_count"] == 1
    assert second["entry_count"] == 1
    assert second["pruned_by_age"] == 1
    with open(journal_path, "r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle.read().strip().splitlines()]
    assert len(entries) == 1
    assert entries[0]["passed"] is True


def test_summarize_research_journal_entries_counts_frequent_items():
    module = _load_script()
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 2,
            "negative_results": [{"metric": "predictive_spike_entropy_reduction_observed"}],
            "regression_watchlist": [{"id": "release_gate_safety_review"}],
            "next_hypotheses": [{"id": "linear_snn_fusion_metric_recovery"}],
            "roadmap_patch_review_decision": "rejected",
            "remeasure_results": [
                {
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "status": "failed",
                    "resolved_timestamp": 150.0,
                    "target_ids": ["predictive_spike_entropy_reduction_observed"],
                },
                {
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "status": "success",
                    "resolved_timestamp": 210.0,
                    "target_ids": ["predictive_spike_entropy_reduction_observed"],
                },
            ],
        },
        {
            "generated_at": 200.0,
            "seen_count": 1,
            "negative_results": [{"metric": "predictive_spike_entropy_reduction_observed"}],
            "regression_watchlist": [],
            "next_hypotheses": [{"id": "linear_snn_fusion_metric_recovery"}],
            "roadmap_patch_review_decision": "approved",
        },
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=250.0)

    assert summary["entry_count"] == 2
    assert summary["total_seen_count"] == 3
    assert summary["stale_age_seconds"] == 50.0
    assert summary["top_negative_results"][0] == {
        "id": "predictive_spike_entropy_reduction_observed",
        "count": 2,
    }
    assert summary["top_next_hypotheses"][0] == {
        "id": "linear_snn_fusion_metric_recovery",
        "count": 2,
    }
    assert summary["suppressed_benchmark_actions"][0]["id"] == (
        "predictive_spike_entropy_reduction_observed"
    )
    assert summary["suppressed_benchmark_actions"][0]["remeasure_trend"] == "recovered"
    assert summary["recommended_benchmark_actions"][0]["command"] == (
        "python scripts/eval/release_gate.py"
    )
    assert summary["recommended_benchmark_actions"][0]["priority"] == "high"
    assert summary["roadmap_patch_review_rejected_count"] == 1
    assert summary["roadmap_patch_review_approved_count"] == 1
    assert summary["roadmap_patch_rejection_reasons"][0] == {
        "reason": "unspecified",
        "count": 1,
    }
    assert summary["roadmap_patch_rejected_item_count"] == 3
    assert summary["roadmap_patch_refreshed_item_count"] == 0
    assert summary["roadmap_patch_refresh_to_rejection_ratio"] == 0.0
    assert summary["roadmap_patch_rejected_items"][0]["id"] == (
        "linear_snn_fusion_metric_recovery"
    )
    assert summary["remeasure_result_count"] == 2
    assert summary["remeasure_status_counts"] == {"failed": 1, "success": 1}
    assert summary["top_remeasured_ids"][0] == {
        "id": "predictive_spike_entropy_reduction_observed",
        "count": 2,
    }
    assert summary["remeasure_trends"][0]["id"] == "predictive_spike_entropy_reduction_observed"
    assert summary["remeasure_trends"][0]["trend"] == "recovered"
    assert summary["experiment_status_summary"]["falsified_item_count"] == 3
    assert summary["experiment_status_summary"]["regressing_item_count"] == 1
    assert summary["experiment_status_summary"]["adoption_candidate_count"] == 0
    assert summary["experiment_priority_plan"]["top_priority_source"] == "experiment_regression_remeasure"
    assert summary["experiment_priority_plan"]["action_count"] == 2


def test_experiment_promotion_target_plan_classifies_known_surfaces():
    module = _load_script()
    status = {
        "adoption_candidate_ids": [
            "delta_memory_erase_write_decoupling_observed",
            "micro_turn_event_budget_observed",
            "future_state_transition_integrity",
            "unmapped_candidate",
        ],
    }

    plan = module.build_experiment_promotion_target_plan(status)
    by_id = {item["id"]: item for item in plan["targets"]}

    assert by_id["delta_memory_erase_write_decoupling_observed"]["target_stage"] == "stage_d"
    assert by_id["delta_memory_erase_write_decoupling_observed"]["target_surface"] == (
        "delta_memory_promotion_candidate"
    )
    assert by_id["micro_turn_event_budget_observed"]["target_stage"] == "stage_e"
    assert by_id["micro_turn_event_budget_observed"]["target_surface"] == "observed_acceptance_candidate"
    assert by_id["future_state_transition_integrity"]["promotion_path"] == "already_minimum"
    assert by_id["unmapped_candidate"]["promotion_path"] == "manual_mapping_review"
    assert plan["review_action_count"] == 3


def test_research_journal_summary_tracks_stage_e_observed_candidate_repair_loop():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "remeasure_results": [
                {
                    "status": "failed",
                    "target_ids": [repair_id],
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "resolved_timestamp": 120.0,
                }
            ],
            "alternative_probe_results": [
                {
                    "status": "failed",
                    "target_ids": [repair_id],
                    "command": (
                        "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q "
                        "tests/test_phase3_accuracy_benchmarks.py::"
                        "test_stage_e_observed_acceptance_candidate_failures_are_structured"
                    ),
                    "resolved_timestamp": 130.0,
                }
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=200000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert repair_loop["id"] == repair_id
    assert repair_loop["negative_result_count"] == 1
    assert repair_loop["regression_watchlist_count"] == 1
    assert repair_loop["next_hypothesis_count"] == 1
    assert repair_loop["remeasure_recommended"] is True
    assert repair_loop["latest_remeasure_trend"] == "still_failing"
    assert repair_loop["latest_alternative_probe_trend"] == "targeted_probe_failed"
    assert repair_loop["needs_followup"] is True
    assert summary["recommended_benchmark_actions"][0]["id"] == repair_id
    assert summary["recommended_benchmark_actions"][0]["command"] == (
        "python scripts/eval/cognitive_runtime_benchmark.py"
    )


def test_research_journal_summary_promotes_recovered_stage_e_observed_candidate_repair():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "remeasure_results": [
                {
                    "status": "failed",
                    "target_ids": [repair_id],
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "resolved_timestamp": 110.0,
                },
                {
                    "status": "success",
                    "target_ids": [repair_id],
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "resolved_timestamp": 150.0,
                },
            ],
            "alternative_probe_results": [
                {
                    "status": "success",
                    "target_ids": [repair_id],
                    "command": (
                        "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q "
                        "tests/test_phase3_accuracy_benchmarks.py::"
                        "test_stage_e_observed_acceptance_candidate_failures_are_structured"
                    ),
                    "resolved_timestamp": 160.0,
                }
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=200.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert repair_loop["latest_remeasure_trend"] == "recovered"
    assert repair_loop["latest_alternative_probe_trend"] == "targeted_probe_passed"
    assert repair_loop["remeasure_suppressed"] is True
    assert repair_loop["needs_followup"] is False
    assert repair_loop["recovery_confirmed"] is True
    assert repair_loop["recovery_source"] == "remeasure,alternative_probe"
    assert repair_loop["promotion_review_recommended"] is True
    assert repair_loop["next_review_action"] == "stage_e_observed_acceptance_candidate_stability"


def test_stage_e_observed_candidate_recovery_review_results_persist_in_journal_entries():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "remeasure_results": [
                {
                    "status": "success",
                    "target_ids": [repair_id],
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "resolved_timestamp": 150.0,
                }
            ],
        }
    ]
    repair_entries = [
        {
            "command": "review_stage_e_observed_acceptance_candidate_recovery_for_stability",
            "status": "success",
            "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review",
            "covered_checks": [
                "stage_e_observed_acceptance_candidate_repair_recovery",
                "stage_e_observed_acceptance_candidate_stability",
                "research_journal_summary",
            ],
            "resolved_timestamp": 900.0,
        }
    ]

    updated, sync = module.attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_entries(
        entries,
        repair_entries,
    )
    repeated, repeat_sync = module.attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_entries(
        updated,
        repair_entries,
    )
    summary = module.summarize_research_journal_entries(repeated, now_timestamp=1000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert sync["linked_count"] == 1
    assert sync["status_counts"] == {"success": 1}
    assert updated[0]["stage_e_observed_acceptance_candidate_recovery_review_results"][0][
        "review_type"
    ] == "stage_e_observed_acceptance_candidate_recovery_review"
    assert repeat_sync["linked_count"] == 0
    assert repeat_sync["skipped_duplicate_count"] == 1
    assert repair_loop["promotion_review_completed"] is True
    assert repair_loop["promotion_review_recommended"] is False
    assert repair_loop["promotion_review_latest_status"] == "success"
    assert summary["stage_e_observed_acceptance_candidate_recovery_review_count"] == 1


def test_stage_e_observed_candidate_recovery_review_followup_results_are_summarized():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "stage_e_observed_acceptance_candidate_recovery_review_results": [
                {
                    "status": "pending",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup",
                    "command": "followup_stage_e_observed_acceptance_candidate_recovery_review",
                    "resolved_timestamp": 900.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_followup",
                }
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=1000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert summary["stage_e_observed_acceptance_candidate_recovery_review_followup_count"] == 1
    assert summary["stage_e_observed_acceptance_candidate_recovery_review_followup_in_progress"] is True
    assert repair_loop["promotion_review_followup_in_progress"] is True
    assert repair_loop["promotion_review_followup_latest_status"] == "pending"


def test_stage_e_observed_candidate_recovery_review_failed_followup_is_summarized():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "stage_e_observed_acceptance_candidate_recovery_review_results": [
                {
                    "status": "failed",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup",
                    "command": "followup_stage_e_observed_acceptance_candidate_recovery_review",
                    "resolved_timestamp": 900.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_followup",
                }
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=1000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert summary["stage_e_observed_acceptance_candidate_recovery_review_followup_failed"] is True
    assert repair_loop["promotion_review_followup_failed"] is True
    assert repair_loop["promotion_review_followup_latest_status"] == "failed"


def test_stage_e_observed_candidate_recovery_review_followup_retry_is_summarized():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "stage_e_observed_acceptance_candidate_recovery_review_results": [
                {
                    "status": "failed",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup",
                    "command": "followup_stage_e_observed_acceptance_candidate_recovery_review",
                    "resolved_timestamp": 900.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_followup",
                },
                {
                    "status": "pending",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                    "command": "retry_stage_e_observed_acceptance_candidate_recovery_review_followup",
                    "resolved_timestamp": 950.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                },
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=1000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert summary["stage_e_observed_acceptance_candidate_recovery_review_followup_count"] == 2
    assert summary["stage_e_observed_acceptance_candidate_recovery_review_followup_retry_count"] == 1
    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_in_progress"
    ] is True
    assert repair_loop["promotion_review_followup_retry_in_progress"] is True
    assert repair_loop["promotion_review_followup_latest_status"] == "pending"
    assert repair_loop["promotion_review_followup_retry_latest_status"] == "pending"


def test_stage_e_observed_candidate_recovery_review_followup_retry_escalation_is_summarized():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "stage_e_observed_acceptance_candidate_recovery_review_results": [
                {
                    "status": "failed",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                    "command": "retry_stage_e_observed_acceptance_candidate_recovery_review_followup",
                    "resolved_timestamp": 900.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                },
                {
                    "status": "pending",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation",
                    "command": "escalate_stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                    "resolved_timestamp": 950.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation",
                },
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=1000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_count"
    ] == 1
    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_in_progress"
    ] is True
    assert repair_loop["promotion_review_followup_retry_failed"] is True
    assert repair_loop["promotion_review_followup_retry_escalation_in_progress"] is True
    assert repair_loop["promotion_review_followup_retry_escalation_latest_status"] == "pending"


def test_stage_e_observed_candidate_recovery_review_evidence_collection_is_summarized():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "stage_e_observed_acceptance_candidate_recovery_review_results": [
                {
                    "status": "failed",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation",
                    "command": "escalate_stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                    "resolved_timestamp": 900.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation",
                },
                {
                    "status": "pending",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_evidence_collection",
                    "command": "collect_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                    "resolved_timestamp": 950.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection",
                },
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=1000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_count"
    ] == 1
    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_in_progress"
    ] is True
    assert repair_loop["promotion_review_followup_retry_escalation_failed"] is True
    assert repair_loop["promotion_review_evidence_collection_in_progress"] is True
    assert repair_loop["promotion_review_evidence_collection_latest_status"] == "pending"


def test_stage_e_observed_candidate_recovery_review_evidence_recheck_is_summarized():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "stage_e_observed_acceptance_candidate_recovery_review_results": [
                {
                    "status": "success",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_evidence_collection",
                    "command": "collect_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                    "resolved_timestamp": 900.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection",
                },
                {
                    "status": "pending",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck",
                    "command": "recheck_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                    "resolved_timestamp": 950.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck",
                },
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=1000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert summary["stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_count"] == 1
    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_in_progress"
    ] is True
    assert repair_loop["promotion_review_evidence_collection_completed"] is True
    assert repair_loop["promotion_review_evidence_recheck_in_progress"] is True
    assert repair_loop["promotion_review_evidence_recheck_latest_status"] == "pending"


def test_stage_e_observed_candidate_recovery_review_targeted_probe_is_summarized():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "stage_e_observed_acceptance_candidate_recovery_review_results": [
                {
                    "status": "failed",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck",
                    "command": "recheck_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                    "resolved_timestamp": 900.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck",
                },
                {
                    "status": "pending",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
                    "command": "probe_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                    "resolved_timestamp": 950.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
                },
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=1000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_count"
    ] == 1
    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_in_progress"
    ] is True
    assert repair_loop["promotion_review_evidence_recheck_failed"] is True
    assert repair_loop["promotion_review_targeted_probe_in_progress"] is True
    assert repair_loop["promotion_review_targeted_probe_latest_status"] == "pending"


def test_stage_e_observed_candidate_recovery_review_targeted_probe_recheck_is_summarized():
    module = _load_script()
    repair_id = "stage_e_observed_acceptance_candidate_repair"
    entries = [
        {
            "generated_at": 100.0,
            "seen_count": 1,
            "negative_results": [{"metric": repair_id}],
            "regression_watchlist": [{"id": repair_id}],
            "next_hypotheses": [{"id": repair_id}],
            "stage_e_observed_acceptance_candidate_recovery_review_results": [
                {
                    "status": "success",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
                    "command": "probe_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                    "resolved_timestamp": 900.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
                },
                {
                    "status": "pending",
                    "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck",
                    "command": "recheck_stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
                    "resolved_timestamp": 950.0,
                    "target_ids": [repair_id],
                    "review_type": "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck",
                },
            ],
        }
    ]

    summary = module.summarize_research_journal_entries(entries, now_timestamp=1000.0)
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]

    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_count"
    ] == 1
    assert summary[
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_in_progress"
    ] is True
    assert repair_loop["promotion_review_targeted_probe_completed"] is True
    assert repair_loop["promotion_review_targeted_probe_recheck_in_progress"] is True
    assert repair_loop["promotion_review_targeted_probe_recheck_latest_status"] == "pending"


def test_attach_remeasure_results_to_research_journal_entries_links_repair_log():
    module = _load_script()
    entries = [
        {
            "generated_at": 100.0,
            "negative_results": [{"metric": "predictive_spike_entropy_reduction_observed"}],
            "regression_watchlist": [],
            "next_hypotheses": [],
        }
    ]
    repair_entries = [
        {
            "command": "python scripts/eval/cognitive_runtime_benchmark.py",
            "status": "success",
            "source": "runbook_action:research_journal_remeasure",
            "covered_checks": [
                "research_journal_summary",
                "predictive_spike_entropy_reduction_observed",
            ],
            "resolved_timestamp": 300.0,
        }
    ]

    updated, sync = module.attach_remeasure_results_to_research_journal_entries(
        entries,
        repair_entries,
    )
    repeated, repeat_sync = module.attach_remeasure_results_to_research_journal_entries(
        updated,
        repair_entries,
    )

    assert sync["linked_count"] == 1
    assert sync["status_counts"] == {"success": 1}
    assert updated[0]["remeasure_results"][0]["target_ids"] == [
        "predictive_spike_entropy_reduction_observed"
    ]
    assert repeat_sync["linked_count"] == 0
    assert repeat_sync["skipped_duplicate_count"] == 1
    assert repeated == updated


def test_attach_alternative_probe_results_to_research_journal_entries_links_separately():
    module = _load_script()
    entries = [
        {
            "generated_at": 100.0,
            "negative_results": [{"metric": "predictive_spike_entropy_reduction_observed"}],
            "regression_watchlist": [],
            "next_hypotheses": [],
        }
    ]
    repair_entries = [
        {
            "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_cognitive_runtime_benchmark_returns_expected_metrics",
            "status": "success",
            "source": "runbook_action:research_journal_alternative_probe",
            "covered_checks": [
                "research_journal_summary",
                "predictive_spike_entropy_reduction_observed",
                "remeasure_quota_hold",
            ],
            "resolved_timestamp": 320.0,
        }
    ]

    remeasured, remeasure_sync = module.attach_remeasure_results_to_research_journal_entries(
        entries,
        repair_entries,
    )
    updated, sync = module.attach_alternative_probe_results_to_research_journal_entries(
        remeasured,
        repair_entries,
    )
    summary = module.summarize_research_journal_entries(updated, now_timestamp=400.0)

    assert remeasure_sync["linked_count"] == 0
    assert sync["linked_count"] == 1
    assert updated[0]["alternative_probe_results"][0]["target_ids"] == [
        "predictive_spike_entropy_reduction_observed"
    ]
    assert "remeasure_results" not in updated[0]
    assert summary["alternative_probe_result_count"] == 1
    assert summary["alternative_probe_status_counts"] == {"success": 1}
    assert summary["alternative_probe_trends"][0]["id"] == (
        "predictive_spike_entropy_reduction_observed"
    )
    assert summary["alternative_probe_trends"][0]["trend"] == "targeted_probe_passed"


def test_rejected_roadmap_patch_reason_suppresses_repeated_patch_suggestion():
    module = _load_script()
    phase3 = {
        "passed": True,
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "predictive_spike_entropy_reduction_observed": 0.4,
                    "phase_binding_coincidence_integrity_observed": 1.0,
                    "forward_only_local_update_stability_observed": 1.0,
                }
            },
            "energy_efficiency": {
                "neuromorphic_profile_trend": {"regression_count": 0, "policy_change_count": 0}
            },
        },
        "linear_snn_fusion_observed_trend": {"regression_count": 0},
    }
    journal_summary = {
        "roadmap_patch_rejected_items": [
            {
                "id": "linear_snn_fusion_metric_recovery",
                "count": 1,
                "latest_reason": "Needs real-data evidence.",
                "latest_timestamp": 100.0,
            }
        ],
    }

    report = module.build_research_review_report(
        phase3_report=phase3,
        release_soak_report={"release_gate": {"passed": True}},
        operational_report={"passed": True},
        input_snapshots=[],
        generated_at=200.0,
        research_journal_summary=journal_summary,
    )
    suggestion = module.build_roadmap_patch_suggestion(report)
    compact = module.compact_research_review_report(report)
    next_hypothesis = report["experiment_planner"]["next_hypotheses"][0]

    assert next_hypothesis["roadmap_patch_review_suppressed"] is True
    assert next_hypothesis["roadmap_patch_review_rejection_reason"] == "Needs real-data evidence."
    assert next_hypothesis["requires_additional_evidence"] is True
    assert all("linear_snn_fusion_metric_recovery" not in line for line in suggestion["suggestions"])
    assert suggestion["suppressed_rejected_items"][0] == {
        "id": "linear_snn_fusion_metric_recovery",
        "reason": "Needs real-data evidence.",
    }
    assert compact["roadmap_patch_rejection_suppressed_count"] == 1
    assert compact["roadmap_patch_rejection_refreshed_count"] == 0


def test_rejected_roadmap_patch_suppression_lifts_after_refresh_evidence():
    module = _load_script()
    phase3 = {
        "passed": True,
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "predictive_spike_entropy_reduction_observed": 0.4,
                    "phase_binding_coincidence_integrity_observed": 1.0,
                    "forward_only_local_update_stability_observed": 1.0,
                }
            },
            "energy_efficiency": {
                "neuromorphic_profile_trend": {"regression_count": 0, "policy_change_count": 0}
            },
        },
        "linear_snn_fusion_observed_trend": {"regression_count": 0},
    }
    journal_summary = {
        "roadmap_patch_rejected_items": [
            {
                "id": "linear_snn_fusion_metric_recovery",
                "count": 1,
                "latest_reason": "Needs real-data evidence.",
                "latest_timestamp": 100.0,
            }
        ],
        "alternative_probe_trends": [
            {
                "id": "linear_snn_fusion_metric_recovery",
                "trend": "targeted_probe_passed",
                "latest_status": "success",
                "latest_timestamp": 180.0,
                "latest_command": "python scripts/eval/cognitive_runtime_benchmark.py",
            }
        ],
    }

    report = module.build_research_review_report(
        phase3_report=phase3,
        release_soak_report={"release_gate": {"passed": True}},
        operational_report={"passed": True},
        input_snapshots=[],
        generated_at=200.0,
        research_journal_summary=journal_summary,
    )
    suggestion = module.build_roadmap_patch_suggestion(report)
    compact = module.compact_research_review_report(report)
    next_hypothesis = report["experiment_planner"]["next_hypotheses"][0]
    policy = report["experiment_planner"]["remeasure_priority_policy"]

    assert next_hypothesis["roadmap_patch_review_suppression_lifted"] is True
    assert next_hypothesis["roadmap_patch_review_refresh_reason"] == "targeted_probe_passed"
    assert "roadmap_patch_review_suppressed" not in next_hypothesis
    assert policy["roadmap_patch_rejection_refreshed_count"] == 1
    assert policy["roadmap_patch_rejection_suppressed_count"] == 0
    assert compact["roadmap_patch_rejection_suppressed_count"] == 0
    assert compact["roadmap_patch_rejection_refreshed_count"] == 1
    assert any("linear_snn_fusion_metric_recovery" in line for line in suggestion["suggestions"])
    assert suggestion["suppressed_rejected_items"] == []


def test_refreshed_roadmap_patch_is_not_resurfaced_without_new_evidence():
    module = _load_script()
    phase3 = {
        "passed": True,
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "predictive_spike_entropy_reduction_observed": 0.4,
                    "phase_binding_coincidence_integrity_observed": 1.0,
                    "forward_only_local_update_stability_observed": 1.0,
                }
            },
            "energy_efficiency": {
                "neuromorphic_profile_trend": {"regression_count": 0, "policy_change_count": 0}
            },
        },
        "linear_snn_fusion_observed_trend": {"regression_count": 0},
    }
    long_run_entries = [
        {
            "generated_at": 100.0,
            "next_hypotheses": [{"id": "linear_snn_fusion_metric_recovery"}],
            "regression_watchlist": [],
            "negative_results": [],
            "roadmap_patch_review_decision": "rejected",
            "roadmap_patch_review_reason": "Needs real-data evidence.",
        },
        {
            "generated_at": 220.0,
            "next_hypotheses": [
                {
                    "id": "linear_snn_fusion_metric_recovery",
                    "roadmap_patch_review_suppression_lifted": True,
                    "roadmap_patch_review_refresh_reason": "targeted_probe_passed",
                    "roadmap_patch_review_refresh_timestamp": 180.0,
                }
            ],
            "regression_watchlist": [],
            "negative_results": [],
            "alternative_probe_results": [
                {
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "status": "success",
                    "resolved_timestamp": 180.0,
                    "target_ids": ["linear_snn_fusion_metric_recovery"],
                }
            ],
        },
    ]
    journal_summary = module.summarize_research_journal_entries(
        long_run_entries,
        now_timestamp=260.0,
    )

    report = module.build_research_review_report(
        phase3_report=phase3,
        release_soak_report={"release_gate": {"passed": True}},
        operational_report={"passed": True},
        input_snapshots=[],
        generated_at=260.0,
        research_journal_summary=journal_summary,
    )
    suggestion = module.build_roadmap_patch_suggestion(report)
    next_hypothesis = report["experiment_planner"]["next_hypotheses"][0]

    assert journal_summary["roadmap_patch_refreshed_items"][0]["id"] == (
        "linear_snn_fusion_metric_recovery"
    )
    assert journal_summary["roadmap_patch_rejected_item_count"] == 1
    assert journal_summary["roadmap_patch_refreshed_item_count"] == 1
    assert journal_summary["roadmap_patch_refresh_to_rejection_ratio"] == 1.0
    assert next_hypothesis["roadmap_patch_review_suppressed"] is True
    assert next_hypothesis["roadmap_patch_review_suppression_reason"] == (
        "targeted_probe_refresh_already_surfaced"
    )
    assert all("linear_snn_fusion_metric_recovery" not in line for line in suggestion["suggestions"])


def test_remeasure_trends_adjust_experiment_planner_priority():
    module = _load_script()
    report = module.build_research_review_report(
        phase3_report=_phase3_report(passed=False, linear_value=0.5),
        release_soak_report={"passed": False},
        operational_report={"passed": False},
        input_snapshots=[],
        generated_at=400.0,
        research_journal_summary={
            "remeasure_trends": [
                {
                    "id": "linear_snn_fusion_metric_recovery",
                    "trend": "recovered",
                    "latest_status": "success",
                    "success_count": 1,
                    "failed_count": 1,
                    "skipped_count": 0,
                },
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "trend": "still_failing",
                    "latest_status": "failed",
                    "success_count": 0,
                    "failed_count": 2,
                    "skipped_count": 0,
                },
            ],
        },
    )

    next_hypothesis = report["experiment_planner"]["next_hypotheses"][0]
    negative_result = [
        item
        for item in report["experiment_planner"]["negative_results"]
        if item["metric"] == "predictive_spike_entropy_reduction_observed"
    ][0]
    policy = report["experiment_planner"]["remeasure_priority_policy"]

    assert next_hypothesis["priority"] == "medium"
    assert next_hypothesis["priority_adjustment"] == "deprioritized_after_remeasure_recovery"
    assert next_hypothesis["recommended_remeasure_interval_seconds"] == (
        module.RECOVERED_REMEASURE_INTERVAL_SECONDS
    )
    assert negative_result["priority"] == "high"
    assert negative_result["priority_adjustment"] == "escalated_after_remeasure_failure"
    assert negative_result["recommended_remeasure_interval_seconds"] == (
        module.FAILED_REMEASURE_INTERVAL_SECONDS
    )
    assert policy["deprioritized_recovered_count"] == 1
    assert policy["escalated_still_failing_count"] == 1


def test_alternative_probe_trends_add_planner_branches_and_patch_suggestions():
    module = _load_script()
    report = module.build_research_review_report(
        phase3_report=_phase3_report(passed=False, linear_value=0.5),
        release_soak_report={"passed": False},
        operational_report={"passed": False},
        input_snapshots=[],
        generated_at=410.0,
        research_journal_summary={
            "alternative_probe_trends": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "trend": "targeted_probe_passed",
                    "latest_status": "success",
                    "latest_command": "targeted-predictive-probe",
                },
                {
                    "id": "phase_binding_coincidence_integrity_observed",
                    "trend": "targeted_probe_failed",
                    "latest_status": "failed",
                    "latest_command": "targeted-phase-probe",
                },
            ]
        },
    )
    planner = report["experiment_planner"]
    compact = module.compact_research_review_report(report)
    suggestion = module.build_roadmap_patch_suggestion(report)

    assert planner["cause_boundary_documentation_tasks"][0]["id"] == (
        "predictive_spike_entropy_reduction_observed"
    )
    assert planner["cause_boundary_documentation_tasks"][0]["latest_command"] == (
        "targeted-predictive-probe"
    )
    assert planner["targeted_fixture_repair_tasks"][0]["id"] == (
        "phase_binding_coincidence_integrity_observed"
    )
    assert planner["targeted_fixture_repair_tasks"][0]["priority"] == "high"
    assert planner["remeasure_priority_policy"]["cause_boundary_documentation_count"] == 1
    assert planner["remeasure_priority_policy"]["targeted_fixture_repair_count"] == 1
    assert compact["cause_boundary_documentation_count"] == 1
    assert compact["targeted_fixture_repair_count"] == 1
    assert any(
        "DOC: document targeted-probe boundary for `predictive_spike_entropy_reduction_observed`"
        in item
        for item in suggestion["suggestions"]
    )
    assert any(
        "FIXTURE: add or repair minimal targeted fixture for `phase_binding_coincidence_integrity_observed`"
        in item
        for item in suggestion["suggestions"]
    )


def test_evidence_collection_history_prioritizes_next_evidence_kind():
    module = _load_script()
    report = module.build_research_review_report(
        phase3_report=_phase3_report(passed=False, linear_value=0.5),
        release_soak_report={"passed": False},
        operational_report={"passed": False},
        input_snapshots=[],
        generated_at=415.0,
        research_journal_summary={
            "roadmap_patch_rejected_items": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "count": 1,
                    "latest_reason": "Needs real-data evidence.",
                    "latest_timestamp": 100.0,
                }
            ],
            "roadmap_patch_evidence_collection_kind_counts": {"targeted_probe": 1},
            "roadmap_patch_evidence_collection_next_required_kind": "real_data_fixture",
            "alternative_probe_trends": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "trend": "targeted_probe_passed",
                    "latest_status": "success",
                    "latest_command": "targeted-predictive-probe",
                },
            ],
        },
    )
    planner = report["experiment_planner"]
    compact = module.compact_research_review_report(report)
    suggestion = module.build_roadmap_patch_suggestion(report)

    assert planner["cause_boundary_documentation_tasks"] == []
    assert planner["roadmap_patch_evidence_collection_tasks"][0]["id"] == (
        "predictive_spike_entropy_reduction_observed"
    )
    assert planner["roadmap_patch_evidence_collection_tasks"][0]["evidence_kind"] == (
        "real_data_fixture"
    )
    assert planner["remeasure_priority_policy"]["roadmap_patch_evidence_collection_count"] == 1
    assert compact["roadmap_patch_evidence_collection_count"] == 1
    assert compact["roadmap_patch_evidence_collection_ids"] == [
        "predictive_spike_entropy_reduction_observed"
    ]
    assert any(
        "EVIDENCE: collect `real_data_fixture` for `predictive_spike_entropy_reduction_observed`"
        in item
        for item in suggestion["suggestions"]
    )
    assert all("DOC: document targeted-probe boundary" not in item for item in suggestion["suggestions"])


def test_completed_evidence_collection_tasks_are_not_reopened():
    module = _load_script()
    item_id = "predictive_spike_entropy_reduction_observed"
    entries = [
        {
            "generated_at": 100.0,
            "roadmap_patch_review_decision": "rejected",
            "roadmap_patch_review_reason": "Needs real-data evidence.",
            "negative_results": [{"metric": item_id}],
            "regression_watchlist": [],
            "next_hypotheses": [],
            "roadmap_patch_evidence_collection_tasks": [
                {
                    "id": item_id,
                    "evidence_kind": "real_data_fixture",
                }
            ],
        }
    ]
    repair_entries = [
        {
            "command": "collect-real-data-fixture",
            "status": "success",
            "source": "manual:roadmap_patch_refresh_policy:evidence_collection",
            "covered_checks": [
                "roadmap_patch_refresh_policy",
                "evidence_collection",
                "real_data_fixture",
                item_id,
            ],
            "resolved_timestamp": 500.0,
        }
    ]

    updated, sync = module.attach_roadmap_patch_evidence_collection_completions_to_research_journal_entries(
        entries,
        repair_entries,
    )
    repeated, repeat_sync = module.attach_roadmap_patch_evidence_collection_completions_to_research_journal_entries(
        updated,
        repair_entries,
    )
    summary = module.summarize_research_journal_entries(updated, now_timestamp=600.0)
    summary["roadmap_patch_evidence_collection_next_required_kind"] = "real_data_fixture"
    report = module.build_research_review_report(
        phase3_report=_phase3_report(passed=False, linear_value=0.5),
        release_soak_report={"passed": False},
        operational_report={"passed": False},
        input_snapshots=[],
        generated_at=610.0,
        research_journal_summary=summary,
    )

    assert sync["linked_count"] == 1
    assert sync["evidence_kind_counts"] == {"real_data_fixture": 1}
    assert repeat_sync["linked_count"] == 0
    assert repeat_sync["skipped_duplicate_count"] == 1
    assert repeated == updated
    assert updated[0]["completed_roadmap_patch_evidence_collection_keys"] == [
        f"{item_id}:real_data_fixture"
    ]
    assert summary["completed_roadmap_patch_evidence_collection_count"] == 1
    assert summary["completed_roadmap_patch_evidence_collection_keys"] == [
        f"{item_id}:real_data_fixture"
    ]
    assert report["experiment_planner"]["roadmap_patch_evidence_collection_tasks"] == []
    assert report["experiment_planner"]["remeasure_priority_policy"][
        "completed_roadmap_patch_evidence_collection_count"
    ] == 1


def test_completed_evidence_collection_review_marks_unreflected_keys():
    module = _load_script()
    summary = {
        "completed_roadmap_patch_evidence_collection_keys": [
            "predictive_spike_entropy_reduction_observed:real_data_fixture",
            "phase_binding_coincidence_integrity_observed:targeted_probe",
        ],
        "roadmap_patch_refreshed_items": [
            {"id": "phase_binding_coincidence_integrity_observed"},
        ],
    }

    review = module.summarize_completed_roadmap_patch_evidence_review(summary)

    assert review["completed_count"] == 2
    assert review["refreshed_id_count"] == 1
    assert review["pending_review_count"] == 1
    assert review["pending_review_keys"] == [
        "predictive_spike_entropy_reduction_observed:real_data_fixture"
    ]
    assert review["needs_review"] is True


def test_completed_alternative_probe_planner_tasks_are_not_reopened():
    module = _load_script()
    report = module.build_research_review_report(
        phase3_report=_phase3_report(passed=False, linear_value=0.5),
        release_soak_report={"passed": False},
        operational_report={"passed": False},
        input_snapshots=[],
        generated_at=420.0,
        research_journal_summary={
            "alternative_probe_trends": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "trend": "targeted_probe_passed",
                    "latest_status": "success",
                    "latest_command": "targeted-predictive-probe",
                },
                {
                    "id": "phase_binding_coincidence_integrity_observed",
                    "trend": "targeted_probe_failed",
                    "latest_status": "failed",
                    "latest_command": "targeted-phase-probe",
                },
            ],
            "completed_cause_boundary_documentation_ids": [
                "predictive_spike_entropy_reduction_observed"
            ],
            "completed_targeted_fixture_repair_ids": [
                "phase_binding_coincidence_integrity_observed"
            ],
        },
    )
    planner = report["experiment_planner"]
    compact = module.compact_research_review_report(report)

    assert planner["cause_boundary_documentation_tasks"] == []
    assert planner["targeted_fixture_repair_tasks"] == []
    assert planner["remeasure_priority_policy"]["completed_cause_boundary_documentation_count"] == 1
    assert planner["remeasure_priority_policy"]["completed_targeted_fixture_repair_count"] == 1
    assert compact["cause_boundary_documentation_count"] == 0
    assert compact["targeted_fixture_repair_count"] == 0


def test_attach_research_planner_task_completions_updates_journal_entries():
    module = _load_script()
    entries = [
        {
            "generated_at": 100.0,
            "negative_results": [{"metric": "predictive_spike_entropy_reduction_observed"}],
            "regression_watchlist": [{"id": "phase_binding_coincidence_integrity_observed"}],
            "next_hypotheses": [],
        }
    ]
    repair_entries = [
        {
            "command": "doc-boundary",
            "status": "success",
            "source": "manual:cause_boundary_documentation",
            "covered_checks": [
                "cause_boundary_documentation",
                "predictive_spike_entropy_reduction_observed",
            ],
            "resolved_timestamp": 500.0,
        },
        {
            "command": "fix-fixture",
            "status": "success",
            "source": "manual:targeted_fixture_repair",
            "covered_checks": [
                "targeted_fixture_repair",
                "phase_binding_coincidence_integrity_observed",
            ],
            "resolved_timestamp": 510.0,
        },
    ]

    updated, sync = module.attach_research_planner_task_completions_to_research_journal_entries(
        entries,
        repair_entries,
    )
    repeated, repeat_sync = module.attach_research_planner_task_completions_to_research_journal_entries(
        updated,
        repair_entries,
    )
    summary = module.summarize_research_journal_entries(updated, now_timestamp=600.0)

    assert sync["linked_count"] == 2
    assert sync["task_type_counts"] == {
        "cause_boundary_documentation": 1,
        "targeted_fixture_repair": 1,
    }
    assert updated[0]["completed_cause_boundary_documentation_ids"] == [
        "predictive_spike_entropy_reduction_observed"
    ]
    assert updated[0]["completed_targeted_fixture_repair_ids"] == [
        "phase_binding_coincidence_integrity_observed"
    ]
    assert repeat_sync["linked_count"] == 0
    assert repeat_sync["skipped_duplicate_count"] == 2
    assert repeated == updated
    assert summary["completed_research_planner_task_count"] == 2
    assert summary["completed_cause_boundary_documentation_ids"] == [
        "predictive_spike_entropy_reduction_observed"
    ]
    assert summary["completed_targeted_fixture_repair_ids"] == [
        "phase_binding_coincidence_integrity_observed"
    ]


def test_research_automation_release_soak_mode_uses_release_gate_without_operational_report():
    module = _load_script()
    report = module.build_research_review_report(
        phase3_report=_phase3_report(passed=True),
        release_soak_report={"release_gate": {"passed": True}},
        operational_report=None,
        input_snapshots=[],
        require_operational_readiness=False,
        generated_at=789.0,
    )

    assert report["signals"]["release_safety"]["ready"] is True
    assert "operational_readiness_passed" not in report["signals"]["release_safety"]["checks"]
    assert report["passed"] is True
