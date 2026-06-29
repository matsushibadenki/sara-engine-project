import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import workspace_path


def _load_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "sara_ann_comparison_report.py",
        )
    )
    spec = importlib.util.spec_from_file_location("sara_ann_comparison_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _external_validity_report(with_real_reference: bool = False):
    return {
        "passed": True,
        "metrics": {
            "real_data_qa_accuracy": 1.0,
            "ann_cost_advantage_proxy": 6.0,
            "bm25_offline_proxy_qa_accuracy": 1.0,
            "dense_embedding_ann_proxy_qa_accuracy": 1.0,
            "dense_embedding_ann_cost_advantage_proxy": 4.0,
            "ann_proxy_qa_accuracy": 1.0,
            "ann_proxy_avg_latency_ms": 4.0,
            "dense_embedding_ann_proxy_avg_latency_ms": 3.0,
            "bm25_offline_cost_advantage_proxy": 3.0,
            "per_task_external_validity_summary_available": 1.0,
            "real_pretrained_embedding_reference_available": 1.0 if with_real_reference else 0.0,
            "real_pretrained_embedding_reference_qa_accuracy": 1.0 if with_real_reference else 0.0,
            "real_pretrained_embedding_reference_cost_advantage_proxy": 2.5 if with_real_reference else 0.0,
            "real_pretrained_embedding_reference_avg_latency_ms": 5.0 if with_real_reference else 0.0,
            "real_pretrained_embedding_faiss_reference_available": 0.0,
            "real_pretrained_embedding_faiss_reference_qa_accuracy": 0.0,
            "real_pretrained_embedding_faiss_reference_cost_advantage_proxy": 0.0,
            "real_pretrained_embedding_faiss_reference_avg_latency_ms": 0.0,
            "real_cross_encoder_reference_available": 0.0,
            "real_cross_encoder_reference_qa_accuracy": 0.0,
            "real_cross_encoder_reference_cost_advantage_proxy": 0.0,
            "real_cross_encoder_reference_avg_latency_ms": 0.0,
            "reference_ready_count": 1.0 if with_real_reference else 0.0,
            "reference_configured_count": 1.0 if with_real_reference else 0.0,
            "reference_dependency_error_count": 0.0,
        },
        "checks": {"trend.no_regressions": True},
        "ann_pretrained_embedding_reference": {
            "available": with_real_reference,
            "reason": "" if with_real_reference else "not_configured",
        },
        "ann_pretrained_embedding_faiss_reference": {
            "available": False,
            "reason": "not_configured",
        },
        "ann_cross_encoder_reference": {
            "available": False,
            "reason": "not_configured",
        },
        "reference_readiness": {
            "status": "partial_reference_ready" if with_real_reference else "proxy_only",
            "configured_reference_count": 1 if with_real_reference else 0,
            "ready_reference_count": 1 if with_real_reference else 0,
            "dependency_error_count": 0,
            "references": [
                {
                    "reference_id": "ann_pretrained_embedding_reference",
                    "label": "Local Pretrained Embedding Reference",
                    "configured_path": "workspace/models/local-embedding" if with_real_reference else "",
                    "available": with_real_reference,
                    "reason": "" if with_real_reference else "not_configured",
                },
                {
                    "reference_id": "ann_pretrained_embedding_faiss_reference",
                    "label": "Local Pretrained Embedding FAISS Reference",
                    "configured_path": "",
                    "available": False,
                    "reason": "not_configured",
                },
                {
                    "reference_id": "ann_cross_encoder_reference",
                    "label": "Local Cross-Encoder Reference",
                    "configured_path": "",
                    "available": False,
                    "reason": "not_configured",
                },
            ],
            "next_actions": [],
        },
        "bm25_offline_proxy": {
            "accuracy": 1.0,
            "avg_latency_ms": 2.0,
        },
    }


def _ladder_report():
    return {
        "passed": True,
        "metrics": {"profile_count": 3.0},
        "checks": {"all_profiles_passed": True},
    }


def _energy_measurement_report(real: bool = False):
    return {
        "real_joule_measurements_present": real,
        "checks": {
            "quality_parity_passed": real,
            "paired_task_rows_balanced": real,
        },
        "metrics": {
            "paired_task_count": 1.0 if real else 0.0,
            "ann_to_sara_joule_efficiency_ratio": 4.0 if real else 0.0,
            "maintenance_trace_rows_present": real,
            "sara_maintenance_event_cost_per_success": 0.06 if real else 0.0,
            "ann_maintenance_event_cost_per_success": 0.02 if real else 0.0,
        },
        "maintenance_alignment": (
            {
                "available": True,
                "sara_physical_maintenance_event_cost_per_selected": 0.05,
                "reference_maintenance_event_cost_per_selected": 1.5,
                "maintenance_event_cost_per_selected_ratio": 0.0333333333,
                "maintenance_event_cost_per_selected_delta": -1.45,
            }
            if real
            else {"available": False}
        ),
    }


def _internal_maintenance_report():
    return {
        "passed": True,
        "observed_only": True,
        "counts": {
            "maintenance_selected_count": 4,
            "maintenance_refresh_count": 2,
            "maintenance_idle_self_state_ok_count": 3,
        },
        "normalized_metrics": {
            "maintenance_event_cost": 6.0,
            "maintenance_event_cost_per_selected": 1.5,
        },
        "metrics": {
            "maintenance_self_state_continuity_observed": 1.0,
            "maintenance_cache_refresh_observed": 1.0,
        },
    }


def _event_memory_report():
    return {
        "passed": True,
        "counts": {
            "observed_events": 4,
            "episodes": 1,
            "verified_relations": 7,
        },
        "metrics": {
            "eventization_emission_ratio": 0.8,
            "candidate_event_acceptance_rate": 1.0,
            "episode_compression_ratio": 5.0,
            "relation_verification_yield": 1.0,
            "lineage_coverage_ratio": 1.0,
            "self_state_continuity": 0.2857142857,
            "self_state_external_event_ratio": 1.4,
        },
    }


def _event_memory_maintenance_coupling_report():
    return {
        "passed": True,
        "profile_count": 3,
        "best_profile": {
            "profile_id": "wide",
        },
        "metrics": {
            "compression_to_maintenance_correlation": 0.5121754721,
            "best_profile_compression_efficiency_per_maintenance": 0.1896551724,
            "best_profile_self_state_continuity": 0.8333333333,
            "best_profile_episode_compression_ratio": 3.6666666667,
        },
    }


def test_sara_ann_comparison_report_marks_proxy_only_surface_when_real_reference_missing():
    module = _load_module()

    report = module.build_sara_ann_comparison_report(
        external_validity_report=_external_validity_report(with_real_reference=False),
        external_ladder_report=_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real=False),
    )

    assert report["schema"] == "sara-ann-comparison-report-v1"
    assert report["passed"] is False
    assert report["status"] == "proxy_only_or_partial_reference_surface"
    assert report["checks"]["bm25_reference_present"] is True
    assert report["checks"]["stronger_real_reference_present"] is False
    assert report["checks"]["reference_readiness_visible"] is True
    assert report["best_available_offline_reference"]["baseline_id"] == "bm25_offline_proxy"
    assert report["artifact_state"]["internal_maintenance_reference_present"] is False
    assert report["maintenance_surface"]["available"] is False
    assert report["compression_surface"]["available"] is False
    assert report["compression_maintenance_coupling_surface"]["available"] is False
    assert report["next_action_count"] >= 3
    assert report["next_actions"][0]["baseline_id"] == "ann_cross_encoder_reference"
    assert any(
        action.get("category") == "missing_internal_maintenance_reference"
        for action in report["next_actions"]
    )
    assert any(
        action.get("category") == "missing_event_memory_compression_surface"
        for action in report["next_actions"]
    )
    assert any(
        action.get("category") == "missing_event_memory_maintenance_coupling_surface"
        for action in report["next_actions"]
    )
    summary = module.format_sara_ann_comparison_summary(report)
    assert "SARA ANN Comparison Report" in summary
    assert "bm25_offline_proxy" in summary
    assert "Reference Readiness:" in summary
    assert "Maintenance Surface:" in summary


def test_sara_ann_comparison_report_accepts_phase6_and_phase8_surface_when_reference_and_physical_exist():
    module = _load_module()

    report = module.build_sara_ann_comparison_report(
        external_validity_report=_external_validity_report(with_real_reference=True),
        external_ladder_report=_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real=True),
        internal_maintenance_report=_internal_maintenance_report(),
        event_memory_report=_event_memory_report(),
        event_memory_maintenance_coupling_report=_event_memory_maintenance_coupling_report(),
    )

    assert report["passed"] is True
    assert report["status"] == "phase6_and_phase8_evidence_surface_ready"
    assert report["checks"]["stronger_real_reference_present"] is True
    assert report["checks"]["physical_quality_guard_passed"] is True
    assert report["best_available_offline_reference"]["baseline_id"] == "ann_pretrained_embedding_reference"
    assert report["maintenance_surface"]["available"] is True
    assert report["maintenance_surface"]["maintenance_event_cost_per_selected"] == 1.5
    assert report["maintenance_surface"]["physical_alignment_available"] is True
    assert report["compression_surface"]["available"] is True
    assert report["compression_surface"]["episode_compression_ratio"] == 5.0
    assert report["compression_maintenance_coupling_surface"]["available"] is True
    assert report["compression_maintenance_coupling_surface"]["best_profile_id"] == "wide"
    assert report["metrics"]["event_memory_episode_compression_ratio"] == 5.0
    assert report["metrics"]["event_memory_maintenance_best_profile"] == "wide"
    assert report["metrics"]["physical_maintenance_event_cost_per_selected"] == 0.05
    assert report["metrics"]["sara_maintenance_event_cost_per_success"] == 0.06
    physical = [card for card in report["baseline_cards"] if card["baseline_id"] == "physical_ann_measurement"][0]
    assert physical["available"] is True
    assert physical["cost_score"] == 4.0
    summary = module.format_sara_ann_comparison_summary(report)
    assert "physical_alignment=True" in summary
    assert "alignment_ratio=0.033" in summary
    assert "Compression Surface:" in summary
    assert "episode_compression_ratio=5.000" in summary
    assert "Compression Maintenance Coupling Surface:" in summary
    assert "best_profile=wide" in summary


def test_sara_ann_comparison_report_prioritizes_missing_directory_over_not_configured():
    module = _load_module()
    external = _external_validity_report(with_real_reference=False)
    external["reference_readiness"]["configured_reference_count"] = 2
    external["reference_readiness"]["references"] = [
        {
            "reference_id": "ann_cross_encoder_reference",
            "label": "Local Cross-Encoder Reference",
            "configured_path": "workspace/models/missing-cross",
            "available": False,
            "reason": "missing_directory",
        },
        {
            "reference_id": "ann_pretrained_embedding_faiss_reference",
            "label": "Local Pretrained Embedding FAISS Reference",
            "configured_path": "",
            "available": False,
            "reason": "not_configured",
        },
        {
            "reference_id": "ann_pretrained_embedding_reference",
            "label": "Local Pretrained Embedding Reference",
            "configured_path": "workspace/models/local-embedding",
            "available": False,
            "reason": "RuntimeError",
        },
    ]
    report = module.build_sara_ann_comparison_report(
        external_validity_report=external,
        external_ladder_report=_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real=False),
        internal_maintenance_report=_internal_maintenance_report(),
        event_memory_report=_event_memory_report(),
        event_memory_maintenance_coupling_report=_event_memory_maintenance_coupling_report(),
    )

    assert report["next_actions"][0]["category"] == "missing_reference_directory"
    assert report["next_actions"][0]["baseline_id"] == "ann_cross_encoder_reference"
    assert report["next_actions"][1]["category"] == "missing_reference_dependency"


def test_sara_ann_comparison_report_main_writes_report():
    module = _load_module()
    external_path = workspace_path("evaluation", "test_sara_ann_external_validity.json")
    ladder_path = workspace_path("evaluation", "test_sara_ann_ladder.json")
    energy_path = workspace_path("evaluation", "test_sara_ann_energy.json")
    maintenance_path = workspace_path("evaluation", "test_sara_ann_internal_maintenance.json")
    report_path = workspace_path("evaluation", "test_sara_ann_comparison_report.json")
    summary_path = workspace_path("evaluation", "test_sara_ann_comparison_report.txt")
    os.makedirs(os.path.dirname(external_path), exist_ok=True)
    with open(external_path, "w", encoding="utf-8") as handle:
        json.dump(_external_validity_report(with_real_reference=False), handle)
    with open(ladder_path, "w", encoding="utf-8") as handle:
        json.dump(_ladder_report(), handle)
    with open(energy_path, "w", encoding="utf-8") as handle:
        json.dump(_energy_measurement_report(real=False), handle)
    with open(maintenance_path, "w", encoding="utf-8") as handle:
        json.dump(_internal_maintenance_report(), handle)
    event_memory_path = workspace_path("evaluation", "test_sara_ann_event_memory.json")
    with open(event_memory_path, "w", encoding="utf-8") as handle:
        json.dump(_event_memory_report(), handle)
    coupling_path = workspace_path(
        "evaluation", "test_sara_ann_event_memory_maintenance_coupling.json"
    )
    with open(coupling_path, "w", encoding="utf-8") as handle:
        json.dump(_event_memory_maintenance_coupling_report(), handle)

    try:
        exit_code = module.main(
            [
                "--external-validity-report-path",
                external_path,
                "--external-ladder-report-path",
                ladder_path,
                "--energy-measurement-report-path",
                energy_path,
                "--internal-maintenance-report-path",
                maintenance_path,
                "--event-memory-report-path",
                event_memory_path,
                "--event-memory-maintenance-coupling-report-path",
                coupling_path,
                "--report-path",
                report_path,
                "--summary-path",
                summary_path,
            ]
        )
        assert exit_code == 1
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["status"] == "proxy_only_or_partial_reference_surface"
        assert report["maintenance_surface"]["available"] is True
        assert report["compression_surface"]["available"] is True
        assert report["compression_maintenance_coupling_surface"]["available"] is True
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = handle.read()
        assert "Next Actions:" in summary
    finally:
        for path in (
            external_path,
            ladder_path,
            energy_path,
            maintenance_path,
            event_memory_path,
            coupling_path,
            report_path,
            summary_path,
        ):
            if os.path.exists(path):
                os.remove(path)


def test_sara_ann_comparison_report_requests_alignment_when_physical_exists_without_surface():
    module = _load_module()
    energy = _energy_measurement_report(real=True)
    energy["maintenance_alignment"] = {"available": False}

    report = module.build_sara_ann_comparison_report(
        external_validity_report=_external_validity_report(with_real_reference=True),
        external_ladder_report=_ladder_report(),
        energy_measurement_report=energy,
        internal_maintenance_report=_internal_maintenance_report(),
        event_memory_report=_event_memory_report(),
        event_memory_maintenance_coupling_report=_event_memory_maintenance_coupling_report(),
    )

    assert any(
        action.get("category") == "missing_physical_maintenance_alignment"
        for action in report["next_actions"]
    )


def test_sara_ann_comparison_report_requests_drift_followup_when_alignment_is_high():
    module = _load_module()
    energy = _energy_measurement_report(real=True)
    energy["maintenance_alignment"] = {
        "available": True,
        "sara_physical_maintenance_event_cost_per_selected": 3.5,
        "reference_maintenance_event_cost_per_selected": 1.5,
        "maintenance_event_cost_per_selected_ratio": 2.3333333333,
        "maintenance_event_cost_per_selected_delta": 2.0,
    }

    report = module.build_sara_ann_comparison_report(
        external_validity_report=_external_validity_report(with_real_reference=True),
        external_ladder_report=_ladder_report(),
        energy_measurement_report=energy,
        internal_maintenance_report=_internal_maintenance_report(),
        event_memory_report=_event_memory_report(),
        event_memory_maintenance_coupling_report=_event_memory_maintenance_coupling_report(),
    )

    assert any(
        action.get("category") == "maintenance_alignment_drift"
        for action in report["next_actions"]
    )


def test_sara_ann_comparison_report_requests_event_memory_surface_when_missing():
    module = _load_module()

    report = module.build_sara_ann_comparison_report(
        external_validity_report=_external_validity_report(with_real_reference=True),
        external_ladder_report=_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real=True),
        internal_maintenance_report=_internal_maintenance_report(),
    )

    assert any(
        action.get("category") == "missing_event_memory_compression_surface"
        for action in report["next_actions"]
    )


def test_sara_ann_comparison_report_requests_event_memory_coupling_surface_when_missing():
    module = _load_module()

    report = module.build_sara_ann_comparison_report(
        external_validity_report=_external_validity_report(with_real_reference=True),
        external_ladder_report=_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real=True),
        internal_maintenance_report=_internal_maintenance_report(),
        event_memory_report=_event_memory_report(),
    )

    assert any(
        action.get("category") == "missing_event_memory_maintenance_coupling_surface"
        for action in report["next_actions"]
    )


def test_sara_ann_comparison_report_requests_event_memory_followup_when_surface_is_weak():
    module = _load_module()
    weak_event_memory = _event_memory_report()
    weak_event_memory["metrics"]["episode_compression_ratio"] = 0.8
    weak_event_memory["metrics"]["relation_verification_yield"] = 0.4

    report = module.build_sara_ann_comparison_report(
        external_validity_report=_external_validity_report(with_real_reference=True),
        external_ladder_report=_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real=True),
        internal_maintenance_report=_internal_maintenance_report(),
        event_memory_report=weak_event_memory,
        event_memory_maintenance_coupling_report=_event_memory_maintenance_coupling_report(),
    )

    assert any(
        action.get("category") == "weak_event_memory_compression_surface"
        for action in report["next_actions"]
    )


def test_sara_ann_comparison_report_requests_event_memory_coupling_followup_when_surface_is_weak():
    module = _load_module()
    weak_coupling = _event_memory_maintenance_coupling_report()
    weak_coupling["metrics"]["best_profile_compression_efficiency_per_maintenance"] = 0.0
    weak_coupling["metrics"]["best_profile_self_state_continuity"] = 0.3

    report = module.build_sara_ann_comparison_report(
        external_validity_report=_external_validity_report(with_real_reference=True),
        external_ladder_report=_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real=True),
        internal_maintenance_report=_internal_maintenance_report(),
        event_memory_report=_event_memory_report(),
        event_memory_maintenance_coupling_report=weak_coupling,
    )

    assert any(
        action.get("category") == "weak_event_memory_maintenance_coupling_surface"
        for action in report["next_actions"]
    )
