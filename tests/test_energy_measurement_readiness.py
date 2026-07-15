import importlib.util
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "energy_measurement_readiness.py")
    spec = importlib.util.spec_from_file_location("energy_measurement_readiness", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    module,
    system,
    task,
    joules,
    *,
    pair_id="pair-1",
    replicate=1,
    success=10,
    trials=10,
    run_order=None,
    maintenance_selected_count=None,
    maintenance_phase_count=None,
    maintenance_refresh_count=None,
    maintenance_event_cost=None,
    measurement_quality="physical_meter",
    physical_evidence=True,
):
    return module.build_measurement_row(
        run_id=f"{system}-{task}-{replicate}",
        system=system,
        task=task,
        success_count=success,
        trial_count=trials,
        joules=joules,
        source="real_energy_session",
        pair_id=pair_id,
        replicate_index=replicate,
        environment_fingerprint="env-sha256",
        task_fixture_hash="fixture-sha256",
        success_criterion_id="exact-match-v1",
        measurement_boundary="warm-index-query-only-v1",
        measurement_tool="powermetrics-v1",
        cpu_model="test-cpu",
        thread_count=1,
        process_affinity="core-0",
        power_mode="ac-fixed",
        warmup_count=2,
        measured_repetitions=10,
        run_order=run_order or (1 if system == "sara" else 2),
        maintenance_selected_count=maintenance_selected_count,
        maintenance_phase_count=maintenance_phase_count,
        maintenance_refresh_count=maintenance_refresh_count,
        maintenance_event_cost=maintenance_event_cost,
        measurement_quality=measurement_quality,
        physical_evidence=physical_evidence,
    )


def test_energy_measurement_readiness_is_protocol_ready_without_rows():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report([])

    assert report["passed"] is True
    assert report["status"] == "protocol_ready_pending_measurements"
    assert report["real_joule_measurements_present"] is False
    assert report["checks"]["schema_ready"] is True
    assert report["checks"]["joule_efficiency_ratio_passed"] is False
    assert report["measurement_plan"]["ready_for_real_joule_claim"] is False
    assert report["measurement_plan"]["pending_pair_count"] == 4
    assert report["measurement_session_plan"]["status"] == "pending_measurement"
    assert report["measurement_session_plan"]["planned_run_count"] == 4
    assert report["measurement_session_plan"]["schema"] == "sara-energy-measurement-session-plan-v2"
    assert report["measurement_session_plan"]["pairing_matrix"]["required_paired_replicates_per_task"] == 3
    assert report["measurement_session_plan"]["planned_runs"][0]["run_id_template"].startswith(
        "ann-efficiency-real-joule-real_data_external_validity-sara-"
    )
    assert "--source real_energy_session" in report["measurement_session_plan"]["planned_runs"][0]["command_template"]
    assert "run-physical-energy-pair" in report["measurement_session_plan"]["planned_runs"][0]["pair_command_template"]
    assert (
        "--event-memory-maintenance-coupling-report-path "
        "workspace/evaluation/event_memory_maintenance_coupling_benchmark.json"
    ) in report["measurement_session_plan"]["planned_runs"][0]["pair_command_template"]
    assert report["measurement_session_plan"]["planned_runs"][0]["meter_template_path"].endswith(
        "_real_data_external_validity_r<replicate>_meter_template.json"
    )
    assert report["measurement_session_plan"]["planned_runs"][0]["replicate_count"] == 3
    assert report["measurement_plan"]["pending_pairs"][0]["command_template"].startswith(
        "python scripts/sara_cli.py record-energy-measurement"
    )
    summary = module.format_energy_measurement_summary(report)
    assert "Measurement Plan:" in summary
    assert "Measurement Session Plan:" in summary
    assert "task=real_data_external_validity" in summary
    assert "python scripts/sara_cli.py record-energy-measurement" in summary


def test_system_estimates_are_recorded_but_do_not_complete_physical_gate():
    module = _load_module()
    rows = [
        _row(module, "sara", "qa", 2.0, measurement_quality="system_estimate", physical_evidence=False),
        _row(module, "ann", "qa", 8.0, measurement_quality="system_estimate", physical_evidence=False),
    ]

    report = module.build_energy_measurement_readiness_report(rows)

    assert report["passed"] is False
    assert report["status"] == "system_estimate_pending_physical_measurement"
    assert report["physical_measurement_count"] == 0
    assert report["system_estimate_measurement_count"] == 2
    assert report["real_joule_measurements_present"] is False


def test_energy_measurement_readiness_accepts_real_joule_advantage():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [
            _row(module, "sara", "qa", 2.0),
            _row(module, "ann", "qa", 8.0),
        ],
        min_ann_to_sara_ratio=2.0,
        min_paired_replicates_per_task=1,
    )

    assert report["passed"] is True
    assert report["status"] == "real_joule_evidence_passed"
    assert report["real_joule_measurements_present"] is True
    assert report["metrics"]["sara_joule_per_success"] == 0.2
    assert report["metrics"]["ann_joule_per_success"] == 0.8
    assert report["metrics"]["ann_to_sara_joule_efficiency_ratio"] == 4.0
    assert report["metrics"]["paired_task_count"] == 1
    assert report["metrics"]["min_paired_task_ann_to_sara_ratio"] == 4.0
    assert report["metrics"]["valid_pair_count"] == 1
    assert report["metrics"]["task_pair_statistics"]["qa"]["median_ann_to_sara_ratio"] == 4.0
    assert report["checks"]["paired_task_measurements_present"] is True
    assert report["checks"]["paired_task_rows_balanced"] is True
    assert report["checks"]["paired_task_efficiency_ratio_passed"] is True
    assert report["measurement_plan"]["ready_for_real_joule_claim"] is True
    assert report["measurement_plan"]["pending_pair_count"] == 0
    assert report["measurement_plan"]["weak_pair_count"] == 0
    assert report["measurement_session_plan"]["status"] == "ready_for_real_joule_claim"
    assert report["measurement_session_plan"]["planned_run_count"] == 0


def test_energy_measurement_readiness_tracks_maintenance_trace_metrics():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [
            _row(
                module,
                "sara",
                "qa",
                2.0,
                maintenance_selected_count=12,
                maintenance_phase_count=4,
                maintenance_refresh_count=2,
                maintenance_event_cost=0.6,
            ),
            _row(
                module,
                "ann",
                "qa",
                8.0,
                maintenance_selected_count=3,
                maintenance_phase_count=1,
                maintenance_refresh_count=0,
                maintenance_event_cost=0.2,
            ),
        ],
        min_ann_to_sara_ratio=2.0,
        min_paired_replicates_per_task=1,
    )

    pair = report["metrics"]["valid_pairs"][0]
    stats = report["metrics"]["task_pair_statistics"]["qa"]
    assert report["metrics"]["maintenance_trace_rows_present"] is True
    assert pair["sara_maintenance_event_cost_per_success"] == 0.06
    assert pair["ann_maintenance_event_cost_per_success"] == 0.02
    assert pair["sara_maintenance_selected_per_success"] == 1.2
    assert pair["ann_maintenance_selected_per_success"] == 0.3
    assert round(pair["sara_maintenance_event_cost_per_selected"], 6) == 0.05
    assert round(pair["ann_maintenance_event_cost_per_selected"], 6) == round(0.2 / 3.0, 6)
    assert stats["sara_median_maintenance_event_cost_per_success"] == 0.06
    assert stats["ann_median_maintenance_event_cost_per_success"] == 0.02
    assert stats["sara_median_maintenance_selected_per_success"] == 1.2
    assert stats["ann_median_maintenance_selected_per_success"] == 0.3
    assert "maintenance_selected_count" in report["measurement_protocol"]["optional_maintenance_fields"]
    summary = module.format_energy_measurement_summary(report)
    assert "maintenance_trace_rows_present: True" in summary

    pending_report = module.build_energy_measurement_readiness_report([])
    assert "--maintenance-selected-count <count>" in pending_report["measurement_session_plan"]["planned_runs"][0]["command_template"]
    assert "--meter-template-path" in pending_report["measurement_session_plan"]["planned_runs"][0]["pair_command_template"]
    assert "--event-memory-maintenance-coupling-report-path" in pending_report["measurement_session_plan"]["planned_runs"][0]["pair_command_template"]


def test_energy_measurement_readiness_surfaces_internal_maintenance_reference():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [],
        internal_maintenance_report={
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
                "maintenance_event_cost_efficiency_observed": 1.0,
            },
        },
    )

    reference = report["internal_maintenance_reference"]
    assert reference["available"] is True
    assert reference["maintenance_event_cost_per_selected"] == 1.5
    summary = module.format_energy_measurement_summary(report)
    assert "internal_maintenance_reference_available: True" in summary
    assert "Internal Maintenance Reference:" in summary


def test_energy_measurement_readiness_surfaces_event_memory_maintenance_coupling_reference():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [],
        event_memory_maintenance_coupling_report={
            "passed": True,
            "observed_only": True,
            "profile_count": 3,
            "best_profile": {
                "profile_id": "wide",
            },
            "metrics": {
                "compression_to_maintenance_correlation": 0.51,
                "best_profile_compression_efficiency_per_maintenance": 0.19,
                "best_profile_self_state_continuity": 0.83,
                "best_profile_episode_compression_ratio": 3.67,
                "best_profile_multimodal_bundle_compression_contribution": 1.83,
            },
        },
    )

    reference = report["event_memory_maintenance_coupling_reference"]
    assert reference["available"] is True
    assert reference["best_profile_id"] == "wide"
    assert reference["best_profile_compression_efficiency_per_maintenance"] == 0.19
    assert reference["best_profile_multimodal_bundle_compression_contribution"] == 1.83
    summary = module.format_energy_measurement_summary(report)
    assert "event_memory_maintenance_coupling_reference_available: True" in summary
    assert "event_memory_maintenance_best_bundle_contribution: 1.830" in summary
    assert "Event Memory Maintenance Coupling Reference:" in summary
    progress = report["measurement_session_progress"]
    assert progress["event_memory_maintenance_coupling_reference"]["best_profile_id"] == "wide"
    progress_summary = module.format_measurement_session_progress_summary(progress)
    assert "Event Memory Maintenance Coupling Reference:" in progress_summary


def test_energy_measurement_readiness_warns_when_bundle_contribution_is_weak():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [],
        event_memory_maintenance_coupling_report={
            "passed": True,
            "observed_only": True,
            "profile_count": 3,
            "best_profile": {
                "profile_id": "wide",
            },
            "metrics": {
                "compression_to_maintenance_correlation": 0.51,
                "best_profile_compression_efficiency_per_maintenance": 0.19,
                "best_profile_self_state_continuity": 0.83,
                "best_profile_episode_compression_ratio": 3.67,
                "best_profile_multimodal_bundle_compression_contribution": 0.1,
            },
        },
    )

    assert "Bundle-backed compression contribution is weak" in report["bundle_contribution_warning"]
    summary = module.format_energy_measurement_summary(report)
    assert "Bundle Contribution Warning:" in summary
    progress_summary = module.format_measurement_session_progress_summary(
        report["measurement_session_progress"]
    )
    assert "Bundle Contribution Warning:" in progress_summary


def test_energy_measurement_readiness_surfaces_physical_internal_alignment():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [
            _row(
                module,
                "sara",
                "qa",
                2.0,
                maintenance_selected_count=12,
                maintenance_refresh_count=2,
                maintenance_event_cost=0.6,
            ),
            _row(
                module,
                "ann",
                "qa",
                8.0,
                maintenance_selected_count=3,
                maintenance_refresh_count=1,
                maintenance_event_cost=0.2,
            ),
        ],
        min_ann_to_sara_ratio=2.0,
        min_paired_replicates_per_task=1,
        internal_maintenance_report={
            "passed": True,
            "observed_only": True,
            "counts": {
                "maintenance_selected_count": 4,
                "maintenance_refresh_count": 2,
            },
            "normalized_metrics": {
                "maintenance_event_cost_per_selected": 0.04,
                "maintenance_event_cost_per_refresh": 0.20,
            },
            "metrics": {
                "maintenance_self_state_continuity_observed": 1.0,
                "maintenance_event_cost_efficiency_observed": 1.0,
            },
        },
    )

    alignment = report["maintenance_alignment"]
    assert alignment["available"] is True
    assert round(alignment["sara_physical_maintenance_event_cost_per_selected"], 6) == 0.05
    assert alignment["reference_maintenance_event_cost_per_selected"] == 0.04
    assert round(alignment["maintenance_event_cost_per_selected_ratio"], 6) == 1.25
    summary = module.format_energy_measurement_summary(report)
    assert "maintenance_alignment_available: True" in summary
    assert "Maintenance Alignment:" in summary
    assert "ratio=1.250" in summary


def test_energy_measurement_readiness_rejects_unpaired_real_measurements():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [
            _row(module, "sara", "qa", 2.0),
            _row(module, "ann", "summary", 8.0),
        ],
        min_ann_to_sara_ratio=2.0,
        min_paired_replicates_per_task=1,
    )

    assert report["passed"] is False
    assert report["real_joule_measurements_present"] is True
    assert report["metrics"]["paired_task_count"] == 0
    assert report["metrics"]["unpaired_task_count"] == 2
    assert report["checks"]["paired_task_measurements_present"] is False
    assert report["checks"]["paired_task_rows_balanced"] is False
    assert report["measurement_plan"]["pending_pair_count"] == 2
    missing = {(item["task"], item["missing_system"]) for item in report["measurement_plan"]["pending_pairs"]}
    assert ("qa", "ann") in missing
    assert ("summary", "sara") in missing


def test_energy_measurement_readiness_rejects_weak_paired_task_ratio():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [
            _row(module, "sara", "qa", 4.0),
            _row(module, "ann", "qa", 5.0),
        ],
        min_ann_to_sara_ratio=2.0,
        min_paired_replicates_per_task=1,
    )

    assert report["passed"] is False
    assert report["metrics"]["min_paired_task_ann_to_sara_ratio"] == 1.25
    assert report["checks"]["paired_task_efficiency_ratio_passed"] is False
    assert report["measurement_plan"]["weak_pair_count"] == 1
    assert report["measurement_plan"]["weak_pairs"][0]["task"] == "qa"
    assert report["measurement_plan"]["weak_pairs"][0]["priority"] == "high"
    assert report["measurement_plan"]["weak_pairs"][0]["severity"] == "high"
    assert report["measurement_plan"]["weak_pairs"][0]["relative_ratio"] == 0.625
    assert report["measurement_plan"]["weak_pairs"][0]["ratio_gap"] == 0.75
    assert report["measurement_session_plan"]["planned_run_count"] == 2
    planned = {(item["category"], item["system"]) for item in report["measurement_session_plan"]["planned_runs"]}
    assert ("repeat_weak_pair", "sara") in planned
    assert ("repeat_weak_pair", "ann") in planned
    summary = module.format_energy_measurement_summary(report)
    assert "weak_pairs:" in summary
    assert "task=qa" in summary
    assert "ratio=1.250" in summary


def test_energy_measurement_readiness_tracks_session_progress_for_partial_pairs():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [
            _row(
                module,
                "sara",
                "real_data_external_validity",
                2.0,
                pair_id="ann-efficiency-real-joule-real_data_external_validity-pair-1",
                replicate=1,
            ),
        ],
        session_id="ann-efficiency-real-joule",
    )

    progress = report["measurement_session_progress"]
    assert progress["planned_pair_count"] >= 1
    assert progress["partial_pair_count"] >= 1
    assert report["checks"]["session_pair_completion_passed"] is False
    summary = module.format_energy_measurement_summary(report)
    assert "Measurement Session Progress:" in summary
    assert "status=partial_pair" in summary


def test_energy_measurement_readiness_classifies_invalid_session_pair_reason():
    module = _load_module()
    pair_id = "ann-efficiency-real-joule-real_data_external_validity-pair-1"
    sara = _row(module, "sara", "real_data_external_validity", 2.0, pair_id=pair_id, replicate=1)
    ann = _row(module, "ann", "real_data_external_validity", 4.0, pair_id=pair_id, replicate=1)
    ann["environment_fingerprint"] = "different-env"
    ann["run_order"] = 1

    report = module.build_energy_measurement_readiness_report(
        [sara, ann],
        session_id="ann-efficiency-real-joule",
    )

    pair = report["measurement_session_progress"]["pair_statuses"][0]
    assert pair["status"] == "invalid_pair"
    assert pair["invalid_reason_category"] == "fairness_and_run_order_conflict"
    assert "environment_fingerprint" in pair["invalid_reason_fields"]


def test_energy_measurement_readiness_accepts_empty_session_progress_without_rows():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report([])

    progress = report["measurement_session_progress"]
    assert progress["schema"] == "sara-physical-energy-session-progress-v1"
    assert progress["planned_pair_count"] == 6
    assert progress["missing_pair_count"] == 6
    assert report["checks"]["session_pair_completion_passed"] is True
    summary = module.format_measurement_session_progress_summary(progress)
    assert "SARA Physical Energy Session Progress" in summary
    assert "missing_pair_count: 6" in summary


def test_energy_measurement_session_plan_uses_custom_session_and_path():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [],
        measurement_path="data/raw/lab_measurements.jsonl",
        session_id="lab session",
    )

    session_plan = report["measurement_session_plan"]
    assert session_plan["measurement_path"] == "data/raw/lab_measurements.jsonl"
    assert session_plan["planned_runs"][0]["run_id_template"].startswith(
        "lab-session-real_data_external_validity-sara-"
    )
    assert report["measurement_protocol"]["recommended_path"] == "data/raw/lab_measurements.jsonl"
    assert session_plan["planned_runs"][0]["pair_id_template"].startswith(
        "lab-session-real_data_external_validity-pair-"
    )


def test_format_measurement_session_plan_summary_lists_commands():
    module = _load_module()
    report = module.build_energy_measurement_readiness_report([])

    summary = module.format_measurement_session_plan_summary(report["measurement_session_plan"])

    assert "SARA Energy Measurement Session Plan" in summary
    assert "planned_run_count: 4" in summary
    assert "real_energy_session" in summary
    assert "record-energy-measurement" in summary
    assert "run-physical-energy-pair" in summary


def test_energy_measurement_readiness_rejects_bad_rows():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [
            {"run_id": "bad", "system": "sara", "task": "qa", "success_count": 0, "joules": -1.0},
        ]
    )

    assert report["passed"] is False
    assert report["status"] == "needs_measurement_repair"
    assert report["row_errors"]


def test_build_measurement_row_derives_joules_from_average_watts():
    module = _load_module()

    row = module.build_measurement_row(
        run_id="sara-watt-run",
        system="sara",
        task="qa",
        success_count=6,
        joules=0.0,
        source="powermetrics",
        duration_seconds=2.5,
        average_watts=0.8,
        pair_id="pair-watt",
        environment_fingerprint="env",
        task_fixture_hash="fixture",
        success_criterion_id="criterion",
        measurement_boundary="boundary",
        measurement_tool="powermetrics",
        cpu_model="cpu",
        process_affinity="core-0",
        power_mode="ac",
        warmup_count=1,
        measured_repetitions=6,
        trial_count=6,
    )

    assert row["joules"] == 2.0
    assert row["average_watts"] == 0.8
    assert row["duration_seconds"] == 2.5
    assert row["joules_derivation"] == "average_watts_x_duration_seconds"


def test_build_measurement_row_accepts_optional_maintenance_fields():
    module = _load_module()

    row = module.build_measurement_row(
        run_id="sara-maintenance-run",
        system="sara",
        task="qa",
        success_count=5,
        joules=1.0,
        source="manual",
        pair_id="pair-maintenance",
        environment_fingerprint="env",
        task_fixture_hash="fixture",
        success_criterion_id="criterion",
        measurement_boundary="boundary",
        measurement_tool="meter",
        cpu_model="cpu",
        process_affinity="core-0",
        power_mode="ac",
        warmup_count=1,
        measured_repetitions=5,
        trial_count=5,
        maintenance_selected_count=7,
        maintenance_phase_count=3,
        maintenance_refresh_count=2,
        maintenance_event_cost=0.4,
    )

    assert row["maintenance_selected_count"] == 7
    assert row["maintenance_phase_count"] == 3
    assert row["maintenance_refresh_count"] == 2
    assert row["maintenance_event_cost"] == 0.4


def test_build_measurement_row_rejects_incomplete_average_watt_input():
    module = _load_module()

    try:
        module.build_measurement_row(
            run_id="bad-watt-run",
            system="sara",
            task="qa",
            success_count=6,
            joules=0.0,
            source="powermetrics",
            average_watts=0.8,
            pair_id="pair-bad",
            environment_fingerprint="env",
            task_fixture_hash="fixture",
            success_criterion_id="criterion",
            measurement_boundary="boundary",
            measurement_tool="powermetrics",
            cpu_model="cpu",
            process_affinity="core-0",
            power_mode="ac",
            warmup_count=1,
            measured_repetitions=6,
            trial_count=6,
        )
    except ValueError as exc:
        assert "joules_must_be_positive" in str(exc)
    else:
        raise AssertionError("Expected invalid watt-only row to be rejected")


def test_energy_measurement_append_round_trip(tmp_path):
    module = _load_module()
    path = os.path.join(
        ROOT,
        "workspace",
        "evaluation",
        f"test_energy_measurements_{os.getpid()}.jsonl",
    )
    row = module.build_measurement_row(
        run_id="sara-run",
        system="sara",
        task="qa",
        success_count=4,
        joules=1.2,
        source="manual_meter",
        duration_seconds=3.0,
        pair_id="pair-round-trip",
        environment_fingerprint="env",
        task_fixture_hash="fixture",
        success_criterion_id="criterion",
        measurement_boundary="boundary",
        measurement_tool="manual-meter",
        cpu_model="cpu",
        process_affinity="core-0",
        power_mode="ac",
        warmup_count=1,
        measured_repetitions=4,
        trial_count=4,
    )

    try:
        resolved = module.append_measurement(str(path), row)
        rows = module.load_measurements(resolved)
    finally:
        if os.path.exists(path):
            os.remove(path)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "sara-run"
    assert rows[0]["system"] == "sara"
    assert rows[0]["joules"] == 1.2
    assert rows[0]["duration_seconds"] == 3.0


def test_energy_measurement_rejects_mismatched_environment_pair():
    module = _load_module()
    sara = _row(module, "sara", "qa", 2.0)
    ann = _row(module, "ann", "qa", 8.0)
    ann["environment_fingerprint"] = "different-env"

    report = module.build_energy_measurement_readiness_report([sara, ann])

    assert report["passed"] is False
    assert report["checks"]["fair_pair_contract_passed"] is False
    assert report["metrics"]["invalid_pair_count"] == 1
    assert "mismatch:environment_fingerprint" in report["metrics"]["pair_errors"][0]["errors"]


def test_energy_measurement_rejects_quality_gap_even_when_ann_uses_more_joules():
    module = _load_module()
    sara = _row(module, "sara", "qa", 2.0, success=8, trials=10)
    ann = _row(module, "ann", "qa", 8.0, success=10, trials=10)

    report = module.build_energy_measurement_readiness_report(
        [sara, ann],
        max_success_rate_delta=0.05,
    )

    assert report["passed"] is False
    assert report["checks"]["quality_parity_passed"] is False
    assert "success_rate_parity_failed" in report["metrics"]["pair_errors"][0]["errors"]


def test_energy_measurement_reports_median_and_mad_across_replicates():
    module = _load_module()
    rows = []
    for replicate, sara_joules, ann_joules in (
        (1, 2.0, 8.0),
        (2, 2.2, 7.7),
        (3, 1.8, 8.1),
    ):
        rows.extend(
            [
                _row(module, "sara", "qa", sara_joules, pair_id=f"pair-{replicate}", replicate=replicate, run_order=1 if replicate % 2 else 2),
                _row(module, "ann", "qa", ann_joules, pair_id=f"pair-{replicate}", replicate=replicate, run_order=2 if replicate % 2 else 1),
            ]
        )

    report = module.build_energy_measurement_readiness_report(
        rows,
        min_ann_to_sara_ratio=2.0,
        min_paired_replicates_per_task=3,
    )

    stats = report["metrics"]["task_pair_statistics"]["qa"]
    assert report["passed"] is True
    assert stats["valid_pair_count"] == 3
    assert stats["sara_joule_per_success_mad"] > 0.0
    assert report["metrics"]["run_order_balance"] == {"sara_first": 2, "ann_first": 1}


def test_safe_numeric_helpers_reject_non_finite_and_boolean_values():
    module = _load_module()
    assert module._safe_float("NaN") == 0.0
    assert module._safe_float("Infinity") == 0.0
    assert module._safe_int(True) == 0
