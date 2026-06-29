import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import raw_data_path, workspace_path


def _load_module(name: str, relative_path: str):
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            relative_path,
        )
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_physical_energy_session_progress_tracks_pair_completion():
    progress_module = _load_module(
        "physical_energy_session_progress",
        "scripts/eval/physical_energy_session_progress.py",
    )
    readiness_module = _load_module(
        "energy_measurement_readiness_for_progress",
        "scripts/eval/energy_measurement_readiness.py",
    )
    batch_report = {
        "session_id": "lab-session",
        "batch_runs": [
            {
                "category": "collect_missing_pair",
                "task": "real_data_external_validity",
                "pair_id": "lab-session-real_data_external_validity-pair-1",
                "replicate_index": 1,
            },
            {
                "category": "collect_missing_pair",
                "task": "energy_efficiency_benchmark",
                "pair_id": "lab-session-energy_efficiency_benchmark-pair-1",
                "replicate_index": 1,
            },
        ],
    }
    rows = [
        readiness_module.build_measurement_row(
            run_id="sara-1",
            system="sara",
            task="real_data_external_validity",
            success_count=8,
            joules=10.0,
            source="real_energy_session",
            pair_id="lab-session-real_data_external_validity-pair-1",
            replicate_index=1,
            environment_fingerprint="env-1",
            task_fixture_hash="fixture-1",
            success_criterion_id="criterion-1",
            measurement_boundary="boundary-1",
            measurement_tool="meter-1",
            cpu_model="cpu-1",
            thread_count=1,
            process_affinity="affinity-1",
            power_mode="power-1",
            warmup_count=1,
            measured_repetitions=2,
            trial_count=8,
            run_order=1,
        ),
        readiness_module.build_measurement_row(
            run_id="ann-1",
            system="ann",
            task="real_data_external_validity",
            success_count=8,
            joules=13.0,
            source="real_energy_session",
            pair_id="lab-session-real_data_external_validity-pair-1",
            replicate_index=1,
            environment_fingerprint="env-1",
            task_fixture_hash="fixture-1",
            success_criterion_id="criterion-1",
            measurement_boundary="boundary-1",
            measurement_tool="meter-1",
            cpu_model="cpu-1",
            thread_count=1,
            process_affinity="affinity-1",
            power_mode="power-1",
            warmup_count=1,
            measured_repetitions=2,
            trial_count=8,
            run_order=2,
        ),
        readiness_module.build_measurement_row(
            run_id="sara-2",
            system="sara",
            task="energy_efficiency_benchmark",
            success_count=5,
            joules=7.0,
            source="real_energy_session",
            pair_id="lab-session-energy_efficiency_benchmark-pair-1",
            replicate_index=1,
            environment_fingerprint="env-2",
            task_fixture_hash="fixture-2",
            success_criterion_id="criterion-2",
            measurement_boundary="boundary-2",
            measurement_tool="meter-2",
            cpu_model="cpu-2",
            thread_count=1,
            process_affinity="affinity-2",
            power_mode="power-2",
            warmup_count=1,
            measured_repetitions=2,
            trial_count=5,
            run_order=1,
        ),
        readiness_module.build_measurement_row(
            run_id="orphan-ann",
            system="ann",
            task="extra_task",
            success_count=3,
            joules=4.0,
            source="real_energy_session",
            pair_id="orphan-pair",
            replicate_index=1,
            environment_fingerprint="env-o",
            task_fixture_hash="fixture-o",
            success_criterion_id="criterion-o",
            measurement_boundary="boundary-o",
            measurement_tool="meter-o",
            cpu_model="cpu-o",
            thread_count=1,
            process_affinity="affinity-o",
            power_mode="power-o",
            warmup_count=1,
            measured_repetitions=1,
            trial_count=3,
            run_order=1,
        ),
    ]

    report = progress_module.build_physical_energy_session_progress(
        batch_report,
        rows,
        internal_maintenance_report={
            "passed": True,
            "observed_only": True,
            "counts": {"maintenance_selected_count": 4},
            "normalized_metrics": {"maintenance_event_cost_per_selected": 1.5},
            "metrics": {"maintenance_self_state_continuity_observed": 1.0},
        },
        event_memory_maintenance_coupling_report={
            "passed": True,
            "observed_only": True,
            "profile_count": 3,
            "best_profile": {"profile_id": "wide"},
            "metrics": {
                "best_efficiency_score": 1.8,
                "best_self_state_continuity": 0.92,
            },
        },
    )

    assert report["planned_pair_count"] == 2
    assert report["complete_valid_pair_count"] == 1
    assert report["partial_pair_count"] == 1
    assert report["missing_pair_count"] == 0
    assert report["orphan_pair_count"] == 1
    assert report["status"] == "in_progress"
    assert report["pair_statuses"][0]["status"] == "complete_valid_pair"
    assert report["pair_statuses"][1]["status"] == "partial_pair"
    assert report["task_progress"]["real_data_external_validity"]["complete_valid_pair_count"] == 1
    assert report["internal_maintenance_reference"]["available"] is True
    assert report["event_memory_maintenance_coupling_reference"]["best_profile_id"] == "wide"


def test_build_physical_energy_session_progress_classifies_invalid_pair_reason():
    progress_module = _load_module(
        "physical_energy_session_progress_invalid",
        "scripts/eval/physical_energy_session_progress.py",
    )
    readiness_module = _load_module(
        "energy_measurement_readiness_for_progress_invalid",
        "scripts/eval/energy_measurement_readiness.py",
    )
    batch_report = {
        "session_id": "lab-session",
        "batch_runs": [
            {
                "category": "collect_missing_pair",
                "task": "real_data_external_validity",
                "pair_id": "lab-session-real_data_external_validity-pair-1",
                "replicate_index": 1,
            }
        ],
    }
    sara_row = readiness_module.build_measurement_row(
        run_id="sara-1",
        system="sara",
        task="real_data_external_validity",
        success_count=8,
        joules=10.0,
        source="real_energy_session",
        pair_id="lab-session-real_data_external_validity-pair-1",
        replicate_index=1,
        environment_fingerprint="env-1",
        task_fixture_hash="fixture-1",
        success_criterion_id="criterion-1",
        measurement_boundary="boundary-1",
        measurement_tool="meter-1",
        cpu_model="cpu-1",
        thread_count=1,
        process_affinity="affinity-1",
        power_mode="power-1",
        warmup_count=1,
        measured_repetitions=2,
        trial_count=8,
        run_order=1,
    )
    ann_row = dict(sara_row)
    ann_row["run_id"] = "ann-1"
    ann_row["system"] = "ann"
    ann_row["joules"] = 13.0
    ann_row["environment_fingerprint"] = "env-2"
    ann_row["run_order"] = 1

    report = progress_module.build_physical_energy_session_progress(
        batch_report,
        [sara_row, ann_row],
    )

    pair = report["pair_statuses"][0]
    assert pair["status"] == "invalid_pair"
    assert pair["invalid_reason_category"] == "fairness_and_run_order_conflict"
    assert "environment_fingerprint" in pair["invalid_reason_fields"]


def test_physical_energy_session_progress_main_writes_report():
    progress_module = _load_module(
        "physical_energy_session_progress_main",
        "scripts/eval/physical_energy_session_progress.py",
    )
    readiness_module = _load_module(
        "energy_measurement_readiness_for_progress_main",
        "scripts/eval/energy_measurement_readiness.py",
    )
    batch_path = workspace_path("evaluation", "test_physical_energy_session_batch.json")
    measurement_path = raw_data_path("test_physical_energy_measurements.jsonl")
    report_path = workspace_path("evaluation", "test_physical_energy_session_progress.json")
    summary_path = workspace_path("evaluation", "test_physical_energy_session_progress.txt")
    maintenance_path = workspace_path("evaluation", "test_internal_maintenance_reference.json")
    coupling_path = workspace_path("evaluation", "test_event_memory_maintenance_coupling_reference.json")
    payload = {
        "session_id": "lab-session",
        "batch_runs": [
            {
                "category": "collect_missing_pair",
                "task": "real_data_external_validity",
                "pair_id": "lab-session-real_data_external_validity-pair-1",
                "replicate_index": 1,
            }
        ],
    }
    row = readiness_module.build_measurement_row(
        run_id="sara-1",
        system="sara",
        task="real_data_external_validity",
        success_count=8,
        joules=10.0,
        source="real_energy_session",
        pair_id="lab-session-real_data_external_validity-pair-1",
        replicate_index=1,
        environment_fingerprint="env-1",
        task_fixture_hash="fixture-1",
        success_criterion_id="criterion-1",
        measurement_boundary="boundary-1",
        measurement_tool="meter-1",
        cpu_model="cpu-1",
        thread_count=1,
        process_affinity="affinity-1",
        power_mode="power-1",
        warmup_count=1,
        measured_repetitions=2,
        trial_count=8,
        run_order=1,
    )
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)
    os.makedirs(os.path.dirname(measurement_path), exist_ok=True)
    with open(batch_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    with open(measurement_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    with open(maintenance_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "passed": True,
                "observed_only": True,
                "counts": {"maintenance_selected_count": 4},
                "normalized_metrics": {"maintenance_event_cost_per_selected": 1.5},
                "metrics": {"maintenance_self_state_continuity_observed": 1.0},
            },
            handle,
        )
    with open(coupling_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "passed": True,
                "observed_only": True,
                "profile_count": 3,
                "best_profile": {"profile_id": "wide"},
                "metrics": {
                    "best_efficiency_score": 1.8,
                    "best_self_state_continuity": 0.92,
                },
            },
            handle,
        )

    try:
        exit_code = progress_module.main(
            [
                "--batch-report-path",
                batch_path,
                "--measurement-path",
                measurement_path,
                "--report-path",
                report_path,
                "--summary-path",
                summary_path,
                "--internal-maintenance-report-path",
                maintenance_path,
                "--event-memory-maintenance-coupling-report-path",
                coupling_path,
            ]
        )
        assert exit_code == 0
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["planned_pair_count"] == 1
        assert report["partial_pair_count"] == 1
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = handle.read()
        assert "SARA Physical Energy Session Progress" in summary
        assert "partial_pair_count: 1" in summary
        assert "Internal Maintenance Reference:" in summary
        assert "Event Memory Maintenance Coupling Reference:" in summary
    finally:
        for path in (
            batch_path,
            measurement_path,
            maintenance_path,
            coupling_path,
            report_path,
            summary_path,
        ):
            if os.path.exists(path):
                os.remove(path)
