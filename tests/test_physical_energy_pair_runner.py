import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path


def _load_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "physical_energy_pair_runner.py",
        )
    )
    spec = importlib.util.spec_from_file_location("physical_energy_pair_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_workload_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "energy_pair_workload.py",
        )
    )
    spec = importlib.util.spec_from_file_location("energy_pair_workload", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_physical_energy_pair_runner_dry_run_freezes_manifest():
    module = _load_module()
    manifest_path = workspace_path("evaluation", "test_physical_pair_manifest.json")
    trace_path = workspace_path("evaluation", "test_physical_pair_trace.jsonl")
    report_path = workspace_path("evaluation", "test_physical_pair_report.json")
    summary_path = workspace_path("evaluation", "test_physical_pair_summary.txt")
    meter_template_path = workspace_path("evaluation", "test_physical_pair_meter_template.json")

    exit_code = module.main(
        [
            "--pair-id",
            "test-pair-1",
            "--replicate-index",
            "1",
            "--corpus-path",
            processed_data_path("corpus.txt"),
            "--manifest-path",
            manifest_path,
            "--trace-path",
            trace_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
            "--meter-template-path",
            meter_template_path,
            "--dry-run",
        ]
    )

    assert exit_code == 0
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["protocol_version"] == "sara-energy-fair-comparison-v2"
    assert manifest["run_order"] == ["sara", "ann"]
    assert len(manifest["task_fixture_hash"]) == 64
    with open(trace_path, "r", encoding="utf-8") as handle:
        traces = [json.loads(line) for line in handle if line.strip()]
    assert [trace["system"] for trace in traces] == ["sara", "ann"]
    assert all(trace["status"] == "planned" for trace in traces)
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["measurement_pending"] is True
    assert report["maintenance_by_system"] == {}
    assert report["meter_template_path"] == meter_template_path
    with open(meter_template_path, "r", encoding="utf-8") as handle:
        meter_template = json.load(handle)
    assert meter_template["pair_id"] == "test-pair-1"
    assert meter_template["replicate_index"] == 1
    assert meter_template["readings"]["sara"]["joules"] is None
    assert meter_template["readings"]["ann"]["average_watts"] is None
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "SARA Physical Energy Pair" in summary
    assert "measurement_pending: True" in summary
    assert "Resume Append Command:" in summary
    assert "meter_template_path:" in summary


def test_physical_energy_pair_runner_alternates_even_replicate_order():
    module = _load_module()
    manifest = module.build_manifest(
        corpus_path=processed_data_path("corpus.txt"),
        replicate_index=2,
        repetitions=3,
        warmup_count=1,
        thread_count=1,
        process_affinity="unbound",
        power_mode="ac",
        measurement_tool="manual",
        pair_id="test-pair-2",
    )

    assert manifest["run_order"] == ["ann", "sara"]


def test_energy_pair_workload_reports_sparse_maintenance_for_sara():
    module = _load_workload_module()

    report = module.run_retrieval_workload(
        system="sara",
        corpus_path=processed_data_path("corpus.txt"),
        max_docs=32,
        max_cases=4,
        repetitions=1,
        warmup_count=0,
    )

    assert report["passed"] is True
    assert report["maintenance_selected_count"] >= 0
    assert report["maintenance_phase_count"] >= 0
    assert report["maintenance_refresh_count"] >= 0
    assert report["maintenance_event_cost"] >= 0.0
    assert report["maintenance_spontaneous_event_count"] >= 0
    assert report["maintenance_predicted_event_count"] >= 0


def test_physical_energy_pair_runner_appends_maintenance_fields():
    module = _load_module()
    measurement_path = raw_data_path("test_physical_energy_measurements.jsonl")
    if os.path.exists(measurement_path):
        os.remove(measurement_path)
    manifest = module.build_manifest(
        corpus_path=processed_data_path("corpus.txt"),
        replicate_index=1,
        repetitions=1,
        warmup_count=0,
        thread_count=1,
        process_affinity="unbound",
        power_mode="ac",
        measurement_tool="manual",
        pair_id="maintenance-pair",
        max_docs=32,
        max_cases=4,
    )
    traces = [
        {
            "system": "sara",
            "run_order": 1,
            "workload_result": {
                "passed": True,
                "success_count": 4,
                "trial_count": 4,
                "duration_seconds": 0.5,
                "maintenance_selected_count": 6,
                "maintenance_phase_count": 2,
                "maintenance_refresh_count": 1,
                "maintenance_event_cost": 0.8,
            },
        },
        {
            "system": "ann",
            "run_order": 2,
            "workload_result": {
                "passed": True,
                "success_count": 4,
                "trial_count": 4,
                "duration_seconds": 0.4,
                "maintenance_selected_count": 0,
                "maintenance_phase_count": 0,
                "maintenance_refresh_count": 0,
                "maintenance_event_cost": 0.0,
            },
        },
    ]

    try:
        rows = module.append_measured_rows(
            manifest,
            traces,
            sara_joules=2.0,
            ann_joules=3.0,
            measurement_path=measurement_path,
        )
    finally:
        if os.path.exists(measurement_path):
            os.remove(measurement_path)

    assert rows[0]["maintenance_selected_count"] == 6
    assert rows[0]["maintenance_phase_count"] == 2
    assert rows[0]["maintenance_refresh_count"] == 1
    assert rows[0]["maintenance_event_cost"] == 0.8
    assert rows[1]["maintenance_event_cost"] == 0.0


def test_build_pair_report_surfaces_maintenance_summary():
    module = _load_module()
    report = module.build_pair_report(
        {
            "pair_id": "pilot-2",
            "replicate_index": 2,
            "task": "paired_retrieval",
            "run_order": ["ann", "sara"],
            "measurement_tool": "powermetrics-v1",
        },
        [
            {
                "system": "sara",
                "status": "passed",
                "workload_result": {
                    "maintenance_selected_count": 5,
                    "maintenance_phase_count": 2,
                    "maintenance_refresh_count": 1,
                    "maintenance_event_cost": 0.7,
                    "maintenance_idle_self_state_ok_count": 1,
                    "maintenance_spontaneous_event_count": 2,
                    "maintenance_predicted_event_count": 3,
                },
            },
            {
                "system": "ann",
                "status": "passed",
                "workload_result": {
                    "maintenance_selected_count": 0,
                    "maintenance_phase_count": 0,
                    "maintenance_refresh_count": 0,
                    "maintenance_event_cost": 0.0,
                    "maintenance_idle_self_state_ok_count": 0,
                    "maintenance_spontaneous_event_count": 0,
                    "maintenance_predicted_event_count": 0,
                },
            },
        ],
        dry_run=False,
        measurement_path="data/raw/energy_measurements.jsonl",
        meter_reading_path="",
        meter_template_path="workspace/evaluation/template.json",
        recorded_rows=[],
        manifest_path="workspace/evaluation/a.json",
        trace_path="workspace/evaluation/b.jsonl",
        report_path="workspace/evaluation/c.json",
        summary_path="workspace/evaluation/d.txt",
        internal_maintenance_report={
            "counts": {
                "maintenance_selected_count": 6,
                "maintenance_phase_count": 3,
                "maintenance_refresh_count": 2,
                "maintenance_idle_self_state_ok_count": 3,
                "maintenance_spontaneous_event_count": 4,
                "maintenance_predicted_event_count": 5,
            },
            "normalized_metrics": {
                "maintenance_event_cost": 1.2,
                "maintenance_event_cost_per_selected": 0.2,
                "maintenance_event_cost_per_refresh": 0.6,
            },
        },
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
            },
        },
    )

    assert report["maintenance_by_system"]["sara"]["maintenance_event_cost"] == 0.7
    assert report["measurement_pending"] is True
    assert report["internal_maintenance_reference"]["maintenance_event_cost_per_selected"] == 0.2
    assert report["event_memory_maintenance_coupling_reference"]["best_profile_id"] == "wide"
    assert round(report["maintenance_alignment"]["sara"]["actual_event_cost_per_selected"], 6) == 0.14
    assert round(report["maintenance_alignment"]["sara"]["event_cost_per_selected_delta"], 6) == -0.06
    assert "--sara-joules <J> --ann-joules <J>" in report["resume_append_command_template"]
    assert len(report["record_measurement_commands"]) == 2
    assert "--maintenance-event-cost 0.700000" in report["record_measurement_commands"][0]["command"]
    summary = module.format_pair_summary(report)
    assert "selected=5" in summary
    assert "event_cost=0.700" in summary
    assert "Internal Maintenance Reference:" in summary
    assert "event_cost_per_selected=0.200" in summary
    assert "Maintenance Alignment:" in summary
    assert "event_cost_per_selected_delta=-0.060" in summary
    assert "Event Memory Maintenance Coupling Reference:" in summary
    assert "best_profile=wide" in summary
    assert "record-energy-measurement" in summary


def test_parse_args_accepts_internal_maintenance_report_path():
    module = _load_module()

    args = module.parse_args(
        [
            "--pair-id",
            "pilot-3",
            "--replicate-index",
            "1",
            "--internal-maintenance-report-path",
            "workspace/evaluation/custom_internal_maintenance.json",
        ]
    )

    assert (
        args.internal_maintenance_report_path
        == "workspace/evaluation/custom_internal_maintenance.json"
    )


def test_parse_args_accepts_event_memory_maintenance_coupling_report_path():
    module = _load_module()

    args = module.parse_args(
        [
            "--pair-id",
            "pilot-4",
            "--replicate-index",
            "1",
            "--event-memory-maintenance-coupling-report-path",
            "workspace/evaluation/custom_event_memory_maintenance_coupling.json",
        ]
    )

    assert (
        args.event_memory_maintenance_coupling_report_path
        == "workspace/evaluation/custom_event_memory_maintenance_coupling.json"
    )


def test_load_meter_joules_accepts_direct_and_computed_readings():
    module = _load_module()
    reading_path = workspace_path("evaluation", "test_physical_meter_readings.json")
    manifest = {
        "pair_id": "meter-pair",
        "replicate_index": 1,
    }
    payload = {
        "schema": "sara-physical-meter-readings-v1",
        "pair_id": "meter-pair",
        "replicate_index": 1,
        "readings": {
            "sara": {"joules": 2.5},
            "ann": {"average_watts": 1.25, "duration_seconds": 4.0},
        },
    }
    os.makedirs(os.path.dirname(reading_path), exist_ok=True)
    with open(reading_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    try:
        joules = module.load_meter_joules(reading_path, manifest=manifest)
    finally:
        if os.path.exists(reading_path):
            os.remove(reading_path)

    assert joules == {"sara": 2.5, "ann": 5.0}


def test_load_meter_joules_rejects_mismatched_pair():
    module = _load_module()
    reading_path = workspace_path("evaluation", "test_physical_meter_mismatch.json")
    payload = {
        "pair_id": "wrong-pair",
        "replicate_index": 1,
        "readings": {
            "sara": {"joules": 2.5},
            "ann": {"joules": 5.0},
        },
    }
    os.makedirs(os.path.dirname(reading_path), exist_ok=True)
    with open(reading_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    try:
        try:
            module.load_meter_joules(
                reading_path,
                manifest={"pair_id": "meter-pair", "replicate_index": 1},
            )
        except ValueError as exc:
            assert "pair_id mismatch" in str(exc)
        else:
            raise AssertionError("Expected mismatched pair_id to be rejected.")
    finally:
        if os.path.exists(reading_path):
            os.remove(reading_path)


def test_build_meter_reading_template_carries_workload_duration():
    module = _load_module()
    template = module.build_meter_reading_template(
        {
            "pair_id": "template-pair",
            "replicate_index": 3,
            "measurement_tool": "powermetrics-v1",
            "measurement_boundary": "query-only-v1",
            "task": "paired_retrieval",
        },
        [
            {
                "system": "sara",
                "run_order": 1,
                "workload_result": {
                    "duration_seconds": 1.25,
                    "trial_count": 8,
                    "success_count": 8,
                },
            },
            {
                "system": "ann",
                "run_order": 2,
                "workload_result": {
                    "duration_seconds": 0.75,
                    "trial_count": 8,
                    "success_count": 8,
                },
            },
        ],
    )

    assert template["schema"] == "sara-physical-meter-readings-v1"
    assert template["readings"]["sara"]["duration_seconds"] == 1.25
    assert template["readings"]["ann"]["duration_seconds"] == 0.75
    assert template["readings"]["sara"]["run_order"] == 1
    assert template["readings"]["ann"]["run_order"] == 2
