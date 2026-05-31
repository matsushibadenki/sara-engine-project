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
    assert report["measurement_session_plan"]["planned_runs"][0]["run_id_template"].startswith(
        "ann-efficiency-real-joule-real_data_external_validity-sara-"
    )
    assert "--source real_energy_session" in report["measurement_session_plan"]["planned_runs"][0]["command_template"]
    assert report["measurement_plan"]["pending_pairs"][0]["command_template"].startswith(
        "python scripts/sara_cli.py record-energy-measurement"
    )
    summary = module.format_energy_measurement_summary(report)
    assert "Measurement Plan:" in summary
    assert "Measurement Session Plan:" in summary
    assert "task=real_data_external_validity" in summary
    assert "python scripts/sara_cli.py record-energy-measurement" in summary


def test_energy_measurement_readiness_accepts_real_joule_advantage():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [
            {"run_id": "sara-1", "system": "sara", "task": "qa", "success_count": 10, "joules": 2.0},
            {"run_id": "ann-1", "system": "ann", "task": "qa", "success_count": 10, "joules": 8.0},
        ],
        min_ann_to_sara_ratio=2.0,
    )

    assert report["passed"] is True
    assert report["status"] == "real_joule_evidence_passed"
    assert report["real_joule_measurements_present"] is True
    assert report["metrics"]["sara_joule_per_success"] == 0.2
    assert report["metrics"]["ann_joule_per_success"] == 0.8
    assert report["metrics"]["ann_to_sara_joule_efficiency_ratio"] == 4.0
    assert report["metrics"]["paired_task_count"] == 1
    assert report["metrics"]["min_paired_task_ann_to_sara_ratio"] == 4.0
    assert report["checks"]["paired_task_measurements_present"] is True
    assert report["checks"]["paired_task_rows_balanced"] is True
    assert report["checks"]["paired_task_efficiency_ratio_passed"] is True
    assert report["measurement_plan"]["ready_for_real_joule_claim"] is True
    assert report["measurement_plan"]["pending_pair_count"] == 0
    assert report["measurement_plan"]["weak_pair_count"] == 0
    assert report["measurement_session_plan"]["status"] == "ready_for_real_joule_claim"
    assert report["measurement_session_plan"]["planned_run_count"] == 0


def test_energy_measurement_readiness_rejects_unpaired_real_measurements():
    module = _load_module()

    report = module.build_energy_measurement_readiness_report(
        [
            {"run_id": "sara-1", "system": "sara", "task": "qa", "success_count": 10, "joules": 2.0},
            {"run_id": "ann-1", "system": "ann", "task": "summary", "success_count": 10, "joules": 8.0},
        ],
        min_ann_to_sara_ratio=2.0,
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
            {"run_id": "sara-1", "system": "sara", "task": "qa", "success_count": 10, "joules": 4.0},
            {"run_id": "ann-1", "system": "ann", "task": "qa", "success_count": 10, "joules": 5.0},
        ],
        min_ann_to_sara_ratio=2.0,
    )

    assert report["passed"] is False
    assert report["metrics"]["min_paired_task_ann_to_sara_ratio"] == 1.25
    assert report["checks"]["paired_task_efficiency_ratio_passed"] is False
    assert report["measurement_plan"]["weak_pair_count"] == 1
    assert report["measurement_plan"]["weak_pairs"][0]["task"] == "qa"
    assert report["measurement_session_plan"]["planned_run_count"] == 2
    planned = {(item["category"], item["system"]) for item in report["measurement_session_plan"]["planned_runs"]}
    assert ("repeat_weak_pair", "sara") in planned
    assert ("repeat_weak_pair", "ann") in planned
    summary = module.format_energy_measurement_summary(report)
    assert "weak_pairs:" in summary
    assert "task=qa" in summary
    assert "ratio=1.250" in summary


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


def test_format_measurement_session_plan_summary_lists_commands():
    module = _load_module()
    report = module.build_energy_measurement_readiness_report([])

    summary = module.format_measurement_session_plan_summary(report["measurement_session_plan"])

    assert "SARA Energy Measurement Session Plan" in summary
    assert "planned_run_count: 4" in summary
    assert "real_energy_session" in summary
    assert "record-energy-measurement" in summary


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
    )

    assert row["joules"] == 2.0
    assert row["average_watts"] == 0.8
    assert row["duration_seconds"] == 2.5
    assert row["joules_derivation"] == "average_watts_x_duration_seconds"


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
