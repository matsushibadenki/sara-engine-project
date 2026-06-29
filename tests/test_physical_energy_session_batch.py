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
            "physical_energy_session_batch.py",
        )
    )
    spec = importlib.util.spec_from_file_location("physical_energy_session_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_physical_energy_session_batch_deduplicates_pair_runs():
    module = _load_module()
    session_plan = {
        "session_id": "lab-session",
        "planned_runs": [
            {
                "category": "collect_missing_pair",
                "task": "real_data_external_validity",
                "system": "sara",
                "priority": "high",
                "replicate_count": 3,
                "pair_id_template": "lab-session-real_data_external_validity-pair-<replicate>",
                "pair_command_template": (
                    "python scripts/sara_cli.py run-physical-energy-pair "
                    "--pair-id lab-session-real_data_external_validity-pair-<replicate> "
                    "--replicate-index <replicate> "
                    "--event-memory-maintenance-coupling-report-path "
                    "workspace/evaluation/event_memory_maintenance_coupling_benchmark.json "
                    "--meter-template-path workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_meter_template.json"
                ),
                "meter_template_path": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_meter_template.json",
                "manifest_path_template": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_manifest.json",
                "trace_path_template": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_trace.jsonl",
                "report_path_template": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_report.json",
                "summary_path_template": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_summary.txt",
            },
            {
                "category": "collect_missing_pair",
                "task": "real_data_external_validity",
                "system": "ann",
                "priority": "high",
                "replicate_count": 3,
                "pair_id_template": "lab-session-real_data_external_validity-pair-<replicate>",
                "pair_command_template": (
                    "python scripts/sara_cli.py run-physical-energy-pair "
                    "--pair-id lab-session-real_data_external_validity-pair-<replicate> "
                    "--replicate-index <replicate> "
                    "--event-memory-maintenance-coupling-report-path "
                    "workspace/evaluation/event_memory_maintenance_coupling_benchmark.json "
                    "--meter-template-path workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_meter_template.json"
                ),
                "meter_template_path": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_meter_template.json",
                "manifest_path_template": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_manifest.json",
                "trace_path_template": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_trace.jsonl",
                "report_path_template": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_report.json",
                "summary_path_template": "workspace/evaluation/lab-session_real_data_external_validity_r<replicate>_summary.txt",
            },
        ],
    }

    report = module.build_physical_energy_session_batch(session_plan)

    assert report["planned_pair_count"] == 3
    assert report["batch_runs"][0]["pair_id"] == "lab-session-real_data_external_validity-pair-1"
    assert report["batch_runs"][0]["systems"] == ["ann", "sara"]
    assert "--replicate-index 1" in report["batch_runs"][0]["command"]
    assert "--event-memory-maintenance-coupling-report-path" in report["batch_runs"][0]["command"]
    assert report["batch_runs"][2]["meter_template_path"].endswith("_r3_meter_template.json")


def test_physical_energy_session_batch_main_writes_report():
    module = _load_module()
    session_plan_path = workspace_path("evaluation", "test_energy_session_batch_plan.json")
    report_path = workspace_path("evaluation", "test_energy_session_batch_report.json")
    summary_path = workspace_path("evaluation", "test_energy_session_batch_summary.txt")
    payload = {
        "session_id": "lab-session",
        "planned_runs": [
            {
                "category": "repeat_weak_pair",
                "task": "energy_efficiency_benchmark",
                "system": "sara",
                "priority": "medium",
                "replicate_count": 1,
                "pair_id_template": "lab-session-energy_efficiency_benchmark-pair-<replicate>",
                "pair_command_template": (
                    "python scripts/sara_cli.py run-physical-energy-pair "
                    "--pair-id lab-session-energy_efficiency_benchmark-pair-<replicate> "
                    "--replicate-index <replicate> "
                    "--event-memory-maintenance-coupling-report-path "
                    "workspace/evaluation/event_memory_maintenance_coupling_benchmark.json"
                ),
                "meter_template_path": "workspace/evaluation/lab-session_energy_efficiency_benchmark_r<replicate>_meter_template.json",
                "manifest_path_template": "workspace/evaluation/lab-session_energy_efficiency_benchmark_r<replicate>_manifest.json",
                "trace_path_template": "workspace/evaluation/lab-session_energy_efficiency_benchmark_r<replicate>_trace.jsonl",
                "report_path_template": "workspace/evaluation/lab-session_energy_efficiency_benchmark_r<replicate>_report.json",
                "summary_path_template": "workspace/evaluation/lab-session_energy_efficiency_benchmark_r<replicate>_summary.txt",
            }
        ],
    }
    os.makedirs(os.path.dirname(session_plan_path), exist_ok=True)
    with open(session_plan_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    try:
        exit_code = module.main(
            [
                "--session-plan-path",
                session_plan_path,
                "--report-path",
                report_path,
                "--summary-path",
                summary_path,
            ]
        )
        assert exit_code == 0
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["planned_pair_count"] == 1
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = handle.read()
        assert "SARA Physical Energy Session Batch" in summary
        assert "run-physical-energy-pair" in summary
    finally:
        for path in (session_plan_path, report_path, summary_path):
            if os.path.exists(path):
                os.remove(path)
