import importlib.util
import json
import os
import sys
import tempfile
from typing import Any, Optional, Tuple
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def _load_sara_cli_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "sara_cli.py")
    )
    spec = importlib.util.spec_from_file_location("sara_cli_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chat_distill_dispatches_to_agent_chat(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "chat-distill", "--model", "models/test_agent"])

    sara_cli.main()

    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[0] == sys.executable
    assert args[1] == "scripts/eval/chat_agent.py"
    assert "--model-dir" in args
    assert "models/test_agent" in args


def test_train_self_org_dispatches_to_training_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "train-self-org"])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once_with([sys.executable, "scripts/train/train_self_organized.py"])


def test_prune_dispatches_to_memory_pruner(monkeypatch):
    sara_cli = _load_sara_cli_module()
    prune_mock = Mock()
    monkeypatch.setattr(sara_cli, "prune_model_memory", prune_mock)
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "prune", "--model", "models/test.msgpack", "--threshold", "12"])

    sara_cli.main()

    prune_mock.assert_called_once_with("models/test.msgpack", 12.0)


def test_inspect_memory_dispatches_to_memory_health_reporter(monkeypatch):
    sara_cli = _load_sara_cli_module()
    inspect_mock = Mock(return_value={"pattern_count": 3})
    monkeypatch.setattr(sara_cli, "inspect_inference_memory", inspect_mock)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sara_cli.py", "inspect-memory", "--model", "models/test.msgpack", "--report", "workspace/tests/health.json"],
    )

    sara_cli.main()

    inspect_mock.assert_called_once_with("models/test.msgpack", "workspace/tests/health.json")


def test_upgrade_memory_dispatches_to_upgrade_utility(monkeypatch):
    sara_cli = _load_sara_cli_module()
    upgrade_mock = Mock(return_value={"pattern_count": 3})
    monkeypatch.setattr(sara_cli, "upgrade_inference_memory", upgrade_mock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "upgrade-memory",
            "--model",
            "models/test.msgpack",
            "--output",
            "models/upgraded/test.msgpack",
            "--report",
            "workspace/tests/upgrade.json",
            "--replay-data",
            "workspace/tests/replay.jsonl",
            "--turboquant",
        ],
    )

    sara_cli.main()

    upgrade_mock.assert_called_once_with(
        "models/test.msgpack",
        "models/upgraded/test.msgpack",
        replay_data_path="workspace/tests/replay.jsonl",
        enable_turboquant=True,
    )


def test_build_replay_data_dispatches_to_replay_builder(monkeypatch):
    sara_cli = _load_sara_cli_module()
    replay_mock = Mock(return_value={"example_count": 2})
    monkeypatch.setattr(sara_cli, "build_replay_data", replay_mock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "build-replay-data",
            "--data",
            "data/raw/chat_data.jsonl",
            "--output",
            "workspace/replay/chat_replay_tokens.jsonl",
            "--tokenizer",
            "stub-tokenizer",
        ],
    )

    sara_cli.main()

    replay_mock.assert_called_once_with(
        "data/raw/chat_data.jsonl",
        "workspace/replay/chat_replay_tokens.jsonl",
        tokenizer_name="stub-tokenizer",
    )


def test_collect_continual_horizon_external_dispatches_bounded_collector(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "collect-continual-horizon-external",
            "--target-horizon",
            "10",
            "--timeout-seconds",
            "5",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    command = mock_run.call_args.args[0]
    assert command[1] == "scripts/data/collect_continual_horizon_external.py"
    assert command[command.index("--target-horizon") + 1] == "10"
    assert command[command.index("--timeout-seconds") + 1] == "5.0"


def test_build_autobot_dataset_dispatches_to_builder_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "build-autobot-dataset",
            "--records-path",
            "data/processed/autobot/test_records.jsonl",
            "--candidate-path",
            "data/interim/autobot/test_candidates.jsonl",
            "--rejected-path",
            "data/interim/autobot/test_rejected.jsonl",
            "--accepted-path",
            "data/processed/autobot/test_materials.jsonl",
            "--curriculum-path",
            "data/processed/autobot/test_curriculum.jsonl",
            "--report-path",
            "workspace/autobot/test_dataset_report.json",
            "--summary-path",
            "workspace/autobot/test_dataset_summary.txt",
            "--fixture-request-plan-path",
            "workspace/autobot/test_fixture_request_plan.json",
            "--collection-targets-path",
            "workspace/autobot/test_collection_targets.json",
            "--evaluation-gap",
            "negative_control",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "bot/dataset_builder.py"]
    assert "--records-path" in args
    assert "data/processed/autobot/test_records.jsonl" in args
    assert "--candidate-path" in args
    assert "data/interim/autobot/test_candidates.jsonl" in args
    assert "--curriculum-path" in args
    assert "data/processed/autobot/test_curriculum.jsonl" in args
    assert "--fixture-request-plan-path" in args
    assert "workspace/autobot/test_fixture_request_plan.json" in args
    assert "--collection-targets-path" in args
    assert "workspace/autobot/test_collection_targets.json" in args
    assert "--evaluation-gap" in args
    assert "negative_control" in args


def test_build_autobot_gap_materials_dispatches_to_builder_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "build-autobot-gap-materials",
            "--accepted-path",
            "data/processed/autobot/test_materials.jsonl",
            "--targets-path",
            "workspace/autobot/test_collection_targets.json",
            "--output-path",
            "data/processed/autobot/test_gap_materials.jsonl",
            "--report-path",
            "workspace/autobot/test_gap_materials_report.json",
            "--summary-path",
            "workspace/autobot/test_gap_materials_summary.txt",
            "--blocked-request-id",
            "fixture_counterexample_gap",
            "--clear-blocked-request-id",
            "fixture_source_diversity_gap",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "bot/gap_materials_builder.py"]
    assert "--accepted-path" in args
    assert "data/processed/autobot/test_materials.jsonl" in args
    assert "--targets-path" in args
    assert "workspace/autobot/test_collection_targets.json" in args
    assert "--blocked-request-id" in args
    assert "fixture_counterexample_gap" in args
    assert "--clear-blocked-request-id" in args
    assert "fixture_source_diversity_gap" in args


def test_enqueue_autobot_gap_curriculum_dispatches_to_builder_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "enqueue-autobot-gap-curriculum",
            "--curriculum-path",
            "data/processed/autobot/test_gap_curriculum.jsonl",
            "--queue-path",
            "workspace/autobot/test_gap_train_queue.json",
            "--report-path",
            "workspace/autobot/test_gap_enqueue_report.json",
            "--summary-path",
            "workspace/autobot/test_gap_enqueue_summary.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "bot/enqueue_curriculum.py"]
    assert "--curriculum-path" in args
    assert "data/processed/autobot/test_gap_curriculum.jsonl" in args


def test_run_autobot_gap_loop_dispatches_to_runner_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "run-autobot-gap-loop",
            "--records-path",
            "data/processed/autobot/test_records.jsonl",
            "--candidate-path",
            "data/interim/autobot/test_candidates.jsonl",
            "--rejected-path",
            "data/interim/autobot/test_rejected.jsonl",
            "--accepted-path",
            "data/processed/autobot/test_materials.jsonl",
            "--curriculum-path",
            "data/processed/autobot/test_curriculum.jsonl",
            "--fixture-request-plan-path",
            "workspace/autobot/test_fixture_request_plan.json",
            "--collection-targets-path",
            "workspace/autobot/test_collection_targets.json",
            "--gap-output-path",
            "data/processed/autobot/test_gap_materials.jsonl",
            "--gap-curriculum-path",
            "data/processed/autobot/test_gap_curriculum.jsonl",
            "--queue-path",
            "workspace/autobot/test_train_queue.json",
            "--report-path",
            "workspace/autobot/test_gap_loop_report.json",
            "--summary-path",
            "workspace/autobot/test_gap_loop_summary.txt",
            "--evaluation-gap",
            "negative_control",
            "--blocked-request-id",
            "fixture_counterexample_gap",
            "--clear-blocked-request-id",
            "fixture_source_diversity_gap",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "bot/run_gap_loop.py"]
    assert "--candidate-path" in args
    assert "data/interim/autobot/test_candidates.jsonl" in args
    assert "--rejected-path" in args
    assert "data/interim/autobot/test_rejected.jsonl" in args
    assert "--gap-curriculum-path" in args
    assert "data/processed/autobot/test_gap_curriculum.jsonl" in args
    assert "--blocked-request-id" in args
    assert "fixture_counterexample_gap" in args
    assert "--clear-blocked-request-id" in args
    assert "fixture_source_diversity_gap" in args


def test_eval_autobot_gap_loop_readiness_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-autobot-gap-loop-readiness",
            "--loop-report-path",
            "workspace/autobot/test_gap_loop_report.json",
            "--collection-targets-path",
            "workspace/autobot/test_collection_targets.json",
            "--dataset-report-path",
            "workspace/autobot/test_dataset_report.json",
            "--gap-report-path",
            "workspace/autobot/test_gap_report.json",
            "--enqueue-report-path",
            "workspace/autobot/test_enqueue_report.json",
            "--report-path",
            "workspace/evaluation/test_autobot_gap_loop_readiness.json",
            "--summary-path",
            "workspace/evaluation/test_autobot_gap_loop_readiness.txt",
            "--min-accepted-count",
            "6",
            "--min-gap-build-coverage",
            "0.75",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/autobot_gap_loop_readiness.py"]
    assert "--loop-report-path" in args
    assert "workspace/autobot/test_gap_loop_report.json" in args
    assert "--collection-targets-path" in args
    assert "workspace/autobot/test_collection_targets.json" in args
    assert "--dataset-report-path" in args
    assert "workspace/autobot/test_dataset_report.json" in args
    assert "--gap-report-path" in args
    assert "workspace/autobot/test_gap_report.json" in args
    assert "--enqueue-report-path" in args
    assert "workspace/autobot/test_enqueue_report.json" in args
    assert "--min-accepted-count" in args
    assert "6" in args
    assert "--min-gap-build-coverage" in args
    assert "0.75" in args


def test_fix_memory_dispatches_to_managed_repair_utility(monkeypatch):
    sara_cli = _load_sara_cli_module()
    fix_mock = Mock(return_value={"matched_token": True})
    monkeypatch.setattr(sara_cli, "fix_inference_memory", fix_mock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "fix-memory",
            "--model",
            "models/test.msgpack",
            "--output",
            "models/repaired/test.msgpack",
            "--report",
            "workspace/tests/fix_memory.json",
            "--context-tokens",
            "1,2,3",
            "--wrong-token-id",
            "7",
            "--decay",
            "0.5",
            "--dry-run",
        ],
    )

    sara_cli.main()

    fix_mock.assert_called_once_with(
        "models/test.msgpack",
        "models/repaired/test.msgpack",
        context_tokens=[1, 2, 3],
        context_text=None,
        wrong_token_id=7,
        wrong_text=None,
        tokenizer_path=None,
        decay=0.5,
        dry_run=True,
        report_path="workspace/tests/fix_memory.json",
    )


def test_eval_external_validity_dispatches_to_benchmark_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-external-validity",
            "--corpus",
            "data/processed/corpus.txt",
            "--max-docs",
            "32",
            "--max-cases",
            "8",
            "--report-path",
            "workspace/evaluation/test_external_validity.json",
            "--summary-path",
            "workspace/evaluation/test_external_validity.txt",
            "--history-path",
            "workspace/evaluation/test_external_validity_history.json",
            "--regression-tolerance",
            "0.1",
            "--pretrained-embedding-model",
            "workspace/models/test-embedding-model",
            "--cross-encoder-model",
            "workspace/models/test-cross-encoder-model",
            "--no-history-update",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/real_data_external_validity.py"]
    assert "--max-docs" in args
    assert "32" in args
    assert "workspace/evaluation/test_external_validity.json" in args
    assert "--history-path" in args
    assert "workspace/evaluation/test_external_validity_history.json" in args
    assert "--regression-tolerance" in args
    assert "0.1" in args
    assert "--pretrained-embedding-model" in args
    assert "workspace/models/test-embedding-model" in args
    assert "--cross-encoder-model" in args
    assert "workspace/models/test-cross-encoder-model" in args
    assert "--no-history-update" in args


def test_eval_external_validity_ladder_dispatches_to_ladder_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-external-validity-ladder",
            "--corpus",
            "data/processed/corpus.txt",
            "--profile",
            "tiny:32:8",
            "--profile",
            "pilot:64:12",
            "--report-path",
            "workspace/evaluation/test_external_validity_ladder.json",
            "--summary-path",
            "workspace/evaluation/test_external_validity_ladder.txt",
            "--regression-tolerance",
            "0.1",
            "--no-history-update",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/real_data_external_validity_ladder.py"]
    assert "--profile" in args
    assert "tiny:32:8" in args
    assert "pilot:64:12" in args
    assert "workspace/evaluation/test_external_validity_ladder.json" in args
    assert "--regression-tolerance" in args
    assert "0.1" in args
    assert "--no-history-update" in args


def test_eval_ann_efficiency_roadmap_dispatches_to_gate_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-ann-efficiency-roadmap",
            "--energy-report-path",
            "workspace/evaluation/test_energy.json",
            "--external-validity-report-path",
            "workspace/evaluation/test_external.json",
            "--external-ladder-report-path",
            "workspace/evaluation/test_ladder.json",
            "--energy-measurement-report-path",
            "workspace/evaluation/test_energy_measurement.json",
            "--operational-report-path",
            "workspace/release/test_operational.json",
            "--output-report-path",
            "workspace/evaluation/test_ann_roadmap.json",
            "--output-summary-path",
            "workspace/evaluation/test_ann_roadmap.txt",
            "--refresh-artifacts",
            "--allow-missing-operational",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/ann_efficiency_roadmap_gate.py"]
    assert "--energy-report-path" in args
    assert "workspace/evaluation/test_energy.json" in args
    assert "--external-validity-report-path" in args
    assert "workspace/evaluation/test_external.json" in args
    assert "--external-ladder-report-path" in args
    assert "workspace/evaluation/test_ladder.json" in args
    assert "--energy-measurement-report-path" in args
    assert "workspace/evaluation/test_energy_measurement.json" in args
    assert "--operational-report-path" in args
    assert "workspace/release/test_operational.json" in args
    assert "--output-report-path" in args
    assert "workspace/evaluation/test_ann_roadmap.json" in args
    assert "--refresh-artifacts" in args
    assert "--allow-missing-operational" in args


def test_record_energy_measurement_dispatches_to_readiness_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "record-energy-measurement",
            "--measurement-path",
            "data/raw/test_energy_measurements.jsonl",
            "--run-id",
            "sara-1",
            "--system",
            "sara",
            "--task",
            "qa",
            "--success-count",
            "5",
            "--joules",
            "1.5",
            "--source",
            "manual_meter",
            "--session-id",
            "lab-session",
            "--duration-seconds",
            "3.0",
            "--average-watts",
            "0.5",
            "--notes",
            "pilot run",
            "--pair-id",
            "qa-pair-1",
            "--replicate-index",
            "1",
            "--environment-fingerprint",
            "env-sha256",
            "--task-fixture-hash",
            "fixture-sha256",
            "--success-criterion-id",
            "exact-match-v1",
            "--measurement-boundary",
            "query-only-v1",
            "--measurement-tool",
            "powermetrics-v1",
            "--cpu-model",
            "test-cpu",
            "--thread-count",
            "1",
            "--process-affinity",
            "core-0",
            "--power-mode",
            "ac-fixed",
            "--warmup-count",
            "2",
            "--measured-repetitions",
            "5",
            "--trial-count",
            "5",
            "--run-order",
            "1",
            "--report-path",
            "workspace/evaluation/test_energy_measurement.json",
            "--summary-path",
            "workspace/evaluation/test_energy_measurement.txt",
            "--session-plan-path",
            "workspace/evaluation/test_energy_measurement_session_plan.json",
            "--session-plan-summary-path",
            "workspace/evaluation/test_energy_measurement_session_plan.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/energy_measurement_readiness.py"]
    assert "--append-measurement" in args
    assert "data/raw/test_energy_measurements.jsonl" in args
    assert "--run-id" in args
    assert "sara-1" in args
    assert "--joules" in args
    assert "1.5" in args
    assert "--duration-seconds" in args
    assert "3.0" in args
    assert "--average-watts" in args
    assert "0.5" in args
    assert "--session-id" in args
    assert "lab-session" in args
    assert "--notes" in args
    assert "pilot run" in args
    assert "--pair-id" in args
    assert "qa-pair-1" in args
    assert "--success-criterion-id" in args
    assert "exact-match-v1" in args
    assert "--trial-count" in args
    assert "--session-plan-path" in args
    assert "workspace/evaluation/test_energy_measurement_session_plan.json" in args
    assert "--session-plan-summary-path" in args
    assert "workspace/evaluation/test_energy_measurement_session_plan.txt" in args


def test_physical_energy_pair_dispatches_to_runner(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "run-physical-energy-pair",
            "--pair-id",
            "pilot-1",
            "--replicate-index",
            "1",
            "--repetitions",
            "2",
            "--report-path",
            "workspace/evaluation/test_physical_pair_report.json",
            "--summary-path",
            "workspace/evaluation/test_physical_pair_summary.txt",
            "--meter-reading-path",
            "workspace/evaluation/test_meter_readings.json",
            "--meter-template-path",
            "workspace/evaluation/test_meter_template.json",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/physical_energy_pair_runner.py",
    ]
    assert "--pair-id" in args
    assert "pilot-1" in args
    assert "--report-path" in args
    assert "workspace/evaluation/test_physical_pair_report.json" in args
    assert "--summary-path" in args
    assert "workspace/evaluation/test_physical_pair_summary.txt" in args
    assert "--meter-reading-path" in args
    assert "workspace/evaluation/test_meter_readings.json" in args
    assert "--meter-template-path" in args
    assert "workspace/evaluation/test_meter_template.json" in args
    assert "--dry-run" in args


def test_physical_energy_session_batch_dispatches_to_runner(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "run-physical-energy-session-batch",
            "--session-plan-path",
            "workspace/evaluation/test_energy_measurement_session_plan.json",
            "--report-path",
            "workspace/evaluation/test_physical_energy_batch.json",
            "--summary-path",
            "workspace/evaluation/test_physical_energy_batch.txt",
            "--execute-dry-run-pairs",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/physical_energy_session_batch.py",
    ]
    assert "--session-plan-path" in args
    assert "workspace/evaluation/test_energy_measurement_session_plan.json" in args
    assert "--execute-dry-run-pairs" in args


def test_physical_energy_session_progress_dispatches_to_runner(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-physical-energy-session-progress",
            "--batch-report-path",
            "workspace/evaluation/test_physical_energy_batch.json",
            "--measurement-path",
            "data/raw/test_energy_measurements.jsonl",
            "--report-path",
            "workspace/evaluation/test_physical_energy_progress.json",
            "--summary-path",
            "workspace/evaluation/test_physical_energy_progress.txt",
            "--internal-maintenance-report-path",
            "workspace/evaluation/test_internal_maintenance_efficiency.json",
            "--event-memory-maintenance-coupling-report-path",
            "workspace/evaluation/test_event_memory_maintenance_coupling.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/physical_energy_session_progress.py",
    ]
    assert "--batch-report-path" in args
    assert "workspace/evaluation/test_physical_energy_batch.json" in args
    assert "--measurement-path" in args
    assert "data/raw/test_energy_measurements.jsonl" in args
    assert "--internal-maintenance-report-path" in args
    assert "workspace/evaluation/test_internal_maintenance_efficiency.json" in args
    assert "--event-memory-maintenance-coupling-report-path" in args
    assert "workspace/evaluation/test_event_memory_maintenance_coupling.json" in args


def test_sara_ann_comparison_dispatches_to_runner(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-sara-ann-comparison",
            "--external-validity-report-path",
            "workspace/evaluation/test_real_data_external_validity.json",
            "--external-ladder-report-path",
            "workspace/evaluation/test_real_data_external_validity_ladder.json",
            "--energy-measurement-report-path",
            "workspace/evaluation/test_energy_measurement_readiness.json",
            "--internal-maintenance-report-path",
            "workspace/evaluation/test_internal_maintenance_efficiency.json",
            "--event-memory-report-path",
            "workspace/evaluation/test_event_memory_ingest_pipeline.json",
            "--event-memory-maintenance-coupling-report-path",
            "workspace/evaluation/test_event_memory_maintenance_coupling.json",
            "--report-path",
            "workspace/evaluation/test_sara_ann_comparison_report.json",
            "--summary-path",
            "workspace/evaluation/test_sara_ann_comparison_report.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/sara_ann_comparison_report.py",
    ]
    assert "--external-validity-report-path" in args
    assert "workspace/evaluation/test_real_data_external_validity.json" in args
    assert "--energy-measurement-report-path" in args
    assert "workspace/evaluation/test_energy_measurement_readiness.json" in args
    assert "--internal-maintenance-report-path" in args
    assert "workspace/evaluation/test_internal_maintenance_efficiency.json" in args
    assert "--event-memory-report-path" in args
    assert "workspace/evaluation/test_event_memory_ingest_pipeline.json" in args
    assert "--event-memory-maintenance-coupling-report-path" in args
    assert "workspace/evaluation/test_event_memory_maintenance_coupling.json" in args


def test_sparse_diffusion_block_readiness_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-sparse-diffusion-block-readiness",
            "--block-count",
            "3",
            "--report-path",
            "workspace/evaluation/test_sparse_diffusion.json",
            "--summary-path",
            "workspace/evaluation/test_sparse_diffusion.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/sparse_diffusion_block_readiness.py"]
    assert "--block-count" in args
    assert "3" in args
    assert "--report-path" in args
    assert "workspace/evaluation/test_sparse_diffusion.json" in args
    assert "--summary-path" in args
    assert "workspace/evaluation/test_sparse_diffusion.txt" in args


def test_rust_core_readiness_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-rust-core-readiness",
            "--report-path",
            "workspace/evaluation/test_rust_readiness.json",
            "--summary-path",
            "workspace/evaluation/test_rust_readiness.txt",
            "--run-cargo-test",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/rust_core_readiness.py"]
    assert "--report-path" in args
    assert "workspace/evaluation/test_rust_readiness.json" in args
    assert "--summary-path" in args
    assert "workspace/evaluation/test_rust_readiness.txt" in args
    assert "--run-cargo-test" in args


def test_rust_core_benchmark_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-rust-core-benchmark",
            "--iterations",
            "7",
            "--report-path",
            "workspace/evaluation/test_rust_benchmark.json",
            "--summary-path",
            "workspace/evaluation/test_rust_benchmark.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/rust_core_benchmark.py"]
    assert "--iterations" in args
    assert "7" in args
    assert "--report-path" in args
    assert "workspace/evaluation/test_rust_benchmark.json" in args
    assert "--summary-path" in args
    assert "workspace/evaluation/test_rust_benchmark.txt" in args


def test_research_benchmark_suite_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-research-benchmark-suite",
            "--dry-run",
            "--rust-iterations",
            "9",
            "--manifest-path",
            "workspace/evaluation/test_research_benchmark_manifest.json",
            "--summary-path",
            "workspace/evaluation/test_research_benchmark_summary.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/research_benchmark_suite.py"]
    assert "--dry-run" in args
    assert "--rust-iterations" in args
    assert "9" in args
    assert "--manifest-path" in args
    assert "workspace/evaluation/test_research_benchmark_manifest.json" in args
    assert "--summary-path" in args
    assert "workspace/evaluation/test_research_benchmark_summary.txt" in args


def test_research_fixture_readiness_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-research-fixture-readiness",
            "--fixture-path",
            "data/processed/benchmark_fixtures/external_validity_cases.jsonl",
            "--report-path",
            "workspace/evaluation/test_fixture_readiness.json",
            "--summary-path",
            "workspace/evaluation/test_fixture_readiness.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/research_fixture_readiness.py"]
    assert "--fixture-path" in args
    assert "data/processed/benchmark_fixtures/external_validity_cases.jsonl" in args
    assert "--report-path" in args
    assert "workspace/evaluation/test_fixture_readiness.json" in args
    assert "--summary-path" in args
    assert "workspace/evaluation/test_fixture_readiness.txt" in args


def test_neuromorphic_capability_matrix_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-neuromorphic-capability-matrix",
            "--profile",
            "lava",
            "--profile",
            "akida",
            "--active-row-count",
            "12",
            "--context-length",
            "24",
            "--total-readout-size",
            "96",
            "--quantization-bits",
            "3",
            "--report-path",
            "workspace/evaluation/test_neuromorphic_matrix.json",
            "--summary-path",
            "workspace/evaluation/test_neuromorphic_matrix.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/neuromorphic_capability_matrix.py"]
    assert "--profile" in args
    assert "lava" in args
    assert "akida" in args
    assert "--active-row-count" in args
    assert "12" in args
    assert "--context-length" in args
    assert "24" in args
    assert "--total-readout-size" in args
    assert "96" in args
    assert "--quantization-bits" in args
    assert "3" in args
    assert "--report-path" in args
    assert "workspace/evaluation/test_neuromorphic_matrix.json" in args
    assert "--summary-path" in args
    assert "workspace/evaluation/test_neuromorphic_matrix.txt" in args


def test_own_latent_learning_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-own-latent-learning",
            "--fixture-path",
            "data/processed/benchmark_fixtures/test_own_latent.jsonl",
            "--report-path",
            "workspace/evaluation/test_own_latent.json",
            "--summary-path",
            "workspace/evaluation/test_own_latent.txt",
            "--history-path",
            "workspace/evaluation/test_own_latent_history.json",
            "--train-sizes",
            "4,8",
            "--no-history-update",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/own_latent_learning_benchmark.py"]
    assert "--fixture-path" in args
    assert "data/processed/benchmark_fixtures/test_own_latent.jsonl" in args
    assert "--report-path" in args
    assert "workspace/evaluation/test_own_latent.json" in args
    assert "--history-path" in args
    assert "workspace/evaluation/test_own_latent_history.json" in args
    assert "--train-sizes" in args
    assert "4,8" in args
    assert "--no-history-update" in args


def test_operator_llm_assistant_readiness_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-operator-llm-assistant-readiness",
            "--report-path",
            "workspace/evaluation/test_operator_llm.json",
            "--summary-path",
            "workspace/evaluation/test_operator_llm.txt",
            "--enabled",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/operator_llm_assistant_readiness.py"]
    assert "--report-path" in args
    assert "workspace/evaluation/test_operator_llm.json" in args
    assert "--summary-path" in args
    assert "workspace/evaluation/test_operator_llm.txt" in args
    assert "--enabled" in args


def test_dendritic_feedback_gate_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-dendritic-feedback-gate",
            "--event-budget",
            "48",
            "--report-path",
            "workspace/evaluation/test_dendritic_gate.json",
            "--summary-path",
            "workspace/evaluation/test_dendritic_gate.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/dendritic_feedback_gate_benchmark.py"]
    assert "--event-budget" in args
    assert "48" in args
    assert "--report-path" in args
    assert "workspace/evaluation/test_dendritic_gate.json" in args
    assert "--summary-path" in args
    assert "workspace/evaluation/test_dendritic_gate.txt" in args


def test_phase33_preregistration_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "register-phase33-structured-edge-preregistration",
            "--draft-path",
            "workspace/evaluation/phase33_draft.json",
            "--output-path",
            "workspace/evaluation/phase33_registered.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase33_structured_edge_preregistration.py",
    ]
    assert "--draft-path" in args
    assert "workspace/evaluation/phase33_draft.json" in args
    assert "--output-path" in args
    assert "workspace/evaluation/phase33_registered.json" in args


def test_phase33_draft_builder_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "build-phase33-structured-edge-preregistration-draft",
            "--fixture-path",
            "data/processed/benchmark_fixtures/phase33_cases.jsonl",
            "--draft-path",
            "workspace/evaluation/phase33_draft.json",
            "--environment-path",
            "workspace/evaluation/phase33_environment.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase33_structured_edge_draft.py",
    ]
    assert "--fixture-path" in args
    assert "data/processed/benchmark_fixtures/phase33_cases.jsonl" in args
    assert "--draft-path" in args
    assert "workspace/evaluation/phase33_draft.json" in args
    assert "--environment-path" in args
    assert "workspace/evaluation/phase33_environment.json" in args


def test_phase33_benchmark_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-phase33-structured-edge",
            "--fixture-path",
            "data/processed/benchmark_fixtures/phase33_cases.jsonl",
            "--preregistration-path",
            "workspace/evaluation/phase33_registered.json",
            "--output-path",
            "workspace/evaluation/phase33_benchmark.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase33_structured_edge_benchmark.py",
    ]
    assert "data/processed/benchmark_fixtures/phase33_cases.jsonl" in args
    assert "workspace/evaluation/phase33_registered.json" in args
    assert "workspace/evaluation/phase33_benchmark.json" in args


def test_phase33_twinprop_draft_dispatches_to_separate_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "build-phase33-twinprop-ablation-preregistration-draft",
            "--fixture-path",
            "data/processed/benchmark_fixtures/twinprop.jsonl",
            "--draft-path",
            "workspace/evaluation/twinprop_draft.json",
            "--environment-path",
            "workspace/evaluation/twinprop_environment.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase33_twinprop_ablation_draft.py",
    ]
    assert "data/processed/benchmark_fixtures/twinprop.jsonl" in args
    assert "workspace/evaluation/twinprop_draft.json" in args
    assert "workspace/evaluation/twinprop_environment.json" in args


def test_phase33_twinprop_registration_dispatches_to_separate_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "register-phase33-twinprop-ablation-preregistration",
            "--draft-path",
            "workspace/evaluation/twinprop_draft.json",
            "--output-path",
            "workspace/evaluation/twinprop_registered.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase33_twinprop_ablation_preregistration.py",
    ]
    assert "workspace/evaluation/twinprop_draft.json" in args
    assert "workspace/evaluation/twinprop_registered.json" in args


def test_phase34_memory_cache_draft_dispatches_to_separate_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "build-phase34-memory-checkpoint-cache-preregistration-draft",
        "--fixture-path", "data/processed/benchmark_fixtures/phase34.jsonl",
        "--draft-path", "workspace/evaluation/phase34_draft.json",
        "--environment-path", "workspace/evaluation/phase34_environment.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/phase34_memory_checkpoint_cache_draft.py"]
    assert "data/processed/benchmark_fixtures/phase34.jsonl" in args
    assert "workspace/evaluation/phase34_draft.json" in args
    assert "workspace/evaluation/phase34_environment.json" in args


def test_phase34_memory_cache_registration_dispatches_to_separate_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "register-phase34-memory-checkpoint-cache-preregistration",
        "--draft-path", "workspace/evaluation/phase34_draft.json",
        "--output-path", "workspace/evaluation/phase34_registered.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/phase34_memory_checkpoint_cache_preregistration.py"]
    assert "workspace/evaluation/phase34_draft.json" in args
    assert "workspace/evaluation/phase34_registered.json" in args


def test_phase34_memory_cache_benchmark_dispatches_to_registered_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "eval-phase34-memory-checkpoint-cache",
        "--fixture-path", "data/processed/benchmark_fixtures/phase34.jsonl",
        "--preregistration-path", "workspace/evaluation/phase34_registered.json",
        "--output-path", "workspace/evaluation/phase34_benchmark.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_checkpoint_cache_benchmark.py",
    ]
    assert "data/processed/benchmark_fixtures/phase34.jsonl" in args
    assert "workspace/evaluation/phase34_registered.json" in args
    assert "workspace/evaluation/phase34_benchmark.json" in args


def test_phase34_separation_draft_dispatches_to_separate_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "build-phase34-memory-cache-separation-preregistration-draft",
        "--fixture-path", "data/processed/benchmark_fixtures/separation.jsonl",
        "--draft-path", "workspace/evaluation/separation_draft.json",
        "--environment-path", "workspace/evaluation/separation_environment.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_separation_draft.py",
    ]
    assert "data/processed/benchmark_fixtures/separation.jsonl" in args
    assert "workspace/evaluation/separation_draft.json" in args


def test_phase34_separation_registration_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "register-phase34-memory-cache-separation-preregistration",
        "--draft-path", "workspace/evaluation/separation_draft.json",
        "--output-path", "workspace/evaluation/separation_registered.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_separation_preregistration.py",
    ]
    assert "workspace/evaluation/separation_draft.json" in args
    assert "workspace/evaluation/separation_registered.json" in args


def test_phase34_separation_benchmark_dispatches_to_registered_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "eval-phase34-memory-cache-separation",
        "--fixture-path", "data/processed/benchmark_fixtures/separation.jsonl",
        "--preregistration-path", "workspace/evaluation/separation_registered.json",
        "--output-path", "workspace/evaluation/separation_benchmark.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_separation_benchmark.py",
    ]
    assert "data/processed/benchmark_fixtures/separation.jsonl" in args
    assert "workspace/evaluation/separation_registered.json" in args
    assert "workspace/evaluation/separation_benchmark.json" in args


def test_phase34_factorial_draft_dispatches_to_separate_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "build-phase34-memory-cache-factorial-preregistration-draft",
        "--fixture-path", "data/processed/benchmark_fixtures/factorial.jsonl",
        "--draft-path", "workspace/evaluation/factorial_draft.json",
        "--environment-path", "workspace/evaluation/factorial_environment.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_factorial_draft.py",
    ]
    assert "data/processed/benchmark_fixtures/factorial.jsonl" in args
    assert "workspace/evaluation/factorial_draft.json" in args


def test_phase34_factorial_registration_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "register-phase34-memory-cache-factorial-preregistration",
        "--draft-path", "workspace/evaluation/factorial_draft.json",
        "--output-path", "workspace/evaluation/factorial_registered.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_factorial_preregistration.py",
    ]
    assert "workspace/evaluation/factorial_draft.json" in args
    assert "workspace/evaluation/factorial_registered.json" in args


def test_phase34_independent_adapter_draft_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "build-phase34-independent-adapter-preregistration-draft",
        "--manifest-path", "data/processed/autobot/external.jsonl",
        "--case-plan-path", "workspace/evaluation/independent_plan.json",
        "--draft-path", "workspace/evaluation/independent_draft.json",
        "--environment-path", "workspace/evaluation/independent_environment.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_factorial_independent_adapter_draft.py",
    ]
    assert "data/processed/autobot/external.jsonl" in args
    assert "workspace/evaluation/independent_plan.json" in args


def test_phase34_independent_adapter_registration_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "register-phase34-independent-adapter-preregistration",
        "--draft-path", "workspace/evaluation/independent_draft.json",
        "--output-path", "workspace/evaluation/independent_registered.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_factorial_independent_adapter_preregistration.py",
    ]
    assert "workspace/evaluation/independent_registered.json" in args


def test_phase34_semantic_delayed_recall_draft_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "build-phase34-semantic-delayed-recall-preregistration-draft",
            "--fixture-path",
            "data/processed/benchmark_fixtures/semantic.jsonl",
            "--draft-path",
            "workspace/evaluation/semantic_draft.json",
            "--environment-path",
            "workspace/evaluation/semantic_environment.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_semantic_delayed_recall_draft.py",
    ]
    assert "data/processed/benchmark_fixtures/semantic.jsonl" in args
    assert "workspace/evaluation/semantic_draft.json" in args


def test_phase34_semantic_delayed_recall_registration_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "register-phase34-semantic-delayed-recall-preregistration",
            "--draft-path",
            "workspace/evaluation/semantic_draft.json",
            "--output-path",
            "workspace/evaluation/semantic_registered.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_semantic_delayed_recall_preregistration.py",
    ]
    assert "workspace/evaluation/semantic_draft.json" in args
    assert "workspace/evaluation/semantic_registered.json" in args


def test_phase34_semantic_delayed_recall_benchmark_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-phase34-semantic-delayed-recall",
            "--fixture-path",
            "data/processed/benchmark_fixtures/semantic.jsonl",
            "--preregistration-path",
            "workspace/evaluation/semantic_registered.json",
            "--request-path",
            "workspace/evaluation/semantic_request.json",
            "--output-path",
            "workspace/evaluation/semantic_benchmark.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_semantic_delayed_recall_benchmark.py",
    ]
    assert "data/processed/benchmark_fixtures/semantic.jsonl" in args
    assert "workspace/evaluation/semantic_benchmark.json" in args


def test_phase34_independent_adapter_benchmark_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "eval-phase34-independent-adapter-v2",
        "--manifest-path", "data/processed/autobot/external.jsonl",
        "--case-plan-path", "workspace/evaluation/independent_plan.json",
        "--preregistration-path", "workspace/evaluation/independent_registered.json",
        "--output-path", "workspace/evaluation/independent_benchmark.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_factorial_independent_adapter_benchmark.py",
    ]
    assert "workspace/evaluation/independent_benchmark.json" in args


def test_phase34_independent_provenance_review_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "review-phase34-independent-provenance",
        "--raw-path", "data/raw/architecture_migration/source_rows.jsonl",
        "--manifest-path", "data/processed/autobot/external.jsonl",
        "--output-path", "workspace/evaluation/provenance.json",
        "--timeout-seconds", "5",
        "--offline-only",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_independent_provenance_review.py",
    ]
    assert "--offline-only" in args
    assert "workspace/evaluation/provenance.json" in args


def test_phase34_cpython_snapshot_registration_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "register-phase34-cpython-snapshot",
        "--case-plan-path", "workspace/evaluation/plan.json",
        "--output-path", "workspace/evaluation/cpython_registration.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_cpython_snapshot_preregistration.py",
    ]
    assert "workspace/evaluation/cpython_registration.json" in args


def test_phase34_cpython_snapshot_collection_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "collect-phase34-cpython-snapshot",
        "--preregistration-path", "workspace/evaluation/cpython_registration.json",
        "--raw-path", "data/raw/phase34_cpython_snapshot/source_rows.jsonl",
        "--manifest-path", "data/processed/autobot/cpython_snapshot.jsonl",
        "--report-path", "workspace/evaluation/cpython_collection.json",
        "--timeout-seconds", "5",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/data/collect_phase34_cpython_snapshot.py",
    ]
    assert "data/processed/autobot/cpython_snapshot.jsonl" in args
    assert "workspace/evaluation/cpython_collection.json" in args


def test_phase34_transcribed_excerpt_review_request_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "build-phase34-transcribed-excerpt-review-request",
        "--raw-path", "data/raw/architecture_migration/source_rows.jsonl",
        "--provenance-path", "workspace/evaluation/provenance.json",
        "--output-path", "workspace/evaluation/manual_review.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_transcribed_excerpt_review_request.py",
    ]
    assert "workspace/evaluation/manual_review.json" in args


def test_phase34_transcribed_excerpt_review_gate_dispatches_human_decision(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "review-phase34-transcribed-excerpts",
        "--request-path", "workspace/evaluation/review_request.json",
        "--ledger-path", "workspace/evaluation/review_decisions.json",
        "--report-path", "workspace/evaluation/review_gate.json",
        "--record-id", "arch-migration-python-001",
        "--authoritative-section-locator", "argparse/module",
        "--authoritative-text-hash", "a" * 64,
        "--alignment-decision", "aligned",
        "--semantic-distortion", "not-found",
        "--reviewer", "human-reviewer",
        "--reviewed-at", "2026-08-20T10:00:00+09:00",
        "--notes", "Compared with the cited section.",
        "--attest-human-review",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_transcribed_excerpt_review_gate.py",
    ]
    assert "--attest-human-review" in args
    assert "workspace/evaluation/review_decisions.json" in args
    assert "workspace/evaluation/review_gate.json" in args


def test_phase34_review_support_commands_dispatch_registered_collection(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "register-phase34-review-support",
        "--request-path", "workspace/evaluation/review_request.json",
        "--output-path", "workspace/evaluation/review_support_registration.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_review_support_preregistration.py",
    ]
    assert "workspace/evaluation/review_support_registration.json" in args

    mock_run.reset_mock()
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "collect-phase34-review-support",
        "--request-path", "workspace/evaluation/review_request.json",
        "--preregistration-path", "workspace/evaluation/review_support_registration.json",
        "--raw-path", "data/raw/phase34_review_support/test_rows.jsonl",
        "--packet-path", "workspace/evaluation/review_packet.json",
        "--report-path", "workspace/evaluation/review_collection.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/data/collect_phase34_review_support.py",
    ]
    assert "data/raw/phase34_review_support/test_rows.jsonl" in args
    assert "workspace/evaluation/review_packet.json" in args


def test_phase34_cpython_git_snapshot_commands_dispatch_to_registered_scripts(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "register-phase34-cpython-git-snapshot",
        "--case-plan-path", "workspace/evaluation/plan.json",
        "--output-path", "workspace/evaluation/git_registration.json",
    ])
    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()
    assert exc_info.value.code == 0
    assert mock_run.call_args.args[0][:2] == [
        sys.executable,
        "scripts/eval/phase34_cpython_git_snapshot_preregistration.py",
    ]

    mock_run.reset_mock()
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "collect-phase34-cpython-git-snapshot",
        "--preregistration-path", "workspace/evaluation/git_registration.json",
        "--raw-path", "data/raw/phase34_cpython_git_snapshot/source_rows.jsonl",
        "--manifest-path", "data/processed/autobot/git_snapshot.jsonl",
        "--report-path", "workspace/evaluation/git_collection.json",
    ])
    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()
    assert exc_info.value.code == 0
    assert mock_run.call_args.args[0][:2] == [
        sys.executable,
        "scripts/data/collect_phase34_cpython_git_snapshot.py",
    ]


def test_phase34_factorial_benchmark_dispatches_to_registered_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "eval-phase34-memory-cache-factorial",
        "--fixture-path", "data/processed/benchmark_fixtures/factorial.jsonl",
        "--preregistration-path", "workspace/evaluation/factorial_registered.json",
        "--output-path", "workspace/evaluation/factorial_benchmark.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_factorial_benchmark.py",
    ]
    assert "data/processed/benchmark_fixtures/factorial.jsonl" in args
    assert "workspace/evaluation/factorial_registered.json" in args
    assert "workspace/evaluation/factorial_benchmark.json" in args


def test_phase34_factorial_independent_gate_dispatches_to_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", [
        "sara_cli.py",
        "gate-phase34-memory-cache-factorial-independent",
        "--preregistration-path", "workspace/evaluation/factorial_registered.json",
        "--factorial-report-path", "workspace/evaluation/factorial_benchmark.json",
        "--external-gate-path", "workspace/evaluation/external_gate.json",
        "--output-path", "workspace/evaluation/factorial_independent_gate.json",
    ])

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase34_memory_cache_factorial_independent_gate.py",
    ]
    assert "workspace/evaluation/factorial_registered.json" in args
    assert "workspace/evaluation/factorial_benchmark.json" in args
    assert "workspace/evaluation/external_gate.json" in args
    assert "workspace/evaluation/factorial_independent_gate.json" in args


def test_phase33_twinprop_benchmark_dispatches_to_registered_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-phase33-twinprop-ablation",
            "--fixture-path",
            "data/processed/benchmark_fixtures/twinprop.jsonl",
            "--preregistration-path",
            "workspace/evaluation/twinprop_registered.json",
            "--output-path",
            "workspace/evaluation/twinprop_benchmark.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/phase33_twinprop_ablation_benchmark.py",
    ]
    assert "data/processed/benchmark_fixtures/twinprop.jsonl" in args
    assert "workspace/evaluation/twinprop_registered.json" in args
    assert "workspace/evaluation/twinprop_benchmark.json" in args


def test_build_own_latent_manifest_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "build-own-latent-manifest",
            "--materials-path",
            "data/processed/autobot/test_materials.jsonl",
            "--manifest-path",
            "data/processed/autobot/test_latent_manifest.jsonl",
            "--report-path",
            "workspace/evaluation/test_latent_manifest.json",
            "--summary-path",
            "workspace/evaluation/test_latent_manifest.txt",
            "--width",
            "512",
            "--max-events",
            "16",
            "--max-terms",
            "8",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/own_latent_manifest_builder.py"]
    assert "--materials-path" in args
    assert "data/processed/autobot/test_materials.jsonl" in args
    assert "--manifest-path" in args
    assert "data/processed/autobot/test_latent_manifest.jsonl" in args
    assert "--width" in args
    assert "512" in args
    assert "--max-events" in args
    assert "16" in args


def test_sparse_plan_trace_verifier_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-sparse-plan-trace-verifier",
            "--fixture-path",
            "data/processed/benchmark_fixtures/test_plan_cases.jsonl",
            "--repair-path",
            "data/processed/autobot/test_plan_repairs.jsonl",
            "--report-path",
            "workspace/evaluation/test_plan_trace.json",
            "--summary-path",
            "workspace/evaluation/test_plan_trace.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/sparse_plan_trace_verifier.py"]
    assert "--fixture-path" in args
    assert "data/processed/benchmark_fixtures/test_plan_cases.jsonl" in args
    assert "--repair-path" in args
    assert "data/processed/autobot/test_plan_repairs.jsonl" in args
    assert "--report-path" in args
    assert "workspace/evaluation/test_plan_trace.json" in args


def test_synesthetic_multimodal_binding_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-synesthetic-multimodal-binding",
            "--fixture-path",
            "data/processed/benchmark_fixtures/test_synesthetic_cases.jsonl",
            "--cross-link-path",
            "data/interim/autobot/test_synesthetic_links.jsonl",
            "--binding-manifest-path",
            "data/processed/autobot/test_synesthetic_manifest.jsonl",
            "--latent-manifest-path",
            "data/processed/autobot/test_latent_manifest.jsonl",
            "--report-path",
            "workspace/evaluation/test_synesthetic_binding.json",
            "--window-ms",
            "40",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/synesthetic_multimodal_binding_benchmark.py",
    ]
    assert "data/processed/benchmark_fixtures/test_synesthetic_cases.jsonl" in args
    assert "data/interim/autobot/test_synesthetic_links.jsonl" in args
    assert "data/processed/autobot/test_synesthetic_manifest.jsonl" in args
    assert "data/processed/autobot/test_latent_manifest.jsonl" in args
    assert "workspace/evaluation/test_synesthetic_binding.json" in args
    assert "40.0" in args


def test_sparse_reasoning_prior_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-sparse-reasoning-prior",
            "--fixture-path",
            "data/processed/benchmark_fixtures/test_reasoning_prior.jsonl",
            "--trace-path",
            "workspace/evaluation/test_reasoning_prior_traces.jsonl",
            "--report-path",
            "workspace/evaluation/test_reasoning_prior.json",
            "--summary-path",
            "workspace/evaluation/test_reasoning_prior.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/sparse_reasoning_prior_benchmark.py"]
    assert "data/processed/benchmark_fixtures/test_reasoning_prior.jsonl" in args
    assert "workspace/evaluation/test_reasoning_prior_traces.jsonl" in args
    assert "workspace/evaluation/test_reasoning_prior.json" in args


def test_resonance_credit_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-resonance-credit",
            "--fixture-path",
            "data/processed/benchmark_fixtures/test_resonance.jsonl",
            "--state-path",
            "workspace/evaluation/test_resonance_state.json",
            "--report-path",
            "workspace/evaluation/test_resonance.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/resonance_credit_benchmark.py"]
    assert "data/processed/benchmark_fixtures/test_resonance.jsonl" in args
    assert "workspace/evaluation/test_resonance_state.json" in args
    assert "workspace/evaluation/test_resonance.json" in args


def test_adaptive_credit_field_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-adaptive-credit-field",
            "--fixture-path",
            "data/processed/benchmark_fixtures/test_adaptive_credit.jsonl",
            "--state-path",
            "workspace/evaluation/test_adaptive_credit_state.json",
            "--report-path",
            "workspace/evaluation/test_adaptive_credit.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/adaptive_credit_field_benchmark.py"]
    assert "data/processed/benchmark_fixtures/test_adaptive_credit.jsonl" in args
    assert "workspace/evaluation/test_adaptive_credit_state.json" in args
    assert "workspace/evaluation/test_adaptive_credit.json" in args


def test_adaptive_credit_event_memory_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-adaptive-credit-event-memory",
            "--fixture-path",
            "data/processed/benchmark_fixtures/test_adaptive_credit_event_memory.jsonl",
            "--report-path",
            "workspace/evaluation/test_adaptive_credit_event_memory.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [sys.executable, "scripts/eval/adaptive_credit_event_memory_benchmark.py"]
    assert "data/processed/benchmark_fixtures/test_adaptive_credit_event_memory.jsonl" in args
    assert "workspace/evaluation/test_adaptive_credit_event_memory.json" in args


def test_resonance_credit_integration_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-resonance-credit-integration",
            "--report-path",
            "workspace/evaluation/test_resonance_integration.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/resonance_credit_integration_benchmark.py",
    ]
    assert "workspace/evaluation/test_resonance_integration.json" in args


def test_event_state_cache_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-event-state-cache",
            "--fixture-path",
            "data/processed/benchmark_fixtures/test_event_cache.jsonl",
            "--state-path",
            "workspace/evaluation/test_event_cache_state.json",
            "--report-path",
            "workspace/evaluation/test_event_cache.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/event_state_cache_benchmark.py",
    ]
    assert "data/processed/benchmark_fixtures/test_event_cache.jsonl" in args
    assert "workspace/evaluation/test_event_cache_state.json" in args
    assert "workspace/evaluation/test_event_cache.json" in args


def test_event_state_cache_integration_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-event-state-cache-integration",
            "--manifest-path",
            "data/processed/autobot/test_latent_manifest.jsonl",
            "--report-path",
            "workspace/evaluation/test_event_cache_integration.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    args = mock_run.call_args.args[0]
    assert args[:2] == [
        sys.executable,
        "scripts/eval/event_state_cache_integration_benchmark.py",
    ]
    assert "data/processed/autobot/test_latent_manifest.jsonl" in args
    assert "workspace/evaluation/test_event_cache_integration.json" in args


def test_db_status_without_database_prints_empty_notice(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()
    original_exists = sara_cli.os.path.exists
    monkeypatch.setattr(
        sara_cli.os.path,
        "exists",
        lambda path: False if path == "data/sara_corpus.db" else original_exists(path),
    )
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "db-status"])

    sara_cli.main()

    captured = capsys.readouterr()
    assert "DBが存在しません" in captured.out


def test_db_status_prints_material_summary(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def get_stats(self):
            return [("document", 2), ("chat", 1)]

        def get_material_summary(self):
            return {
                "total_count": 3,
                "active_count": 2,
                "inactive_count": 1,
                "avg_quality_score": 0.75,
                "categories": [("research", 2), ("dialogue", 1)],
            }

        def get_review_summary(self):
            return {
                "by_category": [{"key": "research", "count": 2, "avg_quality_score": 0.8}],
                "by_source": [{"key": "paper_notes", "count": 2, "avg_quality_score": 0.8}],
                "by_lang": [{"key": "en", "count": 2, "avg_quality_score": 0.8}],
                "by_status": [{"key": "active", "count": 2, "avg_quality_score": 0.85}],
            }

    monkeypatch.setattr(sara_cli, "SaraCorpusDB", StubDB)
    monkeypatch.setattr(sara_cli.os.path, "exists", lambda path: True)
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "db-status"])

    sara_cli.main()

    captured = capsys.readouterr()
    assert "有効素材: 2 件" in captured.out
    assert "無効素材: 1 件" in captured.out
    assert "平均品質スコア: 0.75" in captured.out
    assert "- research: 2 件" in captured.out
    assert "source内訳:" in captured.out
    assert "- paper_notes: 2 件" in captured.out
    assert "lang内訳:" in captured.out
    assert "- en: 2 件" in captured.out


def test_db_status_can_print_json(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def get_stats(self):
            return [("document", 2), ("chat", 1)]

        def get_material_summary(self):
            return {"total_count": 3, "active_count": 2, "inactive_count": 1, "avg_quality_score": 0.75, "categories": []}

        def get_review_summary(self):
            return {"by_category": [], "by_source": [], "by_lang": [], "by_status": []}

    monkeypatch.setattr(sara_cli, "SaraCorpusDB", StubDB)
    monkeypatch.setattr(sara_cli.os.path, "exists", lambda path: True)
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "db-status", "--format", "json"])

    sara_cli.main()

    captured = capsys.readouterr()
    assert '"summary"' in captured.out
    assert '"review_summary"' in captured.out


def test_db_import_dispatches_metadata_options(monkeypatch):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def import_file(self, *args, **kwargs):
            self.called = (args, kwargs)
            return 3

        def get_material_summary(self):
            return {"total_count": 3, "active_count": 2, "inactive_count": 1, "avg_quality_score": 0.8, "categories": []}

        def get_review_summary(self):
            return {"by_category": [], "by_source": [], "by_lang": [], "by_status": []}

    stub = StubDB("data/sara_corpus.db")
    monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "db-import",
            "data/raw/materials.jsonl",
            "--category",
            "research",
            "--lang",
            "en",
            "--source-version",
            "2026-03-31",
            "--quality-score",
            "0.9",
            "--inactive",
        ],
    )

    sara_cli.main()

    args, kwargs = stub.called
    assert args == ("data/raw/materials.jsonl",)
    assert kwargs["category"] == "research"
    assert kwargs["lang"] == "en"
    assert kwargs["source_version"] == "2026-03-31"
    assert kwargs["quality_score"] == 0.9
    assert kwargs["is_active"] is False


def test_db_import_writes_report(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def import_file(self, *args, **kwargs):
            self.called = (args, kwargs)
            return 4

        def get_material_summary(self):
            return {"total_count": 4, "active_count": 4, "inactive_count": 0, "avg_quality_score": 0.8, "categories": [("research", 4)]}

        def get_review_summary(self):
            return {
                "by_category": [{"key": "research", "count": 4, "avg_quality_score": 0.8}],
                "by_source": [{"key": "materials.jsonl", "count": 4, "avg_quality_score": 0.8}],
                "by_lang": [{"key": "en", "count": 4, "avg_quality_score": 0.8}],
                "by_status": [{"key": "active", "count": 4, "avg_quality_score": 0.8}],
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "db_import_report.json")
        stub = StubDB("data/sara_corpus.db")
        monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
        monkeypatch.setattr(sara_cli, "ensure_parent_directory", lambda path: report_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sara_cli.py",
                "db-import",
                "data/raw/materials.jsonl",
                "--category",
                "research",
                "--lang",
                "en",
                "--source-version",
                "2026-04-01",
                "--quality-score",
                "0.8",
                "--report",
                "workspace/tests/db_import_report.json",
            ],
        )

        sara_cli.main()

        captured = capsys.readouterr()
        assert "Saved import report" in captured.out
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["file"] == "data/raw/materials.jsonl"
        assert report["added_count"] == 4
        assert report["metadata"]["category"] == "research"
        assert report["metadata"]["lang"] == "en"
        assert report["metadata"]["source_version"] == "2026-04-01"
        assert report["metadata"]["quality_score"] == 0.8
        assert report["metadata"]["is_active"] is True
        assert report["summary"]["total_count"] == 4
        assert report["review_summary"]["by_category"][0]["key"] == "research"


def test_db_export_dispatches_filter_options(monkeypatch):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            self.self_org_called: Optional[Tuple[Tuple[Any, ...], dict[str, Any]]] = None
            self.distill_called: Optional[Tuple[Tuple[Any, ...], dict[str, Any]]] = None

        def get_material_summary(self):
            return {"total_count": 12, "active_count": 10, "inactive_count": 2, "avg_quality_score": 0.85, "categories": []}

        def get_review_summary(self):
            return {"by_category": [], "by_source": [], "by_lang": [], "by_status": []}

        def summarize_export_plan(self, **kwargs):
            self.plan_kwargs = kwargs
            return {"total_count": 3, "items": []}

        def export_for_self_organized(self, *args, **kwargs):
            self.self_org_called = (args, kwargs)
            return 2

        def export_for_distillation(self, *args, **kwargs):
            self.distill_called = (args, kwargs)
            return 1

    stub = StubDB("data/sara_corpus.db")
    monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "db-export",
            "--category",
            "research",
            "--source",
            "paper_notes",
            "--min-quality-score",
            "0.8",
            "--show-inactive",
        ],
    )

    sara_cli.main()

    assert stub.self_org_called is not None
    args, kwargs = stub.self_org_called
    assert args == ("data/processed/corpus.txt",)
    assert kwargs["category"] == "research"
    assert kwargs["source"] == "paper_notes"
    assert kwargs["min_quality_score"] == 0.8
    assert kwargs["show_inactive"] is True


def test_db_list_prints_preview(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def list_materials(self, **kwargs):
            self.kwargs = kwargs
            return [
                {
                    "text_type": "document",
                    "category": "research",
                    "quality_score": 0.9,
                    "source": "paper_notes",
                    "source_version": "v2",
                    "lang": "en",
                    "is_active": False,
                    "preview": "Research note about STDP.",
                }
            ]

    stub = StubDB("data/sara_corpus.db")
    monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
    monkeypatch.setattr(sara_cli.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "db-list",
            "--category",
            "research",
            "--source",
            "paper_notes",
            "--min-quality-score",
            "0.8",
            "--show-inactive",
            "--limit",
            "5",
        ],
    )

    sara_cli.main()

    captured = capsys.readouterr()
    assert "SARA Corpus Material Preview" in captured.out
    assert "Research note about STDP." in captured.out
    assert "status=inactive" in captured.out
    assert stub.kwargs["source"] == "paper_notes"
    assert stub.kwargs["show_inactive"] is True


def test_db_list_can_print_json(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def list_materials(self, **kwargs):
            self.kwargs = kwargs
            return [
                {
                    "text_type": "document",
                    "category": "research",
                    "quality_score": 0.9,
                    "source": "paper_notes",
                    "source_version": "v2",
                    "lang": "en",
                    "is_active": True,
                    "preview": "Research note about STDP.",
                }
            ]

    stub = StubDB("data/sara_corpus.db")
    monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
    monkeypatch.setattr(sara_cli.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sara_cli.py", "db-list", "--format", "json", "--limit", "5"],
    )

    sara_cli.main()

    captured = capsys.readouterr()
    assert '"items"' in captured.out
    assert '"preview": "Research note about STDP."' in captured.out


def test_db_export_dry_run_prints_summary(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def get_material_summary(self):
            return {"total_count": 10, "active_count": 8, "inactive_count": 2, "avg_quality_score": 0.85, "categories": []}

        def get_review_summary(self):
            return {"by_category": [], "by_source": [], "by_lang": [], "by_status": []}

        def summarize_export_plan(self, **kwargs):
            self.kwargs = kwargs
            return {
                "total_count": 2,
                "items": [
                    {
                        "text_type": "document",
                        "category": "research",
                        "count": 2,
                        "avg_quality_score": 0.9,
                    }
                ],
            }

    stub = StubDB("data/sara_corpus.db")
    monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "db-export",
            "--category",
            "research",
            "--source",
            "paper_notes",
            "--min-quality-score",
            "0.8",
            "--show-inactive",
            "--dry-run",
        ],
    )

    sara_cli.main()

    captured = capsys.readouterr()
    assert "SARA Export Dry Run" in captured.out
    assert "total_count: 2" in captured.out
    assert "document/research: 2 件" in captured.out
    assert stub.kwargs["category"] == "research"
    assert stub.kwargs["source"] == "paper_notes"
    assert stub.kwargs["min_quality_score"] == 0.8
    assert stub.kwargs["show_inactive"] is True


def test_db_export_dry_run_writes_report(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def get_material_summary(self):
            return {"total_count": 5, "active_count": 5, "inactive_count": 0, "avg_quality_score": 0.9, "categories": []}

        def get_review_summary(self):
            return {
                "by_category": [{"key": "research", "count": 1, "avg_quality_score": 0.9}],
                "by_source": [{"key": "paper_notes", "count": 1, "avg_quality_score": 0.9}],
                "by_lang": [{"key": "en", "count": 1, "avg_quality_score": 0.9}],
                "by_status": [{"key": "active", "count": 1, "avg_quality_score": 0.9}],
            }

        def summarize_export_plan(self, **kwargs):
            self.kwargs = kwargs
            return {
                "total_count": 1,
                "items": [
                    {
                        "text_type": "document",
                        "category": "research",
                        "count": 1,
                        "avg_quality_score": 0.9,
                    }
                ],
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "db_export_report.json")
        stub = StubDB("data/sara_corpus.db")
        monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
        monkeypatch.setattr(sara_cli, "ensure_parent_directory", lambda path: report_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sara_cli.py",
                "db-export",
                "--category",
                "research",
                "--dry-run",
                "--report",
                "workspace/tests/db_export_report.json",
            ],
        )

        sara_cli.main()

        captured = capsys.readouterr()
        assert "Saved export report" in captured.out
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["dry_run"] is True
        assert report["filters"]["category"] == "research"
        assert report["material_summary"]["total_count"] == 5
        assert report["review_summary"]["by_source"][0]["key"] == "paper_notes"
        assert report["plan"]["total_count"] == 1
        assert report["delta"]["selected_count"] == 1
        assert report["delta"]["total_material_count"] == 5
        assert abs(report["delta"]["selected_ratio"] - 0.2) < 1e-9


def test_db_activate_dispatches_filter_options(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def set_material_active_state(self, *args, **kwargs):
            self.called = (args, kwargs)
            return 3

    stub = StubDB("data/sara_corpus.db")
    monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "db-activate",
            "--category",
            "research",
            "--source",
            "paper_notes",
            "--min-quality-score",
            "0.8",
        ],
    )

    sara_cli.main()

    captured = capsys.readouterr()
    args, kwargs = stub.called
    assert args == (True,)
    assert kwargs["category"] == "research"
    assert kwargs["source"] == "paper_notes"
    assert kwargs["min_quality_score"] == 0.8
    assert kwargs["include_inactive"] is True
    assert "3 件の素材を active" in captured.out


def test_db_deactivate_dispatches_filter_options(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def set_material_active_state(self, *args, **kwargs):
            self.called = (args, kwargs)
            return 2

    stub = StubDB("data/sara_corpus.db")
    monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "db-deactivate",
            "--category",
            "research",
            "--source",
            "paper_notes",
            "--min-quality-score",
            "0.8",
        ],
    )

    sara_cli.main()

    captured = capsys.readouterr()
    args, kwargs = stub.called
    assert args == (False,)
    assert kwargs["category"] == "research"
    assert kwargs["source"] == "paper_notes"
    assert kwargs["min_quality_score"] == 0.8
    assert kwargs["include_inactive"] is True
    assert "2 件の素材を inactive" in captured.out


def test_db_export_writes_default_report(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()

    class StubDB:
        def __init__(self, _path: str):
            pass

        def get_material_summary(self):
            return {"total_count": 8, "active_count": 8, "inactive_count": 0, "avg_quality_score": 0.9, "categories": []}

        def get_review_summary(self):
            return {
                "by_category": [{"key": "research", "count": 2, "avg_quality_score": 0.9}],
                "by_source": [{"key": "paper_notes", "count": 2, "avg_quality_score": 0.9}],
                "by_lang": [{"key": "en", "count": 2, "avg_quality_score": 0.9}],
                "by_status": [{"key": "active", "count": 2, "avg_quality_score": 0.9}],
            }

        def summarize_export_plan(self, **kwargs):
            return {
                "total_count": 2,
                "items": [
                    {
                        "text_type": "document",
                        "category": "research",
                        "count": 2,
                        "avg_quality_score": 0.9,
                    }
                ],
            }

        def export_for_self_organized(self, *args, **kwargs):
            return 2

        def export_for_distillation(self, *args, **kwargs):
            return 2

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "db_export_report.json")
        stub = StubDB("data/sara_corpus.db")
        monkeypatch.setattr(sara_cli, "SaraCorpusDB", lambda _path: stub)
        monkeypatch.setattr(
            sara_cli,
            "workspace_path",
            lambda *parts: os.path.join(tmpdir, parts[-1]),
        )
        monkeypatch.setattr(sara_cli, "ensure_parent_directory", lambda path: report_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sara_cli.py",
                "db-export",
                "--category",
                "research",
            ],
        )

        sara_cli.main()

        captured = capsys.readouterr()
        assert "Saved export report" in captured.out
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["dry_run"] is False
        assert report["material_summary"]["total_count"] == 8
        assert report["review_summary"]["by_category"][0]["key"] == "research"
        assert report["outputs"]["corpus_count"] == 2
        assert report["outputs"]["chat_count"] == 2
        assert report["filters"]["category"] == "research"
        assert report["delta"]["selected_count"] == 2
        assert report["delta"]["total_material_count"] == 8
        assert abs(report["delta"]["selected_ratio"] - 0.25) < 1e-9


def test_clean_removes_non_gitkeep_items(monkeypatch):
    sara_cli = _load_sara_cli_module()
    removed_files = []
    removed_dirs = []

    def fake_exists(path: str) -> bool:
        return path in {"data/interim", "data/processed", "data/interim/tmp.txt", "data/processed/subdir"}

    def fake_listdir(path: str):
        if path == "data/interim":
            return [".gitkeep", "tmp.txt"]
        if path == "data/processed":
            return ["subdir"]
        return []

    monkeypatch.setattr(sara_cli.os.path, "exists", fake_exists)
    monkeypatch.setattr(sara_cli.os, "listdir", fake_listdir)
    monkeypatch.setattr(sara_cli.os.path, "isdir", lambda path: path == "data/processed/subdir")
    monkeypatch.setattr(sara_cli.os, "remove", lambda path: removed_files.append(path))
    monkeypatch.setattr(sara_cli.shutil, "rmtree", lambda path: removed_dirs.append(path))
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "clean"])

    sara_cli.main()

    assert removed_files == ["data/interim/tmp.txt"]
    assert removed_dirs == ["data/processed/subdir"]


def test_eval_event_memory_ingest_pipeline_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-event-memory-ingest-pipeline",
            "--report-path",
            "workspace/evaluation/test_event_memory_ingest_pipeline.json",
            "--summary-path",
            "workspace/evaluation/test_event_memory_ingest_pipeline.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once_with(
        [
            sys.executable,
            "scripts/eval/event_memory_ingest_pipeline.py",
            "--report-path",
            "workspace/evaluation/test_event_memory_ingest_pipeline.json",
            "--summary-path",
            "workspace/evaluation/test_event_memory_ingest_pipeline.txt",
        ]
    )


def test_eval_event_memory_maintenance_coupling_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-event-memory-maintenance-coupling",
            "--report-path",
            "workspace/evaluation/test_event_memory_maintenance_coupling.json",
            "--summary-path",
            "workspace/evaluation/test_event_memory_maintenance_coupling.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once_with(
        [
            sys.executable,
            "scripts/eval/event_memory_maintenance_coupling_benchmark.py",
            "--report-path",
            "workspace/evaluation/test_event_memory_maintenance_coupling.json",
            "--summary-path",
            "workspace/evaluation/test_event_memory_maintenance_coupling.txt",
        ]
    )


def test_eval_persistent_self_state_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-persistent-self-state",
            "--report-path",
            "workspace/evaluation/test_persistent_self_state.json",
            "--summary-path",
            "workspace/evaluation/test_persistent_self_state.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once_with(
        [
            sys.executable,
            "scripts/eval/persistent_self_state_benchmark.py",
            "--report-path",
            "workspace/evaluation/test_persistent_self_state.json",
            "--summary-path",
            "workspace/evaluation/test_persistent_self_state.txt",
        ]
    )


def test_eval_idle_replay_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-idle-replay",
            "--report-path",
            "workspace/evaluation/test_idle_replay.json",
            "--summary-path",
            "workspace/evaluation/test_idle_replay.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once_with(
        [
            sys.executable,
            "scripts/eval/idle_replay_benchmark.py",
            "--report-path",
            "workspace/evaluation/test_idle_replay.json",
            "--summary-path",
            "workspace/evaluation/test_idle_replay.txt",
        ]
    )


def test_eval_internal_maintenance_efficiency_dispatches_to_eval_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sara_cli.py",
            "eval-internal-maintenance-efficiency",
            "--report-path",
            "workspace/evaluation/test_internal_maintenance.json",
            "--summary-path",
            "workspace/evaluation/test_internal_maintenance.txt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sara_cli.main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once_with(
        [
            sys.executable,
            "scripts/eval/internal_maintenance_efficiency_benchmark.py",
            "--report-path",
            "workspace/evaluation/test_internal_maintenance.json",
            "--summary-path",
            "workspace/evaluation/test_internal_maintenance.txt",
        ]
    )
