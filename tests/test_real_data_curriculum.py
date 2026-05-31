import importlib.util
import os


def _load_script():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "train",
        "run_real_data_curriculum.py",
    )
    spec = importlib.util.spec_from_file_location("run_real_data_curriculum_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_curriculum_commands_small_includes_phase5_and_no_phase4():
    module = _load_script()
    profile = module._profiles()["small"]
    commands = module.build_curriculum_commands(profile, skip_gates=False)
    command_lines = [" ".join(command) for command in commands]

    assert any("scripts/sara_cli.py db-export" in line for line in command_lines)
    assert any("scripts/train/train_self_organized.py --corpus data/processed/corpus.txt" in line for line in command_lines)
    assert any("phase3_accuracy_suite.py --history-path" in line for line in command_lines)
    assert any("--regression-tolerance 0.050000" in line for line in command_lines)
    assert any("scripts/train/train_snn_lm.py" in line for line in command_lines)
    assert any("phase5_predictive_coding_benchmark.py" in line for line in command_lines)
    assert any("real_data_external_validity.py" in line for line in command_lines)
    assert any("real_data_external_validity_small.json" in line for line in command_lines)
    assert any("real_data_external_validity_small_history.json" in line for line in command_lines)
    assert not any("phase4_scale_continual_benchmark.py" in line for line in command_lines)


def test_build_curriculum_commands_large_includes_operational_readiness():
    module = _load_script()
    profile = module._profiles()["large"]
    commands = module.build_curriculum_commands(profile, skip_gates=False)
    command_lines = [" ".join(command) for command in commands]

    assert any("phase4_scale_continual_benchmark.py" in line for line in command_lines)
    assert any("phase5_completion_gate.py" in line for line in command_lines)
    assert any("--max-docs 4096 --max-cases 128" in line for line in command_lines)
    assert any("real_data_external_validity_large_history.json" in line for line in command_lines)
    assert any("operational_readiness.py --refresh-artifacts" in line for line in command_lines)


def test_run_real_data_curriculum_dry_run_passes_without_execution():
    module = _load_script()
    report = module.run_real_data_curriculum(stage="medium", dry_run=True, skip_gates=True)

    assert report["suite_name"] == "RealDataCurriculumRunner"
    assert report["stage"] == "medium"
    assert report["dry_run"] is True
    assert report["passed"] is True
    assert "preflight" in report
    assert report["command_count"] >= 3


def test_preflight_reports_missing_corpus_db_without_creating_it():
    module = _load_script()
    profile = module._profiles()["small"]
    report = module.build_preflight_report(profile, db_path="workspace/tests/missing_curriculum.db")

    assert report["passed"] is False
    assert report["db_exists"] is False
    assert report["selected_count"] == 0
    assert any("Corpus DB not found" in error for error in report["errors"])


def test_preflight_only_does_not_build_commands_when_data_is_missing():
    module = _load_script()
    original = module.build_preflight_report
    module.build_preflight_report = lambda _profile: {
        "passed": False,
        "errors": ["missing fixture data"],
        "warnings": [],
    }
    try:
        report = module.run_real_data_curriculum(stage="small", preflight_only=True)
    finally:
        module.build_preflight_report = original

    assert report["passed"] is False
    assert report["preflight_only"] is True
    assert report["command_count"] == 0
    assert report["commands"] == []
