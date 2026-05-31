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


def test_build_operational_command_for_extended_enables_strict_and_accuracy():
    module = _load_script("phase4_operational_cycle.py")
    command = module._build_operational_command(
        "extended",
        include_accuracy=False,
        runbook_action_limit=3,
        runbook_max_actions=25,
        runbook_max_per_source=1,
        runbook_drop_rate_threshold=0.8,
        v1_actions_max_age_seconds=7200.0,
    )
    text = " ".join(command)
    assert "--soak-profile extended" in text
    assert "--strict-production" in text
    assert "--include-accuracy" in text
    assert "--append-runbook-actions-max 3" in text
    assert "--runbook-max-actions 25" in text
    assert "--runbook-max-per-source 1" in text
    assert "--runbook-drop-rate-threshold 0.800" in text
    assert "--v1-actions-max-age-seconds 7200.0" in text


def test_run_phase4_operational_cycle_dry_run_has_release_and_extended():
    module = _load_script("phase4_operational_cycle.py")
    report = module.run_phase4_operational_cycle(
        profiles=["release", "extended"],
        include_accuracy=False,
        runbook_action_limit=2,
        runbook_max_actions=25,
        runbook_max_per_source=1,
        runbook_drop_rate_threshold=0.8,
        v1_actions_max_age_seconds=7200.0,
        dry_run=True,
    )
    assert report["passed"] is True
    assert report["dry_run"] is True
    assert len(report["runs"]) == 2
    assert report["runs"][0]["profile"] == "release"
    assert report["runs"][1]["profile"] == "extended"
    assert "--strict-production" in report["runs"][1]["command"]
    assert "--runbook-max-actions 25" in report["runs"][0]["command"]
    assert "--runbook-max-per-source 1" in report["runs"][0]["command"]
    assert "--runbook-drop-rate-threshold 0.800" in report["runs"][0]["command"]
    assert "--v1-actions-max-age-seconds 7200.0" in report["runs"][0]["command"]


def test_summarize_cycle_contains_cycle_rows():
    module = _load_script("phase4_operational_cycle.py")
    summary = module._summarize_cycle(
        {
            "passed": True,
            "dry_run": True,
            "runs": [
                {"profile": "release", "passed": True, "duration_seconds": 1.0},
                {"profile": "extended", "passed": False, "duration_seconds": 2.0},
            ],
        }
    )
    assert "Phase4 Operational Cycle Summary" in summary
    assert "profile=release status=PASS" in summary
    assert "profile=extended status=FAIL" in summary
