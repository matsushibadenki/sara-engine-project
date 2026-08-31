import importlib.util
from pathlib import Path

import sara_engine.sara_rust_core as rust_core


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval" / "phase27_observed_feedback_cycle.py"
SPEC = importlib.util.spec_from_file_location("phase27_observed_feedback_cycle", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_observed_feedback_cycle_matches_python_and_rust():
    report = MODULE.build_report(MODULE.load_rows(str(MODULE.DEFAULT_SOURCE)), rust_core)

    assert report["passed"] is True
    assert report["independent_evidence"] is True
    assert report["observed_change_path"] == [1.0, 1.0, 0.0]
    assert report["feedback_action_path"] == [
        "strengthen_relation",
        "request_more_evidence",
        "cut_relation",
    ]
    assert all(report["checks"].values())


def test_feedback_cycle_rejects_an_unobserved_expected_transition():
    rows = [dict(row) for row in MODULE.load_rows(str(MODULE.DEFAULT_SOURCE))]
    rows[-1]["material_hash"] = "different-material"

    report = MODULE.build_report(rows, rust_core)

    assert report["passed"] is False
    assert report["checks"]["observed_change_then_stability"] is False
