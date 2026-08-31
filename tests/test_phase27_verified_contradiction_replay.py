import importlib.util
from pathlib import Path

import sara_engine.sara_rust_core as rust_core


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval" / "phase27_verified_contradiction_replay.py"
SPEC = importlib.util.spec_from_file_location("phase27_verified_contradiction_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_verified_erratum_contradiction_freezes_across_python_and_rust():
    report = MODULE.build_report(MODULE.load_rows(str(MODULE.DEFAULT_SOURCE)), rust_core)

    assert report["passed"] is True
    assert report["independent_evidence"] is True
    assert report["metrics"] == {"claim_count": 2, "decision_count": 1}
    assert all(report["checks"].values())


def test_same_polarity_cannot_satisfy_contradiction_evidence():
    rows = [dict(row) for row in MODULE.load_rows(str(MODULE.DEFAULT_SOURCE))]
    rows[1]["polarity"] = "supports"
    rows[1]["claim_value"] = True

    report = MODULE.build_report(rows, rust_core)

    assert report["passed"] is False
    assert report["checks"]["opposite_polarities"] is False
