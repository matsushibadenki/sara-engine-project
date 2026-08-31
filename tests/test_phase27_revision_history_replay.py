import importlib.util
from pathlib import Path

import sara_engine.sara_rust_core as rust_core


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval" / "phase27_revision_history_replay.py"
SPEC = importlib.util.spec_from_file_location("phase27_revision_history_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_independent_revision_history_replays_across_python_and_rust():
    rows = MODULE.load_rows(str(MODULE.DEFAULT_SOURCE))
    report = MODULE.build_report(rows, rust_core)

    assert report["passed"] is True
    assert report["independent_evidence"] is True
    assert report["metrics"] == {"source_revision_count": 2, "decision_count": 1}
    assert all(report["checks"].values())


def test_revision_history_rejects_same_material_as_independent_change():
    rows = [dict(row) for row in MODULE.load_rows(str(MODULE.DEFAULT_SOURCE))]
    rows[1]["material_hash"] = rows[0]["material_hash"]

    report = MODULE.build_report(rows, rust_core)

    assert report["passed"] is False
    assert report["checks"]["distinct_materials"] is False
