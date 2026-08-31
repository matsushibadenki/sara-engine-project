from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_independent_external_history_replays_across_python_and_rust():
    path = ROOT / "scripts" / "eval" / "phase27_independent_decision_replay.py"
    spec = importlib.util.spec_from_file_location("phase27_independent_decision_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rust = importlib.import_module("sara_engine.sara_rust_core")

    report = module.build_report(module.load_rows(module.DEFAULT_SOURCE), rust)

    assert report["passed"] is True
    assert report["metrics"]["source_row_count"] == 6
    assert report["metrics"]["decision_count"] == 21
    assert report["metrics"]["cache_entry_count"] <= 4
    assert report["checks"]["canonical_bytes_equivalent"] is True
    assert report["checks"]["digest_equivalent"] is True
    assert report["checks"]["controlled_revision_replaced"] is True
    assert report["checks"]["controlled_contradiction_frozen"] is True
    assert report["checks"]["controlled_feedback_oscillation_frozen"] is True
    assert report["checks"]["controlled_digest_equivalent"] is True
    assert report["controlled_perturbation"]["independent_evidence"] is False
    assert report["production_path_changed"] is False
