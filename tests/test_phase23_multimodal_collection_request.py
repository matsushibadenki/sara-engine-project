from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "phase23_multimodal_collection_request.py"
    spec = importlib.util.spec_from_file_location("phase23_multimodal_collection_request", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase23_collection_request_describes_all_blocked_case_families():
    module = _load_module()
    targets = module.build_targets({"promotion_allowed": False})

    assert targets["target_count"] == 4
    assert sum(item["minimum_case_count"] for item in targets["targets"]) == 5
    assert all(
        item["quality_constraints"]["evidence_scope"] == "independent_external"
        for item in targets["targets"]
    )


def test_phase23_collection_request_is_empty_after_gate_passes():
    module = _load_module()
    targets = module.build_targets({"promotion_allowed": True})

    assert targets["target_count"] == 0
    assert targets["targets"] == []
