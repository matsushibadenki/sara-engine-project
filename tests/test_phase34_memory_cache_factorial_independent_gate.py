from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase34_memory_cache_factorial_independent_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase34_memory_cache_factorial_independent_gate", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(name: str):
    return json.loads((PROJECT_ROOT / "workspace" / "evaluation" / name).read_text())


def test_current_external_coverage_allows_independent_factorial_execution():
    module = _load_script()
    report = module.build_report(
        _read("phase34_memory_cache_factorial_preregistration.json"),
        _read("phase34_memory_cache_factorial_benchmark.json"),
        _read("continual_horizon_external_gate.json"),
    )

    assert report["independent_execution_ready"] is True
    assert report["promotion_ready"] is False
    assert report["selector_retuning_allowed"] is False
    assert report["query_aware_retention_allowed"] is False
    assert report["metrics"]["source_domain_count"] == 2
    assert report["metrics"]["missing_horizon_target_count"] == 0
    assert report["blockers"] == []
    assert report["missing_collection_targets"] == []


def test_complete_external_coverage_allows_execution_but_not_promotion():
    module = _load_script()
    external = copy.deepcopy(_read("continual_horizon_external_gate.json"))
    external["promotion_allowed"] = True
    external["domain_horizons"] = {
        "docs.python.org": [0, 10, 30, 100],
        "www.ietf.org": [0, 10, 30, 100],
    }
    report = module.build_report(
        _read("phase34_memory_cache_factorial_preregistration.json"),
        _read("phase34_memory_cache_factorial_benchmark.json"),
        external,
    )

    assert report["independent_execution_ready"] is True
    assert report["promotion_ready"] is False
    assert report["missing_collection_targets"] == []
    assert report["blockers"] == []


def test_protocol_mismatch_blocks_independent_execution():
    module = _load_script()
    factorial = copy.deepcopy(_read("phase34_memory_cache_factorial_benchmark.json"))
    factorial["protocol_fingerprint"] = "0" * 64
    report = module.build_report(
        _read("phase34_memory_cache_factorial_preregistration.json"),
        factorial,
        _read("continual_horizon_external_gate.json"),
    )

    assert report["independent_execution_ready"] is False
    assert report["checks"]["factorial_protocol_matches"] is False
    assert "factorial_protocol_matches" in report["blockers"]
