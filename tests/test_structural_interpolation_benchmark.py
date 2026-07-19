import importlib.util
import os


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "structural_interpolation_benchmark.py"))
    spec = importlib.util.spec_from_file_location("structural_interpolation_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structural_interpolation_benchmark_passes_frozen_cases():
    module = _load_module()
    rows = module._load(module.DEFAULT_FIXTURE)
    report = module.build_report(rows)
    assert report["passed"] is True
    assert report["metrics"]["independent_merge_proposal"] == 1.0
    assert report["metrics"]["same_source_duplicate_block"] == 1.0
    assert report["metrics"]["contradiction_block"] == 1.0
    assert report["metrics"]["source_revision_recovery"] == 1.0
    assert report["metrics"]["context_separation"] == 1.0
    assert report["metrics"]["unsupported_neighbor_abstention"] == 1.0
    assert report["metrics"]["oscillation_rollback_freeze"] == 1.0
