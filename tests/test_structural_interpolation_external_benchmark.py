import importlib.util
import os


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "structural_interpolation_external_benchmark.py"))
    spec = importlib.util.spec_from_file_location("structural_interpolation_external_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structural_interpolation_external_manifest_passes():
    module = _load_module()
    report = module.build_report(module._load(module.DEFAULT_MANIFEST))
    assert report["passed"] is True
    assert report["metrics"]["record_count"] == 202
    assert report["metrics"]["source_domain_count"] == 2
    assert report["metrics"]["proposal_count"] == 2
    assert report["metrics"]["accepted_evidence_count"] == 16
    assert report["metrics"]["rejected_evidence_count"] == 186
    assert all(report["checks"].values())
