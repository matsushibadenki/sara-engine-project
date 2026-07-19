import importlib.util
import os


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "structural_interpolation_event_memory_benchmark.py"))
    spec = importlib.util.spec_from_file_location("structural_interpolation_event_memory_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structural_interpolation_event_memory_boundary_passes():
    module = _load_module()
    report = module.build_report(module._load(module.DEFAULT_MANIFEST))
    assert report["passed"] is True
    assert report["metrics"]["retrieval_recall"] == 1.0
    assert report["metrics"]["contradiction_block_count"] == 2
