import importlib.util
import os


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "next_level_structural_benchmark.py"))
    spec = importlib.util.spec_from_file_location("next_level_structural_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_level_structural_benchmark_passes():
    module = _load_module()
    with open(module.DEFAULT_FIXTURE, "r", encoding="utf-8") as handle:
        rows = [module.json.loads(line) for line in handle if line.strip()]
    report = module.build_report(rows)
    assert report["passed"] is True
    assert all(value == 1.0 for value in report["metrics"].values())
