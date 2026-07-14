import importlib.util
import json
import os


def _load_module():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase10_completion_gate.py")
    )
    spec = importlib.util.spec_from_file_location("phase10_completion_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase10_gate_accepts_readiness_and_equivalence(tmp_path):
    module = _load_module()
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "source_readiness_passed": True,
                "built_extension_readiness_passed": True,
                "checks": {name: True for name in [
                    "versions_match", "cargo_feature_split_ready", "pymodule_exports_registered",
                    "rust_core_comments_english", "batch_sdr_parallelized", "python_extension_available",
                    "python_exports_complete",
                ]},
                "cargo_test": {"passed": True},
                "cargo_test_test_count": 9,
            }
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "rust_extension_available": True,
                "comparable_case_count": 4,
                "output_equivalence_passed": True,
                "cases": [{"name": name} for name in [
                    "sdr_overlap", "sparse_propagate_threshold", "build_direct_synapses", "batch_tokens_to_sdr",
                ]],
            }
        ),
        encoding="utf-8",
    )
    report = module.build_report(readiness_path=str(readiness), benchmark_path=str(benchmark))
    assert report["phase10_complete"] is True


def test_phase10_gate_rejects_unrun_cargo_test(tmp_path):
    module = _load_module()
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"source_readiness_passed": True, "built_extension_readiness_passed": True, "checks": {}, "cargo_test": {"status": "not_run"}}), encoding="utf-8")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({}), encoding="utf-8")
    report = module.build_report(readiness_path=str(readiness), benchmark_path=str(benchmark))
    assert report["phase10_complete"] is False
