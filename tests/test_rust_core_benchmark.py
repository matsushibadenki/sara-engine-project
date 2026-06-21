import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "eval" / "rust_core_benchmark.py"
    spec = importlib.util.spec_from_file_location("rust_core_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_python_reference_sparse_primitives_are_deterministic():
    module = _load_module()

    assert module.python_sdr_overlap([1, 2, 2, 3], [2, 3, 4]) == 2 / 3
    assert module.python_sparse_propagate_threshold(
        [0], [{1: 0.4, 2: 0.9}], 4, 0.5
    ) == [2]
    assert module.python_batch_tokens_to_sdr([[7]], 64, 0.1, 123) == module.python_batch_tokens_to_sdr(
        [[7]], 64, 0.1, 123
    )


def test_python_reference_direct_synapses_match_expected_delays():
    module = _load_module()
    synapses = module.python_build_direct_synapses([1, 2, 1, 3], 2)

    assert 2 in synapses[1][1]
    assert 3 in synapses[2][2]


def test_benchmark_report_runs_without_built_rust_extension():
    module = _load_module()
    report = module.build_report(iterations=2)

    assert report["schema"] == "sara-rust-core-benchmark-v1"
    assert report["case_count"] == 4
    assert len(report["cases"]) == 4
    assert all(case["python_seconds"] >= 0.0 for case in report["cases"])
    summary = module.summarize_report(report)
    assert "Rust core benchmark:" in summary
