import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "eval" / "rust_core_readiness.py"
    spec = importlib.util.spec_from_file_location("rust_core_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rust_core_readiness_source_checks_pass_without_running_cargo():
    module = _load_module()
    report = module.build_report(run_cargo_test=False)

    assert report["schema"] == "sara-rust-core-readiness-v1"
    assert report["source_readiness_passed"] is True
    assert report["checks"]["versions_match"] is True
    assert report["checks"]["cargo_feature_split_ready"] is True
    assert report["checks"]["pymodule_exports_registered"] is True
    assert report["checks"]["rust_core_comments_english"] is True
    assert report["checks"]["batch_sdr_parallelized"] is True
    assert "maturin_build_backend_available" in report["checks"]
    assert "benchmark_report_present" in report["checks"]
    assert report["cargo_test"]["status"] == "not_run"


def test_rust_core_readiness_summary_mentions_build_state():
    module = _load_module()
    report = module.build_report(run_cargo_test=False)
    summary = module.summarize_report(report)

    assert "Rust core readiness:" in summary
    assert "Source readiness: True" in summary
    assert "Batch SDR parallelized: True" in summary
    assert "Maturin build backend available:" in summary
    assert "Benchmark report present:" in summary
    assert "Python extension available:" in summary
