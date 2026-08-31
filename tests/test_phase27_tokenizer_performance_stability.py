import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval" / "phase27_tokenizer_performance_stability.py"
SPEC = importlib.util.spec_from_file_location("phase27_tokenizer_performance_stability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _report(speedup: float):
    return {
        "schema": MODULE.EXPECTED_SCHEMA,
        "passed": True,
        "observed_only": True,
        "production_path_changed": False,
        "rust_build_profile": "release",
        "rust_snapshot_reference_equivalent": True,
        "tokenizer_fingerprint": "frozen-tokenizer",
        "checks": {
            "large_trace_snapshot_equivalent": True,
            "rust_snapshot_downstream_replay_equivalent": True,
            "snapshot_state_bounded": True,
            "peak_rss_growth_bounded": True,
        },
        "resource_measurement": {
            "median_trace_count": 300,
            "median_repetitions": 7,
            "rust_snapshot_median_speedup_vs_python": speedup,
            "python_median_elapsed_ns": 100,
            "rust_snapshot_median_elapsed_ns": 90,
            "peak_rss_delta_bytes": 1024,
        },
    }


def test_stability_gate_requires_every_fresh_trial_to_pass_threshold():
    report = MODULE.aggregate_reports(
        [_report(value) for value in (1.10, 1.09, 1.08, 1.07, 1.06)],
        fixture_digest="fixture",
    )
    assert report["passed"] is True
    assert report["promotion_ready"] is True
    assert report["metrics"]["worst_speedup"] == 1.06


def test_stability_gate_retains_and_rejects_one_slow_trial():
    report = MODULE.aggregate_reports(
        [_report(value) for value in (1.10, 1.09, 1.08, 1.07, 1.04)],
        fixture_digest="fixture",
    )
    assert report["passed"] is True
    assert report["promotion_ready"] is False
    assert report["checks"]["every_trial_above_threshold"] is False
    assert len(report["trials"]) == 5


def test_stability_gate_fails_integrity_for_mixed_fingerprint():
    reports = [_report(1.10) for _ in range(5)]
    reports[-1]["tokenizer_fingerprint"] = "changed"
    report = MODULE.aggregate_reports(reports, fixture_digest="fixture")
    assert report["passed"] is False
    assert report["promotion_ready"] is False
