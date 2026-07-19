from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "continual_horizon_benchmark.py"
    spec = importlib.util.spec_from_file_location("continual_horizon_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_continual_horizon_benchmark_passes_all_frozen_horizons():
    module = _load_module()
    fixture = PROJECT_ROOT / "data" / "processed" / "benchmark_fixtures" / "continual_horizon_cases.jsonl"
    report = module.build_report(module._load(str(fixture)))

    assert report["passed"] is True
    assert report["horizons"] == [10, 30, 100]
    assert report["checks"]["active_beats_frozen_recall"] is True
    assert report["checks"]["independent_source_hashes_unique"] is True
    assert report["checks"]["independent_domains_present"] is True
    assert report["metrics"]["max_state_growth"] <= 8
    assert set(report["metrics"]["retention_profile_recall"]) == {"fixed", "linear", "logarithmic"}
    assert report["checks"]["protected_knowledge_survives"] is True
    assert report["checks"]["delayed_correction_is_measurable"] is True


def test_continual_horizon_benchmark_blocks_contradiction_and_tracks_replay():
    module = _load_module()
    fixture = PROJECT_ROOT / "data" / "processed" / "benchmark_fixtures" / "continual_horizon_cases.jsonl"
    report = module.build_report(module._load(str(fixture)))
    case = report["cases"]["horizon_100"]["profiles"]["event_memory"]

    assert case["contradiction_blocked"] is True
    assert case["revision_uptake_latency"] is not None
    assert case["replay_count"] > 0
    assert case["checks"]["state_budget_bounded"] is True


def test_continual_horizon_benchmark_routes_correction_through_structural_feedback():
    module = _load_module()
    fixture = PROJECT_ROOT / "data" / "processed" / "benchmark_fixtures" / "continual_horizon_cases.jsonl"
    report = module.build_report(module._load(str(fixture)))
    profile = report["cases"]["horizon_10"]["profiles"]["structural_feedback_event_memory"]

    assert profile["feedback_edit_count"] == 1
    assert report["metrics"]["structural_feedback_edit_count"] == 3


def test_continual_horizon_benchmark_applies_distractor_pressure():
    module = _load_module()
    fixture = PROJECT_ROOT / "data" / "processed" / "benchmark_fixtures" / "continual_horizon_cases.jsonl"
    report = module.build_report(module._load(str(fixture)))
    profile = report["cases"]["horizon_100"]["retention_profiles"]["logarithmic"]

    assert profile["eviction_count"] > 0
    assert profile["state_growth"] <= 8


def test_continual_horizon_benchmark_reports_delayed_correction_ablation():
    module = _load_module()
    fixture = PROJECT_ROOT / "data" / "processed" / "benchmark_fixtures" / "continual_horizon_cases.jsonl"
    report = module.build_report(module._load(str(fixture)))
    ablation = report["delayed_correction_ablation"]["horizon_100"]

    assert ablation["latency_delta"] == 62
    assert ablation["delayed_useful_recall"] == 1.0
    assert ablation["immediate_useful_recall"] == 1.0
