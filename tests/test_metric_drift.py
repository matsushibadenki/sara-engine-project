from __future__ import annotations

from sara_engine.evaluation.metric_drift import (
    build_metric_snapshot,
    classify_metric_drift,
)


def _report(*, accuracy=0.9, source_hash="source-a", passed=True):
    return {
        "schema": "benchmark-v1",
        "passed": passed,
        "metrics": {"decision_accuracy": accuracy, "event_cost": 4},
        "cases": {"case-a": {"source_hash": source_hash}},
    }


def test_metric_drift_starts_with_a_baseline():
    current = build_metric_snapshot(
        {"phase": _report()},
        implementation_fingerprints={"phase": "code-a"},
    )

    drift = classify_metric_drift(current, None)

    assert drift["classification"] == "baseline"
    assert drift["history_available"] is False
    assert drift["code_regression_detected"] is False


def test_metric_drift_distinguishes_data_drift_from_code_regression():
    previous = build_metric_snapshot(
        {"phase": _report(accuracy=0.9, source_hash="source-a")},
        implementation_fingerprints={"phase": "code-a"},
    )
    changed_data = build_metric_snapshot(
        {"phase": _report(accuracy=0.8, source_hash="source-b")},
        implementation_fingerprints={"phase": "code-a"},
    )
    changed_code = build_metric_snapshot(
        {"phase": _report(accuracy=0.8, source_hash="source-a")},
        implementation_fingerprints={"phase": "code-b"},
    )

    data_drift = classify_metric_drift(changed_data, previous)
    code_regression = classify_metric_drift(changed_code, previous)

    assert data_drift["classification"] == "data_drift"
    assert data_drift["data_drift_detected"] is True
    assert data_drift["code_regression_detected"] is False
    assert code_regression["classification"] == "code_regression"
    assert code_regression["code_regression_detected"] is True


def test_metric_drift_separates_improving_code_change_and_mixed_drift():
    previous = build_metric_snapshot(
        {"phase": _report(accuracy=0.8, source_hash="source-a")},
        implementation_fingerprints={"phase": "code-a"},
    )
    improved_code = build_metric_snapshot(
        {"phase": _report(accuracy=0.9, source_hash="source-a")},
        implementation_fingerprints={"phase": "code-b"},
    )
    mixed = build_metric_snapshot(
        {"phase": _report(accuracy=0.7, source_hash="source-b")},
        implementation_fingerprints={"phase": "code-b"},
    )

    code_change = classify_metric_drift(improved_code, previous)
    mixed_drift = classify_metric_drift(mixed, previous)

    assert code_change["classification"] == "code_change"
    assert code_change["code_regression_detected"] is False
    assert mixed_drift["classification"] == "mixed_drift"
    assert mixed_drift["data_drift_detected"] is True


def test_metric_drift_flags_unexplained_repeat_degradation():
    previous = build_metric_snapshot(
        {"phase": _report(accuracy=0.9)},
        implementation_fingerprints={"phase": "code-a"},
    )
    current = build_metric_snapshot(
        {"phase": _report(accuracy=0.7)},
        implementation_fingerprints={"phase": "code-a"},
    )

    drift = classify_metric_drift(current, previous)

    assert drift["classification"] == "nondeterministic_regression"
    assert drift["code_regression_detected"] is True


def test_metric_drift_treats_removed_metric_as_regression_and_ignores_non_finite():
    previous = build_metric_snapshot(
        {"phase": _report(accuracy=0.9)},
        implementation_fingerprints={"phase": "code-a"},
    )
    current_report = _report(accuracy=0.9)
    current_report["metrics"] = {"event_cost": 4, "invalid": float("nan")}
    current = build_metric_snapshot(
        {"phase": current_report},
        implementation_fingerprints={"phase": "code-b"},
    )

    drift = classify_metric_drift(current, previous)

    assert "invalid" not in current["phases"]["phase"]["metrics"]
    assert drift["classification"] == "code_regression"
    assert "decision_accuracy" in drift["phase_results"]["phase"][
        "degraded_metrics"
    ]
