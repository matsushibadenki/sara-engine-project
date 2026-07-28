"""Deterministic metric drift classification for managed benchmark reports."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

from sara_engine.memory.verification_receipt import evidence_digest


_PROVENANCE_KEYS = {
    "case_count",
    "case_id",
    "dataset",
    "dataset_id",
    "domain_count",
    "domains",
    "eligible_case_count",
    "evidence_scope",
    "fixture",
    "fixture_path",
    "independent_source_scope",
    "manifest",
    "manifest_path",
    "near_duplicate_signature",
    "source_domain",
    "source_domains",
    "source_hash",
    "source_hashes",
    "source_ref",
    "source_refs",
    "source_revision",
    "source_revisions",
    "source_scope",
}


def _numeric_metrics(report: Mapping[str, Any]) -> Dict[str, float]:
    metrics = report.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return {}
    normalized = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            normalized[str(key)] = numeric
    return normalized


def _provenance_surface(value: Any, *, key: str = "") -> Any:
    normalized_key = str(key).lower()
    if normalized_key in _PROVENANCE_KEYS:
        return value
    if isinstance(value, Mapping):
        selected = {
            str(child_key): surface
            for child_key, child_value in value.items()
            if (
                surface := _provenance_surface(
                    child_value,
                    key=str(child_key),
                )
            )
            not in ({}, [], (), None, "")
        }
        if normalized_key == "cases":
            selected["case_ids"] = sorted(str(item) for item in value)
        return selected
    if isinstance(value, (list, tuple)):
        selected = [
            surface
            for item in value
            if (surface := _provenance_surface(item, key=key))
            not in ({}, [], (), None, "")
        ]
        return selected
    return None


def data_fingerprint(report: Mapping[str, Any]) -> str:
    """Hash only declared input identity and provenance fields."""
    surface = _provenance_surface(report)
    return evidence_digest(surface if surface else {"no_provenance": True})


def build_metric_snapshot(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    implementation_fingerprints: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    implementation = dict(implementation_fingerprints or {})
    phases = {
        str(name): {
            "schema": str(report.get("schema", "")),
            "passed": bool(report.get("passed", False)),
            "metrics": _numeric_metrics(report),
            "data_fingerprint": data_fingerprint(report),
            "implementation_fingerprint": str(implementation.get(name, "")),
        }
        for name, report in sorted(reports.items())
    }
    return {
        "schema": "sara-next-level-metric-snapshot-v1",
        "phases": phases,
        "snapshot_digest": evidence_digest(phases),
    }


def _metric_degraded(metric: str, previous: float, current: float) -> bool:
    name = metric.lower()
    lower_is_better = any(
        token in name
        for token in (
            "cost",
            "error",
            "false_",
            "latency",
            "overflow",
            "state_growth",
        )
    )
    return current > previous if lower_is_better else current < previous


def classify_metric_drift(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if not previous or not isinstance(previous.get("phases"), Mapping):
        return {
            "schema": "sara-next-level-metric-drift-v1",
            "classification": "baseline",
            "history_available": False,
            "code_regression_detected": False,
            "data_drift_detected": False,
            "phase_results": {},
        }
    current_phases = current.get("phases", {})
    previous_phases = previous.get("phases", {})
    phase_results: Dict[str, Any] = {}
    classifications = set()
    for phase in sorted(set(current_phases) | set(previous_phases)):
        current_phase = current_phases.get(phase, {})
        previous_phase = previous_phases.get(phase, {})
        current_metrics = current_phase.get("metrics", {})
        previous_metrics = previous_phase.get("metrics", {})
        metric_changes = {}
        degraded_metrics = []
        for metric in sorted(set(current_metrics) | set(previous_metrics)):
            if metric not in current_metrics or metric not in previous_metrics:
                metric_changes[metric] = {
                    "previous": previous_metrics.get(metric),
                    "current": current_metrics.get(metric),
                    "delta": None,
                }
                if metric in previous_metrics and metric not in current_metrics:
                    degraded_metrics.append(metric)
                continue
            old = float(previous_metrics[metric])
            new = float(current_metrics[metric])
            if old == new:
                continue
            metric_changes[metric] = {
                "previous": old,
                "current": new,
                "delta": round(new - old, 12),
            }
            if _metric_degraded(metric, old, new):
                degraded_metrics.append(metric)
        passed_regressed = bool(
            previous_phase.get("passed", False)
            and not current_phase.get("passed", False)
        )
        data_changed = (
            current_phase.get("data_fingerprint")
            != previous_phase.get("data_fingerprint")
        )
        implementation_changed = (
            current_phase.get("implementation_fingerprint")
            != previous_phase.get("implementation_fingerprint")
        )
        changed = bool(metric_changes or passed_regressed)
        degraded = bool(degraded_metrics or passed_regressed)
        classification = _classify_phase(
            changed=changed,
            degraded=degraded,
            data_changed=data_changed,
            implementation_changed=implementation_changed,
        )
        classifications.add(classification)
        phase_results[phase] = {
            "classification": classification,
            "metric_changes": metric_changes,
            "degraded_metrics": degraded_metrics,
            "passed_regressed": passed_regressed,
            "data_fingerprint_changed": data_changed,
            "implementation_fingerprint_changed": implementation_changed,
        }
    overall = _overall_classification(classifications)
    return {
        "schema": "sara-next-level-metric-drift-v1",
        "classification": overall,
        "history_available": True,
        "code_regression_detected": any(
            item["classification"] in {"code_regression", "nondeterministic_regression"}
            for item in phase_results.values()
        ),
        "data_drift_detected": any(
            item["classification"] in {"data_drift", "mixed_drift"}
            for item in phase_results.values()
        ),
        "phase_results": phase_results,
    }


def _classify_phase(
    *,
    changed: bool,
    degraded: bool,
    data_changed: bool,
    implementation_changed: bool,
) -> str:
    if not changed:
        return "stable"
    if data_changed and implementation_changed:
        return "mixed_drift"
    if data_changed:
        return "data_drift"
    if implementation_changed:
        return "code_regression" if degraded else "code_change"
    return "nondeterministic_regression" if degraded else "unexplained_metric_drift"


def _overall_classification(classifications: set[str]) -> str:
    priority: Tuple[str, ...] = (
        "code_regression",
        "nondeterministic_regression",
        "mixed_drift",
        "data_drift",
        "unexplained_metric_drift",
        "code_change",
        "stable",
    )
    return next((item for item in priority if item in classifications), "stable")


__all__ = [
    "build_metric_snapshot",
    "classify_metric_drift",
    "data_fingerprint",
]
