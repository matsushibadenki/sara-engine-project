from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


REQUIRED_REPORTS = (
    "reasoning_prior",
    "plan_verifier",
    "multimodal_binding",
    "dendritic_feedback",
    "own_latent",
    "metabolic_budget",
)


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    value = report.get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def _find_resource_pressure(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        if "metabolic_budget_report" in value:
            report = value.get("metabolic_budget_report", {})
            if isinstance(report, Mapping) and "resource_pressure" in report:
                return _clamp01(report.get("resource_pressure"))
        for child in value.values():
            found = _find_resource_pressure(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_resource_pressure(child)
            if found is not None:
                return found
    return None


@dataclass(frozen=True)
class ResonanceEvidenceBundle:
    signals: Dict[str, Any]
    source_status: Dict[str, Dict[str, Any]]
    event_cost: int
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": dict(self.signals),
            "source_status": {
                key: dict(value) for key, value in sorted(self.source_status.items())
            },
            "event_cost": self.event_cost,
            "trace": dict(self.trace),
        }


def build_resonance_evidence(
    reports: Mapping[str, Mapping[str, Any]],
) -> ResonanceEvidenceBundle:
    source_status: Dict[str, Dict[str, Any]] = {}
    for name in REQUIRED_REPORTS:
        report = reports.get(name, {})
        present = isinstance(report, Mapping) and bool(report)
        schema = str(report.get("schema", "")) if present else ""
        observed_only = bool(report.get("observed_only", False)) if present else False
        passed = bool(report.get("passed", False)) if present else False
        if name == "metabolic_budget" and present:
            report_metrics = _metrics(report)
            passed = bool(
                _clamp01(report_metrics.get("metabolic_budget_integrity")) == 1.0
                and _clamp01(report_metrics.get("plasticity_reserve_integrity")) == 1.0
            )
            observed_only = bool(report.get("observed_only", False))
            schema = str(report.get("schema", "sara-structural-metabolic-budget-v1"))
        source_status[name] = {
            "present": present,
            "schema": schema,
            "observed_only": observed_only,
            "passed": passed,
            "trusted": bool(present and schema and observed_only and passed),
        }

    reasoning = reports.get("reasoning_prior", {})
    reasoning_metrics = _metrics(reasoning)
    own_latent = reports.get("own_latent", {})
    own_latent_metrics = _metrics(own_latent)
    plan = reports.get("plan_verifier", {})
    multimodal_metrics = _metrics(reports.get("multimodal_binding", {}))
    dendritic = reports.get("dendritic_feedback", {})

    prediction_consistency = min(
        _clamp01(reasoning_metrics.get("logic_to_state_consistency")),
        _clamp01(own_latent_metrics.get("own_latent_sample_efficiency_ok")),
    )
    plan_case_count = max(1, int(plan.get("case_count", 0) or 0))
    verifier_confidence = _clamp01(
        float(plan.get("expected_match_count", 0) or 0) / float(plan_case_count)
    )
    cross_modal_agreement = min(
        _clamp01(multimodal_metrics.get("cross_modal_link_precision")),
        _clamp01(multimodal_metrics.get("route_traceability")),
    )
    local_coincidence = _clamp01(dendritic.get("gated_precision", 0.0))
    novelty_signal = _clamp01(
        own_latent_metrics.get("own_latent_sample_efficiency_ok", 0.0)
    )
    trusted_count = sum(1 for status in source_status.values() if status["trusted"])
    reward_signal = float(trusted_count) / float(len(REQUIRED_REPORTS))
    contradiction = max(
        1.0 - verifier_confidence,
        1.0 - _clamp01(reasoning_metrics.get("logic_to_state_consistency")),
        1.0 - _clamp01(multimodal_metrics.get("route_traceability")),
    )
    abstained = bool(
        _clamp01(reasoning_metrics.get("external_event_missing_abstention")) < 1.0
        or _clamp01(multimodal_metrics.get("missing_modality_abstention_integrity")) < 1.0
    )
    resource_pressure = _find_resource_pressure(
        {"metabolic_budget_report": reports.get("metabolic_budget", {})}
    )
    metabolic_headroom = 1.0 - resource_pressure if resource_pressure is not None else 0.0
    source_backed = trusted_count == len(REQUIRED_REPORTS)
    signals = {
        "local_coincidence": round(local_coincidence, 6),
        "prediction_consistency": round(prediction_consistency, 6),
        "verifier_confidence": round(verifier_confidence, 6),
        "cross_modal_agreement": round(cross_modal_agreement, 6),
        "reward_signal": round(reward_signal, 6),
        "novelty_signal": round(novelty_signal, 6),
        "reward_polarity": 1.0,
        "contradiction": round(contradiction, 6),
        "metabolic_headroom": round(max(0.0, metabolic_headroom), 6),
        "abstained": abstained,
        "source_backed": source_backed,
    }
    return ResonanceEvidenceBundle(
        signals=signals,
        source_status=source_status,
        event_cost=len(REQUIRED_REPORTS) + len(signals),
        trace={
            "trusted_source_count": trusted_count,
            "required_source_count": len(REQUIRED_REPORTS),
            "resource_pressure": resource_pressure,
            "derivation": {
                "local_coincidence": "dendritic_feedback.gated_precision",
                "prediction_consistency": "min(reasoning logic consistency, own-latent sample efficiency)",
                "verifier_confidence": "plan expected-match ratio",
                "cross_modal_agreement": "min(cross-modal precision, route traceability)",
                "reward_signal": "trusted report ratio",
                "novelty_signal": "own-latent sample efficiency",
            },
        },
    )
