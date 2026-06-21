from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DIRECTION_VALUES = {"up": 1, "flat": 0, "down": -1}
MAGNITUDE_WEIGHTS = {"small": 1, "moderate": 2, "large": 3}


def stable_reason_event_id(value: str, width: int = 4096) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(1, int(width))


@dataclass(frozen=True)
class SparseReasoningPriorResult:
    case_id: str
    predicted_direction: str
    predicted_magnitude: str
    selected_route: str
    confidence: float
    uncertainty: float
    abstained: bool
    abstention_reason: str
    relevant_evidence_count: int
    external_event_present: bool
    logic_to_state_consistent: bool
    event_relevance: float
    event_cost: int
    state_budget_units: int
    sparse_prior_signature: Tuple[int, ...]
    trace: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "predicted_direction": self.predicted_direction,
            "predicted_magnitude": self.predicted_magnitude,
            "selected_route": self.selected_route,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "relevant_evidence_count": self.relevant_evidence_count,
            "external_event_present": self.external_event_present,
            "logic_to_state_consistent": self.logic_to_state_consistent,
            "event_relevance": self.event_relevance,
            "event_cost": self.event_cost,
            "state_budget_units": self.state_budget_units,
            "sparse_prior_signature": list(self.sparse_prior_signature),
            "trace": [dict(row) for row in self.trace],
        }


def _bounded_evidence(case: Mapping[str, Any], max_evidence: int) -> List[Mapping[str, Any]]:
    raw = case.get("evidence", [])
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, Mapping)][: max(1, int(max_evidence))]


def _magnitude_bucket(score: int) -> str:
    absolute = abs(int(score))
    if absolute <= 1:
        return "small"
    if absolute <= 2:
        return "moderate"
    return "large"


def build_sparse_reasoning_prior(
    case: Mapping[str, Any],
    *,
    max_evidence: int = 16,
    signature_width: int = 4096,
) -> SparseReasoningPriorResult:
    case_id = str(case.get("case_id", "unknown") or "unknown")
    target = str(case.get("target", "state") or "state")
    sudden_shift = bool(case.get("sudden_shift", False))
    evidence = _bounded_evidence(case, max_evidence)
    direction_score = 0
    relevant_count = 0
    external_event_present = False
    trace: List[Dict[str, Any]] = []
    signature_terms = [f"case:{case_id}", f"target:{target}"]

    for index, row in enumerate(evidence):
        direction = str(row.get("direction", "flat") or "flat").lower()
        magnitude = str(row.get("magnitude", "small") or "small").lower()
        relevance = max(0.0, min(1.0, float(row.get("relevance", 0.0) or 0.0)))
        source_ref = str(row.get("source_ref", "") or "")
        external = bool(row.get("external_event", False))
        accepted = bool(source_ref) and direction in DIRECTION_VALUES and magnitude in MAGNITUDE_WEIGHTS
        relevant = accepted and relevance >= 0.5
        contribution = 0
        if relevant:
            contribution = DIRECTION_VALUES[direction] * MAGNITUDE_WEIGHTS[magnitude]
            direction_score += contribution
            relevant_count += 1
            external_event_present = external_event_present or external
            signature_terms.extend(
                [
                    f"direction:{direction}",
                    f"magnitude:{magnitude}",
                    f"source:{source_ref}",
                    f"external:{external}",
                ]
            )
        trace.append(
            {
                "evidence_index": index,
                "source_ref": source_ref,
                "accepted": accepted,
                "relevant": relevant,
                "relevance": round(relevance, 6),
                "direction": direction,
                "magnitude": magnitude,
                "external_event": external,
                "contribution": contribution,
            }
        )

    abstention_reason = ""
    if relevant_count == 0:
        abstention_reason = "no_relevant_source_backed_evidence"
    elif sudden_shift and not external_event_present:
        abstention_reason = "external_event_missing"
    abstained = bool(abstention_reason)

    if abstained:
        predicted_direction = "abstain"
        predicted_magnitude = "unknown"
        selected_route = "request_external_context" if sudden_shift else "abstain"
        confidence = 0.0
    else:
        predicted_direction = "up" if direction_score > 0 else "down" if direction_score < 0 else "flat"
        predicted_magnitude = _magnitude_bucket(direction_score)
        selected_route = f"forecast_{predicted_direction}"
        confidence = min(1.0, abs(direction_score) / float(max(1, relevant_count * 3)))

    expected_direction = str(case.get("expected_direction", "") or "")
    expected_magnitude = str(case.get("expected_magnitude", "") or "")
    expected_abstain = bool(case.get("expected_abstain", False))
    logic_consistent = bool(
        (expected_abstain and abstained)
        or (
            not expected_abstain
            and not abstained
            and (not expected_direction or predicted_direction == expected_direction)
            and (not expected_magnitude or predicted_magnitude == expected_magnitude)
        )
    )
    event_relevance = float(relevant_count) / float(max(1, len(evidence)))
    signature = tuple(
        sorted({stable_reason_event_id(term, width=signature_width) for term in signature_terms})
    )
    return SparseReasoningPriorResult(
        case_id=case_id,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        selected_route=selected_route,
        confidence=round(confidence, 6),
        uncertainty=round(1.0 - confidence, 6),
        abstained=abstained,
        abstention_reason=abstention_reason,
        relevant_evidence_count=relevant_count,
        external_event_present=external_event_present,
        logic_to_state_consistent=logic_consistent,
        event_relevance=round(event_relevance, 6),
        event_cost=len(evidence) + relevant_count + len(signature),
        state_budget_units=len(signature) + len(trace),
        sparse_prior_signature=signature,
        trace=tuple(trace),
    )


def evaluate_sparse_reasoning_cases(
    cases: Sequence[Mapping[str, Any]],
) -> List[SparseReasoningPriorResult]:
    return [build_sparse_reasoning_prior(case) for case in cases]
