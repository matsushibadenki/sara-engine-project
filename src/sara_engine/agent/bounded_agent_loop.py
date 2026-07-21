from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional

from sara_engine.memory.event_state_cache import EventStateCandidate
from sara_engine.memory.verification_receipt import evidence_digest, issue_verification_receipt
from sara_engine.reasoning.sparse_plan_trace import verify_sparse_plan_trace


@dataclass(frozen=True)
class AgentPlanDecision:
    decision: str
    accepted: bool
    goal: str
    structural_prediction: str
    expected_outcome: str
    rollback_action: str
    risk: float
    plan_trace_valid: bool
    durable_mutation_allowed: bool
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "accepted": self.accepted,
            "goal": self.goal,
            "structural_prediction": self.structural_prediction,
            "expected_outcome": self.expected_outcome,
            "rollback_action": self.rollback_action,
            "risk": self.risk,
            "plan_trace_valid": self.plan_trace_valid,
            "durable_mutation_allowed": self.durable_mutation_allowed,
            "trace": dict(self.trace),
        }


class BoundedAgentLoop:
    """Validates structural plans without executing side effects or mutating memory."""

    def __init__(self, *, max_risk: float = 0.75) -> None:
        self.max_risk = max(0.0, min(1.0, float(max_risk)))

    def evaluate_plan(
        self,
        *,
        goal: str,
        structural_prediction: str,
        expected_outcome: str,
        rollback_action: str,
        risk: float,
        plan_case: Mapping[str, Any],
        active_goal: Optional[str] = None,
    ) -> AgentPlanDecision:
        trace_result = verify_sparse_plan_trace(plan_case)
        bounded_risk = max(0.0, min(1.0, float(risk)))
        goal_changed = active_goal is not None and str(active_goal) != str(goal)
        errors = []
        if not structural_prediction:
            errors.append("missing_structural_prediction")
        if not expected_outcome:
            errors.append("missing_expected_outcome")
        if not rollback_action:
            errors.append("missing_rollback_action")
        if bounded_risk > self.max_risk:
            errors.append("risk_above_policy_limit")
        if goal_changed:
            errors.append("stale_goal")
        if not trace_result.valid:
            errors.append("invalid_plan_trace")
        accepted = not errors
        return AgentPlanDecision(
            decision="propose_observed_action" if accepted else "reject_or_gather_information",
            accepted=accepted,
            goal=str(goal),
            structural_prediction=str(structural_prediction),
            expected_outcome=str(expected_outcome),
            rollback_action=str(rollback_action),
            risk=bounded_risk,
            plan_trace_valid=trace_result.valid,
            durable_mutation_allowed=False,
            trace={
                "errors": errors,
                "plan_event_cost": trace_result.event_cost,
                "plan_state_budget_units": trace_result.state_budget_units,
                "goal_changed": goal_changed,
                "side_effects_executed": False,
            },
        )

    def verify_outcome(
        self,
        decision: AgentPlanDecision,
        *,
        observed_outcome: str,
        observation_verified: bool = False,
        observation_evidence: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        evidence_payload = dict(observation_evidence or {})
        observation_digest = evidence_digest(evidence_payload) if evidence_payload else ""
        matched = bool(
            decision.accepted
            and observation_verified
            and observation_digest
            and observed_outcome == decision.expected_outcome
        )
        return {
            "observed": bool(observation_verified and observation_digest),
            "outcome": str(observed_outcome),
            "expected_outcome": decision.expected_outcome,
            "outcome_matches": matched,
            "rollback_required": bool(decision.accepted and not matched),
            "rollback_action": decision.rollback_action if decision.accepted and not matched else "",
            "event_memory_candidate_allowed": matched,
            "observation_digest": observation_digest,
            "durable_mutation_allowed": False,
        }

    def outcome_event_state_candidate(
        self,
        decision: AgentPlanDecision,
        *,
        observed_outcome: str,
        source_ref: str,
        observation_verified: bool = False,
        observation_evidence: Optional[Mapping[str, Any]] = None,
        time_segment: int = 0,
    ) -> Optional[EventStateCandidate]:
        """Create a verified observed candidate only after outcome confirmation."""
        evidence_payload = dict(observation_evidence or {})
        observation_digest = evidence_digest(evidence_payload) if evidence_payload else ""
        if (
            not decision.accepted
            or observed_outcome != decision.expected_outcome
            or not source_ref
            or not observation_verified
            or not observation_digest
        ):
            return None
        source_id = int.from_bytes(sha256(source_ref.encode("utf-8")).digest()[:2], "big") % 4096
        outcome_id = int.from_bytes(sha256(observed_outcome.encode("utf-8")).digest()[:2], "big") % 4096
        source_revision = observation_digest
        receipt = issue_verification_receipt(
            verifier_id="bounded-agent-outcome-verifier",
            verifier_version="v2",
            decision="verified_expected_outcome",
            evidence={
                "observation": evidence_payload,
                "observed_outcome": observed_outcome,
                "expected_outcome": decision.expected_outcome,
                "plan_trace": decision.trace,
            },
            source_refs=(source_ref,),
            source_revision=source_revision,
            observed=True,
            source_backed=True,
            verified=True,
        )
        return EventStateCandidate(
            entry_id=f"outcome:{source_ref}:{observed_outcome}",
            signature=tuple(sorted({source_id, outcome_id})),
            source_ref=str(source_ref),
            source_revision=source_revision,
            time_segment=int(time_segment),
            own_latent_id=f"outcome:{observed_outcome}",
            causal_predecessors=(decision.structural_prediction,),
            confidence=1.0,
            uncertainty=0.0,
            source_reliability=1.0,
            resonance_score=0.9,
            sequence_support_score=1.0,
            sequence_support_count=1,
            credit_score=0.9,
            credit_responsibility=0.9,
            credit_confidence=0.9,
            credit_longevity=0.9,
            metabolic_headroom=1.0,
            observed=True,
            source_backed=True,
            verified=True,
            event_cost=4,
            verification_receipt=receipt,
        )
