from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional, Sequence

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


@dataclass(frozen=True)
class ActionSelectionArm:
    selected_action: str
    score: float
    abstained: bool
    reason: str
    charged_event_budget: int
    scanned_candidate_count: int
    scanned_feedback_count: int
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "score": self.score,
            "abstained": self.abstained,
            "reason": self.reason,
            "charged_event_budget": self.charged_event_budget,
            "scanned_candidate_count": self.scanned_candidate_count,
            "scanned_feedback_count": self.scanned_feedback_count,
            "trace": dict(self.trace),
        }


@dataclass(frozen=True)
class ActionSelectionAblation:
    control: ActionSelectionArm
    structural_feedback: ActionSelectionArm
    equal_event_budget: bool
    event_budget_per_arm: int
    event_envelope_cost: int
    state_budget_units: int
    abstained: bool
    reason: str
    durable_mutation_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-action-selection-ablation-v1",
            "control": self.control.to_dict(),
            "structural_feedback": self.structural_feedback.to_dict(),
            "equal_event_budget": self.equal_event_budget,
            "event_budget_per_arm": self.event_budget_per_arm,
            "event_envelope_cost": self.event_envelope_cost,
            "state_budget_units": self.state_budget_units,
            "abstained": self.abstained,
            "reason": self.reason,
            "durable_mutation_allowed": False,
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

    def compare_action_selection(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        structural_feedback: Sequence[Mapping[str, Any]],
        event_budget_per_arm: int,
        max_state_budget_units: int = 32,
    ) -> ActionSelectionAblation:
        """Compare masked and structural scoring over one equal-cost event envelope."""
        candidate_rows = tuple(candidates)
        feedback_rows = tuple(structural_feedback)
        event_budget = max(1, int(event_budget_per_arm))
        state_limit = max(1, int(max_state_budget_units))
        envelope_cost = sum(
            max(1, int(item.get("event_cost", 1) or 1))
            for item in (*candidate_rows, *feedback_rows)
        )
        state_units = len(candidate_rows) + len(feedback_rows)
        malformed = any(
            not all(
                str(item.get(key, "")).strip()
                for key in (
                    "action",
                    "concept",
                    "evidence_ref",
                    "structural_prediction",
                    "expected_outcome",
                )
            )
            for item in candidate_rows
        )
        if (
            not candidate_rows
            or malformed
            or envelope_cost > event_budget
            or state_units > state_limit
        ):
            reason = (
                "no_action_candidates"
                if not candidate_rows
                else "malformed_action_candidate"
                if malformed
                else "action_selection_event_budget_exceeded"
                if envelope_cost > event_budget
                else "action_selection_state_budget_exceeded"
            )
            empty = self._abstained_selection_arm(
                reason=reason,
                charged_event_budget=min(envelope_cost, event_budget),
                candidate_count=len(candidate_rows),
                feedback_count=len(feedback_rows),
            )
            return ActionSelectionAblation(
                control=empty,
                structural_feedback=empty,
                equal_event_budget=True,
                event_budget_per_arm=event_budget,
                event_envelope_cost=envelope_cost,
                state_budget_units=state_units,
                abstained=True,
                reason=reason,
            )

        eligible = tuple(
            item
            for item in candidate_rows
            if max(0.0, min(1.0, float(item.get("risk", 1.0)))) <= self.max_risk
        )
        if not eligible:
            empty = self._abstained_selection_arm(
                reason="no_policy_eligible_action",
                charged_event_budget=envelope_cost,
                candidate_count=len(candidate_rows),
                feedback_count=len(feedback_rows),
            )
            return ActionSelectionAblation(
                control=empty,
                structural_feedback=empty,
                equal_event_budget=True,
                event_budget_per_arm=event_budget,
                event_envelope_cost=envelope_cost,
                state_budget_units=state_units,
                abstained=True,
                reason="no_policy_eligible_action",
            )

        feedback_by_action: Dict[str, list[Mapping[str, Any]]] = {}
        for item in feedback_rows:
            action = str(item.get("action", "")).strip()
            if action:
                feedback_by_action.setdefault(action, []).append(item)

        control = self._select_action_arm(
            eligible,
            feedback_by_action=feedback_by_action,
            use_structural_feedback=False,
            charged_event_budget=envelope_cost,
            scanned_feedback_count=len(feedback_rows),
        )
        structural = self._select_action_arm(
            eligible,
            feedback_by_action=feedback_by_action,
            use_structural_feedback=True,
            charged_event_budget=envelope_cost,
            scanned_feedback_count=len(feedback_rows),
        )
        return ActionSelectionAblation(
            control=control,
            structural_feedback=structural,
            equal_event_budget=(
                control.charged_event_budget
                == structural.charged_event_budget
                == envelope_cost
            ),
            event_budget_per_arm=event_budget,
            event_envelope_cost=envelope_cost,
            state_budget_units=state_units,
            abstained=False,
            reason="equal_budget_action_selection_compared",
        )

    def _select_action_arm(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        feedback_by_action: Mapping[str, Sequence[Mapping[str, Any]]],
        use_structural_feedback: bool,
        charged_event_budget: int,
        scanned_feedback_count: int,
    ) -> ActionSelectionArm:
        scored = []
        for item in candidates:
            action = str(item.get("action", ""))
            base_score = max(0.0, min(1.0, float(item.get("base_score", 0.0))))
            adjustment = 0.0
            used_feedback_refs = []
            if use_structural_feedback:
                for feedback in feedback_by_action.get(action, ()):
                    if not (
                        bool(feedback.get("verified", False))
                        and bool(feedback.get("feedback_stable", True))
                        and str(feedback.get("source_ref", "")).strip()
                    ):
                        continue
                    confidence = max(
                        0.0, min(1.0, float(feedback.get("confidence", 0.0)))
                    )
                    direction = -1.0 if bool(feedback.get("contradicted", False)) else 1.0
                    adjustment += direction * 0.25 * confidence
                    used_feedback_refs.append(str(feedback.get("source_ref", "")))
            score = max(0.0, min(1.0, base_score + adjustment))
            scored.append(
                {
                    "action": action,
                    "score": score,
                    "risk": max(0.0, min(1.0, float(item.get("risk", 1.0)))),
                    "concept": str(item.get("concept", "")),
                    "evidence_ref": str(item.get("evidence_ref", "")),
                    "structural_prediction": str(item.get("structural_prediction", "")),
                    "expected_outcome": str(item.get("expected_outcome", "")),
                    "feedback_refs": sorted(set(used_feedback_refs)),
                }
            )
        winner = sorted(
            scored,
            key=lambda item: (-item["score"], item["risk"], item["action"]),
        )[0]
        return ActionSelectionArm(
            selected_action=str(winner["action"]),
            score=round(float(winner["score"]), 6),
            abstained=False,
            reason=(
                "structural_feedback_score"
                if use_structural_feedback
                else "masked_structural_control_score"
            ),
            charged_event_budget=charged_event_budget,
            scanned_candidate_count=len(candidates),
            scanned_feedback_count=scanned_feedback_count,
            trace={
                "concept": winner["concept"],
                "evidence_ref": winner["evidence_ref"],
                "structural_prediction": winner["structural_prediction"],
                "expected_outcome": winner["expected_outcome"],
                "feedback_refs": winner["feedback_refs"],
                "structural_feedback_enabled": use_structural_feedback,
                "side_effects_executed": False,
                "scores": scored,
            },
        )

    @staticmethod
    def _abstained_selection_arm(
        *,
        reason: str,
        charged_event_budget: int,
        candidate_count: int,
        feedback_count: int,
    ) -> ActionSelectionArm:
        return ActionSelectionArm(
            selected_action="",
            score=0.0,
            abstained=True,
            reason=reason,
            charged_event_budget=charged_event_budget,
            scanned_candidate_count=candidate_count,
            scanned_feedback_count=feedback_count,
            trace={"side_effects_executed": False},
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
