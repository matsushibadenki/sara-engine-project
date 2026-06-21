"""Readiness checks for the optional local LLM operator assistant."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Union

from .llm_proposal_schema import LLMProposalValidation, validate_llm_proposal


@dataclass(frozen=True)
class OperatorAssistantReadiness:
    """Aggregate readiness result for proposal validation."""

    passed: bool
    disabled_by_default: bool
    llm_runtime_required: bool
    proposal_count: int
    accepted_count: int
    rejected_count: int
    rejection_counts: Dict[str, int]
    validations: List[LLMProposalValidation]
    schema: str = "sara-operator-llm-assistant-readiness-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "disabled_by_default": self.disabled_by_default,
            "llm_runtime_required": self.llm_runtime_required,
            "proposal_count": self.proposal_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "proposal_acceptance_rate": (
                float(self.accepted_count) / float(self.proposal_count)
                if self.proposal_count
                else 0.0
            ),
            "latency_ms": 0.0,
            "token_budget": {
                "configured_tokens": 0,
                "used_tokens": 0,
                "runtime": "not_required",
            },
            "fallback_behavior": "Reject invalid proposals and continue without an LLM assistant.",
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "validations": [item.to_dict() for item in self.validations],
            "policy_notes": [
                "The local LLM assistant is optional and disabled by default.",
                "Validated proposals are not executed by this readiness check.",
                "Direct data, model, release, or file mutation actions are rejected.",
                "Generated readiness artifacts must stay under workspace/evaluation.",
            ],
        }


def evaluate_operator_proposals(
    proposals: Sequence[Union[str, Mapping[str, Any]]],
    *,
    disabled_by_default: bool = True,
) -> OperatorAssistantReadiness:
    validations = [validate_llm_proposal(item) for item in proposals]
    rejection_counter: Counter[str] = Counter()
    for validation in validations:
        rejection_counter.update(validation.rejection_reasons)

    accepted_count = sum(1 for item in validations if item.accepted)
    rejected_count = len(validations) - accepted_count
    passed = bool(
        disabled_by_default
        and accepted_count > 0
        and rejected_count > 0
        and rejection_counter.get("direct_mutation_action", 0) > 0
        and rejection_counter.get("unmanaged_output_path", 0) > 0
        and rejection_counter.get("secret_like_text", 0) > 0
    )

    return OperatorAssistantReadiness(
        passed=passed,
        disabled_by_default=disabled_by_default,
        llm_runtime_required=False,
        proposal_count=len(validations),
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        rejection_counts=dict(rejection_counter),
        validations=validations,
    )


def build_readiness_report(
    proposals: Sequence[Union[str, Mapping[str, Any]]],
    *,
    disabled_by_default: bool = True,
) -> Dict[str, Any]:
    return evaluate_operator_proposals(
        proposals,
        disabled_by_default=disabled_by_default,
    ).to_dict()


def summarize_readiness_report(report: Mapping[str, Any]) -> str:
    lines = [
        "Operator LLM assistant readiness: {status}".format(
            status="PASS" if report.get("passed") else "FAIL"
        ),
        f"Disabled by default: {report.get('disabled_by_default')}",
        f"LLM runtime required: {report.get('llm_runtime_required')}",
        "Proposals: {accepted}/{total} accepted, {rejected} rejected".format(
            accepted=report.get("accepted_count"),
            total=report.get("proposal_count"),
            rejected=report.get("rejected_count"),
        ),
        f"Acceptance rate: {report.get('proposal_acceptance_rate')}",
        f"Latency ms: {report.get('latency_ms')}",
        "Token budget: configured={configured}, used={used}, runtime={runtime}".format(
            configured=report.get("token_budget", {}).get("configured_tokens"),
            used=report.get("token_budget", {}).get("used_tokens"),
            runtime=report.get("token_budget", {}).get("runtime"),
        ),
        f"Fallback: {report.get('fallback_behavior')}",
        "Rejection reasons:",
    ]
    rejection_counts = report.get("rejection_counts", {})
    if isinstance(rejection_counts, Mapping) and rejection_counts:
        lines.extend(f"- {reason}: {count}" for reason, count in sorted(rejection_counts.items()))
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
