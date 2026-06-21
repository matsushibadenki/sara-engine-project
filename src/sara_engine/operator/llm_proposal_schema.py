"""Strict proposal validation for optional local LLM operator assistance."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from sara_engine.utils.project_paths import ensure_allowed_output_path


ALLOWED_PROPOSAL_TYPES = frozenset(
    {
        "roadmap_patch",
        "evaluation_summary",
        "dataset_candidate_review",
        "collector_request",
        "triage_note",
        "operator_next_action",
    }
)

SAFE_ACTION_TYPES = frozenset(
    {
        "draft_patch",
        "summarize",
        "triage",
        "request_collection",
        "review_dataset_candidate",
        "recommend_next_action",
    }
)

DIRECT_MUTATION_ACTION_TYPES = frozenset(
    {
        "write_file",
        "delete_file",
        "modify_file",
        "apply_patch",
        "train_model",
        "modify_model",
        "modify_training_data",
        "run_release_gate",
        "commit_changes",
        "push_changes",
    }
)

SECRET_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

PATH_KEYS = frozenset(
    {
        "output_path",
        "report_path",
        "summary_path",
        "artifact_path",
        "managed_output_path",
    }
)


@dataclass(frozen=True)
class LLMProposalValidation:
    """Validation result for a single operator-assistant proposal."""

    accepted: bool
    rejection_reasons: List[str]
    proposal_id: Optional[str]
    proposal_type: Optional[str]
    source_ref_count: int
    action_count: int
    managed_output_count: int
    schema_version: str = "sara-operator-llm-proposal-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "source_ref_count": self.source_ref_count,
            "action_count": self.action_count,
            "managed_output_count": self.managed_output_count,
        }


def _load_payload(payload: Union[str, Mapping[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    if isinstance(payload, str):
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            return None, ["invalid_json"]
        if not isinstance(loaded, dict):
            return None, ["proposal_not_object"]
        return dict(loaded), []
    if isinstance(payload, Mapping):
        return dict(payload), []
    return None, ["proposal_not_object"]


def _contains_secret_like_text(value: Any) -> bool:
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in SECRET_PATTERNS)
    if isinstance(value, Mapping):
        return any(_contains_secret_like_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_secret_like_text(item) for item in value)
    return False


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _validate_managed_paths(payload: Mapping[str, Any]) -> Tuple[int, List[str]]:
    managed_count = 0
    reasons: List[str] = []

    def visit(value: Any) -> None:
        nonlocal managed_count
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                if key in PATH_KEYS and isinstance(nested_value, str):
                    try:
                        ensure_allowed_output_path(nested_value)
                    except ValueError:
                        reasons.append("unmanaged_output_path")
                    else:
                        managed_count += 1
                visit(nested_value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                visit(item)

    visit(payload)
    return managed_count, sorted(set(reasons))


def _validate_actions(actions: Sequence[Any]) -> List[str]:
    reasons: List[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            reasons.append("action_not_object")
            continue
        action_type = action.get("action_type")
        if not isinstance(action_type, str) or not action_type:
            reasons.append("missing_action_type")
            continue
        if action_type in DIRECT_MUTATION_ACTION_TYPES:
            reasons.append("direct_mutation_action")
        elif action_type not in SAFE_ACTION_TYPES:
            reasons.append("unsupported_action_type")
    return sorted(set(reasons))


def validate_llm_proposal(payload: Union[str, Mapping[str, Any]]) -> LLMProposalValidation:
    """Validate an optional LLM proposal without executing it."""

    loaded, reasons = _load_payload(payload)
    if loaded is None:
        return LLMProposalValidation(
            accepted=False,
            rejection_reasons=reasons,
            proposal_id=None,
            proposal_type=None,
            source_ref_count=0,
            action_count=0,
            managed_output_count=0,
        )

    proposal_id = loaded.get("proposal_id")
    if not isinstance(proposal_id, str) or not proposal_id.strip():
        reasons.append("missing_proposal_id")
        proposal_id = None

    proposal_type = loaded.get("proposal_type")
    if not isinstance(proposal_type, str):
        reasons.append("missing_proposal_type")
        proposal_type = None
    elif proposal_type not in ALLOWED_PROPOSAL_TYPES:
        reasons.append("unsupported_proposal_type")

    source_refs = _as_list(loaded.get("source_refs"))
    if not source_refs:
        reasons.append("missing_source_refs")
    elif not all(isinstance(item, str) and item.strip() for item in source_refs):
        reasons.append("invalid_source_refs")

    actions = _as_list(loaded.get("actions"))
    if not actions:
        reasons.append("missing_actions")
    reasons.extend(_validate_actions(actions))

    managed_output_count, path_reasons = _validate_managed_paths(loaded)
    reasons.extend(path_reasons)

    if _contains_secret_like_text(loaded):
        reasons.append("secret_like_text")

    unique_reasons = sorted(set(reasons))
    return LLMProposalValidation(
        accepted=not unique_reasons,
        rejection_reasons=unique_reasons,
        proposal_id=proposal_id,
        proposal_type=proposal_type if isinstance(proposal_type, str) else None,
        source_ref_count=len(source_refs),
        action_count=len(actions),
        managed_output_count=managed_output_count,
    )
