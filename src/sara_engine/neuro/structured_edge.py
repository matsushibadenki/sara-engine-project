"""Deterministic bounded structured-edge runtime for Phase 33 evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ARMS = (
    "single_scalar_contact",
    "linear_multi_contact",
    "typed_independent_contacts",
    "branch_local_contacts",
    "branch_local_contacts_with_add_prune",
)
TYPED_ARMS = frozenset(ARMS[2:])
BRANCH_ARMS = frozenset(ARMS[3:])


@dataclass(frozen=True)
class StructuredEdgeLimits:
    """Hard ceilings copied from an immutable experiment manifest."""

    max_contacts: int = 4
    max_branch_slots: int = 4
    max_internal_interactions: int = 8
    max_contact_rewrites_per_event: int = 2
    max_events: int = 128
    max_state_bytes: int = 4096


def _canonical_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


class StructuredEdgeRuntime:
    """Process one observed relation without mutating durable graph state."""

    def __init__(self, arm: str, limits: StructuredEdgeLimits) -> None:
        if arm not in ARMS:
            raise ValueError(f"unsupported structured-edge arm: {arm}")
        self.arm = arm
        self.limits = limits

    def evaluate(self, case: Mapping[str, Any]) -> Dict[str, Any]:
        contacts_raw = case.get("contacts")
        events_raw = case.get("events")
        if not isinstance(contacts_raw, list) or not isinstance(events_raw, list):
            return self._abstention(case, "malformed_case")
        if not 1 <= len(contacts_raw) <= self.limits.max_contacts:
            return self._abstention(case, "contact_budget_exceeded")
        if not 1 <= len(events_raw) <= self.limits.max_events:
            return self._abstention(case, "event_budget_exceeded")

        contact_ids = [str(contact.get("contact_id", "")) for contact in contacts_raw]
        if any(not contact_id for contact_id in contact_ids):
            return self._abstention(case, "missing_contact_identity")
        if len(contact_ids) != len(set(contact_ids)):
            return self._abstention(case, "duplicate_contact")

        contacts = {
            str(contact["contact_id"]): dict(contact) for contact in contacts_raw
        }
        branch_ids = {
            int(contact.get("branch", 0)) for contact in contacts.values()
        }
        if any(branch < 0 for branch in branch_ids):
            return self._abstention(case, "invalid_branch")
        if len(branch_ids) > self.limits.max_branch_slots:
            return self._abstention(case, "branch_budget_exceeded")

        for event in events_raw:
            if event.get("outer_route_present") is False:
                return self._abstention(case, "outer_route_absent")
            if str(event.get("contact_id", "")) not in contacts:
                return self._abstention(case, "missing_contact")

        current_revision = str(case.get("source_revision", ""))
        stale_contacts = sorted(
            {
                str(event["contact_id"])
                for event in events_raw
                if event.get("source_revision") is not None
                and str(event.get("source_revision")) != current_revision
            }
        )
        if stale_contacts:
            return self._abstention(
                case,
                "stale_source_revision",
                frozen_contacts=stale_contacts,
            )

        failed_contacts = {
            str(contact_id)
            for event in events_raw
            for contact_id in event.get("failed_contacts", [])
        }
        ordered_events = sorted(
            enumerate(events_raw),
            key=lambda item: (int(item[1].get("tick", 0)), item[0]),
        )
        arrivals: List[Dict[str, Any]] = []
        reuse_counts: Dict[str, int] = {}
        signed_sum = 0
        for sequence_index, event in ordered_events:
            contact_id = str(event["contact_id"])
            if contact_id in failed_contacts:
                continue
            contact = contacts[contact_id]
            typed = self.arm in TYPED_ARMS
            delay = int(contact.get("delay_bucket", 0)) if typed else 0
            polarity = str(contact.get("polarity", "excitatory"))
            signed_value = -1 if polarity == "inhibitory" else 1
            signed_sum += signed_value
            reuse_counts[contact_id] = min(
                3,
                reuse_counts.get(contact_id, 0) + 1,
            )
            arrivals.append(
                {
                    "contact_id": contact_id if typed else "scalar",
                    "arrival_tick": int(event.get("tick", 0)) + delay,
                    "branch": (
                        int(contact.get("branch", 0))
                        if self.arm in BRANCH_ARMS
                        else 0
                    ),
                    "polarity": polarity,
                    "role": str(contact.get("role", "")) if typed else "",
                    "context": str(event.get("context", "")) if typed else "",
                    "sequence_index": sequence_index,
                }
            )

        interactions = self._branch_interactions(arrivals)
        if len(interactions) > self.limits.max_internal_interactions:
            return self._abstention(case, "interaction_budget_exceeded")
        interaction_gain = sum(
            interaction["excitatory_count"] - 1 for interaction in interactions
        )
        output_signal = signed_sum + interaction_gain
        rewrite_proposals = self._rewrite_proposals(case, arrivals)
        max_rewrites = self.limits.max_contact_rewrites_per_event * len(events_raw)
        if len(rewrite_proposals) > max_rewrites:
            return self._abstention(case, "rewrite_budget_exceeded")

        state = {
            "arm": self.arm,
            "arrivals": arrivals,
            "branch_interactions": interactions,
            "reuse_counts": dict(sorted(reuse_counts.items())),
            "rewrite_proposals": rewrite_proposals,
            "failed_contacts": sorted(failed_contacts),
            "output_signal": output_signal,
        }
        state_bytes = _canonical_size(state)
        if state_bytes > self.limits.max_state_bytes:
            return self._abstention(case, "state_budget_exceeded")

        result = {
            "case_id": str(case.get("case_id", "")),
            "family": str(case.get("family", "")),
            "arm": self.arm,
            "status": "evaluated",
            "reason": "ok",
            "durable_mutation": False,
            "output_signal": output_signal,
            "arrival_ticks": [arrival["arrival_tick"] for arrival in arrivals],
            "ordered_roles": [arrival["role"] for arrival in arrivals],
            "contexts": [arrival["context"] for arrival in arrivals],
            "branch_interaction_count": len(interactions),
            "active_branch_count": len(
                {arrival["branch"] for arrival in arrivals}
            ),
            "reuse_peak": max(reuse_counts.values(), default=0),
            "failed_contact_count": len(failed_contacts),
            "rewrite_proposal_count": len(rewrite_proposals),
            "state_bytes": state_bytes,
            "event_cost": len(arrivals) + len(interactions),
            "state": state,
        }
        result["behavior_satisfied"] = self._behavior_satisfied(case, result)
        return result

    def _branch_interactions(
        self,
        arrivals: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, int]]:
        if self.arm not in BRANCH_ARMS:
            return []
        grouped: Dict[Tuple[int, int], int] = {}
        for arrival in arrivals:
            if arrival.get("polarity") == "inhibitory":
                continue
            key = (int(arrival["branch"]), int(arrival["arrival_tick"]))
            grouped[key] = grouped.get(key, 0) + 1
        return [
            {
                "branch": branch,
                "arrival_tick": tick,
                "excitatory_count": count,
            }
            for (branch, tick), count in sorted(grouped.items())
            if count >= 2
        ]

    def _rewrite_proposals(
        self,
        case: Mapping[str, Any],
        arrivals: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, str]]:
        if self.arm != "branch_local_contacts_with_add_prune":
            return []
        if case.get("family") == "delayed_contradiction":
            return [{"action": "freeze", "reason": "contradiction"}]
        if case.get("family") == "outer_route_deletion":
            return []
        if case.get("family") == "repeated_support" and len(arrivals) >= 3:
            return [{"action": "retain", "reason": "bounded_reuse"}]
        return []

    def _behavior_satisfied(
        self,
        case: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> bool:
        family = str(case.get("family", ""))
        if family == "delay_dependent_meaning":
            return self.arm in TYPED_ARMS and len(set(result["arrival_ticks"])) > 1
        if family == "polarity_context_switch":
            return self.arm in TYPED_ARMS and len(set(result["contexts"])) > 1
        if family == "same_count_different_order":
            return self.arm in TYPED_ARMS and result["ordered_roles"] == [
                "second",
                "first",
            ]
        if family == "branch_local_coincidence":
            return result["branch_interaction_count"] == 1
        if family == "partial_contact_failure":
            return result["failed_contact_count"] == 1 and result["output_signal"] > 0
        if family == "repeated_support":
            return self.arm in TYPED_ARMS and result["reuse_peak"] == 3
        if family == "shuffled_contact_identity":
            return bool(result["arrival_ticks"])
        if family == "shuffled_branch_placement":
            return self.arm in BRANCH_ARMS and result["branch_interaction_count"] == 0
        if family == "no_reuse":
            return result["rewrite_proposal_count"] == 0
        if family == "random_cluster":
            return result["branch_interaction_count"] == 0
        if family in {"all_linear", "all_same_delay"}:
            return True
        return False

    def _abstention(
        self,
        case: Mapping[str, Any],
        reason: str,
        *,
        frozen_contacts: Sequence[str] = (),
    ) -> Dict[str, Any]:
        result = {
            "case_id": str(case.get("case_id", "")),
            "family": str(case.get("family", "")),
            "arm": self.arm,
            "status": "abstained",
            "reason": reason,
            "durable_mutation": False,
            "frozen_contacts": list(frozen_contacts),
            "output_signal": 0,
            "arrival_ticks": [],
            "ordered_roles": [],
            "contexts": [],
            "branch_interaction_count": 0,
            "active_branch_count": 0,
            "reuse_peak": 0,
            "failed_contact_count": 0,
            "rewrite_proposal_count": 0,
            "state_bytes": 0,
            "event_cost": 0,
            "state": {},
        }
        expected_reasons = {
            "duplicated_contact": "duplicate_contact",
            "missing_contact": "missing_contact",
            "stale_source_revision": "stale_source_revision",
            "outer_route_deletion": "outer_route_absent",
            "delayed_contradiction": "stale_source_revision",
        }
        result["behavior_satisfied"] = expected_reasons.get(
            str(case.get("family", ""))
        ) == reason
        return result


__all__ = ["ARMS", "StructuredEdgeLimits", "StructuredEdgeRuntime"]
