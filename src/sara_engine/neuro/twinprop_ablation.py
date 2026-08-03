"""Bounded sparse runtime for the registered TwinProp-inspired ablation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.evaluation.phase33_twinprop_preregistration import ABLATION_ARMS


@dataclass(frozen=True)
class TwinPropAblationLimits:
    max_contacts: int
    max_branches: int
    max_slow_state_slots: int
    max_events: int
    max_interactions: int
    max_state_bytes: int
    decision_window_ticks: int
    readout_threshold: int
    slow_window_ticks: int = 2


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


class TwinPropAblationRuntime:
    """Evaluate one relation with no learning or durable graph mutation."""

    def __init__(self, arm: str, limits: TwinPropAblationLimits) -> None:
        if arm not in ABLATION_ARMS:
            raise ValueError(f"unsupported TwinProp-inspired arm: {arm}")
        self.arm = arm
        self.limits = limits

    def evaluate(self, case: Mapping[str, Any]) -> Dict[str, Any]:
        contacts_raw = case.get("contacts")
        events_raw = case.get("events")
        if not isinstance(contacts_raw, list) or not isinstance(events_raw, list):
            return self._abstain(case, "malformed_case")
        if not 1 <= len(contacts_raw) <= self.limits.max_contacts:
            return self._abstain(case, "contact_budget_exceeded")
        if not 1 <= len(events_raw) <= self.limits.max_events:
            return self._abstain(case, "event_budget_exceeded")

        contact_ids = [str(contact.get("contact_id", "")) for contact in contacts_raw]
        if any(not contact_id for contact_id in contact_ids):
            return self._abstain(case, "missing_contact_identity")
        if len(contact_ids) != len(set(contact_ids)):
            return self._abstain(case, "duplicate_contact")
        contacts = {
            str(contact["contact_id"]): dict(contact) for contact in contacts_raw
        }
        if any(str(event.get("contact_id", "")) not in contacts for event in events_raw):
            return self._abstain(case, "missing_contact")
        current_revision = str(case.get("source_revision", ""))
        if any(
            event.get("source_revision") is not None
            and str(event.get("source_revision")) != current_revision
            for event in events_raw
        ):
            return self._abstain(case, "stale_source_revision")

        branch_ids = {int(contact.get("branch", 0)) for contact in contacts.values()}
        if any(branch < 0 for branch in branch_ids):
            return self._abstain(case, "invalid_branch")
        if len(branch_ids) > self.limits.max_branches:
            return self._abstain(case, "branch_budget_exceeded")

        first_tick = min(int(event.get("tick", 0)) for event in events_raw)
        window_end = first_tick + self.limits.decision_window_ticks - 1
        arrivals: List[Dict[str, Any]] = []
        for sequence_index, event in sorted(
            enumerate(events_raw),
            key=lambda item: (int(item[1].get("tick", 0)), item[0]),
        ):
            contact = contacts[str(event["contact_id"])]
            arrival_tick = int(event.get("tick", 0)) + int(
                contact.get("delay_bucket", 0)
            )
            if arrival_tick > window_end:
                continue
            branch = int(contact.get("branch", 0))
            if self.arm in {
                "topology_collapsed_aggregation",
                "point_neuron_control",
            }:
                branch = 0
            arrivals.append(
                {
                    "contact_id": str(event["contact_id"]),
                    "branch": branch,
                    "tick": arrival_tick,
                    "polarity": str(contact.get("polarity", "excitatory")),
                    "sequence_index": sequence_index,
                }
            )

        if self.arm == "point_neuron_control":
            computation = self._point_computation(arrivals)
        elif self.arm == "passive_linear_branches":
            computation = self._passive_computation(arrivals)
        else:
            computation = self._branch_computation(arrivals)
        event_cost = len(arrivals) + int(computation["interaction_count"])
        if event_cost > self.limits.max_interactions:
            return self._abstain(case, "interaction_budget_exceeded")

        readout_count = int(computation["readout_count"])
        prediction = readout_count >= self.limits.readout_threshold
        state = {
            "arm": self.arm,
            "arrivals": arrivals,
            "branch_outputs": computation["branch_outputs"],
            "slow_state_slots": computation["slow_state_slots"],
            "readout_count": readout_count,
        }
        state_bytes = _canonical_size(state)
        if state_bytes > self.limits.max_state_bytes:
            return self._abstain(case, "state_budget_exceeded")

        expected = case.get("expected", {})
        target = bool(expected.get("readout_target", False))
        return {
            "case_id": str(case.get("case_id", "")),
            "family": str(case.get("family", "")),
            "arm": self.arm,
            "status": "evaluated",
            "reason": "ok",
            "prediction": prediction,
            "target": target,
            "target_match": prediction is target,
            "readout_count": readout_count,
            "readout_threshold": self.limits.readout_threshold,
            "active_branch_count": int(computation["active_branch_count"]),
            "branch_event_overlap": int(computation["branch_event_overlap"]),
            "slow_state_slot_count": len(computation["slow_state_slots"]),
            "slow_state_saturation": (
                len(computation["slow_state_slots"])
                / self.limits.max_slow_state_slots
                if self.limits.max_slow_state_slots
                else 0.0
            ),
            "state_bytes": state_bytes,
            "event_cost": event_cost,
            "durable_mutation": False,
            "state": state,
        }

    def _point_computation(
        self,
        arrivals: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        signed_sum = sum(
            -1 if arrival["polarity"] == "inhibitory" else 1
            for arrival in arrivals
        )
        readout_count = max(0, signed_sum)
        return {
            "readout_count": readout_count,
            "branch_outputs": {"0": readout_count},
            "slow_state_slots": [],
            "interaction_count": 0,
            "active_branch_count": int(readout_count > 0),
            "branch_event_overlap": 0,
        }

    def _passive_computation(
        self,
        arrivals: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        branch_sums: Dict[int, int] = {}
        for arrival in arrivals:
            branch = int(arrival["branch"])
            signed = -1 if arrival["polarity"] == "inhibitory" else 1
            branch_sums[branch] = branch_sums.get(branch, 0) + signed
        outputs = {
            str(branch): max(0, value) for branch, value in sorted(branch_sums.items())
        }
        return {
            "readout_count": sum(outputs.values()),
            "branch_outputs": outputs,
            "slow_state_slots": [],
            "interaction_count": 0,
            "active_branch_count": sum(value > 0 for value in outputs.values()),
            "branch_event_overlap": 0,
        }

    def _branch_computation(
        self,
        arrivals: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        by_branch: Dict[int, List[Mapping[str, Any]]] = {}
        for arrival in arrivals:
            by_branch.setdefault(int(arrival["branch"]), []).append(arrival)
        outputs: Dict[str, int] = {}
        slow_slots: List[Dict[str, int]] = []
        interaction_count = 0
        branch_overlap = 0
        slow_enabled = self.arm in {
            "intact_bounded_branches",
            "topology_collapsed_aggregation",
        }
        for branch, branch_arrivals in sorted(by_branch.items()):
            excitatory_ticks = sorted(
                int(arrival["tick"])
                for arrival in branch_arrivals
                if arrival["polarity"] != "inhibitory"
            )
            inhibitory_ticks = sorted(
                int(arrival["tick"])
                for arrival in branch_arrivals
                if arrival["polarity"] == "inhibitory"
            )
            candidate_count = 0
            tick_counts: Dict[int, int] = {}
            for tick in excitatory_ticks:
                tick_counts[tick] = tick_counts.get(tick, 0) + 1
            for tick, count in sorted(tick_counts.items()):
                if count >= 2:
                    candidate_count += count
                    branch_overlap += count - 1
                    interaction_count += count - 1
            if slow_enabled and candidate_count == 0:
                slow_pair = self._first_slow_pair(excitatory_ticks)
                if slow_pair is not None:
                    candidate_count = 2
                    branch_overlap += 1
                    interaction_count += 1
                    slow_slots.append(
                        {
                            "branch": branch,
                            "start_tick": slow_pair[0],
                            "end_tick": slow_pair[1],
                        }
                    )
            if candidate_count:
                if slow_enabled and any(
                    abs(inhibitory_tick - excitatory_tick)
                    <= self.limits.slow_window_ticks
                    for inhibitory_tick in inhibitory_ticks
                    for excitatory_tick in excitatory_ticks
                ):
                    interaction_count += 1
                    candidate_count = 0
                elif not slow_enabled and any(
                    inhibitory_tick in tick_counts for inhibitory_tick in inhibitory_ticks
                ):
                    interaction_count += 1
                    candidate_count = 0
            outputs[str(branch)] = candidate_count
        if len(slow_slots) > self.limits.max_slow_state_slots:
            return {
                "readout_count": 0,
                "branch_outputs": {},
                "slow_state_slots": slow_slots,
                "interaction_count": self.limits.max_interactions + 1,
                "active_branch_count": 0,
                "branch_event_overlap": branch_overlap,
            }
        return {
            "readout_count": sum(outputs.values()),
            "branch_outputs": outputs,
            "slow_state_slots": slow_slots,
            "interaction_count": interaction_count,
            "active_branch_count": sum(value > 0 for value in outputs.values()),
            "branch_event_overlap": branch_overlap,
        }

    def _first_slow_pair(self, ticks: Sequence[int]) -> Tuple[int, int] | None:
        for index, start in enumerate(ticks):
            for end in ticks[index + 1 :]:
                delta = end - start
                if 0 < delta <= self.limits.slow_window_ticks:
                    return start, end
        return None

    def _abstain(self, case: Mapping[str, Any], reason: str) -> Dict[str, Any]:
        expected_reason = {
            "missing_contact": "missing_contact",
            "stale_source_revision": "stale_source_revision",
        }.get(str(case.get("family", "")))
        return {
            "case_id": str(case.get("case_id", "")),
            "family": str(case.get("family", "")),
            "arm": self.arm,
            "status": "abstained",
            "reason": reason,
            "prediction": None,
            "target": bool(case.get("expected", {}).get("readout_target", False)),
            "target_match": expected_reason == reason,
            "readout_count": 0,
            "readout_threshold": self.limits.readout_threshold,
            "active_branch_count": 0,
            "branch_event_overlap": 0,
            "slow_state_slot_count": 0,
            "slow_state_saturation": 0.0,
            "state_bytes": 0,
            "event_cost": 0,
            "durable_mutation": False,
            "state": {},
        }


__all__ = ["TwinPropAblationLimits", "TwinPropAblationRuntime"]
