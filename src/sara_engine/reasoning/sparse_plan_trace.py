from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableSet, Optional, Sequence, Set


def stable_fact_event_id(value: str, width: int = 4096) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(1, int(width))


def _fact_set(values: Iterable[Any]) -> Set[str]:
    return {str(item).strip() for item in values if str(item).strip()}


def _sorted(values: Iterable[str]) -> List[str]:
    return sorted(str(item) for item in values)


@dataclass(frozen=True)
class SparsePlanStepResult:
    step_index: int
    action: str
    valid: bool
    errors: List[str]
    state_before: List[str]
    expected_next_state: List[str]
    claimed_next_state: List[str]
    preconditions_missing: List[str]
    invariant_violations: List[str]
    event_cost: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": self.action,
            "valid": self.valid,
            "errors": list(self.errors),
            "state_before": list(self.state_before),
            "expected_next_state": list(self.expected_next_state),
            "claimed_next_state": list(self.claimed_next_state),
            "preconditions_missing": list(self.preconditions_missing),
            "invariant_violations": list(self.invariant_violations),
            "event_cost": self.event_cost,
        }


@dataclass(frozen=True)
class SparsePlanTraceResult:
    case_id: str
    valid: bool
    errors: List[str]
    step_results: List[SparsePlanStepResult]
    final_state: List[str]
    goal_missing: List[str]
    invalid_step_count: int
    event_cost: int
    state_budget_units: int
    sparse_trace_signature: List[int]
    abstained: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "valid": self.valid,
            "errors": list(self.errors),
            "step_results": [item.to_dict() for item in self.step_results],
            "final_state": list(self.final_state),
            "goal_missing": list(self.goal_missing),
            "invalid_step_count": self.invalid_step_count,
            "event_cost": self.event_cost,
            "state_budget_units": self.state_budget_units,
            "sparse_trace_signature": list(self.sparse_trace_signature),
            "abstained": self.abstained,
        }


def _action_spec(actions: Mapping[str, Any], action_name: str) -> Mapping[str, Any]:
    spec = actions.get(action_name, {})
    return spec if isinstance(spec, Mapping) else {}


def _violated_invariants(state: Set[str], invariants: Sequence[Any]) -> List[str]:
    violations: List[str] = []
    for idx, invariant in enumerate(invariants):
        if not isinstance(invariant, Mapping):
            continue
        requires = _fact_set(invariant.get("requires", []))
        forbids = _fact_set(invariant.get("forbids", []))
        missing = sorted(requires.difference(state))
        present_forbidden = sorted(forbids.intersection(state))
        if missing or present_forbidden:
            name = str(invariant.get("name", f"invariant_{idx}"))
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if present_forbidden:
                details.append("forbidden=" + ",".join(present_forbidden))
            violations.append(f"{name}:{';'.join(details)}")
    return violations


def _step_action(step: Any) -> str:
    if isinstance(step, Mapping):
        return str(step.get("action", "")).strip()
    return str(step).strip()


def _claimed_next_state(step: Any) -> Optional[Set[str]]:
    if isinstance(step, Mapping) and "claimed_next_state" in step:
        value = step.get("claimed_next_state", [])
        if isinstance(value, list):
            return _fact_set(value)
    return None


def verify_sparse_plan_trace(case: Mapping[str, Any], *, signature_width: int = 4096) -> SparsePlanTraceResult:
    case_id = str(case.get("case_id", "unknown") or "unknown")
    actions = case.get("actions", {})
    if not isinstance(actions, Mapping):
        actions = {}
    plan_steps = case.get("plan", [])
    if not isinstance(plan_steps, list):
        plan_steps = []
    invariants = case.get("invariants", [])
    if not isinstance(invariants, list):
        invariants = []

    state: MutableSet[str] = set(_fact_set(case.get("initial_state", [])))
    goal = _fact_set(case.get("goal", []))
    step_results: List[SparsePlanStepResult] = []
    trace_facts: List[str] = [f"case:{case_id}"] + [f"initial:{fact}" for fact in state]
    event_cost = len(state) + len(goal)
    all_facts: Set[str] = set(state).union(goal)
    errors: List[str] = []
    abstained = False

    if not plan_steps:
        abstained = True
        errors.append("empty_plan")

    for step_index, step in enumerate(plan_steps):
        action_name = _step_action(step)
        spec = _action_spec(actions, action_name)
        step_errors: List[str] = []
        if not action_name:
            step_errors.append("missing_action")
        if not spec:
            step_errors.append("unknown_action")

        preconditions = _fact_set(spec.get("pre", []))
        add_effects = _fact_set(spec.get("add", []))
        del_effects = _fact_set(spec.get("del", []))
        missing_preconditions = _sorted(preconditions.difference(state))
        if missing_preconditions:
            step_errors.append("precondition_unsatisfied")

        expected_next = set(state)
        if not missing_preconditions and spec:
            expected_next.difference_update(del_effects)
            expected_next.update(add_effects)
        claimed_next = _claimed_next_state(step)
        if claimed_next is None:
            claimed_next = set(expected_next)
        else:
            missing_frame = sorted((state.difference(del_effects)).difference(claimed_next))
            if missing_frame:
                step_errors.append("missing_frame_persistence")
            if claimed_next != expected_next:
                step_errors.append("wrong_effects")

        invariant_violations = _violated_invariants(claimed_next, invariants)
        if invariant_violations:
            step_errors.append("invariant_violation")

        event_cost += len(state) + len(preconditions) + len(add_effects) + len(del_effects) + len(claimed_next)
        all_facts.update(preconditions)
        all_facts.update(add_effects)
        all_facts.update(del_effects)
        all_facts.update(claimed_next)
        trace_facts.extend(
            [
                f"step:{step_index}",
                f"action:{action_name}",
                *[f"pre:{fact}" for fact in sorted(preconditions)],
                *[f"add:{fact}" for fact in sorted(add_effects)],
                *[f"del:{fact}" for fact in sorted(del_effects)],
                *[f"next:{fact}" for fact in sorted(claimed_next)],
            ]
        )

        step_result = SparsePlanStepResult(
            step_index=step_index,
            action=action_name,
            valid=not step_errors,
            errors=sorted(set(step_errors)),
            state_before=_sorted(state),
            expected_next_state=_sorted(expected_next),
            claimed_next_state=_sorted(claimed_next),
            preconditions_missing=missing_preconditions,
            invariant_violations=sorted(invariant_violations),
            event_cost=len(state) + len(preconditions) + len(add_effects) + len(del_effects) + len(claimed_next),
        )
        step_results.append(step_result)
        if step_errors:
            errors.extend(f"step_{step_index}:{error}" for error in sorted(set(step_errors)))
        state = set(claimed_next)

    goal_missing = _sorted(goal.difference(state))
    if goal_missing:
        errors.append("unmet_goal")
    invalid_step_count = sum(1 for item in step_results if not item.valid)
    valid = not errors and invalid_step_count == 0 and not goal_missing
    signature = sorted({stable_fact_event_id(item, width=signature_width) for item in trace_facts})
    return SparsePlanTraceResult(
        case_id=case_id,
        valid=valid,
        errors=sorted(set(errors)),
        step_results=step_results,
        final_state=_sorted(state),
        goal_missing=goal_missing,
        invalid_step_count=invalid_step_count,
        event_cost=event_cost,
        state_budget_units=len(all_facts) + len(actions),
        sparse_trace_signature=signature,
        abstained=abstained,
    )


def build_repair_materials(
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[SparsePlanTraceResult],
) -> List[Dict[str, Any]]:
    repairs: List[Dict[str, Any]] = []
    case_by_id = {str(case.get("case_id", "unknown") or "unknown"): case for case in cases}
    for result in results:
        if result.valid:
            continue
        case = case_by_id.get(result.case_id, {})
        source_ref = str(case.get("source_ref", "synthetic_sparse_plan_trace_fixture"))
        repairs.append(
            {
                "schema": "sara-plan-trace-repair-material-v1",
                "material_type": "plan_trace_repair",
                "case_id": result.case_id,
                "source_ref": source_ref,
                "prompt": "Repair the sparse plan trace using deterministic verifier feedback.",
                "content": {
                    "errors": result.errors,
                    "goal_missing": result.goal_missing,
                    "invalid_step_count": result.invalid_step_count,
                    "step_results": [item.to_dict() for item in result.step_results if not item.valid],
                },
                "expected_behavior": "repair_or_abstain",
                "observed_only": True,
                "accepted": True,
                "quality_score": 0.8,
            }
        )
    return repairs
