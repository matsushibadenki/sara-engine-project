# Directory Path: src/sara_engine/nn/common_spike_space.py
# English Title: Common Spike Space Runtime Primitives
# Purpose/Content: Provides lightweight sparse-event primitives for multimodal spike normalization, temporal compression, and traceable high-order reasoning without backpropagation or dense matrix operations.

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sara_engine.nn.plastic_submodel_registry import (
    PlasticSubmodelRegistry,
    build_default_plastic_submodel_registry,
)


@dataclass(frozen=True)
class SparseSpikeEvent:
    """A single sparse event in the shared spike interface."""

    modality: str
    spike_id: int
    timestep: int
    channel: str
    confidence: float = 1.0
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "spike_id": self.spike_id,
            "timestep": self.timestep,
            "channel": self.channel,
            "confidence": self.confidence,
            "tags": list(self.tags),
        }


def _stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class CommonSpikeSpaceEncoder:
    """Normalizes heterogeneous inputs into one bounded sparse event schema."""

    def __init__(
        self,
        dimension: int = 4096,
        active_bits: int = 4,
        modality_offsets: Optional[Mapping[str, int]] = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if active_bits <= 0:
            raise ValueError("active_bits must be positive")
        self.dimension = int(dimension)
        self.active_bits = int(active_bits)
        self.modality_offsets = dict(
            modality_offsets
            or {
                "text": 0,
                "state": 997,
                "image": 1999,
                "audio": 2999,
                "sensor": 3511,
            }
        )

    def _spike_id(self, modality: str, key: str, bit: int) -> int:
        offset = int(self.modality_offsets.get(modality, _stable_hash(modality) % self.dimension))
        return (_stable_hash(f"{modality}:{key}:{bit}") + offset + bit * 131) % self.dimension

    def encode_text(self, text: str, *, timestep: int = 0, confidence: float = 1.0) -> List[SparseSpikeEvent]:
        tokens = [token.lower() for token in text.replace("\n", " ").split() if token.strip()]
        events: List[SparseSpikeEvent] = []
        for token_index, token in enumerate(tokens):
            for bit in range(self.active_bits):
                events.append(
                    SparseSpikeEvent(
                        modality="text",
                        spike_id=self._spike_id("text", f"{token}:{token_index}", bit),
                        timestep=timestep + token_index,
                        channel="semantic",
                        confidence=float(confidence),
                        tags=(f"token:{token}",),
                    )
                )
        return self._dedupe(events)

    def encode_structured_state(
        self,
        state: Mapping[str, Any],
        *,
        timestep: int = 0,
        confidence: float = 1.0,
    ) -> List[SparseSpikeEvent]:
        events: List[SparseSpikeEvent] = []
        for item_index, key in enumerate(sorted(state.keys())):
            value = state[key]
            state_key = f"{key}={value}"
            for bit in range(self.active_bits):
                events.append(
                    SparseSpikeEvent(
                        modality="state",
                        spike_id=self._spike_id("state", state_key, bit),
                        timestep=timestep + item_index,
                        channel="state",
                        confidence=float(confidence),
                        tags=(f"field:{key}",),
                    )
                )
        return self._dedupe(events)

    def encode_adapter_features(
        self,
        modality: str,
        features: Sequence[str],
        *,
        timestep: int = 0,
        confidence: float = 1.0,
    ) -> List[SparseSpikeEvent]:
        """Encodes lightweight adapter features from sensors, actions, or external modules."""

        events: List[SparseSpikeEvent] = []
        for feature_index, feature in enumerate(features):
            for bit in range(max(1, self.active_bits // 2)):
                events.append(
                    SparseSpikeEvent(
                        modality=modality,
                        spike_id=self._spike_id(modality, f"{feature}:{feature_index}", bit),
                        timestep=timestep + feature_index,
                        channel="adapter",
                        confidence=float(confidence),
                        tags=("adapter:v1", f"feature:{feature}", f"feature_index:{feature_index}"),
                    )
                )
        return self._dedupe(events)

    def encode_adapter_stub(
        self,
        modality: str,
        features: Sequence[str],
        *,
        timestep: int = 0,
        confidence: float = 1.0,
    ) -> List[SparseSpikeEvent]:
        """Backward-compatible alias for older benchmarks."""

        return self.encode_adapter_features(
            modality,
            features,
            timestep=timestep,
            confidence=confidence,
        )

    def _dedupe(self, events: Iterable[SparseSpikeEvent]) -> List[SparseSpikeEvent]:
        by_key: Dict[Tuple[str, int, int, str], SparseSpikeEvent] = {}
        for event in events:
            key = (event.modality, event.spike_id, event.timestep, event.channel)
            current = by_key.get(key)
            if current is None or event.confidence > current.confidence:
                by_key[key] = event
        return sorted(by_key.values(), key=lambda e: (e.timestep, e.modality, e.spike_id))


class TemporalCompressionPolicy:
    """Keeps sparse events inside a bounded event window."""

    def __init__(self, max_window: int = 8, max_events_per_modality: int = 24) -> None:
        if max_window <= 0:
            raise ValueError("max_window must be positive")
        if max_events_per_modality <= 0:
            raise ValueError("max_events_per_modality must be positive")
        self.max_window = int(max_window)
        self.max_events_per_modality = int(max_events_per_modality)

    def compress(self, events: Sequence[SparseSpikeEvent]) -> Tuple[List[SparseSpikeEvent], Dict[str, float]]:
        if not events:
            return [], {
                "input_event_count": 0.0,
                "compressed_event_count": 0.0,
                "compression_ratio": 1.0,
                "max_timestep": 0.0,
            }

        buckets: Dict[str, List[SparseSpikeEvent]] = {}
        for event in events:
            buckets.setdefault(event.modality, []).append(event)

        compressed: List[SparseSpikeEvent] = []
        for modality, bucket in buckets.items():
            ranked = sorted(
                bucket,
                key=lambda e: (-e.confidence, e.timestep, e.spike_id),
            )[: self.max_events_per_modality]
            for event in ranked:
                compressed.append(
                    SparseSpikeEvent(
                        modality=event.modality,
                        spike_id=event.spike_id,
                        timestep=event.timestep % self.max_window,
                        channel=event.channel,
                        confidence=event.confidence,
                        tags=event.tags + ("compressed",),
                    )
                )

        compressed = sorted(compressed, key=lambda e: (e.timestep, e.modality, e.spike_id))
        ratio = len(compressed) / max(len(events), 1)
        return compressed, {
            "input_event_count": float(len(events)),
            "compressed_event_count": float(len(compressed)),
            "compression_ratio": float(ratio),
            "max_timestep": float(max((event.timestep for event in compressed), default=0)),
        }


class ModalityTemporalBudget:
    """Allocates bounded timestep budgets without treating every modality equally."""

    def __init__(self, base_budget: Optional[Mapping[str, int]] = None, max_budget: int = 8) -> None:
        self.base_budget = dict(base_budget or {"text": 3, "state": 2, "image": 2, "audio": 4, "sensor": 2})
        self.max_budget = int(max_budget)
        if self.max_budget <= 0:
            raise ValueError("max_budget must be positive")

    def allocate(
        self,
        modality: str,
        *,
        confidence: float = 1.0,
        surprise: float = 0.0,
    ) -> Dict[str, Any]:
        base = int(self.base_budget.get(modality, 2))
        surprise_boost = 1 if surprise >= 0.5 else 0
        confidence_boost = 1 if confidence < 0.5 else 0
        budget = max(1, min(self.max_budget, base + surprise_boost + confidence_boost))
        return {
            "modality": modality,
            "budget": budget,
            "base_budget": base,
            "bounded": budget <= self.max_budget,
            "surprise": float(surprise),
            "confidence": float(confidence),
        }


class DendriticContextGate:
    """A lightweight multi-channel context gate inspired by multi-compartment neurons."""

    def __init__(
        self,
        short_term_limit: int = 32,
        long_term_limit: int = 64,
        error_limit: int = 16,
    ) -> None:
        self.short_term_limit = int(short_term_limit)
        self.long_term_limit = int(long_term_limit)
        self.error_limit = int(error_limit)
        self.short_term: List[int] = []
        self.long_term: List[int] = []
        self.prediction_error: List[int] = []

    def update(
        self,
        current_events: Sequence[SparseSpikeEvent],
        *,
        prediction_events: Optional[Sequence[SparseSpikeEvent]] = None,
        consolidate: bool = False,
    ) -> Dict[str, Any]:
        current_ids = [event.spike_id for event in current_events]
        predicted_ids = [event.spike_id for event in (prediction_events or [])]
        current_set = set(current_ids)
        predicted_set = set(predicted_ids)

        error_ids = sorted(current_set.symmetric_difference(predicted_set))[: self.error_limit]
        self.short_term = self._bounded_unique(current_ids + self.short_term, self.short_term_limit)
        if consolidate:
            self.long_term = self._bounded_unique(self.short_term + self.long_term, self.long_term_limit)
        self.prediction_error = self._bounded_unique(error_ids, self.error_limit)

        integrated = self._bounded_unique(
            self.short_term[: self.short_term_limit // 2]
            + self.long_term[: self.long_term_limit // 2]
            + self.prediction_error,
            self.short_term_limit + self.long_term_limit + self.error_limit,
        )
        overlap = len(current_set.intersection(set(self.long_term)))
        stability = 1.0 if len(self.prediction_error) <= max(self.error_limit, 1) and overlap <= len(current_set) else 0.0
        return {
            "integrated_spikes": integrated,
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "prediction_error_count": len(self.prediction_error),
            "context_stability": stability,
        }

    @staticmethod
    def _bounded_unique(values: Iterable[int], limit: int) -> List[int]:
        result: List[int] = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(int(value))
            if len(result) >= limit:
                break
        return result


@dataclass(frozen=True)
class CognitiveModuleResult:
    """Traceable output from one cognitive runtime module."""

    module: str
    emitted_events: Tuple[SparseSpikeEvent, ...] = field(default_factory=tuple)
    trace: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "emitted_events": [event.as_dict() for event in self.emitted_events],
            "trace": self.trace,
        }


class ModularCognitiveRuntime:
    """Coordinates planner, memory, world model, and actor over sparse events."""

    def __init__(
        self,
        *,
        encoder: Optional[CommonSpikeSpaceEncoder] = None,
        compressor: Optional[TemporalCompressionPolicy] = None,
        budgeter: Optional[ModalityTemporalBudget] = None,
        gate: Optional[DendriticContextGate] = None,
        plastic_registry: Optional[PlasticSubmodelRegistry] = None,
    ) -> None:
        self.encoder = encoder or CommonSpikeSpaceEncoder()
        self.compressor = compressor or TemporalCompressionPolicy()
        self.budgeter = budgeter or ModalityTemporalBudget()
        self.gate = gate or DendriticContextGate()
        self.plastic_registry = plastic_registry or build_default_plastic_submodel_registry()

    def run(
        self,
        *,
        text: str,
        state: Mapping[str, Any],
        candidate_actions: Sequence[str],
        action_feedback: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        text_events = self.encoder.encode_text(text, timestep=0, confidence=0.92)
        state_events = self.encoder.encode_structured_state(state, timestep=1, confidence=0.88)
        input_events = text_events + state_events
        compressed_events, compression = self.compressor.compress(input_events)
        memory_report = self.gate.update(compressed_events, consolidate=True)

        encoder_result = CognitiveModuleResult(
            module="encoder",
            emitted_events=tuple(input_events),
            trace={
                "text_event_count": len(text_events),
                "state_event_count": len(state_events),
                "schema": "common_spike_space.v1",
            },
        )
        memory_result = CognitiveModuleResult(
            module="memory_controller",
            emitted_events=tuple(compressed_events),
            trace={
                "compression": compression,
                "context": memory_report,
            },
        )

        world_model_result, candidates = self._run_world_model(
            state=state,
            compressed_events=compressed_events,
            candidate_actions=candidate_actions,
        )
        planner_result, selected, alternative = self._run_planner(candidates)
        feedback_report = self._apply_action_feedback(
            selected=selected,
            alternative=alternative,
            compressed_events=compressed_events,
            action_feedback=action_feedback or {},
        )
        actor_result = self._run_actor(selected)

        module_results = [
            encoder_result,
            memory_result,
            world_model_result,
            planner_result,
            actor_result,
        ]
        module_order = [result.module for result in module_results]
        return {
            "module_order": module_order,
            "modules": {result.module: result.as_dict() for result in module_results},
            "selected_action": selected,
            "counterfactual_action": alternative,
            "candidate_count": len(candidates),
            "action_feedback": feedback_report,
            "module_orchestration_complete": module_order
            == ["encoder", "memory_controller", "world_model", "planner", "actor"],
            "counterfactual_lane_complete": bool(
                selected
                and alternative
                and selected.get("branch_id") != alternative.get("branch_id")
                and selected.get("relation_trace", {}).get("trace_complete")
                and alternative.get("reverse_trace", {}).get("trace_complete")
                and selected.get("causal_trace", {}).get("trace_complete")
                and alternative.get("causal_trace", {}).get("trace_complete")
            ),
            "action_trace_complete": bool(actor_result.trace.get("trace_complete", False)),
        }

    def _run_world_model(
        self,
        *,
        state: Mapping[str, Any],
        compressed_events: Sequence[SparseSpikeEvent],
        candidate_actions: Sequence[str],
    ) -> Tuple[CognitiveModuleResult, List[Dict[str, Any]]]:
        goal = str(state.get("goal", "unknown"))
        status = str(state.get("status", "unknown"))
        candidates: List[Dict[str, Any]] = []
        for index, action in enumerate(candidate_actions):
            branch_id = "primary" if index == 0 else f"counterfactual-{index}"
            budget = self.budgeter.allocate(
                "state",
                confidence=0.9 if index == 0 else 0.7,
                surprise=0.2 if index == 0 else 0.6,
            )
            relation_trace = build_event_relation_trace(
                cause=f"{status}:{action}",
                relation="projects",
                effect=f"{goal}:ready",
                branch_id=branch_id,
            )
            reverse_trace = build_reverse_reasoning_trace(
                outcome=f"{goal}:blocked",
                candidate_causes=[f"{action}:missing", f"{status}:unchanged"],
                selected_cause=f"{status}:unchanged",
                branch_id=branch_id,
            )
            causal_trace = build_causal_candidate_trace(
                relation_trace=relation_trace,
                reverse_trace=reverse_trace,
                selected_action=action,
                branch_id=branch_id,
            )
            route_goal = self._submodel_route_goal(
                goal=goal,
                status=status,
                action=action,
                branch_id=branch_id,
            )
            submodel_route = self.plastic_registry.route(compressed_events, goal=route_goal)
            support_submodels = [
                str(item.get("submodel_id", ""))
                for item in submodel_route.get("selected_submodels", [])
                if isinstance(item, Mapping) and str(item.get("submodel_id", ""))
            ]
            action_tokens = set(action.lower().split())
            goal_bonus = 1.0 if goal.lower() in action_tokens or "release" in action_tokens else 0.5
            cost_penalty = float(budget["budget"]) / max(float(self.budgeter.max_budget), 1.0)
            submodel_bonus = min(len(support_submodels), 4) * 0.025
            score = max(0.0, min(1.0, 0.55 + 0.30 * goal_bonus - 0.10 * cost_penalty))
            score = max(0.0, min(1.0, score + submodel_bonus))
            candidates.append(
                {
                    "branch_id": branch_id,
                    "action": action,
                    "projected_state": f"{goal}:ready",
                    "score": score,
                    "budget": budget,
                    "relation_trace": relation_trace,
                    "reverse_trace": reverse_trace,
                    "causal_trace": causal_trace,
                    "submodel_route": submodel_route,
                    "support_submodels": support_submodels,
                }
            )

        event_sample = tuple(compressed_events[: min(len(compressed_events), 8)])
        concept_trace = self.plastic_registry.concept_trace()
        return (
            CognitiveModuleResult(
                module="world_model",
                emitted_events=event_sample,
                trace={
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "plastic_submodel_concept_trace": concept_trace,
                },
            ),
            candidates,
        )

    @staticmethod
    def _submodel_route_goal(
        *,
        goal: str,
        status: str,
        action: str,
        branch_id: str,
    ) -> str:
        action_text = str(action).lower()
        route_terms = ["world", "memory", str(goal), str(status), str(branch_id)]
        if "defer" in action_text or "hold" in action_text:
            route_terms.extend(["value", "self_monitor", "language"])
        else:
            route_terms.extend(["value", "body", "math"])
        return " ".join(route_terms)

    def _run_planner(
        self,
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[CognitiveModuleResult, Dict[str, Any], Dict[str, Any]]:
        ordered = sorted(candidates, key=lambda item: (-float(item.get("score", 0.0)), str(item.get("branch_id", ""))))
        selected = dict(ordered[0]) if ordered else {}
        alternative = dict(ordered[1]) if len(ordered) > 1 else {}
        return (
            CognitiveModuleResult(
                module="planner",
                trace={
                    "selected_branch": selected.get("branch_id"),
                    "alternative_branch": alternative.get("branch_id"),
                    "selected_score": float(selected.get("score", 0.0)),
                    "alternative_score": float(alternative.get("score", 0.0)),
                    "decision_observable": bool(selected and alternative),
                },
            ),
            selected,
            alternative,
        )

    def _run_actor(self, selected: Mapping[str, Any]) -> CognitiveModuleResult:
        action = str(selected.get("action", ""))
        action_events = tuple(
            self.encoder.encode_adapter_features(
                "action",
                [action or "noop"],
                timestep=0,
                confidence=float(selected.get("score", 0.0)),
            )
        )
        trace_complete = bool(action and action_events and selected.get("relation_trace", {}).get("trace_complete"))
        return CognitiveModuleResult(
            module="actor",
            emitted_events=action_events,
            trace={
                "action": action,
                "branch_id": selected.get("branch_id"),
                "event_count": len(action_events),
                "trace_complete": trace_complete,
            },
        )

    def _apply_action_feedback(
        self,
        *,
        selected: Dict[str, Any],
        alternative: Dict[str, Any],
        compressed_events: Sequence[SparseSpikeEvent],
        action_feedback: Mapping[str, float],
    ) -> Dict[str, Any]:
        if not action_feedback:
            return {
                "applied": False,
                "feedback_count": 0,
                "state_budget_ok": self.plastic_registry.state_budget_ok(),
            }
        feedback_records = []
        for candidate in (selected, alternative):
            branch_id = str(candidate.get("branch_id", "") or "")
            if not branch_id or branch_id not in action_feedback:
                continue
            route = candidate.get("submodel_route", {})
            if not isinstance(route, Mapping):
                continue
            feedback = self.plastic_registry.apply_route_credit(
                route,
                support_events=compressed_events,
                credit=float(action_feedback.get(branch_id, 0.0)),
                reason=f"runtime feedback for {branch_id}",
            )
            candidate["feedback_trace"] = feedback
            feedback_records.append(
                {
                    "branch_id": branch_id,
                    "credit": feedback["credit"],
                    "updated_submodel_count": feedback["updated_submodel_count"],
                    "state_budget_ok": feedback["state_budget_ok"],
                }
            )
        return {
            "applied": bool(feedback_records),
            "feedback_count": len(feedback_records),
            "records": feedback_records,
            "state_budget_ok": self.plastic_registry.state_budget_ok(),
        }


def build_event_relation_trace(
    *,
    cause: str,
    relation: str,
    effect: str,
    branch_id: str = "primary",
) -> Dict[str, Any]:
    return {
        "branch_id": branch_id,
        "cause": cause,
        "relation": relation,
        "effect": effect,
        "trace_complete": bool(cause and relation and effect),
    }


def build_reverse_reasoning_trace(
    *,
    outcome: str,
    candidate_causes: Sequence[str],
    selected_cause: str,
    branch_id: str = "primary",
) -> Dict[str, Any]:
    candidates = [cause for cause in candidate_causes if cause]
    return {
        "branch_id": branch_id,
        "outcome": outcome,
        "candidate_causes": candidates,
        "selected_cause": selected_cause,
        "trace_complete": bool(outcome and selected_cause and selected_cause in candidates),
    }


def build_causal_candidate_trace(
    *,
    relation_trace: Mapping[str, Any],
    reverse_trace: Mapping[str, Any],
    selected_action: str,
    branch_id: str = "primary",
) -> Dict[str, Any]:
    """Links forward relation evidence and reverse cause evidence for one action branch."""

    relation_branch = str(relation_trace.get("branch_id", ""))
    reverse_branch = str(reverse_trace.get("branch_id", ""))
    relation_complete = bool(relation_trace.get("trace_complete", False))
    reverse_complete = bool(reverse_trace.get("trace_complete", False))
    candidate_causes = reverse_trace.get("candidate_causes", [])
    if not isinstance(candidate_causes, Sequence) or isinstance(candidate_causes, (str, bytes)):
        candidate_causes = []
    selected_cause = str(reverse_trace.get("selected_cause", ""))
    cause = str(relation_trace.get("cause", ""))
    effect = str(relation_trace.get("effect", ""))
    action_tokens = {token for token in selected_action.lower().split() if token}
    cause_tokens = {token for token in cause.lower().replace(":", " ").split() if token}
    causal_alignment = bool(action_tokens.intersection(cause_tokens)) or bool(selected_action and selected_cause)
    trace_complete = bool(
        branch_id
        and relation_branch == branch_id
        and reverse_branch == branch_id
        and relation_complete
        and reverse_complete
        and selected_cause in [str(item) for item in candidate_causes]
        and causal_alignment
        and effect
    )
    return {
        "branch_id": branch_id,
        "selected_action": selected_action,
        "relation_branch": relation_branch,
        "reverse_branch": reverse_branch,
        "selected_cause": selected_cause,
        "candidate_cause_count": len(candidate_causes),
        "causal_alignment": causal_alignment,
        "trace_complete": trace_complete,
    }


def build_runtime_trace_digest(runtime_report: Mapping[str, Any]) -> Dict[str, Any]:
    """Builds a stable audit digest for deterministic sparse runtime traces."""

    selected = runtime_report.get("selected_action", {})
    if not isinstance(selected, Mapping):
        selected = {}
    counterfactual = runtime_report.get("counterfactual_action", {})
    if not isinstance(counterfactual, Mapping):
        counterfactual = {}
    module_order = runtime_report.get("module_order", [])
    if not isinstance(module_order, list):
        module_order = []
    payload = {
        "module_order": [str(item) for item in module_order],
        "selected_branch": str(selected.get("branch_id", "")),
        "selected_action": str(selected.get("action", "")),
        "counterfactual_branch": str(counterfactual.get("branch_id", "")),
        "counterfactual_action": str(counterfactual.get("action", "")),
        "candidate_count": int(runtime_report.get("candidate_count", 0) or 0),
        "module_orchestration_complete": bool(runtime_report.get("module_orchestration_complete", False)),
        "counterfactual_lane_complete": bool(runtime_report.get("counterfactual_lane_complete", False)),
        "action_trace_complete": bool(runtime_report.get("action_trace_complete", False)),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "trace_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def compare_runtime_trace_digests(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compares two runtime audit digests without retaining full event payloads."""

    first_digest = build_runtime_trace_digest(first)
    second_digest = build_runtime_trace_digest(second)
    matching_fields = {
        "module_order": first_digest["module_order"] == second_digest["module_order"],
        "selected_branch": first_digest["selected_branch"] == second_digest["selected_branch"],
        "selected_action": first_digest["selected_action"] == second_digest["selected_action"],
        "counterfactual_branch": first_digest["counterfactual_branch"] == second_digest["counterfactual_branch"],
        "counterfactual_action": first_digest["counterfactual_action"] == second_digest["counterfactual_action"],
        "candidate_count": first_digest["candidate_count"] == second_digest["candidate_count"],
        "trace_digest": first_digest["trace_digest"] == second_digest["trace_digest"],
    }
    return {
        "consistent": all(matching_fields.values()),
        "matching_fields": matching_fields,
        "first_digest": first_digest,
        "second_digest": second_digest,
    }


def _event_id_set(events: Sequence[SparseSpikeEvent]) -> set[int]:
    return {int(event.spike_id) for event in events}


def _event_fingerprint(events: Sequence[SparseSpikeEvent]) -> str:
    payload = [
        {
            "modality": event.modality,
            "spike_id": event.spike_id,
            "channel": event.channel,
            "tags": list(event.tags),
        }
        for event in sorted(events, key=lambda item: (item.modality, item.channel, item.spike_id, item.tags))
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_spiking_hjepa_transition_trace(
    *,
    source_events: Sequence[SparseSpikeEvent],
    predicted_events: Sequence[SparseSpikeEvent],
    observed_events: Sequence[SparseSpikeEvent],
    correction_events: Sequence[SparseSpikeEvent],
    operator: str,
    branch_id: str = "primary",
    max_prediction_error: int = 32,
) -> Dict[str, Any]:
    """Audits a lightweight latent-transition prediction without dense reconstruction."""

    source_ids = _event_id_set(source_events)
    predicted_ids = _event_id_set(predicted_events)
    observed_ids = _event_id_set(observed_events)
    correction_ids = _event_id_set(correction_events)
    aligned_ids = sorted(predicted_ids.intersection(observed_ids))
    prediction_error_ids = sorted(observed_ids.difference(predicted_ids))[:max_prediction_error]
    false_positive_ids = sorted(predicted_ids.difference(observed_ids))[:max_prediction_error]
    correction_coverage = bool(prediction_error_ids) and set(prediction_error_ids).issubset(correction_ids)
    alignment_ratio = len(aligned_ids) / max(len(predicted_ids), 1)
    transition_fingerprints = {
        "source": _event_fingerprint(source_events),
        "predicted": _event_fingerprint(predicted_events),
        "observed": _event_fingerprint(observed_events),
        "correction": _event_fingerprint(correction_events),
    }
    anti_collapse_diversity = len(set(transition_fingerprints.values())) >= 3
    trace_complete = bool(
        branch_id
        and operator
        and source_ids
        and predicted_ids
        and observed_ids
        and correction_ids
        and aligned_ids
        and prediction_error_ids
        and correction_coverage
        and anti_collapse_diversity
    )
    return {
        "branch_id": branch_id,
        "operator": operator,
        "source_count": len(source_ids),
        "predicted_count": len(predicted_ids),
        "observed_count": len(observed_ids),
        "correction_count": len(correction_ids),
        "alignment_ratio": float(alignment_ratio),
        "aligned_ids": aligned_ids,
        "prediction_error_ids": prediction_error_ids,
        "false_positive_ids": false_positive_ids,
        "correction_coverage": correction_coverage,
        "anti_collapse_diversity": anti_collapse_diversity,
        "transition_fingerprints": transition_fingerprints,
        "trace_complete": trace_complete,
    }


def compare_spiking_hjepa_transition_branches(
    primary_trace: Mapping[str, Any],
    counterfactual_trace: Mapping[str, Any],
) -> Dict[str, Any]:
    """Checks that primary and counterfactual latent transitions stay separable."""

    primary_fingerprints = primary_trace.get("transition_fingerprints", {})
    if not isinstance(primary_fingerprints, Mapping):
        primary_fingerprints = {}
    counterfactual_fingerprints = counterfactual_trace.get("transition_fingerprints", {})
    if not isinstance(counterfactual_fingerprints, Mapping):
        counterfactual_fingerprints = {}
    different_branch = str(primary_trace.get("branch_id", "")) != str(counterfactual_trace.get("branch_id", ""))
    different_prediction = str(primary_fingerprints.get("predicted", "")) != str(
        counterfactual_fingerprints.get("predicted", "")
    )
    different_observation = str(primary_fingerprints.get("observed", "")) != str(
        counterfactual_fingerprints.get("observed", "")
    )
    both_complete = bool(primary_trace.get("trace_complete", False)) and bool(
        counterfactual_trace.get("trace_complete", False)
    )
    return {
        "different_branch": different_branch,
        "different_prediction": different_prediction,
        "different_observation": different_observation,
        "both_complete": both_complete,
        "separable": bool(different_branch and different_prediction and different_observation and both_complete),
    }


def _trace_id_count(trace: Mapping[str, Any], key: str) -> int:
    value = trace.get(key, [])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def evaluate_lejepa_sparse_latent_health_trace(
    primary_trace: Mapping[str, Any],
    counterfactual_trace: Mapping[str, Any],
    branch_comparison: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluates LeJEPA-inspired latent health without dense SIGReg or backprop."""

    comparison = (
        dict(branch_comparison)
        if isinstance(branch_comparison, Mapping)
        else compare_spiking_hjepa_transition_branches(primary_trace, counterfactual_trace)
    )
    primary_fingerprints = primary_trace.get("transition_fingerprints", {})
    if not isinstance(primary_fingerprints, Mapping):
        primary_fingerprints = {}
    counterfactual_fingerprints = counterfactual_trace.get("transition_fingerprints", {})
    if not isinstance(counterfactual_fingerprints, Mapping):
        counterfactual_fingerprints = {}

    primary_alignment = float(primary_trace.get("alignment_ratio", 0.0) or 0.0)
    counterfactual_alignment = float(counterfactual_trace.get("alignment_ratio", 0.0) or 0.0)
    primary_role_diversity = len(set(str(value) for value in primary_fingerprints.values() if value))
    counterfactual_role_diversity = len(set(str(value) for value in counterfactual_fingerprints.values() if value))
    prediction_error_count = _trace_id_count(primary_trace, "prediction_error_ids")
    false_positive_count = _trace_id_count(primary_trace, "false_positive_ids")

    linear_identifiability = bool(
        primary_trace.get("trace_complete", False)
        and primary_alignment >= 1.0
        and primary_trace.get("correction_coverage", False)
        and prediction_error_count > 0
    )
    latent_whitening_health = bool(
        primary_trace.get("anti_collapse_diversity", False)
        and counterfactual_trace.get("anti_collapse_diversity", False)
        and primary_role_diversity >= 3
        and counterfactual_role_diversity >= 3
    )
    factor_disentanglement = bool(
        comparison.get("separable", False)
        and comparison.get("different_prediction", False)
        and comparison.get("different_observation", False)
    )
    latent_planning_consistency = bool(
        linear_identifiability
        and factor_disentanglement
        and false_positive_count == 0
        and str(primary_trace.get("operator", ""))
        and str(counterfactual_trace.get("operator", ""))
    )
    positive_pair_alignment = bool(primary_alignment >= 1.0 and counterfactual_alignment >= 1.0)

    metrics = {
        "lejepa_linear_identifiability_proxy": 1.0 if linear_identifiability else 0.0,
        "lejepa_latent_whitening_health": 1.0 if latent_whitening_health else 0.0,
        "lejepa_factor_disentanglement": 1.0 if factor_disentanglement else 0.0,
        "lejepa_latent_planning_consistency": 1.0 if latent_planning_consistency else 0.0,
        "lejepa_positive_pair_alignment": 1.0 if positive_pair_alignment else 0.0,
    }
    return {
        "observed_only": True,
        "metrics": metrics,
        "trace": {
            "primary_branch_id": str(primary_trace.get("branch_id", "")),
            "counterfactual_branch_id": str(counterfactual_trace.get("branch_id", "")),
            "primary_alignment": primary_alignment,
            "counterfactual_alignment": counterfactual_alignment,
            "primary_role_diversity": primary_role_diversity,
            "counterfactual_role_diversity": counterfactual_role_diversity,
            "prediction_error_count": prediction_error_count,
            "false_positive_count": false_positive_count,
            "branch_comparison": comparison,
        },
    }


def evaluate_micro_turn_interaction_trace(
    micro_turns: Sequence[Mapping[str, Any]],
    *,
    max_turns: int = 8,
    max_event_budget: int = 18,
) -> Dict[str, Any]:
    """Audits a bounded foreground/background interaction stream."""

    turns = [dict(turn) for turn in micro_turns if isinstance(turn, Mapping)]
    turn_count = len(turns)
    event_cost = sum(int(turn.get("event_cost", 1) or 0) for turn in turns)
    streams = {str(turn.get("stream", "")) for turn in turns if str(turn.get("stream", ""))}
    foreground_turns = [turn for turn in turns if str(turn.get("lane", "")) == "foreground"]
    background_turns = [turn for turn in turns if str(turn.get("lane", "")) == "background"]
    handoff_turns = [turn for turn in turns if bool(turn.get("handoff", False))]
    interrupt_turns = [turn for turn in turns if str(turn.get("event_type", "")) == "interrupt"]
    recovery_turns = [turn for turn in turns if str(turn.get("event_type", "")) == "interrupt_recovery"]
    backchannel_turns = [turn for turn in turns if bool(turn.get("backchannel", False))]
    simultaneous_groups: Dict[str, set[str]] = {}
    for turn in turns:
        bucket = str(turn.get("time_bucket", ""))
        stream = str(turn.get("stream", ""))
        if bucket and stream:
            simultaneous_groups.setdefault(bucket, set()).add(stream)

    micro_turn_event_budget = bool(turns and turn_count <= max_turns and event_cost <= max_event_budget)
    foreground_background_handoff = bool(foreground_turns and background_turns and handoff_turns)
    interrupt_recovery = bool(
        interrupt_turns
        and recovery_turns
        and int(recovery_turns[0].get("time_bucket", 0) or 0) >= int(interrupt_turns[0].get("time_bucket", 0) or 0)
    )
    simultaneous_stream_route = any(len(group_streams) >= 2 for group_streams in simultaneous_groups.values())
    time_aligned_backchannel = any(
        bool(turn.get("backchannel", False))
        and str(turn.get("policy", "")) in {"acknowledge", "hold", "yield"}
        and int(turn.get("latency_ms", 9999) or 9999) <= 250
        for turn in backchannel_turns
    )
    metrics = {
        "micro_turn_event_budget": 1.0 if micro_turn_event_budget else 0.0,
        "foreground_background_context_handoff": 1.0 if foreground_background_handoff else 0.0,
        "interrupt_recovery_trace": 1.0 if interrupt_recovery else 0.0,
        "simultaneous_stream_route_integrity": 1.0 if simultaneous_stream_route else 0.0,
        "time_aligned_backchannel_policy": 1.0 if time_aligned_backchannel else 0.0,
    }
    return {
        "observed_only": True,
        "metrics": metrics,
        "trace": {
            "turn_count": turn_count,
            "event_cost": event_cost,
            "streams": sorted(streams),
            "foreground_count": len(foreground_turns),
            "background_count": len(background_turns),
            "handoff_count": len(handoff_turns),
            "interrupt_count": len(interrupt_turns),
            "recovery_count": len(recovery_turns),
            "backchannel_count": len(backchannel_turns),
            "simultaneous_bucket_count": sum(
                1 for group_streams in simultaneous_groups.values() if len(group_streams) >= 2
            ),
            "max_turns": int(max_turns),
            "max_event_budget": int(max_event_budget),
        },
    }


def evaluate_phase_assigned_submodel_block_trace(
    blocks: Sequence[Mapping[str, Any]],
    *,
    max_event_budget: int = 24,
) -> Dict[str, Any]:
    """Audits DiffusionBlocks-inspired phase routing without score matching."""

    block_records = [dict(block) for block in blocks if isinstance(block, Mapping)]
    event_cost = sum(int(block.get("event_cost", 1) or 0) for block in block_records)
    phases = {str(block.get("phase", "")) for block in block_records if str(block.get("phase", ""))}
    submodels = {str(block.get("submodel", "")) for block in block_records if str(block.get("submodel", ""))}
    uncertainty_buckets = {
        str(block.get("uncertainty_bucket", ""))
        for block in block_records
        if str(block.get("uncertainty_bucket", ""))
    }
    correction_blocks = [
        block
        for block in block_records
        if str(block.get("phase", "")) in {"prediction_error", "correction"}
        or bool(block.get("correction_event", False))
    ]
    local_credit_blocks = [block for block in block_records if bool(block.get("local_credit", False))]
    independent_blocks = [
        block
        for block in block_records
        if bool(block.get("independent_update", False)) and not bool(block.get("backprop_required", False))
    ]
    phase_assigned_route = bool(
        block_records
        and phases.issuperset({"memory_phase", "prediction_error", "correction"})
        and len(submodels) >= 3
        and all(str(block.get("phase", "")) and str(block.get("submodel", "")) for block in block_records)
    )
    uncertainty_specialization = bool(
        uncertainty_buckets.issuperset({"low", "medium", "high"})
        and len({
            (str(block.get("uncertainty_bucket", "")), str(block.get("submodel", "")))
            for block in block_records
            if str(block.get("uncertainty_bucket", "")) and str(block.get("submodel", ""))
        }) >= 3
    )
    denoising_correction_trace = bool(
        correction_blocks
        and all(bool(block.get("correction_event", False)) for block in correction_blocks)
        and any(float(block.get("residual_reduction", 0.0) or 0.0) > 0.0 for block in correction_blocks)
    )
    block_independent_local_update_budget = bool(
        block_records
        and len(independent_blocks) == len(block_records)
        and local_credit_blocks
        and event_cost <= max_event_budget
    )
    metrics = {
        "phase_assigned_submodel_route": 1.0 if phase_assigned_route else 0.0,
        "uncertainty_bucket_specialization": 1.0 if uncertainty_specialization else 0.0,
        "denoising_correction_trace_integrity": 1.0 if denoising_correction_trace else 0.0,
        "block_independent_local_update_budget": 1.0 if block_independent_local_update_budget else 0.0,
    }
    return {
        "observed_only": True,
        "metrics": metrics,
        "trace": {
            "block_count": len(block_records),
            "event_cost": event_cost,
            "max_event_budget": int(max_event_budget),
            "phases": sorted(phases),
            "submodels": sorted(submodels),
            "uncertainty_buckets": sorted(uncertainty_buckets),
            "correction_block_count": len(correction_blocks),
            "local_credit_block_count": len(local_credit_blocks),
            "independent_block_count": len(independent_blocks),
        },
    }


def build_spiking_hjepa_multistep_trace(
    transition_traces: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Audits chained latent transitions without dense rollout state."""

    traces = [dict(trace) for trace in transition_traces if isinstance(trace, Mapping)]
    step_count = len(traces)
    complete_steps = sum(1 for trace in traces if bool(trace.get("trace_complete", False)))
    error_counts = [
        len(trace.get("prediction_error_ids", []))
        if isinstance(trace.get("prediction_error_ids", []), list)
        else 0
        for trace in traces
    ]
    correction_counts = [
        int(trace.get("correction_count", 0) or 0)
        for trace in traces
    ]
    operators = [str(trace.get("operator", "")) for trace in traces]
    branches = [str(trace.get("branch_id", "")) for trace in traces]
    observed_fingerprints = [
        str(trace.get("transition_fingerprints", {}).get("observed", ""))
        for trace in traces
        if isinstance(trace.get("transition_fingerprints", {}), Mapping)
    ]
    correction_coverage_steps = sum(1 for trace in traces if bool(trace.get("correction_coverage", False)))
    total_errors = sum(error_counts)
    total_corrections = sum(correction_counts)
    error_reduction_ratio = 1.0
    if len(error_counts) >= 2 and error_counts[0] > 0:
        error_reduction_ratio = max(0.0, min(1.0, 1.0 - (error_counts[-1] / max(error_counts[0], 1))))

    chain_complete = bool(
        step_count >= 2
        and complete_steps == step_count
        and all(operator for operator in operators)
        and len(set(observed_fingerprints)) == len(observed_fingerprints)
    )
    correction_converged = bool(
        total_errors > 0
        and total_corrections >= total_errors
        and correction_coverage_steps == step_count
        and (error_counts[-1] <= error_counts[0] if len(error_counts) >= 2 else True)
    )
    return {
        "step_count": step_count,
        "complete_steps": complete_steps,
        "operators": operators,
        "branches": branches,
        "error_counts": error_counts,
        "correction_counts": correction_counts,
        "total_prediction_errors": total_errors,
        "total_corrections": total_corrections,
        "correction_coverage_steps": correction_coverage_steps,
        "error_reduction_ratio": float(error_reduction_ratio),
        "chain_complete": chain_complete,
        "correction_converged": correction_converged,
        "trace_complete": bool(chain_complete and correction_converged),
    }
