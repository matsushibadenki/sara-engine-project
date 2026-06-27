from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from sara_engine.dynamics.oscillation import OscillationManager
from sara_engine.nn.multi_timescale_leak_state import MultiTimescaleLeakState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def stable_self_state_id(value: Any, modulus: int = 4096) -> int:
    text = str(value).strip()
    if not text:
        return 0
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % max(1, int(modulus))


def relation_self_state_alignment(
    source_event_id: str,
    target_event_id: str,
    self_state_ids: Sequence[int],
) -> float:
    anchors = (
        stable_self_state_id(source_event_id),
        stable_self_state_id(target_event_id),
    )
    active = {int(value) for value in self_state_ids}
    if not active:
        return 0.0
    matched = sum(1 for anchor in anchors if int(anchor) in active)
    return float(matched) / float(len(anchors))


def concept_self_state_alignment(
    concept_key: str,
    self_state_ids: Sequence[int],
) -> float:
    text = str(concept_key)
    if ":" not in text or "->" not in text:
        return 0.0
    _, rest = text.split(":", 1)
    source_event_id, target_event_id = rest.split("->", 1)
    return relation_self_state_alignment(source_event_id, target_event_id, self_state_ids)


def memory_self_state_alignment(
    *,
    own_latent_id: str,
    source_ref: str,
    self_state_ids: Sequence[int],
) -> float:
    if own_latent_id:
        concept_score = concept_self_state_alignment(own_latent_id, self_state_ids)
        if concept_score > 0.0:
            return concept_score
        return float(stable_self_state_id(own_latent_id) in {int(value) for value in self_state_ids})
    if not source_ref:
        return 0.0
    return float(stable_self_state_id(source_ref) in {int(value) for value in self_state_ids})


def _bounded_unique_ints(values: Iterable[int], limit: int) -> Tuple[int, ...]:
    ordered: List[int] = []
    seen = set()
    for value in values:
        normalized = int(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
        if len(ordered) >= max(1, int(limit)):
            break
    return tuple(ordered)


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set)) / float(len(union))


@dataclass(frozen=True)
class SelfStateConfig:
    max_state_units: int = 96
    max_prediction_links: int = 64
    max_self_ids: int = 8
    spontaneous_floor: int = 1
    tonic_boost: float = 0.65
    recurrent_boost: float = 0.55
    predictive_boost: float = 0.45
    memory_boost: float = 0.75
    persistence_decay: float = 0.92
    idle_continuity_target: float = 0.35


class SparseInternalPredictor:
    """Learns sparse next-state transitions from local active-id sequences."""

    def __init__(self, *, max_links: int = 64) -> None:
        self.max_links = max(1, int(max_links))
        self._transition_counts: Dict[int, Dict[int, float]] = {}

    def observe(self, previous_ids: Sequence[int], next_ids: Sequence[int]) -> None:
        prev = tuple(int(value) for value in previous_ids)
        nxt = tuple(int(value) for value in next_ids)
        if not prev or not nxt:
            return
        for left in prev:
            bucket = self._transition_counts.setdefault(left, {})
            for right in nxt:
                bucket[right] = float(bucket.get(right, 0.0)) + 1.0
            self._transition_counts[left] = self._trim_bucket(bucket)

    def predict(self, active_ids: Sequence[int], *, limit: int = 4) -> Tuple[int, ...]:
        scores: Dict[int, float] = {}
        for left in active_ids:
            for right, count in self._transition_counts.get(int(left), {}).items():
                scores[int(right)] = float(scores.get(int(right), 0.0)) + float(count)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return tuple(event_id for event_id, _ in ranked[: max(1, int(limit))])

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema": "sara-sparse-internal-predictor-v1",
            "transition_counts": {
                str(left): {str(right): float(count) for right, count in sorted(bucket.items())}
                for left, bucket in sorted(self._transition_counts.items())
            },
        }

    def _trim_bucket(self, bucket: Mapping[int, float]) -> Dict[int, float]:
        ranked = sorted(bucket.items(), key=lambda item: (-item[1], item[0]))
        return {int(right): float(count) for right, count in ranked[: self.max_links]}


class PersistentSelfStateController:
    """Maintains bounded self-state continuity across time with no external input."""

    def __init__(
        self,
        *,
        core_event_ids: Sequence[int] = (),
        config: SelfStateConfig | None = None,
        leak_state: MultiTimescaleLeakState | None = None,
        oscillation: OscillationManager | None = None,
        predictor: SparseInternalPredictor | None = None,
    ) -> None:
        self.config = config or SelfStateConfig()
        self.core_event_ids = _bounded_unique_ints(core_event_ids, self.config.max_self_ids)
        self.leak_state = leak_state or MultiTimescaleLeakState(max_state_units=self.config.max_state_units)
        self.oscillation = oscillation or OscillationManager()
        self.predictor = predictor or SparseInternalPredictor(max_links=self.config.max_prediction_links)
        self._persistence: Dict[int, float] = {
            int(event_id): 1.0 for event_id in self.core_event_ids
        }
        self._clock_ms = 0.0
        self._last_active_ids: Tuple[int, ...] = self.core_event_ids
        self._last_prediction_ids: Tuple[int, ...] = ()

    def step(
        self,
        *,
        external_event_ids: Sequence[int] = (),
        reactivation_hints: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        self._clock_ms += 25.0
        previous_active_ids = self.active_event_ids()
        if not previous_active_ids and self._last_active_ids:
            previous_active_ids = self._last_active_ids

        memory_ids = self._memory_hint_ids(reactivation_hints)
        predicted_ids = self.predictor.predict(
            previous_active_ids or self.core_event_ids,
            limit=self.config.max_self_ids,
        )
        spontaneous_ids = self._spontaneous_ids(
            previous_active_ids,
            memory_ids,
            predicted_ids,
        )
        recurrent_ids = _bounded_unique_ints(
            tuple(previous_active_ids) + tuple(spontaneous_ids) + tuple(predicted_ids),
            self.config.max_self_ids,
        )
        input_events = self._weighted_event_ids(
            tuple(int(value) for value in external_event_ids) + tuple(memory_ids),
            gating_factor=0.0,
            memory_count=len(memory_ids),
        )
        recurrent_events = self._weighted_event_ids(
            recurrent_ids,
            gating_factor=self.oscillation.get_gating_factor(self._clock_ms),
            memory_count=0,
        )

        leak_trace = self.leak_state.update(
            input_events=input_events,
            recurrent_events=recurrent_events,
            input_strength=1.0,
            recurrent_strength=1.0,
        )
        current_active_ids = self.active_event_ids()
        if not current_active_ids:
            current_active_ids = spontaneous_ids or self.core_event_ids

        self.predictor.observe(previous_active_ids, current_active_ids)
        self._update_persistence(current_active_ids)
        self._last_active_ids = tuple(current_active_ids)
        self._last_prediction_ids = tuple(predicted_ids)
        continuity = _jaccard(previous_active_ids, current_active_ids)
        self_state_ids = self.self_state_ids()
        return {
            "schema": "sara-persistent-self-state-step-v1",
            "clock_ms": float(self._clock_ms),
            "external_event_ids": list(int(value) for value in external_event_ids),
            "memory_event_ids": list(memory_ids),
            "predicted_event_ids": list(predicted_ids),
            "spontaneous_event_ids": list(spontaneous_ids),
            "recurrent_event_ids": list(recurrent_ids),
            "current_active_ids": list(current_active_ids),
            "self_state_ids": list(self_state_ids),
            "continuity_score": float(continuity),
            "idle_self_state_ok": bool(
                not external_event_ids and continuity >= self.config.idle_continuity_target
            ),
            "leak_trace": leak_trace,
        }

    def active_event_ids(self) -> Tuple[int, ...]:
        active = self.leak_state.active_events()
        ranked: List[Tuple[float, int]] = []
        for event_id in {
            int(item)
            for values in active.values()
            for item in values
        }:
            values = self.leak_state.read_event(event_id)
            ranked.append((sum(abs(float(value)) for value in values.values()), int(event_id)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(event_id for _, event_id in ranked[: self.config.max_self_ids])

    def self_state_ids(self) -> Tuple[int, ...]:
        ranked = sorted(
            self._persistence.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(event_id for event_id, score in ranked[: self.config.max_self_ids] if score > 0.0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema": "sara-persistent-self-state-v1",
            "clock_ms": float(self._clock_ms),
            "core_event_ids": list(self.core_event_ids),
            "last_active_ids": list(self._last_active_ids),
            "last_prediction_ids": list(self._last_prediction_ids),
            "self_state_ids": list(self.self_state_ids()),
            "persistence": {
                str(event_id): float(score)
                for event_id, score in sorted(self._persistence.items())
            },
            "predictor": self.predictor.snapshot(),
            "leak_state": self.leak_state.snapshot(),
        }

    def _memory_hint_ids(
        self,
        reactivation_hints: Sequence[Mapping[str, Any]],
    ) -> Tuple[int, ...]:
        ranked: List[Tuple[float, int]] = []
        for hint in reactivation_hints:
            entry_id = str(hint.get("entry_id", "") or hint.get("source_ref", "") or "")
            if not entry_id:
                continue
            activation = _clamp01(hint.get("activation", 0.0) or 0.0)
            ranked.append((activation, stable_self_state_id(entry_id)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(event_id for _, event_id in ranked[: self.config.max_self_ids])

    def _spontaneous_ids(
        self,
        previous_active_ids: Sequence[int],
        memory_ids: Sequence[int],
        predicted_ids: Sequence[int],
    ) -> Tuple[int, ...]:
        seeds = self.self_state_ids() or self.core_event_ids
        ranked = list(seeds)
        ranked.extend(int(value) for value in previous_active_ids)
        ranked.extend(int(value) for value in memory_ids)
        ranked.extend(int(value) for value in predicted_ids)
        spontaneous = _bounded_unique_ints(ranked, self.config.max_self_ids)
        return spontaneous[: max(1, self.config.spontaneous_floor)]

    def _weighted_event_ids(
        self,
        event_ids: Sequence[int],
        *,
        gating_factor: float,
        memory_count: int,
    ) -> Tuple[int, ...]:
        weighted: List[int] = []
        tonic_repeats = max(1, int(round(self.config.tonic_boost * 2.0)))
        recurrent_repeats = max(1, int(round(self.config.recurrent_boost * (1.0 + max(0.0, gating_factor)))))
        predictive_repeats = max(1, int(round(self.config.predictive_boost * 2.0)))
        memory_repeats = max(1, int(round(self.config.memory_boost * 2.0)))
        for index, event_id in enumerate(event_ids):
            repeats = recurrent_repeats
            if index < max(0, int(memory_count)):
                repeats = max(repeats, memory_repeats)
            elif int(event_id) in self.core_event_ids:
                repeats = max(repeats, tonic_repeats)
            elif int(event_id) in self._last_prediction_ids:
                repeats = max(repeats, predictive_repeats)
            weighted.extend([int(event_id)] * repeats)
        return tuple(weighted)

    def _update_persistence(self, current_active_ids: Sequence[int]) -> None:
        retained: Dict[int, float] = {}
        active = {int(value) for value in current_active_ids}
        for event_id, score in self._persistence.items():
            decayed = float(score) * float(self.config.persistence_decay)
            if event_id in active:
                decayed += 1.0
            if decayed > 0.01:
                retained[int(event_id)] = decayed
        for event_id in active:
            retained[int(event_id)] = float(retained.get(int(event_id), 0.0)) + 1.0
        for event_id in self.core_event_ids:
            retained[int(event_id)] = max(
                float(retained.get(int(event_id), 0.0)),
                float(self.config.tonic_boost),
            )
        ranked = sorted(retained.items(), key=lambda item: (-item[1], item[0]))
        self._persistence = {
            int(event_id): float(score)
            for event_id, score in ranked[: self.config.max_state_units]
        }


def evaluate_persistent_self_state() -> Dict[str, Any]:
    controller = PersistentSelfStateController(core_event_ids=(101, 202))
    prime_a = controller.step(external_event_ids=(101,))
    prime_b = controller.step(external_event_ids=(202,))
    idle_a = controller.step()
    idle_b = controller.step()
    replay_hint = controller.step(
        reactivation_hints=(
            {
                "entry_id": "episodic-memory-anchor",
                "activation": 0.9,
                "mutates_durable_state": False,
            },
        )
    )
    prediction_probe = controller.step(external_event_ids=(101,))
    prediction_idle = controller.step()
    persistent_nonzero = bool(idle_a["current_active_ids"]) and bool(idle_b["current_active_ids"])
    continuity_ok = (
        float(idle_a["continuity_score"]) >= 0.20
        and float(idle_b["continuity_score"]) >= 0.20
    )
    memory_reactivation_ok = bool(replay_hint["memory_event_ids"]) and bool(replay_hint["current_active_ids"])
    predictive_reactivation_ok = bool(prediction_idle["predicted_event_ids"])
    return {
        "observed_only": True,
        "metrics": {
            "persistent_self_state_idle_activity": 1.0 if persistent_nonzero else 0.0,
            "persistent_self_state_continuity": 1.0 if continuity_ok else 0.0,
            "persistent_self_state_memory_reactivation": 1.0 if memory_reactivation_ok else 0.0,
            "persistent_self_state_internal_prediction": 1.0 if predictive_reactivation_ok else 0.0,
        },
        "traces": {
            "prime_a": prime_a,
            "prime_b": prime_b,
            "idle_a": idle_a,
            "idle_b": idle_b,
            "replay_hint": replay_hint,
            "prediction_probe": prediction_probe,
            "prediction_idle": prediction_idle,
            "snapshot": controller.snapshot(),
        },
    }
