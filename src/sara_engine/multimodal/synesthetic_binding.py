from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from sara_engine.learning.own_latent import build_sparse_signature, stable_event_id, tokenize_sparse_text
from sara_engine.nn.common_spike_space import SparseSpikeEvent


SUPPORTED_MODALITIES = frozenset({"language", "vision", "audio", "tactile", "interoception"})
MODALITY_ALIASES = {
    "text": "language",
    "image": "vision",
    "sensor": "tactile",
    "body": "interoception",
    "internal": "interoception",
}


def normalize_modality(value: str) -> str:
    normalized = MODALITY_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if normalized not in SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported modality: {value}")
    return normalized


@dataclass(frozen=True)
class SparseMultimodalEvent:
    modality: str
    timestamp_ms: float
    time_chunk_id: int
    source_id: str
    sparse_signature: Tuple[int, ...]
    confidence: float
    uncertainty: float
    event_cost: int
    observed: bool = True
    label: str = ""
    source_ref: str = ""
    latent_cluster_id: str = ""
    specialization_factors: Tuple[str, ...] = ()
    event_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "modality_id": self.modality,
            "modality": self.modality,
            "timestamp_ms": self.timestamp_ms,
            "time_chunk_id": self.time_chunk_id,
            "source_id": self.source_id,
            "sparse_signature": list(self.sparse_signature),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "event_cost": self.event_cost,
            "observed": self.observed,
            "label": self.label,
            "source_ref": self.source_ref,
            "latent_cluster_id": self.latent_cluster_id,
            "specialization_factors": list(self.specialization_factors),
        }


@dataclass(frozen=True)
class BindingAuditRecord:
    event_id: str
    time_chunk_id: int
    evidence_types: Tuple[str, ...]
    modality_ids: Tuple[str, ...]
    source_ids: Tuple[str, ...]
    confidence: float
    binding_strength: float
    payload_separable: bool
    admitted: bool
    reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "time_chunk_id": self.time_chunk_id,
            "evidence_types": list(self.evidence_types),
            "modality_ids": list(self.modality_ids),
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "binding_strength": self.binding_strength,
            "payload_separable": self.payload_separable,
            "admitted": self.admitted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SparseEventBundle:
    event_id: str
    time_chunk_id: int
    modality_ids: Tuple[str, ...]
    binding_strength: float
    uncertainty: float
    evidence_types: Tuple[str, ...]
    child_records: Tuple[SparseMultimodalEvent, ...]
    route_trace: Tuple[Dict[str, Any], ...] = ()
    audit: Optional[BindingAuditRecord] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-synesthetic-event-bundle-v1",
            "event_id": self.event_id,
            "time_chunk_id": self.time_chunk_id,
            "modality_ids": list(self.modality_ids),
            "binding_strength": self.binding_strength,
            "uncertainty": self.uncertainty,
            "evidence_types": list(self.evidence_types),
            "route_trace": list(self.route_trace),
            "child_records": [record.to_dict() for record in self.child_records],
            "audit": self.audit.to_dict() if self.audit is not None else None,
        }


class SparseTemporalBinder:
    """Normalizes sparse events into deterministic bounded time chunks."""

    def __init__(self, *, window_ms: float = 32.0, max_events_per_chunk: int = 32) -> None:
        if window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if max_events_per_chunk <= 0:
            raise ValueError("max_events_per_chunk must be positive")
        self.window_ms = float(window_ms)
        self.max_events_per_chunk = int(max_events_per_chunk)

    def chunk_id(self, timestamp_ms: float) -> int:
        return int(max(0.0, float(timestamp_ms)) // self.window_ms)

    def normalize_event(
        self,
        *,
        modality: str,
        timestamp_ms: float,
        source_id: str,
        sparse_signature: Iterable[int],
        confidence: float = 1.0,
        uncertainty: Optional[float] = None,
        observed: bool = True,
        label: str = "",
        source_ref: str = "",
        latent_cluster_id: str = "",
        specialization_factors: Iterable[str] = (),
    ) -> SparseMultimodalEvent:
        signature = tuple(sorted(set(int(item) for item in sparse_signature)))[: self.max_events_per_chunk]
        bounded_confidence = max(0.0, min(1.0, float(confidence)))
        bounded_uncertainty = (
            max(0.0, min(1.0, float(uncertainty)))
            if uncertainty is not None
            else round(1.0 - bounded_confidence, 6)
        )
        normalized_modality = normalize_modality(modality)
        chunk_id = self.chunk_id(timestamp_ms)
        stable_id = f"{normalized_modality}:{source_id}:{chunk_id}:{stable_event_id((normalized_modality, source_id, chunk_id, signature), width=1_000_000):06d}"
        return SparseMultimodalEvent(
            modality=normalized_modality,
            timestamp_ms=float(timestamp_ms),
            time_chunk_id=chunk_id,
            source_id=str(source_id),
            sparse_signature=signature,
            confidence=bounded_confidence,
            uncertainty=bounded_uncertainty,
            event_cost=len(signature),
            observed=bool(observed),
            label=str(label),
            source_ref=str(source_ref),
            latent_cluster_id=str(latent_cluster_id),
            specialization_factors=tuple(
                sorted(set(str(item) for item in specialization_factors if str(item).strip()))
            ),
            event_id=stable_id,
        )

    def from_spike_events(
        self,
        events: Sequence[SparseSpikeEvent],
        *,
        source_id: str,
        timestep_ms: float = 1.0,
    ) -> List[SparseMultimodalEvent]:
        grouped: MutableMapping[Tuple[str, int], List[SparseSpikeEvent]] = defaultdict(list)
        for event in events:
            timestamp_ms = float(event.timestep) * float(timestep_ms)
            grouped[(normalize_modality(event.modality), self.chunk_id(timestamp_ms))].append(event)
        normalized: List[SparseMultimodalEvent] = []
        for (modality, chunk_id), bucket in sorted(grouped.items()):
            timestamp_ms = float(chunk_id) * self.window_ms
            confidence = sum(float(item.confidence) for item in bucket) / float(len(bucket))
            normalized.append(
                self.normalize_event(
                    modality=modality,
                    timestamp_ms=timestamp_ms,
                    source_id=source_id,
                    sparse_signature=[item.spike_id for item in bucket],
                    confidence=confidence,
                    label="common_spike_space",
                )
            )
        return normalized

    def bind(self, events: Sequence[SparseMultimodalEvent]) -> Dict[int, List[SparseMultimodalEvent]]:
        buckets: MutableMapping[int, List[SparseMultimodalEvent]] = defaultdict(list)
        for event in events:
            buckets[int(event.time_chunk_id)].append(event)
        return {
            chunk_id: sorted(
                bucket[: self.max_events_per_chunk],
                key=lambda item: (item.modality, item.source_id, item.sparse_signature),
            )
            for chunk_id, bucket in sorted(buckets.items())
        }

    def bundle_events(
        self,
        events: Sequence[SparseMultimodalEvent],
        *,
        route_trace: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[SparseEventBundle]:
        trace_rows = list(route_trace or ())
        bundles: List[SparseEventBundle] = []
        for chunk_id, bucket in self.bind(events).items():
            modalities = tuple(sorted({event.modality for event in bucket}))
            labels = [event.label for event in bucket if event.label]
            evidence_types = ["direct_synchrony"]
            if len(modalities) > 1:
                evidence_types.append("cross_modal_group")
            if labels and len(set(labels)) == 1:
                evidence_types.append("repeated_coactivation")
            if any(event.source_ref for event in bucket):
                evidence_types.append("source_backed")
            confidence = round(
                sum(float(event.confidence) for event in bucket) / float(max(1, len(bucket))),
                6,
            )
            binding_strength = round(
                min(
                    1.0,
                    confidence * (0.5 + (0.125 * len(modalities))) + (0.05 if "source_backed" in evidence_types else 0.0),
                ),
                6,
            )
            uncertainty = round(
                sum(float(event.uncertainty) for event in bucket) / float(max(1, len(bucket))),
                6,
            )
            payload_separable = len({(event.modality, event.source_id) for event in bucket}) == len(bucket)
            bundle_event_id = f"bundle:{chunk_id}:{stable_event_id(tuple(event.event_id for event in bucket), width=1_000_000):06d}"
            bundle_trace = tuple(
                dict(row)
                for row in trace_rows
                if int(row.get("time_chunk_id", chunk_id)) == chunk_id
                or str(row.get("source_id", "")) in {event.source_id for event in bucket}
            )
            reasons = [
                f"modality_count:{len(modalities)}",
                f"record_count:{len(bucket)}",
            ]
            if payload_separable:
                reasons.append("payloads_preserved_per_modality")
            audit = BindingAuditRecord(
                event_id=bundle_event_id,
                time_chunk_id=int(chunk_id),
                evidence_types=tuple(sorted(set(evidence_types))),
                modality_ids=modalities,
                source_ids=tuple(sorted(event.source_id for event in bucket)),
                confidence=confidence,
                binding_strength=binding_strength,
                payload_separable=payload_separable,
                admitted=bool(len(modalities) > 1 and payload_separable),
                reasons=tuple(reasons),
            )
            bundles.append(
                SparseEventBundle(
                    event_id=bundle_event_id,
                    time_chunk_id=int(chunk_id),
                    modality_ids=modalities,
                    binding_strength=binding_strength,
                    uncertainty=uncertainty,
                    evidence_types=tuple(sorted(set(evidence_types))),
                    child_records=tuple(bucket),
                    route_trace=bundle_trace,
                    audit=audit,
                )
            )
        return bundles


class SparsePluggableCorticalColumn:
    """Shared sparse cortical primitive with modality-agnostic local updates."""

    def __init__(
        self,
        *,
        activation_threshold: float = 0.8,
        learning_rate: float = 0.1,
        max_state_units: int = 256,
        homeostatic_clip: float = 1.5,
    ) -> None:
        self.activation_threshold = float(activation_threshold)
        self.learning_rate = max(0.0, min(1.0, float(learning_rate)))
        self.max_state_units = max(1, int(max_state_units))
        self.homeostatic_clip = max(0.1, float(homeostatic_clip))
        self.event_weights: MutableMapping[int, float] = defaultdict(float)
        self.coactivation: MutableMapping[Tuple[int, int], float] = defaultdict(float)
        self.update_count = 0

    def process(self, event: SparseMultimodalEvent, *, learn: bool = True) -> Dict[str, Any]:
        scores: List[Tuple[int, float]] = []
        signature = list(event.sparse_signature)
        for event_id in signature:
            score = float(event.confidence) + self.event_weights.get(int(event_id), 0.0)
            scores.append((int(event_id), score))
        active_ids = sorted(event_id for event_id, score in scores if score >= self.activation_threshold)

        if learn and signature:
            for event_id in signature:
                current = self.event_weights.get(int(event_id), 0.0) * 0.98 + self.learning_rate
                self.event_weights[int(event_id)] = max(
                    -self.homeostatic_clip,
                    min(self.homeostatic_clip, current),
                )
            for left in signature:
                for right in signature:
                    if left == right or len(self.coactivation) >= self.max_state_units:
                        continue
                    key = (int(left), int(right))
                    updated = self.coactivation.get(key, 0.0) * 0.98 + self.learning_rate
                    self.coactivation[key] = max(
                        -self.homeostatic_clip,
                        min(self.homeostatic_clip, updated),
                    )
            self.update_count += 1

        return {
            "modality": event.modality,
            "active_event_ids": active_ids,
            "input_event_count": len(signature),
            "event_cost": len(signature) + len(scores),
            "state_budget_units": self.state_budget_units(),
            "learning_rule": "shared_local_hebbian",
            "update_count": self.update_count,
        }

    def state_budget_units(self) -> int:
        return min(self.max_state_units, len(self.event_weights) + len(self.coactivation))


class SparseModalityAdapter:
    """Converts modality-specific feature names into the shared sparse event IR."""

    modality = ""
    feature_prefix = "feature"

    def __init__(self, *, width: int = 4096, max_events: int = 16) -> None:
        self.width = max(16, int(width))
        self.max_events = max(1, int(max_events))

    def feature_terms(self, value: Any) -> List[str]:
        if isinstance(value, str):
            return tokenize_sparse_text(value)
        if isinstance(value, Mapping):
            return [f"{key}:{item}" for key, item in sorted(value.items())]
        if isinstance(value, Iterable):
            return [str(item) for item in value]
        return [str(value)]

    def encode(
        self,
        value: Any,
        *,
        binder: SparseTemporalBinder,
        timestamp_ms: float,
        source_id: str,
        confidence: float = 1.0,
        label: str = "",
        source_ref: str = "",
        latent_cluster_id: str = "",
        latent_signature: Iterable[int] = (),
        topology_terms: Iterable[str] = (),
        gate_history_terms: Iterable[str] = (),
    ) -> SparseMultimodalEvent:
        terms = [
            f"{self.feature_prefix}:{term}"
            for term in self.feature_terms(value)
            if str(term).strip()
        ]
        signature = build_sparse_signature(
            terms,
            width=self.width,
            max_events=self.max_events,
        )
        latent_events = [self.width + int(item) for item in latent_signature][: self.max_events]
        topology_events = [
            (self.width * 2) + stable_event_id(str(item), width=self.width)
            for item in topology_terms
            if str(item).strip()
        ]
        gate_events = [
            (self.width * 3) + stable_event_id(str(item), width=self.width)
            for item in gate_history_terms
            if str(item).strip()
        ]
        combined = (signature + latent_events + topology_events + gate_events)[: self.max_events]
        factors = ["input_statistics", "timing_profile"]
        if topology_events:
            factors.append("topology")
        if latent_events:
            factors.append("own_latent")
        if gate_events:
            factors.append("gate_history")
        return binder.normalize_event(
            modality=self.modality,
            timestamp_ms=timestamp_ms,
            source_id=source_id,
            sparse_signature=combined,
            confidence=confidence,
            label=label,
            source_ref=source_ref,
            latent_cluster_id=latent_cluster_id,
            specialization_factors=factors,
        )


class LanguageEventAdapter(SparseModalityAdapter):
    modality = "language"
    feature_prefix = "token"


class VisionEventAdapter(SparseModalityAdapter):
    modality = "vision"
    feature_prefix = "visual"


class AudioEventAdapter(SparseModalityAdapter):
    modality = "audio"
    feature_prefix = "audio"


class TactileEventAdapter(SparseModalityAdapter):
    modality = "tactile"
    feature_prefix = "tactile"


class SparseSynestheticLinker:
    """Learns bounded cross-modal links from same-chunk sparse co-activation."""

    def __init__(self, *, max_links_per_event: int = 4, max_total_links: int = 256) -> None:
        self.max_links_per_event = max(1, int(max_links_per_event))
        self.max_total_links = max(1, int(max_total_links))
        self.link_counts: MutableMapping[Tuple[str, int, str, int], int] = defaultdict(int)

    def update(self, events: Sequence[SparseMultimodalEvent]) -> None:
        by_chunk: MutableMapping[int, List[SparseMultimodalEvent]] = defaultdict(list)
        for event in events:
            by_chunk[event.time_chunk_id].append(event)
        for bucket in by_chunk.values():
            for left in bucket:
                for right in bucket:
                    if left.modality == right.modality:
                        continue
                    for left_id in left.sparse_signature:
                        for right_id in right.sparse_signature:
                            if len(self.link_counts) >= self.max_total_links:
                                return
                            self.link_counts[(left.modality, left_id, right.modality, right_id)] += 1

    def predict(
        self,
        event: SparseMultimodalEvent,
        *,
        target_modality: str,
        min_link_count: int = 1,
    ) -> Dict[str, Any]:
        target = normalize_modality(target_modality)
        candidates: Counter[int] = Counter()
        for source_id in event.sparse_signature:
            links = [
                (target_id, count)
                for (source_modality, linked_source_id, target_name, target_id), count in self.link_counts.items()
                if source_modality == event.modality
                and linked_source_id == source_id
                and target_name == target
                and count >= min_link_count
            ]
            for target_id, count in sorted(links, key=lambda item: (-item[1], item[0]))[
                : self.max_links_per_event
            ]:
                candidates[target_id] += count
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        signature = [event_id for event_id, _count in ranked[: self.max_links_per_event]]
        total = sum(candidates.values())
        confidence = float(sum(count for _event_id, count in ranked[: self.max_links_per_event])) / float(
            max(1, total)
        )
        return {
            "source_modality": event.modality,
            "target_modality": target,
            "predicted_missing_modality_events": signature,
            "confidence": round(confidence, 6) if signature else 0.0,
            "uncertainty": round(1.0 - confidence, 6) if signature else 1.0,
            "abstained": not signature,
            "observed": False,
            "event_cost": len(event.sparse_signature) + len(candidates),
        }

    def state_budget_units(self) -> int:
        return len(self.link_counts)


@dataclass(frozen=True)
class ThalamicGateResult:
    routed_events: List[SparseMultimodalEvent]
    suppressed_count: int
    mode: str
    event_cost: int
    trace: List[Dict[str, Any]]


class SparseThalamicGate:
    """Selects sparse multimodal routes without dense softmax weighting."""

    def __init__(self, *, route_threshold: float = 0.35, max_routes: int = 16) -> None:
        self.route_threshold = float(route_threshold)
        self.max_routes = max(1, int(max_routes))

    def route(
        self,
        events: Sequence[SparseMultimodalEvent],
        *,
        mode: str = "equal",
        focused_modality: Optional[str] = None,
        route_hints: Optional[Mapping[str, float]] = None,
    ) -> ThalamicGateResult:
        if mode not in {"equal", "focused"}:
            raise ValueError("mode must be equal or focused")
        focus = normalize_modality(focused_modality) if focused_modality else None
        hints = dict(route_hints or {})
        ranked: List[Tuple[float, SparseMultimodalEvent, Dict[str, Any]]] = []
        for event in events:
            focus_gain = 0.2 if mode == "focused" and focus == event.modality else 0.0
            route_hint = max(-0.25, min(0.25, float(hints.get(event.source_id, 0.0))))
            cost_penalty = min(0.25, 0.01 * event.event_cost)
            score = event.confidence - event.uncertainty + focus_gain + route_hint - cost_penalty
            trace = {
                "modality": event.modality,
                "source_id": event.source_id,
                "score": round(score, 6),
                "focus_gain": focus_gain,
                "route_hint": round(route_hint, 6),
                "cost_penalty": round(cost_penalty, 6),
                "routed": score >= self.route_threshold,
            }
            ranked.append((score, event, trace))
        ranked.sort(key=lambda item: (-item[0], item[1].modality, item[1].source_id))
        routed_pairs = [item for item in ranked if item[0] >= self.route_threshold][: self.max_routes]
        routed_events = [item[1] for item in routed_pairs]
        routed_keys = {(item.modality, item.source_id, item.time_chunk_id) for item in routed_events}
        trace_rows = []
        for _score, event, trace in ranked:
            row = dict(trace)
            row["routed"] = (event.modality, event.source_id, event.time_chunk_id) in routed_keys
            trace_rows.append(row)
        return ThalamicGateResult(
            routed_events=routed_events,
            suppressed_count=max(0, len(events) - len(routed_events)),
            mode=mode,
            event_cost=len(events),
            trace=trace_rows,
        )
