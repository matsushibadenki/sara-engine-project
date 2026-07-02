from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from sara_engine.dynamics import PersistentSelfStateController, memory_self_state_alignment
from sara_engine.learning.astro_modulator import AstroReplayModulator
from sara_engine.memory.event_state_cache import (
    EventStateEntry,
    VerifiedHierarchicalEventStateCache,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _hint_activation(
    entry: EventStateEntry,
    hints: Sequence[Mapping[str, Any]],
) -> float:
    best = 0.0
    for hint in hints:
        hint_entry_id = str(hint.get("entry_id", "") or "")
        hint_source_ref = str(hint.get("source_ref", "") or "")
        if hint_entry_id and hint_entry_id == entry.entry_id:
            best = max(best, _clamp01(hint.get("activation", 0.0) or 0.0))
        elif hint_source_ref and hint_source_ref == entry.source_ref:
            best = max(best, _clamp01(hint.get("activation", 0.0) or 0.0))
    return best


def _event_cost(entry: EventStateEntry) -> int:
    return max(1, len(entry.signature) + len(entry.causal_predecessors))


def _multimodal_bundle_affinity(entry: EventStateEntry) -> float:
    if str(entry.entry_id).startswith("bundle:") or str(entry.own_latent_id).startswith("bundle:"):
        return 1.0
    if str(entry.source_ref).startswith("bundle::"):
        return 0.8
    return 0.0


@dataclass(frozen=True)
class IdleReplayConfig:
    max_candidates: int = 3
    top_k_trace: int = 8
    event_budget: int = 24
    min_replay_score: float = 0.45


def plan_idle_replay(
    cache: VerifiedHierarchicalEventStateCache,
    *,
    persistent_self_state: PersistentSelfStateController | None = None,
    reactivation_hints: Sequence[Mapping[str, Any]] = (),
    astro_modulator: AstroReplayModulator | None = None,
    now_segment: int | None = None,
    config: IdleReplayConfig | None = None,
) -> Dict[str, Any]:
    """Select bounded replay candidates during idle periods.

    The selector treats replay as an internal maintenance action:
    verified Event Memory stays authoritative, while the current bounded
    self-state only biases which memories are most worth reactivating now.
    """

    cfg = config or IdleReplayConfig()
    if now_segment is not None:
        cache.expire(now_segment)

    self_state_trace = (
        persistent_self_state.step(reactivation_hints=reactivation_hints)
        if persistent_self_state is not None
        else {
            "self_state_ids": [],
            "current_active_ids": [],
            "predicted_event_ids": [],
            "memory_event_ids": [],
            "continuity_score": 0.0,
            "idle_self_state_ok": False,
        }
    )
    active_self_state_ids = tuple(
        int(value)
        for value in (
            list(self_state_trace.get("self_state_ids", ()))
            + list(self_state_trace.get("current_active_ids", ()))
            + list(self_state_trace.get("predicted_event_ids", ()))
        )
    )

    scored: List[Dict[str, Any]] = []
    for entry in cache.entries.values():
        self_state_alignment = _clamp01(
            memory_self_state_alignment(
                own_latent_id=entry.own_latent_id,
                source_ref=entry.source_ref,
                self_state_ids=active_self_state_ids,
            )
        )
        hint_activation = _hint_activation(entry, reactivation_hints)
        temporal_relevance = (
            0.0
            if now_segment is None
            else 1.0 / float(1 + abs(int(now_segment) - int(entry.time_segment)))
        )
        access_factor = float(entry.access_count) / float(1 + max(0, int(entry.access_count)))
        sequence_support = _clamp01(entry.sequence_support_score)
        bundle_affinity = _multimodal_bundle_affinity(entry)
        base_score = _clamp01(
            0.31 * _clamp01(entry.utility)
            + 0.18 * _clamp01(entry.confidence)
            + 0.12 * _clamp01(entry.source_reliability)
            + 0.12 * sequence_support
            + 0.14 * self_state_alignment
            + 0.06 * hint_activation
            + 0.02 * access_factor
            + 0.02 * temporal_relevance
            + 0.03 * bundle_affinity
        )
        replay_score = (
            _clamp01(astro_modulator.modulate_replay_weight(base_score))
            if astro_modulator is not None
            else base_score
        )
        scored.append(
            {
                "entry_id": entry.entry_id,
                "source_ref": entry.source_ref,
                "tier": entry.tier,
                "own_latent_id": entry.own_latent_id,
                "time_segment": int(entry.time_segment),
                "event_cost": _event_cost(entry),
                "base_score": round(base_score, 6),
                "replay_score": round(replay_score, 6),
                "suggested_action": "consolidate" if replay_score >= 0.75 else "replay",
                "components": {
                    "utility": round(_clamp01(entry.utility), 6),
                    "confidence": round(_clamp01(entry.confidence), 6),
                    "source_reliability": round(_clamp01(entry.source_reliability), 6),
                    "sequence_support": round(sequence_support, 6),
                    "multimodal_bundle_affinity": round(bundle_affinity, 6),
                    "self_state_alignment": round(self_state_alignment, 6),
                    "hint_activation": round(hint_activation, 6),
                    "access_factor": round(access_factor, 6),
                    "temporal_relevance": round(temporal_relevance, 6),
                },
                "selected_branch": f"{entry.tier}:{'bundle' if bundle_affinity > 0.0 else ('self_state' if self_state_alignment > 0.0 else 'memory')}",
                "mutates_durable_state": False,
            }
        )

    ranked = sorted(
        scored,
        key=lambda item: (
            -float(item["replay_score"]),
            -float(item["components"]["multimodal_bundle_affinity"]),
            -float(item["components"]["self_state_alignment"]),
            -float(item["components"]["sequence_support"]),
            item["entry_id"],
        ),
    )

    selected: List[Dict[str, Any]] = []
    total_event_cost = 0
    for candidate in ranked:
        if len(selected) >= cfg.max_candidates:
            break
        candidate_cost = int(candidate["event_cost"])
        if float(candidate["replay_score"]) < cfg.min_replay_score:
            continue
        if total_event_cost + candidate_cost > cfg.event_budget:
            continue
        selected.append(candidate)
        total_event_cost += candidate_cost

    top_trace = ranked[: max(1, int(cfg.top_k_trace))]
    selected_alignment = max(
        [float(item["components"]["self_state_alignment"]) for item in selected] + [0.0]
    )
    selected_hint = max(
        [float(item["components"]["hint_activation"]) for item in selected] + [0.0]
    )
    selected_bundle_affinity = max(
        [float(item["components"]["multimodal_bundle_affinity"]) for item in selected] + [0.0]
    )
    metrics = {
        "idle_replay_candidate_selection_observed": 1.0 if selected else 0.0,
        "idle_replay_budget_observed": 1.0 if total_event_cost <= cfg.event_budget else 0.0,
        "idle_replay_self_state_alignment_observed": 1.0 if selected_alignment > 0.0 else 0.0,
        "idle_replay_memory_reactivation_observed": 1.0
        if selected_hint > 0.0 or bool(self_state_trace.get("memory_event_ids", ()))
        else 0.0,
        "idle_replay_multimodal_bundle_observed": 1.0 if selected_bundle_affinity > 0.0 else 0.0,
        "idle_replay_state_continuity_observed": 1.0
        if bool(self_state_trace.get("idle_self_state_ok", False))
        else 0.0,
    }
    return {
        "observed_only": True,
        "config": {
            "max_candidates": int(cfg.max_candidates),
            "top_k_trace": int(cfg.top_k_trace),
            "event_budget": int(cfg.event_budget),
            "min_replay_score": float(cfg.min_replay_score),
        },
        "self_state_trace": self_state_trace,
        "astro_state": astro_modulator.snapshot() if astro_modulator is not None else None,
        "candidates": top_trace,
        "selected": selected,
        "total_event_cost": int(total_event_cost),
        "event_budget_ok": bool(total_event_cost <= cfg.event_budget),
        "metrics": metrics,
    }
