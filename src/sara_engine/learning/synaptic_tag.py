# Directory Path: src/sara_engine/learning/synaptic_tag.py
# English Title: Gradient-Free Synaptic Tagging
# Purpose/Content: Scores synaptic traces from local spike statistics for replay, pruning, and consolidation decisions.

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class SynapticTagConfig:
    coincidence_window: int = 2
    consolidate_threshold: float = 0.72
    replay_threshold: float = 0.50
    prune_threshold: float = 0.35
    min_persistent_weight: float = 0.20
    state_budget: int = 16


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _coefficient_of_variation(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) <= 1e-12:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return sqrt(variance) / abs(mean)


def _event_pair(event: Dict[str, Any]) -> Tuple[str, str]:
    return str(event.get("pre_id", event.get("pre", ""))), str(event.get("post_id", event.get("post", "")))


def _event_step(event: Dict[str, Any]) -> int:
    return int(event.get("step", event.get("time", 0)) or 0)


def evaluate_synaptic_tags(
    events: Iterable[Dict[str, Any]],
    config: SynapticTagConfig | None = None,
) -> Dict[str, Any]:
    """Evaluate local synaptic importance without gradients or dense global scans."""

    cfg = config or SynapticTagConfig()
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for event in events:
        pair = _event_pair(event)
        if not pair[0] or not pair[1]:
            continue
        grouped.setdefault(pair, []).append(dict(event))

    tags: List[Dict[str, Any]] = []
    for pair, pair_events in grouped.items():
        ordered = sorted(pair_events, key=_event_step)
        steps = [_event_step(event) for event in ordered]
        intervals = [float(b - a) for a, b in zip(steps, steps[1:]) if b >= a]
        isi_cv = _coefficient_of_variation(intervals)
        isi_stability = _clamp01(1.0 / (1.0 + isi_cv))

        coincidence_hits = 0
        replay_total = 0
        replay_useful = 0
        weights: List[float] = []
        for event in ordered:
            pre_step = int(event.get("pre_step", _event_step(event)) or 0)
            post_step = int(event.get("post_step", _event_step(event)) or 0)
            if 0 <= post_step - pre_step <= cfg.coincidence_window:
                coincidence_hits += 1
            weights.append(float(event.get("weight", 0.0) or 0.0))
            if bool(event.get("replayed", False)) or "replay_useful" in event:
                replay_total += 1
                if bool(event.get("replay_useful", False)):
                    replay_useful += 1

        pre_post_correlation = _clamp01(coincidence_hits / max(len(ordered), 1))
        max_weight = max([abs(weight) for weight in weights] + [1e-9])
        latest_weight = abs(weights[-1]) if weights else 0.0
        persistent_fraction = sum(
            1 for weight in weights if abs(weight) >= cfg.min_persistent_weight
        ) / max(len(weights), 1)
        weight_persistence = _clamp01(0.5 * (latest_weight / max_weight) + 0.5 * persistent_fraction)
        recent_replay_usefulness = _clamp01(replay_useful / max(replay_total, 1)) if replay_total else 0.0

        importance_score = _clamp01(
            0.25 * isi_stability
            + 0.25 * pre_post_correlation
            + 0.25 * weight_persistence
            + 0.25 * recent_replay_usefulness
        )
        if importance_score >= cfg.consolidate_threshold:
            tag = "consolidate"
        elif importance_score >= cfg.replay_threshold:
            tag = "replay"
        elif importance_score <= cfg.prune_threshold:
            tag = "prune"
        else:
            tag = "watch"

        tags.append(
            {
                "pre_id": pair[0],
                "post_id": pair[1],
                "tag": tag,
                "importance_score": importance_score,
                "replay_priority": _clamp01(0.65 * importance_score + 0.35 * recent_replay_usefulness),
                "pruning_candidate": tag == "prune",
                "components": {
                    "isi_cv": float(isi_cv),
                    "isi_stability": isi_stability,
                    "pre_post_correlation": pre_post_correlation,
                    "weight_persistence": weight_persistence,
                    "recent_replay_usefulness": recent_replay_usefulness,
                    "event_count": len(ordered),
                },
            }
        )

    tags.sort(key=lambda item: (-float(item["importance_score"]), str(item["pre_id"]), str(item["post_id"])))
    top = tags[0] if tags else {}
    weak = tags[-1] if tags else {}
    metrics = {
        "synaptic_tag_importance_score_observed": 1.0
        if tags and str(top.get("tag")) in {"consolidate", "replay"}
        else 0.0,
        "synaptic_tag_replay_priority_observed": 1.0
        if tags and float(top.get("replay_priority", 0.0)) >= cfg.replay_threshold
        else 0.0,
        "synaptic_tag_pruning_candidate_observed": 1.0
        if tags and any(bool(item.get("pruning_candidate", False)) for item in tags)
        else 0.0,
        "synaptic_tag_state_budget_observed": 1.0 if len(tags) <= cfg.state_budget else 0.0,
        "synaptic_tag_integrity": 1.0
        if tags
        and float(top.get("importance_score", 0.0)) > float(weak.get("importance_score", 0.0))
        and str(top.get("tag")) == "consolidate"
        and any(bool(item.get("pruning_candidate", False)) for item in tags)
        and len(tags) <= cfg.state_budget
        else 0.0,
    }
    return {
        "observed_only": True,
        "config": {
            "coincidence_window": cfg.coincidence_window,
            "consolidate_threshold": cfg.consolidate_threshold,
            "replay_threshold": cfg.replay_threshold,
            "prune_threshold": cfg.prune_threshold,
            "min_persistent_weight": cfg.min_persistent_weight,
            "state_budget": cfg.state_budget,
        },
        "tags": tags,
        "metrics": metrics,
    }
