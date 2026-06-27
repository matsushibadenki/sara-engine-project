from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.dynamics import PersistentSelfStateController
from sara_engine.ingest import CandidateRelation, FrequentSequence
from sara_engine.learning.astro_modulator import AstroReplayModulator
from sara_engine.learning.delta_retention_policy import (
    DeltaRetentionPolicyConfig,
    build_delta_retention_events,
    evaluate_delta_retention_policy,
)
from sara_engine.learning.idle_replay import IdleReplayConfig, plan_idle_replay
from sara_engine.learning.memory_phase import (
    MemoryPhaseConfig,
    build_memory_phase_observations,
    evaluate_memory_phase_transitions,
)
from sara_engine.learning.sleep_consolidation import (
    SleepConsolidationConfig,
    evaluate_sleep_consolidation,
)
from sara_engine.memory.concept_admission import ConceptRevalidationEntry
from sara_engine.memory.concept_review_loop import (
    ConceptReviewLoop,
    ConceptReviewLoopResult,
)
from sara_engine.memory.event_state_cache import VerifiedHierarchicalEventStateCache


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _is_concept_key(value: str) -> bool:
    text = str(value)
    return ":" in text and "->" in text


@dataclass(frozen=True)
class IdleConsolidationLoopResult:
    idle_replay_report: Dict[str, Any]
    sleep_consolidation_report: Dict[str, Any]
    memory_phase_report: Dict[str, Any]
    delta_retention_policy_report: Dict[str, Any]
    concept_review_result: ConceptReviewLoopResult
    prioritized_concept_keys: Tuple[str, ...]
    cache_refresh: Tuple[Dict[str, Any], ...]
    schema: str = "sara-idle-consolidation-loop-result-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "idle_replay_report": dict(self.idle_replay_report),
            "sleep_consolidation_report": dict(self.sleep_consolidation_report),
            "memory_phase_report": dict(self.memory_phase_report),
            "delta_retention_policy_report": dict(self.delta_retention_policy_report),
            "concept_review_result": self.concept_review_result.to_dict(),
            "prioritized_concept_keys": list(self.prioritized_concept_keys),
            "cache_refresh": [dict(item) for item in self.cache_refresh],
        }


class IdleConsolidationLoop:
    """Connects idle replay selection to consolidation and concept review."""

    def __init__(
        self,
        *,
        concept_review_loop: ConceptReviewLoop | None = None,
    ) -> None:
        self.concept_review_loop = concept_review_loop or ConceptReviewLoop()

    def run(
        self,
        cache: VerifiedHierarchicalEventStateCache,
        queue_entries: Sequence[ConceptRevalidationEntry],
        relations: Sequence[CandidateRelation],
        *,
        current_segment: int,
        frequent_sequences: Sequence[FrequentSequence] = (),
        persistent_self_state: PersistentSelfStateController | None = None,
        reactivation_hints: Sequence[Mapping[str, Any]] = (),
        astro_modulator: AstroReplayModulator | None = None,
        replay_config: IdleReplayConfig | None = None,
        sleep_config: SleepConsolidationConfig | None = None,
        memory_phase_config: MemoryPhaseConfig | None = None,
        delta_retention_config: DeltaRetentionPolicyConfig | None = None,
        apply_cache_refresh: bool = True,
    ) -> IdleConsolidationLoopResult:
        idle_replay_report = plan_idle_replay(
            cache,
            persistent_self_state=persistent_self_state,
            reactivation_hints=reactivation_hints,
            astro_modulator=astro_modulator,
            now_segment=current_segment,
            config=replay_config,
        )
        prioritized_concept_keys = tuple(
            item.get("own_latent_id", "")
            for item in idle_replay_report.get("selected", ())
            if _is_concept_key(str(item.get("own_latent_id", "")))
        )
        ordered_queue = self._prioritize_queue(queue_entries, prioritized_concept_keys)
        concept_review_result = self.concept_review_loop.run(
            ordered_queue,
            relations,
            current_segment=current_segment,
            frequent_sequences=frequent_sequences,
            persistent_self_state=persistent_self_state,
        )
        sleep_events = self._sleep_events_from_idle_replay(
            idle_replay_report.get("selected", ()),
            concept_review_result=concept_review_result,
        )
        sleep_consolidation_report = evaluate_sleep_consolidation(
            sleep_events,
            config=sleep_config,
        )
        memory_phase_report = evaluate_memory_phase_transitions(
            build_memory_phase_observations(sleep_events, step=current_segment),
            config=memory_phase_config,
        )
        phase_by_memory_id = {
            str(track.get("memory_id", "")): str(track.get("final_phase", "liquid"))
            for track in memory_phase_report.get("phase_tracks", ())
        }
        phase_enriched_sleep_events = [
            {
                **event,
                "phase": phase_by_memory_id.get(str(event.get("memory_id", "")), "liquid"),
            }
            for event in sleep_events
        ]
        astro_stability = (
            float(astro_modulator.snapshot().get("stability_level", 1.0))
            if astro_modulator is not None
            else 1.0
        )
        delta_retention_policy_report = evaluate_delta_retention_policy(
            build_delta_retention_events(
                phase_enriched_sleep_events,
                astro_stability=astro_stability,
            ),
            config=delta_retention_config,
        )
        cache_refresh = (
            tuple(
                item.to_dict()
                for item in cache.refresh_from_consolidation(phase_enriched_sleep_events)
            )
            if apply_cache_refresh
            else ()
        )
        return IdleConsolidationLoopResult(
            idle_replay_report=idle_replay_report,
            sleep_consolidation_report=sleep_consolidation_report,
            memory_phase_report=memory_phase_report,
            delta_retention_policy_report=delta_retention_policy_report,
            concept_review_result=concept_review_result,
            prioritized_concept_keys=prioritized_concept_keys,
            cache_refresh=cache_refresh,
        )

    def _prioritize_queue(
        self,
        queue_entries: Sequence[ConceptRevalidationEntry],
        prioritized_concept_keys: Sequence[str],
    ) -> Tuple[ConceptRevalidationEntry, ...]:
        priority = {key: index for index, key in enumerate(prioritized_concept_keys)}
        ordered = sorted(
            queue_entries,
            key=lambda entry: (
                0 if entry.concept_key in priority else 1,
                priority.get(entry.concept_key, 10_000),
                entry.retry_after_segment,
                entry.concept_key,
            ),
        )
        return tuple(ordered)

    def _sleep_events_from_idle_replay(
        self,
        selected_candidates: Sequence[Mapping[str, Any]],
        *,
        concept_review_result: ConceptReviewLoopResult,
    ) -> List[Dict[str, Any]]:
        admitted_keys = {
            candidate.own_latent_id
            for candidate in concept_review_result.admission_plan.admitted_candidates
            if candidate.own_latent_id
        }
        replay_events: List[Dict[str, Any]] = []
        for candidate in selected_candidates:
            components = dict(candidate.get("components", {}))
            replay_score = _clamp01(candidate.get("replay_score", 0.0) or 0.0)
            utility = _clamp01(components.get("utility", 0.0) or 0.0)
            confidence = _clamp01(components.get("confidence", 0.0) or 0.0)
            sequence_support = _clamp01(components.get("sequence_support", 0.0) or 0.0)
            self_state_alignment = _clamp01(components.get("self_state_alignment", 0.0) or 0.0)
            hint_activation = _clamp01(components.get("hint_activation", 0.0) or 0.0)
            baseline_retention = _clamp01(
                0.45 * utility
                + 0.30 * confidence
                + 0.15 * sequence_support
                + 0.10 * self_state_alignment
            )
            admission_bonus = 0.08 if str(candidate.get("own_latent_id", "")) in admitted_keys else 0.0
            post_retention = _clamp01(
                baseline_retention
                + 0.12 * replay_score
                + 0.06 * hint_activation
                + admission_bonus
            )
            baseline_noise = _clamp01(
                1.0 - (0.55 * utility + 0.25 * sequence_support + 0.20 * confidence)
            )
            post_noise = _clamp01(
                baseline_noise
                - (0.18 * replay_score + 0.08 * self_state_alignment + admission_bonus)
            )
            health_before = _clamp01(0.60 * utility + 0.25 * confidence + 0.15 * sequence_support)
            health_after = _clamp01(
                health_before
                + 0.10 * replay_score
                + 0.05 * self_state_alignment
                + admission_bonus
            )
            replay_events.append(
                {
                    "memory_id": str(candidate.get("entry_id", "")),
                    "baseline_retention": baseline_retention,
                    "post_retention": post_retention,
                    "baseline_noise": baseline_noise,
                    "post_noise": post_noise,
                    "health_before": health_before,
                    "health_after": health_after,
                    "event_cost": float(candidate.get("event_cost", 0)),
                    "latent_branch_count": 3
                    if str(candidate.get("own_latent_id", "")) in admitted_keys
                    else (2 if replay_score >= 0.75 else 1),
                    "selected_branch": str(candidate.get("selected_branch", "")),
                }
            )
        return replay_events
