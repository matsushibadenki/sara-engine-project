"""Observed-only repetition-aware reranking for verified cache candidates."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from sara_engine.learning.repetition_consolidation import (
    RepetitionDependentConsolidator,
)
from sara_engine.memory.event_state_cache import CacheRetrievalResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class CandidateRepetitionReranker:
    """Rerank verified retrieval candidates without mutating durable state."""

    def __init__(
        self,
        consolidator: RepetitionDependentConsolidator,
        *,
        enabled: bool = False,
        max_boost: float = 0.15,
        max_matches: int = 16,
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not math.isfinite(float(max_boost)) or not 0.0 <= float(
            max_boost
        ) <= 1.0:
            raise ValueError("max_boost must be between 0.0 and 1.0")
        if (
            isinstance(max_matches, bool)
            or not isinstance(max_matches, int)
            or max_matches < 1
        ):
            raise ValueError("max_matches must be a positive integer")
        self.consolidator = consolidator
        self.enabled = enabled
        self.max_boost = float(max_boost)
        self.max_matches = max_matches
        self.last_trace: Tuple[Dict[str, Any], ...] = ()

    def observe(
        self,
        *,
        entry_id: str,
        timestep: int,
        source_ref: str,
        recall_success: bool,
        verified: bool,
        contradiction: bool = False,
    ) -> Dict[str, Any]:
        """Record a local retrieval outcome in the bounded consolidator."""
        return self.consolidator.observe(
            memory_id=entry_id,
            timestep=timestep,
            source_ref=source_ref,
            outcome="contradiction" if contradiction else "support",
            recall_success=bool(recall_success and not contradiction),
            verified=bool(verified and not contradiction),
        )

    def rerank(
        self,
        retrieval: CacheRetrievalResult,
        *,
        timestep: Optional[int] = None,
    ) -> CacheRetrievalResult:
        """Return a candidate-only reranked result under an explicit flag."""
        if not self.enabled:
            self.last_trace = ()
            return retrieval
        if len(retrieval.matches) > self.max_matches:
            raise ValueError("retrieval match count exceeds reranker ceiling")

        target_timestep = (
            self.consolidator.clock
            if timestep is None
            else timestep
        )
        reranked: List[Dict[str, Any]] = []
        traces: List[Dict[str, Any]] = []
        for original_position, match in enumerate(retrieval.matches):
            entry_id = str(match.get("entry_id", ""))
            base_score = _clamp01(float(match.get("score", 0.0)))
            state = self.consolidator.read(
                entry_id,
                timestep=target_timestep,
            )
            eligible = bool(
                state is not None
                and float(state["verification_strength"]) > 0.0
            )
            retrieval_strength = (
                float(state["retrieval_strength"]) if eligible else 0.0
            )
            stability = float(state["stability"]) if eligible else 0.0
            memory_signal = _clamp01(
                0.65 * retrieval_strength + 0.35 * stability
            )
            boost = (
                self.max_boost * memory_signal * (1.0 - base_score)
                if eligible
                else 0.0
            )
            candidate_score = _clamp01(base_score + boost)
            components = dict(match.get("components", {}))
            components.update(
                {
                    "repetition_candidate_enabled": 1.0,
                    "repetition_candidate_eligible": float(eligible),
                    "repetition_retrieval_strength": round(
                        retrieval_strength, 6
                    ),
                    "repetition_stability": round(stability, 6),
                    "repetition_candidate_boost": round(boost, 6),
                }
            )
            reranked.append(
                {
                    **dict(match),
                    "score": round(candidate_score, 6),
                    "components": components,
                }
            )
            traces.append(
                {
                    "entry_id": entry_id,
                    "original_position": original_position,
                    "base_score": round(base_score, 6),
                    "candidate_score": round(candidate_score, 6),
                    "eligible": eligible,
                    "verification_strength": (
                        round(float(state["verification_strength"]), 6)
                        if state is not None
                        else 0.0
                    ),
                    "mutates_durable_state": False,
                }
            )

        reranked.sort(
            key=lambda match: (
                -float(match["score"]),
                -float(match.get("utility", 0.0)),
                str(match.get("entry_id", "")),
            )
        )
        hint_by_id = {
            str(hint.get("entry_id", "")): dict(hint)
            for hint in retrieval.reactivation_hints
        }
        hints = tuple(
            {
                **hint_by_id.get(str(match["entry_id"]), {}),
                "entry_id": str(match["entry_id"]),
                "activation": float(match["score"]),
                "candidate_repetition_rerank": True,
                "mutates_durable_state": False,
            }
            for match in reranked
        )
        self.last_trace = tuple(traces)
        return CacheRetrievalResult(
            abstained=not reranked,
            decision=(
                "abstain_insufficient_evidence"
                if not reranked
                else "retrieve_verified_candidate_repetition_rerank"
            ),
            matches=tuple(reranked),
            event_cost=retrieval.event_cost + len(reranked),
            scanned_entries=retrieval.scanned_entries,
            reactivation_hints=hints,
        )
