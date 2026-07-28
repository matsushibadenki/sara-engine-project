from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sara_engine.learning.repetition_candidate_reranker import (
    CandidateRepetitionReranker,
)
from sara_engine.learning.repetition_consolidation import (
    RepetitionConsolidationConfig,
    RepetitionDependentConsolidator,
)
from sara_engine.memory.event_state_cache import CacheRetrievalResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _retrieval(
    target_score: float = 0.60,
    distractor_score: float = 0.62,
) -> CacheRetrievalResult:
    matches = (
        {
            "entry_id": "distractor",
            "score": distractor_score,
            "tier": "consolidated",
            "source_ref": "source-d",
            "utility": 0.70,
            "components": {"sparse_overlap": distractor_score},
        },
        {
            "entry_id": "target",
            "score": target_score,
            "tier": "consolidated",
            "source_ref": "source-t",
            "utility": 0.70,
            "components": {"sparse_overlap": target_score},
        },
    )
    return CacheRetrievalResult(
        abstained=False,
        decision="retrieve_verified",
        matches=matches,
        event_cost=10,
        scanned_entries=2,
        reactivation_hints=tuple(
            {
                "entry_id": match["entry_id"],
                "activation": match["score"],
                "mutates_durable_state": False,
            }
            for match in matches
        ),
    )


def _train_target(
    reranker: CandidateRepetitionReranker,
    *,
    verified: bool,
) -> None:
    for index, timestep in enumerate((0, 5, 10, 15)):
        reranker.observe(
            entry_id="target",
            timestep=timestep,
            source_ref="source-t",
            recall_success=index > 0,
            verified=verified,
        )


def test_candidate_flag_disabled_returns_original_retrieval() -> None:
    reranker = CandidateRepetitionReranker(
        RepetitionDependentConsolidator(),
        enabled=False,
    )
    retrieval = _retrieval()

    assert reranker.rerank(retrieval) is retrieval
    assert reranker.last_trace == ()


def test_verified_spaced_repetition_can_rerank_delayed_candidates() -> None:
    reranker = CandidateRepetitionReranker(
        RepetitionDependentConsolidator(),
        enabled=True,
    )
    _train_target(reranker, verified=True)

    result = reranker.rerank(_retrieval(), timestep=64)

    assert result.matches[0]["entry_id"] == "target"
    assert result.matches[0]["score"] > 0.60
    assert result.matches[0]["components"][
        "repetition_candidate_eligible"
    ] == 1.0
    assert result.event_cost == 12
    assert result.reactivation_hints[0]["mutates_durable_state"] is False


def test_unverified_repetition_cannot_change_candidate_order() -> None:
    reranker = CandidateRepetitionReranker(
        RepetitionDependentConsolidator(),
        enabled=True,
    )
    _train_target(reranker, verified=False)

    result = reranker.rerank(_retrieval(), timestep=64)

    assert result.matches[0]["entry_id"] == "distractor"
    target = next(
        match for match in result.matches if match["entry_id"] == "target"
    )
    assert target["score"] == 0.60
    assert target["components"]["repetition_candidate_eligible"] == 0.0
    assert target["components"]["repetition_candidate_boost"] == 0.0


def test_contradiction_reduces_candidate_boost() -> None:
    reranker = CandidateRepetitionReranker(
        RepetitionDependentConsolidator(),
        enabled=True,
    )
    _train_target(reranker, verified=True)
    before = reranker.rerank(
        _retrieval(target_score=0.60, distractor_score=0.20),
        timestep=15,
    )
    reranker.observe(
        entry_id="target",
        timestep=20,
        source_ref="source-c",
        recall_success=False,
        verified=False,
        contradiction=True,
    )

    after = reranker.rerank(
        _retrieval(target_score=0.60, distractor_score=0.20),
        timestep=20,
    )

    assert after.matches[0]["score"] < before.matches[0]["score"]


def test_reranking_preserves_verified_cache_fields() -> None:
    reranker = CandidateRepetitionReranker(
        RepetitionDependentConsolidator(),
        enabled=True,
    )
    _train_target(reranker, verified=True)
    original = _retrieval(target_score=0.60, distractor_score=0.20)

    result = reranker.rerank(original, timestep=15)
    target_before = next(
        match for match in original.matches if match["entry_id"] == "target"
    )
    target_after = next(
        match for match in result.matches if match["entry_id"] == "target"
    )

    for field in ("tier", "source_ref", "utility"):
        assert target_after[field] == target_before[field]
    assert reranker.last_trace[1]["mutates_durable_state"] is False


def test_reranker_enforces_match_ceiling() -> None:
    reranker = CandidateRepetitionReranker(
        RepetitionDependentConsolidator(
            RepetitionConsolidationConfig(capacity=2)
        ),
        enabled=True,
        max_matches=1,
    )

    with pytest.raises(ValueError, match="ceiling"):
        reranker.rerank(_retrieval())


def test_phase31_repetition_reranking_benchmark_passes() -> None:
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase31_repetition_reranking_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase31_repetition_reranking_benchmark",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.build_report(module.load_cases(module.DEFAULT_FIXTURE))

    assert report["passed"] is True
    assert report["production_path_changed"] is False
    assert report["checks"]["unverified_repetition_cannot_rerank"] is True
    assert report["checks"]["durable_state_not_mutated"] is True
