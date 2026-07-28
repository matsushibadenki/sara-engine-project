from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sara_engine.learning.repetition_consolidation import (
    RepetitionConsolidationConfig,
    RepetitionDependentConsolidator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repetition_strengthens_and_saturates_sparse_memory() -> None:
    consolidator = RepetitionDependentConsolidator()
    strengths = []

    for timestep in range(64):
        update = consolidator.observe(
            memory_id="memory-a",
            timestep=timestep,
            source_ref="source-a",
        )
        strengths.append(update["after"]["retrieval_strength"])

    assert strengths[-1] > strengths[0]
    assert strengths[-1] <= 1.0
    assert strengths[-1] - strengths[-2] < strengths[1] - strengths[0]
    assert consolidator.snapshot()["state_budget_ok"] is True


def test_spaced_successful_retrieval_outperforms_massed_repetition() -> None:
    massed = RepetitionDependentConsolidator()
    spaced = RepetitionDependentConsolidator()

    for timestep in (0, 1, 2, 3):
        massed.observe(
            memory_id="memory-a",
            timestep=timestep,
            source_ref="source-a",
        )
    for index, timestep in enumerate((0, 5, 10, 15)):
        spaced.observe(
            memory_id="memory-a",
            timestep=timestep,
            source_ref="source-a",
            recall_success=index > 0,
        )

    assert spaced.read("memory-a")["retrieval_strength"] > massed.read(
        "memory-a"
    )["retrieval_strength"]
    assert spaced.read("memory-a")["stability"] > massed.read("memory-a")[
        "stability"
    ]


def test_forgetting_is_projected_without_global_memory_rewrite() -> None:
    consolidator = RepetitionDependentConsolidator()
    consolidator.observe(
        memory_id="memory-a",
        timestep=0,
        source_ref="source-a",
    )
    before = consolidator.read("memory-a")

    advance = consolidator.advance(128)
    after = consolidator.read("memory-a")

    assert after["retrieval_strength"] < before["retrieval_strength"]
    assert after["stability"] < before["stability"]
    assert advance["global_memory_rewrite"] is False


def test_contradiction_locally_depresses_memory() -> None:
    consolidator = RepetitionDependentConsolidator()
    consolidator.observe(
        memory_id="memory-a",
        timestep=0,
        source_ref="source-a",
        verified=True,
    )
    consolidator.observe(
        memory_id="memory-a",
        timestep=5,
        source_ref="source-b",
        verified=True,
        recall_success=True,
    )
    before = consolidator.read("memory-a")

    consolidator.observe(
        memory_id="memory-a",
        timestep=10,
        source_ref="source-c",
        outcome="contradiction",
    )
    after = consolidator.read("memory-a")

    assert after["retrieval_strength"] < before["retrieval_strength"]
    assert after["stability"] < before["stability"]
    assert after["verification_strength"] < before["verification_strength"]


def test_duplicate_source_only_strengthens_access_not_verification() -> None:
    consolidator = RepetitionDependentConsolidator()

    for timestep in (0, 5, 10):
        consolidator.observe(
            memory_id="memory-a",
            timestep=timestep,
            source_ref="source-a",
            verified=True,
            recall_success=timestep > 0,
        )
    duplicate = consolidator.read("memory-a")
    for offset, source_ref in enumerate(("source-b", "source-c"), start=3):
        consolidator.observe(
            memory_id="memory-a",
            timestep=offset * 5,
            source_ref=source_ref,
            verified=True,
            recall_success=True,
        )
    distinct = consolidator.read("memory-a")

    assert duplicate["verified_source_count"] == 1
    assert duplicate["verification_strength"] == pytest.approx(0.30)
    assert distinct["verified_source_count"] == 3
    assert distinct["verification_strength"] > duplicate[
        "verification_strength"
    ]


def test_capacity_source_and_event_budgets_are_enforced() -> None:
    consolidator = RepetitionDependentConsolidator(
        RepetitionConsolidationConfig(
            capacity=2,
            max_events=3,
            max_sources_per_memory=1,
        )
    )
    consolidator.observe(
        memory_id="memory-a",
        timestep=0,
        source_ref="source-a",
        verified=True,
    )
    source_overflow = consolidator.observe(
        memory_id="memory-a",
        timestep=1,
        source_ref="source-b",
        verified=True,
    )
    consolidator.observe(
        memory_id="memory-b",
        timestep=2,
        source_ref="source-b",
    )
    rejected = consolidator.observe(
        memory_id="memory-c",
        timestep=3,
        source_ref="source-c",
    )

    assert source_overflow["source_budget_exhausted"] is True
    assert source_overflow["new_verified_source"] is False
    assert rejected["mutation_allowed"] is False
    assert rejected["reason"] == "event_budget_exhausted"
    assert consolidator.snapshot()["state_budget_ok"] is True


def test_invalid_repetition_events_are_rejected() -> None:
    consolidator = RepetitionDependentConsolidator()

    with pytest.raises(ValueError, match="monotonic"):
        consolidator.advance(1)
        consolidator.observe(memory_id="memory-a", timestep=0)
    with pytest.raises(ValueError, match="successful recall"):
        consolidator.observe(
            memory_id="memory-a",
            timestep=2,
            outcome="contradiction",
            recall_success=True,
        )
    with pytest.raises(ValueError, match="requires source_ref"):
        consolidator.observe(
            memory_id="memory-a",
            timestep=2,
            verified=True,
        )


def test_phase31_repetition_benchmark_passes() -> None:
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase31_repetition_consolidation_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase31_repetition_consolidation_benchmark",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.build_report(module.load_cases(module.DEFAULT_FIXTURE))

    assert report["passed"] is True
    assert report["observed_only"] is True
    assert report["production_path_changed"] is False
    assert report["checks"][
        "duplicate_source_does_not_inflate_verification"
    ] is True


def test_repetition_consolidator_is_exposed_from_learning_package() -> None:
    import sara_engine.learning as learning

    consolidator = learning.RepetitionDependentConsolidator()
    consolidator.observe(memory_id="memory-a", timestep=0)

    assert consolidator.read("memory-a")["retrieval_strength"] > 0.0
