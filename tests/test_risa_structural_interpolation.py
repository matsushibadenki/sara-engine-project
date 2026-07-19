from sara_engine.risa.structural_interpolation import (
    StructuralEvidence,
    StructuralInterpolationEngine,
)


def _evidence(**overrides):
    values = {
        "source_node": "concept:dog",
        "target_node": "concept:mammal",
        "relation_type": "instance_of",
        "confidence": 0.7,
        "source_ref": "https://source-a.example/item",
        "source_hash": "hash-a",
        "source_revision": "r1",
        "context_tags": ("biology",),
        "acquired_at": 10,
        "metabolic_cost": 3,
    }
    values.update(overrides)
    return StructuralEvidence(**values)


def test_structural_interpolation_requires_independent_verified_sources():
    result = StructuralInterpolationEngine().propose((_evidence(), _evidence(source_ref="same", source_hash="hash-a")))
    assert result.proposals == ()
    assert result.rejected_count == 1


def test_structural_interpolation_proposes_a_non_mutating_merge_candidate():
    result = StructuralInterpolationEngine().propose(
        (
            _evidence(),
            _evidence(
                confidence=0.9,
                source_ref="https://source-b.example/item",
                source_hash="hash-b",
                source_revision="r2",
                acquired_at=20,
            ),
        ),
        current_segment=20,
    )
    proposal = result.proposals[0]
    assert proposal.action == "merge_candidate"
    assert proposal.distinct_source_count == 2
    assert proposal.confidence_after > proposal.confidence_before
    assert proposal.durable_mutation_allowed is False
    assert proposal.source_revisions == ("r1", "r2")


def test_structural_interpolation_freezes_contradiction_and_expiry():
    result = StructuralInterpolationEngine().propose(
        (
            _evidence(contradiction_count=1),
            _evidence(source_hash="hash-b", source_ref="b", expiry_segment=5),
        ),
        current_segment=10,
    )
    assert result.proposals == ()
    assert result.rejected_count == 2
