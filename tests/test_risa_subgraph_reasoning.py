from sara_engine.risa.subgraph_reasoning import (
    BoundedSubgraphComposer,
    StructuralAnalogyEngine,
    SubgraphEdge,
)


def _edge(source, target, relation, **kwargs):
    return SubgraphEdge(
        source=source,
        target=target,
        relation_type=relation,
        confidence=kwargs.pop("confidence", 0.9),
        evidence_count=kwargs.pop("evidence_count", 2),
        context_tags=kwargs.pop("context_tags", ("test",)),
        **kwargs,
    )


def test_bounded_composer_builds_verified_two_hop_proposal_without_mutation():
    result = BoundedSubgraphComposer().compose(
        (
            _edge("concept:dog", "concept:mammal", "instance_of"),
            _edge("concept:mammal", "concept:animal", "instance_of"),
        ),
        source="concept:dog",
        target="concept:animal",
        context_tags=("test",),
    )
    proposal = result.proposals[0]
    assert result.abstained is False
    assert proposal.relation_type == "composed::instance_of+instance_of"
    assert proposal.durable_mutation_allowed is False
    assert proposal.path[0][0] == "concept:dog"


def test_bounded_composer_abstains_for_unverified_or_missing_path():
    result = BoundedSubgraphComposer().compose(
        (_edge("concept:dog", "concept:mammal", "instance_of", verified=False),),
        source="concept:dog",
        target="concept:animal",
    )
    assert result.abstained is True
    assert result.reason == "verified_path_not_found"


def test_structural_analogy_uses_relation_signature_and_abstains_on_mismatch():
    engine = StructuralAnalogyEngine()
    analogous = engine.compare(
        (_edge("a", "b", "predicts"), _edge("b", "c", "precedes")),
        (_edge("x", "y", "predicts"), _edge("y", "z", "precedes")),
    )
    unrelated = engine.compare((_edge("a", "b", "observes"),), (_edge("x", "y", "causes"),))
    assert analogous.score == 1.0
    assert analogous.abstained is False
    assert unrelated.abstained is True
