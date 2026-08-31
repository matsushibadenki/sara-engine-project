from sara_engine.risa.structural_invariant import CanonicalTypedMotifStore, MotifEdge


def _edge(source, relation, target, evidence="e1"):
    return MotifEdge(source, relation, target, evidence)


def test_motif_identity_ignores_node_names_but_preserves_typed_topology():
    left = CanonicalTypedMotifStore()
    right = CanonicalTypedMotifStore()
    left.observe("a", [_edge("a", "supports", "b"), _edge("a", "contains", "c")], context="x")
    right.observe("b", [_edge("x", "supports", "y"), _edge("x", "contains", "z")], context="x")
    assert left.patterns[0].topology_signature == right.patterns[0].topology_signature
    assert left.patterns[0].relation_signature == right.patterns[0].relation_signature
    assert left.patterns[0].pattern_id == right.patterns[0].pattern_id


def test_motif_proposals_are_bounded_provisional_and_evidence_linked():
    store = CanonicalTypedMotifStore(max_patterns=2, max_candidate_patterns=2)
    store.observe("a", [_edge("a", "supports", "b"), _edge("a", "contains", "c")], context="x")
    result = store.propose([_edge("q", "supports", "r")], context="x", context_aware=False)
    assert result.event_cost <= 2
    assert all(not proposal.durable_mutation_allowed and proposal.evidence_ids for proposal in result.proposals)
    assert result.trace["durable_mutation_allowed"] is False


def test_context_mismatch_and_missing_roles_abstain():
    store = CanonicalTypedMotifStore()
    store.observe("a", [_edge("a", "supports", "b")], context="x")
    missing = store.propose([], context="x", context_aware=True)
    mismatch = store.propose([_edge("q", "supports", "r")], context="other", context_aware=True)
    assert missing.abstained is True
    assert mismatch.abstained is True
