from sara_engine.risa.structural_interpolation import (
    PredictiveStructuralFeedbackEngine,
    StructuralFeedbackSignal,
)


def _signal(**overrides):
    values = {
        "predicting_concept": "concept:animal",
        "source_node": "concept:dog",
        "target_node": "concept:mammal",
        "relation_type": "instance_of",
        "predicted_confidence": 0.5,
        "observed_confidence": 0.9,
        "evidence_ids": ("evidence-1",),
        "context_tags": ("biology",),
    }
    values.update(overrides)
    return StructuralFeedbackSignal(**values)


def test_predictive_feedback_emits_typed_strengthening_without_mutation():
    proposal = PredictiveStructuralFeedbackEngine().propose((_signal(),))[0]
    assert proposal.edit_type == "strengthen_relation"
    assert proposal.durable_mutation_allowed is False
    assert proposal.frozen is False
    assert proposal.rollback_state == "verified_snapshot"


def test_predictive_feedback_freezes_contradiction_and_oscillation():
    engine = PredictiveStructuralFeedbackEngine()
    contradiction = engine.propose((_signal(contradiction_count=1),))[0]
    oscillation = engine.propose(
        (_signal(recent_actions=("strengthen_relation", "cut_relation", "strengthen_relation", "cut_relation")),)
    )[0]
    assert contradiction.edit_type == "freeze_subgraph"
    assert contradiction.frozen is True
    assert oscillation.edit_type == "freeze_subgraph"
    assert oscillation.reason == "oscillating_feedback"


def test_predictive_feedback_requests_more_evidence_before_editing():
    proposal = PredictiveStructuralFeedbackEngine().propose(
        (_signal(evidence_ids=(), eligible=False),)
    )[0]
    assert proposal.edit_type == "request_more_evidence"
