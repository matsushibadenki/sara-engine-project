from sara_engine.language.semantic_echo import SparseSemanticEchoField
from sara_engine.language.semantic_events import LanguageEvent, SparseLanguageEventAdapter


def test_adapter_emits_bounded_surface_events():
    events = SparseLanguageEventAdapter(max_events=3).encode("A small bird flies.")
    assert len(events) == 3
    assert events[0].axis == "orthographic"
    assert events[0].evidence_type == "observed"


def test_adapter_emits_bounded_negation_feature_without_external_parser():
    events = SparseLanguageEventAdapter(max_events=8).encode("not safe")
    negation = [event for event in events if event.axis == "semantic"]
    assert len(negation) == 1
    assert negation[0].feature == "negation"
    assert negation[0].evidence_type == "dictionary_assisted"
    assert negation[0].role == "scope"


def test_role_binding_is_local_and_bounded():
    field = SparseSemanticEchoField(max_echoes=2, max_comparisons=2)
    first = field.step(type("Event", (), {"axis": "orthographic", "feature": "subject", "role": "agent"})())
    second = field.step(type("Event", (), {"axis": "orthographic", "feature": "predicate", "role": "agent"})(), gap=8)
    assert first.active_echoes <= 2
    assert any(decision.kind == "role_binding" for decision in second.decisions)
    assert second.comparisons <= 2


def test_expired_echo_cannot_bind_from_recent_history():
    field = SparseSemanticEchoField(tiers=("fast",), threshold=0.35)
    field.step(LanguageEvent(1, "orthographic", "subject", role="agent"))
    trace = field.step(LanguageEvent(51, "orthographic", "predicate", role="agent"), gap=50)
    assert not any(decision.kind == "role_binding" for decision in trace.decisions)


def test_echo_state_round_trip_preserves_reactivation_and_limits():
    field = SparseSemanticEchoField(max_echoes=6, max_comparisons=4)
    field.step(LanguageEvent(1, "orthographic", "river", role="place"))
    state = field.state_dict()
    restored = SparseSemanticEchoField(max_echoes=6, max_comparisons=4)
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    trace = restored.step(LanguageEvent(2, "orthographic", "river", role="place"))
    assert any(decision.kind == "reactivation" for decision in trace.decisions)
