from sara_engine.language.semantic_echo import SparseSemanticEchoField
from sara_engine.language.semantic_events import SparseLanguageEventAdapter


def test_adapter_emits_bounded_surface_events():
    events = SparseLanguageEventAdapter(max_events=3).encode("A small bird flies.")
    assert len(events) == 3
    assert events[0].axis == "orthographic"
    assert events[0].evidence_type == "observed"


def test_role_binding_is_local_and_bounded():
    field = SparseSemanticEchoField(max_echoes=2, max_comparisons=2)
    first = field.step(type("Event", (), {"axis": "orthographic", "feature": "subject", "role": "agent"})())
    second = field.step(type("Event", (), {"axis": "orthographic", "feature": "predicate", "role": "agent"})(), gap=8)
    assert first.active_echoes <= 2
    assert any(decision.kind == "role_binding" for decision in second.decisions)
    assert second.comparisons <= 2
