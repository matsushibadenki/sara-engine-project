from sara_engine.learning.adaptive_credit import AdaptiveCreditField


def _signals(**overrides):
    base = {
        "prediction_error": 0.8,
        "novelty": 0.6,
        "reward": 0.7,
        "verifier_disagreement": 0.2,
        "contradiction": 0.0,
        "metabolic_headroom": 0.9,
        "source_backed": True,
    }
    base.update(overrides)
    return base


def test_adaptive_credit_updates_only_active_routes():
    field = AdaptiveCreditField(max_routes=8)
    result = field.apply(
        active_routes={(1, 2): 0.9, (2, 3): 0.8},
        signals=_signals(),
        route_regions={(1, 2): "vision", (2, 3): "audio"},
        region_credit={"vision": 1.0, "audio": 0.0},
    )

    assert result.update_allowed is True
    assert result.decision == "update"
    assert result.updated_route_count == 1
    assert result.skipped_by_region_count == 1
    assert (1, 2) in field.routes
    assert (2, 3) not in field.routes


def test_adaptive_credit_freezes_without_learning_event():
    field = AdaptiveCreditField()
    result = field.apply(
        active_routes={(1, 2): 0.9},
        signals=_signals(prediction_error=0.1, novelty=0.1, reward=0.0, verifier_disagreement=0.1),
    )

    assert result.update_allowed is False
    assert result.decision == "freeze_no_learning_event"
    assert field.routes == {}


def test_adaptive_credit_freezes_on_contradiction_and_unverified_source():
    field = AdaptiveCreditField()
    contradiction = field.apply(
        active_routes={(1, 2): 0.9},
        signals=_signals(contradiction=0.9),
    )
    unverified = field.apply(
        active_routes={(2, 3): 0.9},
        signals=_signals(source_backed=False),
    )

    assert contradiction.decision == "freeze_contradiction"
    assert unverified.decision == "freeze_unverified_source"
    assert field.freeze_count == 2


def test_adaptive_credit_quantized_mode_uses_bucketed_credit():
    field = AdaptiveCreditField(quantize_credit=True)
    field.apply(active_routes={(1, 2): 0.9}, signals=_signals())

    state = field.routes[(1, 2)]
    assert state.responsibility in {0.0, 0.33, 0.66, 1.0}
    assert state.confidence in {0.0, 0.33, 0.66, 1.0}
