from sara_engine.learning.resonance_credit import SparseResonanceCreditAssigner


def _resonant_signals():
    return {
        "local_coincidence": 0.9,
        "prediction_consistency": 0.8,
        "verifier_confidence": 1.0,
        "cross_modal_agreement": 0.7,
        "reward_signal": 0.8,
        "novelty_signal": 0.6,
        "reward_polarity": 1.0,
        "metabolic_headroom": 0.8,
        "source_backed": True,
    }


def test_resonance_credit_updates_only_after_multi_channel_agreement():
    assigner = SparseResonanceCreditAssigner()
    result = assigner.apply({(1, 2): 0.75}, _resonant_signals())

    assert result.update_allowed is True
    assert result.decision == "reinforce"
    assert result.active_channel_count >= 3
    assert result.updates[(1, 2)] > 0.0
    assert assigner.weights[(1, 2)] > 0.0


def test_resonance_credit_freezes_on_verifier_contradiction():
    assigner = SparseResonanceCreditAssigner()
    signals = _resonant_signals()
    signals["contradiction"] = 0.9
    result = assigner.apply({(1, 2): 1.0}, signals)

    assert result.update_allowed is False
    assert result.decision == "freeze_contradiction"
    assert result.updates == {}
    assert assigner.weights == {}


def test_resonance_credit_freezes_abstention_and_low_budget():
    assigner = SparseResonanceCreditAssigner()
    abstained = _resonant_signals()
    abstained["abstained"] = True
    low_budget = _resonant_signals()
    low_budget["metabolic_headroom"] = 0.1

    first = assigner.apply({(1, 2): 1.0}, abstained)
    second = assigner.apply({(1, 2): 1.0}, low_budget)

    assert first.decision == "freeze_abstention"
    assert second.decision == "freeze_metabolic_budget"
    assert assigner.freeze_count == 2


def test_resonance_credit_bounds_links_and_weights():
    assigner = SparseResonanceCreditAssigner(
        learning_rate=1.0,
        max_links=1,
        weight_clip=0.5,
    )
    for _ in range(4):
        assigner.apply({(1, 2): 10.0, (2, 3): 10.0}, _resonant_signals())

    assert len(assigner.weights) == 1
    assert assigner.weights[(1, 2)] == 0.5
