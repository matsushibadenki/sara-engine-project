"""Protocol and gate tests use fabricated scores, not candidate evaluation."""
from hashlib import sha256
from pathlib import Path

import pytest

from sara_engine.evaluation.cartpole_refractory_mechanism import ARMS
from sara_engine.evaluation.cartpole_refractory_runner import (
    SEEDS, audit_generated_splits, development_decision, load_protocol,
)


PROTOCOL = (Path(__file__).resolve().parents[1] / "data/processed/benchmark_fixtures"
            / "cartpole_refractory_mechanism_v1.json")


def test_canonical_protocol_is_source_pinned_and_fail_closed():
    digest = sha256(PROTOCOL.read_bytes()).hexdigest()
    assert load_protocol(digest)["heldout_exists"] is False
    with pytest.raises(ValueError, match="digest mismatch"):
        load_protocol("0" * 64)


def _fake_results(spiking_correct=32):
    results = []
    for seed in SEEDS:
        for arm in ARMS:
            correct = (spiking_correct if arm == "spiking_refractory" else
                       32 if arm == "compact_gap" else 16)
            results.append({
                "seed": seed, "arm": arm, "development_correct": correct,
                "development_count": 32, "development_predictions": (0,) * 32,
                "development_feature_one_count": (
                    16 if arm in ("spiking_refractory", "rate_matched_mask") else 0
                ),
                "training_updates": 1, "final_weights": (0, 0, 0, 0),
            })
    return results


def test_split_audit_is_balanced_and_disjoint():
    assert audit_generated_splits() == {
        "seed_count": 5, "training_episodes": 320,
        "development_episodes": 160, "episode_identity_overlap": 0,
        "exact_pattern_overlap": 0,
    }


def test_gate_requires_causal_controls_and_never_opens_heldout():
    positive = development_decision(_fake_results())
    assert positive["passed"] is True
    assert positive["heldout_opening_authorized"] is False
    negative = development_decision(_fake_results(spiking_correct=16))
    assert negative["passed"] is False
    assert negative["checks"]["spiking_accuracy"] is False
    with pytest.raises(ValueError):
        development_decision(_fake_results()[:-1])
