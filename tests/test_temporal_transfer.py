"""Pre-run structural checks for pattern-disjoint temporal transfer."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.event_unit_causal_isolation import UnitEpisode, UnitEvent
from sara_engine.evaluation.temporal_transfer import (
    DEVELOPMENT_NUISANCE_ROUTES, TRAIN_NUISANCE_ROUTES,
    generate_temporal_transfer, labeled_pattern, run_frozen_development_arm,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/temporal_transfer_v1.json"


def test_preregistration_binds_sources_and_structural_gate():
    config = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sources = config["candidate_sources"]
    assert sha256((ROOT / sources["generator_and_evaluator_path"]).read_bytes()).hexdigest() == sources[
        "generator_and_evaluator_sha256"]
    assert sha256((ROOT / sources["event_unit_path"]).read_bytes()).hexdigest() == sources[
        "event_unit_sha256"]
    generator = config["generator"]
    assert len(set(generator["training_seeds"] + generator["development_seeds"])) == 10
    assert generator["training_count_per_seed"] == 40
    assert generator["development_count_per_seed"] == 20
    assert config["pre_execution_identity_gate"]["exact_labeled_pattern_overlap_required"] == 0
    assert config["pre_execution_identity_gate"]["development_true_labels_must_not_update_learner"] is True
    assert config["capture"]["head_published_before_each_evaluation"] is True
    assert config["capture"]["automatic_export_or_approval"] is False


def test_generator_separates_content_and_balances_labels_without_outcome_scoring():
    training = generate_temporal_transfer(
        seeds=(950001, 950113, 950227, 950341, 950457),
        count_per_seed=40, split="training",
    )
    development = generate_temporal_transfer(
        seeds=(960001, 960113, 960227, 960341, 960457),
        count_per_seed=20, split="development",
    )
    assert len(training) == 200
    assert len(development) == 100
    assert sum(row.label for row in training) == 100
    assert sum(row.label for row in development) == 50
    assert not {row.identity for row in training} & {row.identity for row in development}
    assert not {labeled_pattern(row) for row in training} & {
        labeled_pattern(row) for row in development
    }
    assert {row.events[-1].route for row in training} <= set(TRAIN_NUISANCE_ROUTES)
    assert {row.events[-1].route for row in development} <= set(DEVELOPMENT_NUISANCE_ROUTES)
    assert set(TRAIN_NUISANCE_ROUTES).isdisjoint(DEVELOPMENT_NUISANCE_ROUTES)
    assert all(row.events[1].time - row.events[0].time == (1 if row.label else 4)
               for row in (*training, *development))


def test_frozen_evaluator_does_not_learn_development_truth():
    training = [
        UnitEpisode("training:1", "pair_gap_nuisance_transfer",
                    (UnitEvent(0, 1, 0), UnitEvent(1, 2, 0), UnitEvent(20, 5, 0)), 1),
        UnitEpisode("training:2", "pair_gap_nuisance_transfer",
                    (UnitEvent(0, 1, 0), UnitEvent(1, 5, 0), UnitEvent(20, 8, 0)), 0),
    ]
    development = [
        UnitEpisode("development:1", "pair_gap_nuisance_transfer",
                    (UnitEvent(0, 20, 0), UnitEvent(1, 21, 0), UnitEvent(24, 24, 0)), 1),
        UnitEpisode("development:2", "pair_gap_nuisance_transfer",
                    (UnitEvent(0, 21, 0), UnitEvent(1, 25, 0), UnitEvent(25, 28, 0)), 0),
    ]
    original = run_frozen_development_arm("C_temporal_state", training, development)
    relabeled = run_frozen_development_arm(
        "C_temporal_state", training,
        [replace(row, label=1 - row.label) for row in development],
    )
    assert original["prediction_rows"] == relabeled["prediction_rows"]
    assert original["prediction_trace_sha256"] == relabeled["prediction_trace_sha256"]
    assert original["state_bytes"] == relabeled["state_bytes"]
    with pytest.raises(ValueError, match="intervention"):
        run_frozen_development_arm("B_compact_event", training, development,
                                   intervention="state_reset")
