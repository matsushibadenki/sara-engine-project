"""Pattern-disjoint temporal transfer task with frozen development readout."""
from __future__ import annotations

from hashlib import sha256
import json
import random
from typing import Sequence

from sara_engine.evaluation.event_unit_causal_isolation import (
    EventUnitV2Learner, UnitEpisode, UnitEvent,
)


TRAIN_NUISANCE_ROUTES = (20, 21, 22, 23)
DEVELOPMENT_NUISANCE_ROUTES = (24, 25, 26, 27)


def generate_temporal_transfer(
    *, seeds: Sequence[int], count_per_seed: int, split: str,
    namespace: str = "temporal-transfer-v1",
) -> list[UnitEpisode]:
    """Same learned pair-gap rule, disjoint nuisance routes and time origins."""
    if split not in ("training", "development") or count_per_seed < 2 or count_per_seed % 2:
        raise ValueError("Temporal transfer requires an even count and a development split")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("Temporal transfer seeds must be unique integers")
    nuisance_routes = (TRAIN_NUISANCE_ROUTES if split == "training"
                       else DEVELOPMENT_NUISANCE_ROUTES)
    start_time = 1 if split == "training" else 20
    rows = []
    for seed in seeds:
        for index in range(count_per_seed):
            rng = random.Random(seed * 1_000_003 + index * 101)
            label = index % 2
            gap = 1 if label else 4
            origin = start_time + rng.randrange(4)
            nuisance = nuisance_routes[rng.randrange(len(nuisance_routes))]
            rows.append(UnitEpisode(
                identity=f"{namespace}:{split}:{seed}:{index}",
                family="pair_gap_nuisance_transfer",
                events=(
                    UnitEvent(0, origin, 0),
                    UnitEvent(1, origin + gap, 0),
                    UnitEvent(nuisance, origin + gap + 3, 0),
                ),
                label=label,
            ))
    random.Random(sum(seeds) + (0 if split == "training" else 1)).shuffle(rows)
    return rows


def labeled_pattern(episode: UnitEpisode) -> tuple:
    """Content identity excluding a generated episode ID."""
    return episode.family, episode.events, episode.label


def run_frozen_development_arm(
    arm: str, training: Sequence[UnitEpisode], development: Sequence[UnitEpisode],
    *, intervention: str = "none",
) -> dict:
    """Train locally, then score development without observing true labels."""
    if arm not in ("B_compact_event", "C_temporal_state"):
        raise ValueError("Only fixed B/C temporal transfer arms are allowed")
    if intervention not in ("none", "time_shuffle", "state_reset"):
        raise ValueError("Temporal transfer intervention is invalid")
    if arm == "B_compact_event" and intervention != "none":
        raise ValueError("Compact-event control does not accept an intervention")
    if not training or not development:
        raise ValueError("Training and development episodes are required")
    learner = EventUnitV2Learner(arm, intervention=intervention)
    maximum_work = 0
    for episode in training:
        prediction = learner.predict(episode)
        maximum_work = max(maximum_work, prediction.event_work)
        learner.observe(prediction, episode.label)
    rows = []
    correct = 0
    for episode in development:
        prediction = learner.predict(episode)
        maximum_work = max(maximum_work, prediction.event_work)
        rows.append((episode.identity, prediction.predicted))
        correct += int(prediction.predicted == episode.label)
        # Resolve the receipt with its own prediction; this makes no weight update
        # and never supplies the true development outcome to the learner.
        learner.observe(prediction, prediction.predicted)
    trace = sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    return {
        "accuracy": correct / len(development),
        "prediction_rows": rows,
        "prediction_trace_sha256": trace,
        "predictions": len(development),
        "maximum_event_work": maximum_work,
        "state_bytes": learner.state_bytes(),
    }
