#!/usr/bin/env python3
"""Run the one-shot Compact Event prospective incident development gate."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random

from sara_engine.evaluation.event_unit_causal_isolation import EventUnitV2Learner, UnitEpisode, UnitEvent
from sara_engine.utils.project_paths import processed_data_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "uci_incident_compact_development_protocol_v1.json"))
MATERIALIZED = Path(processed_data_path("benchmark_fixtures", "uci_incident_compact_development_v1.jsonl"))
MATERIALIZATION_REPORT = Path(workspace_path("evaluation", "uci_incident_compact_materialize_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "uci_incident_compact_development_v1.json"))
PROTOCOL_SHA256 = "68ad7c0435ec1ad2d5f2553994d536742218cc0a0ff180c9d52224959f2befab"
MATERIALIZED_SHA256 = "1931bb0379ceb538d2fc8fbe7fe87def93173ff20a8d4405239f8b9194c56841"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def accuracy(labels: list[int], predictions: list[int]) -> float:
    values = []
    for label in (0, 1):
        positions = [index for index, value in enumerate(labels) if value == label]
        values.append(sum(predictions[index] == label for index in positions) / len(positions))
    return sum(values) / 2


def _lookup_table(training: list[dict], positions: tuple[int, ...]) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    for row in training:
        counts[tuple(row["tokens"][position] for position in positions)][row["label"]] += 1
    return {key: int(value[1] > value[0]) for key, value in counts.items()}


def run_learner(training: list[dict], development: list[dict], routes: dict[str, int],
                *, shuffled: bool = False) -> dict:
    learner = EventUnitV2Learner("B_compact_event", max_routes=256, max_features=8192)
    labels = [row["label"] for row in training]
    if shuffled:
        random.Random(20160922).shuffle(labels)
    updates = 0
    maximum_work = 0
    unknown = 0
    for rows, targets in ((training, labels), (development, None)):
        predictions = []
        for row in rows:
            event_routes = [routes.get(token, 255) for token in row["tokens"]]
            if targets is None:
                unknown += sum(route == 255 for route in event_routes)
            episode = UnitEpisode(row["case_id"], "prospective_incident", tuple(
                UnitEvent(route, index + 1, 0) for index, route in enumerate(event_routes)), row["label"])
            receipt = learner.predict(episode)
            maximum_work = max(maximum_work, receipt.event_work)
            predictions.append(receipt.predicted)
            target = targets[len(predictions) - 1] if targets is not None else receipt.predicted
            updates += learner.observe(receipt, target)
        if targets is None:
            development_predictions = predictions
    return {"balanced_accuracy": accuracy([row["label"] for row in development], development_predictions),
            "state_bytes": learner.state_bytes(), "features": len(learner._feature_ids),
            "maximum_event_work": maximum_work, "updates": updates,
            "unknown_token_fraction": unknown / (len(development) * 9)}


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("The one-shot development result already exists")
    if digest(PROTOCOL) != PROTOCOL_SHA256 or digest(MATERIALIZED) != MATERIALIZED_SHA256:
        raise ValueError("Frozen compact protocol or features changed")
    materialization = json.loads(MATERIALIZATION_REPORT.read_text())
    if not materialization["passed"] or materialization["materialized_sha256"] != MATERIALIZED_SHA256:
        raise ValueError("Pre-scoring materialization gate failed")
    protocol = json.loads(PROTOCOL.read_text())
    rows = [json.loads(line) for line in MATERIALIZED.read_text().splitlines()]
    if len(rows) != 6574 or {row["split"] for row in rows} != {"training", "development"}:
        raise ValueError("Development split seal failed")
    training = sorted((row for row in rows if row["split"] == "training"), key=lambda row: sha256(
        f"{protocol['protocol_id']}|{row['case_id']}".encode()).hexdigest())
    development = sorted((row for row in rows if row["split"] == "development"), key=lambda row: row["case_id"])
    vocabulary = sorted({token for row in training for token in row["tokens"]})
    if len(vocabulary) > 255:
        raise ValueError("Training route budget exceeded")
    routes = {token: index for index, token in enumerate(vocabulary)}
    labels = [row["label"] for row in development]
    last_state = _lookup_table(training, (1,))
    snapshot = _lookup_table(training, (1, 4, 5, 6))
    controls = {
        "majority_constant": 0.5,
        "last_state_table": accuracy(labels, [last_state.get((row["tokens"][1],), 0) for row in development]),
        "cutoff_snapshot_table": accuracy(labels, [snapshot.get(tuple(row["tokens"][i] for i in (1, 4, 5, 6)), 0)
                                                   for row in development]),
    }
    candidate = run_learner(training, development, routes)
    shuffle = run_learner(training, development, routes, shuffled=True)
    limits = protocol["acceptance"]
    strongest = max(controls.values())
    checks = {
        "minimum_balanced_accuracy": candidate["balanced_accuracy"] >= limits["minimum_balanced_accuracy"],
        "gain_over_simple_control": candidate["balanced_accuracy"] - strongest >= limits["minimum_gain_over_strongest_simple_control"],
        "outcome_shuffle_drop": candidate["balanced_accuracy"] - shuffle["balanced_accuracy"] >= limits["minimum_shuffle_drop"],
        "unknown_token_fraction": candidate["unknown_token_fraction"] <= limits["maximum_unknown_token_fraction"],
        "state_bytes": candidate["state_bytes"] <= limits["maximum_state_bytes"],
        "features": candidate["features"] <= limits["maximum_features"],
        "events_per_case": all(len(row["tokens"]) <= limits["maximum_events_per_case"] for row in rows),
    }
    result = {"schema": "sara-uci-incident-compact-development-result-v1", "protocol_sha256": PROTOCOL_SHA256,
              "materialized_sha256": MATERIALIZED_SHA256, "training_cases": len(training),
              "development_cases": len(development), "training_routes": len(vocabulary),
              "controls": controls, "candidate": candidate, "training_outcome_shuffle": shuffle,
              "checks": checks, "passed": all(checks.values()), "heldout_features_materialized": False,
              "heldout_scored": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
