#!/usr/bin/env python3
"""Run the frozen Backblaze train/development event-unit comparison."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random

from sara_engine.evaluation.event_unit_causal_isolation import (
    EventUnitLearner,
    EventUnitV2Learner,
    UnitEpisode,
    UnitEvent,
)
from sara_engine.utils.project_paths import processed_data_path, workspace_path

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2013_event_unit_protocol_v1.json"))
DATASET = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2013_event_unit_development_v1.jsonl"))
OUTPUT = Path(workspace_path("evaluation", "backblaze_2013_event_unit_development_v1.json"))
PROTOCOL_SHA256 = "a10f1d904d36f6d9771f3b837b1e8d76092eb15a82cd59c76358c8b653213fe0"
DATASET_SHA256 = "3cd3605049965a402976eac83fe54f55766e1434f0c2a3d28621e4b5d2659dd9"


def _episodes(rows: list[dict], split: str, protocol_id: str) -> list[UnitEpisode]:
    selected = [row for row in rows if row["split"] == split]
    selected.sort(key=lambda row: sha256(
        f"{protocol_id}|{row['case_id']}".encode()).hexdigest())
    return [UnitEpisode(
        identity=row["case_id"],
        family="backblaze_drive_failure",
        events=tuple(UnitEvent(event["route"], event["time"], event["branch"])
                     for event in row["events"]),
        label=row["label"],
    ) for row in selected]


def _balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    recalls = []
    for target in (0, 1):
        indices = [index for index, label in enumerate(labels) if label == target]
        recalls.append(sum(predictions[index] == target for index in indices) / len(indices))
    return sum(recalls) / 2


def _run(arm: str, training: list[UnitEpisode], development: list[UnitEpisode],
         *, intervention: str = "none", shuffle_outcomes: bool = False,
         collapse_time: bool = False) -> dict[str, object]:
    learner = (EventUnitLearner(arm, intervention=intervention)
               if arm == "A_scalar_local"
               else EventUnitV2Learner(arm, intervention=intervention))
    outcomes = [episode.label for episode in training]
    if shuffle_outcomes:
        random.Random(20_130_920).shuffle(outcomes)
    updates = 0
    training_work = 0
    for episode, target in zip(training, outcomes):
        receipt = learner.predict(episode)
        training_work += receipt.event_work
        updates += learner.observe(receipt, target)
    labels = []
    predictions = []
    event_work = 0
    trace = []
    for episode in development:
        evaluated = episode
        if collapse_time:
            evaluated = UnitEpisode(episode.identity, episode.family,
                                    tuple(UnitEvent(event.route, 0, event.branch)
                                          for event in episode.events), episode.label)
        receipt = learner.predict(evaluated)
        labels.append(episode.label)
        predictions.append(receipt.predicted)
        event_work += receipt.event_work
        trace.append((episode.identity, receipt.predicted))
        learner.observe(receipt, receipt.predicted)
    return {
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "predicted_positive": sum(predictions),
        "training_updates": updates,
        "training_event_work": training_work,
        "development_event_work": event_work,
        "maximum_events": max(len(episode.events) for episode in development),
        "learned_features": len(learner.weights),
        "state_bytes": learner.state_bytes(),
        "prediction_trace_sha256": sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest(),
    }


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Backblaze event-unit development attempt already exists")
    if sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Backblaze event-unit protocol changed")
    if sha256(DATASET.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Backblaze event dataset changed")
    protocol = json.loads(PROTOCOL.read_text())
    rows = [json.loads(line) for line in DATASET.read_text().splitlines()]
    training = _episodes(rows, "training", protocol["protocol_id"])
    development = _episodes(rows, "development", protocol["protocol_id"])
    arms = {arm: _run(arm, training, development) for arm in protocol["arms"]}
    controls = {
        "outcome_shuffle_D": _run("D_temporal_dendritic", training, development,
                                  shuffle_outcomes=True),
        "time_collapse_C": _run("C_temporal_state", training, development,
                                collapse_time=True),
        "state_reset_C": _run("C_temporal_state", training, development,
                              intervention="state_reset"),
        "branch_assignment_shuffle_D": _run(
            "D_temporal_dendritic", training, development,
            intervention="branch_assignment_shuffle"),
    }
    compact = arms["B_compact_event"]["balanced_accuracy"]
    temporal = arms["C_temporal_state"]["balanced_accuracy"]
    dendritic = arms["D_temporal_dendritic"]["balanced_accuracy"]
    acceptance = protocol["acceptance"]
    best_advanced = max(temporal, dendritic)
    best_control_drop = max(
        temporal - controls["time_collapse_C"]["balanced_accuracy"],
        temporal - controls["state_reset_C"]["balanced_accuracy"],
        dendritic - controls["branch_assignment_shuffle_D"]["balanced_accuracy"],
    )
    budgets = protocol["resource_budgets"]
    checks = {
        "balanced_development": all(
            sum(episode.label == label for episode in development) == 33 for label in (0, 1)),
        "scalar_compact_prediction_equivalence": (
            arms["A_scalar_local"]["prediction_trace_sha256"]
            == arms["B_compact_event"]["prediction_trace_sha256"]),
        "minimum_quality": best_advanced >= acceptance["minimum_development_balanced_accuracy"],
        "gain_over_compact": best_advanced - compact >= acceptance["minimum_gain_over_compact"],
        "targeted_control_drop": best_control_drop >= acceptance["minimum_control_drop"],
        "outcome_shuffle_not_better": (
            controls["outcome_shuffle_D"]["balanced_accuracy"] <= dendritic),
        "event_budget": all(row["maximum_events"] <= budgets["maximum_events_per_case"]
                            for row in arms.values()),
        "feature_budget": all(row["learned_features"] <= budgets["maximum_features"]
                              for row in arms.values()),
        "state_budget": all(row["state_bytes"] <= budgets["maximum_state_bytes"]
                            for row in arms.values()),
    }
    report = {
        "schema": "sara-backblaze-2013-event-unit-development-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "training_cases": len(training),
        "development_cases": len(development),
        "arms": arms,
        "controls": controls,
        "contrasts": {"C_minus_B": temporal - compact, "D_minus_B": dendritic - compact,
                      "D_minus_C": dendritic - temporal,
                      "best_targeted_control_drop": best_control_drop},
        "checks": checks,
        "development_gate_passed": all(checks.values()),
        "heldout_consumed": False,
        "global_backpropagation": False,
        "matrix_learning": False,
        "gpu": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"arms": {key: value["balanced_accuracy"] for key, value in arms.items()},
                      "controls": {key: value["balanced_accuracy"] for key, value in controls.items()},
                      "contrasts": report["contrasts"], "checks": checks,
                      "development_gate_passed": report["development_gate_passed"]}, sort_keys=True))
    return 0 if report["development_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
