"""Training-only applicability audit for logical-step refractory routing."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterator
import xml.etree.ElementTree as ET

from sara_engine.evaluation.bpi2012_data import DEV_BOUNDARY
from sara_engine.evaluation.sepsis_data import TRAIN_END
from sara_engine.utils.project_paths import ensure_allowed_output_path, project_path, raw_data_path


PROTOCOL_PATH = "data/processed/benchmark_fixtures/real_event_refractory_applicability_v1.json"
OUTPUT_PATH = "workspace/evaluation/real_event_refractory_applicability_v1.json"
SOURCES = {
    "bpi2012": ("r2_bpi2012", "BPI_Challenge_2012.xes.gz",
                "5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3"),
    "sepsis": ("r2_sepsis", "Sepsis Cases - Event Log.xes.gz",
               "709c52340306415952811b9b9c5dc6bcc8f8d47d583eba39df9a538459dc543a"),
}
PINNED_SOURCES = {
    "src/sara_engine/evaluation/real_event_refractory_applicability.py",
    "src/sara_engine/evaluation/bpi2012_data.py",
    "src/sara_engine/evaluation/sepsis_data.py",
}
SCREEN = {
    "minimum_masked_prediction_fraction": 0.02,
    "maximum_masked_prediction_fraction": 0.50,
    "minimum_eligible_contexts": 3,
    "minimum_eligible_prediction_support": 200,
    "minimum_weighted_conditional_tv": 0.10,
    "maximum_masked_recurrence_over_one_day_fraction": 0.05,
    "minimum_per_status_context_support": 20,
}
EXPECTED_TRAINING = {"bpi2012": (9160, 177026), "sepsis": (735, 10083)}


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def load_protocol(expected_sha256: str) -> dict:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("A frozen applicability protocol digest is required")
    raw = Path(project_path(PROTOCOL_PATH)).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Applicability protocol digest mismatch")
    protocol = json.loads(raw)
    if (raw != _canonical_bytes(protocol)
            or protocol.get("schema") != "sara-real-event-refractory-applicability-v1"
            or protocol.get("datasets") != ["bpi2012", "sepsis"]
            or protocol.get("source_sha256") is None
            or set(protocol["source_sha256"]) != PINNED_SOURCES
            or protocol.get("screen") != SCREEN
            or protocol.get("output_path") != OUTPUT_PATH
            or protocol.get("training_only") is not True
            or protocol.get("candidate_scoring_authorized") is not False):
        raise ValueError("Applicability protocol boundaries changed")
    for path, digest in protocol["source_sha256"].items():
        if sha256(Path(project_path(path)).read_bytes()).hexdigest() != digest:
            raise ValueError("Applicability source changed")
    return protocol


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attrs(element: ET.Element) -> dict[str, str]:
    return {child.attrib["key"]: child.attrib.get("value", "")
            for child in element if "key" in child.attrib and _local(child.tag) in ("string", "date")}


def _training_traces(dataset: str) -> Iterator[tuple[tuple[str, datetime], ...]]:
    """Materialize event activities only after a case is classified as training."""
    if dataset not in SOURCES:
        raise ValueError("Unknown real-event dataset")
    directory, filename, expected_sha = SOURCES[dataset]
    source = Path(raw_data_path(directory, filename))
    if sha256(source.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Real-event source digest changed")
    with gzip.open(source, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if _local(element.tag) != "trace":
                continue
            first_event = next((child for child in element if _local(child.tag) == "event"), None)
            if first_event is None:
                raise ValueError("Empty real-event trace")
            start = datetime.fromisoformat(_attrs(first_event)["time:timestamp"])
            if dataset == "bpi2012":
                is_training = start < DEV_BOUNDARY
            else:
                case_id = _attrs(element)["concept:name"]
                is_training = (start, case_id) <= TRAIN_END
            if is_training:
                events = []
                for child in element:
                    if _local(child.tag) == "event":
                        row = _attrs(child)
                        events.append((row["concept:name"],
                                       datetime.fromisoformat(row["time:timestamp"])))
                training_events = tuple(events)
                element.clear()
                yield training_events
                continue
            element.clear()


def _total_variation(left: Counter[str], right: Counter[str]) -> float:
    left_total, right_total = sum(left.values()), sum(right.values())
    labels = set(left) | set(right)
    return 0.5 * sum(abs(left[label] / left_total - right[label] / right_total)
                     for label in labels)


def audit_training_dataset(dataset: str) -> dict:
    """Report descriptive training-only repetition and target-association facts."""
    context_counts: dict[tuple[str, str, str], list[Counter[str]]] = defaultdict(
        lambda: [Counter(), Counter()]
    )
    case_count = event_count = prediction_count = masked_events = masked_predictions = 0
    masked_recurrence_seconds = []
    for events in _training_traces(dataset):
        case_count += 1
        event_count += len(events)
        next_eligible: dict[str, int] = {}
        last_emitted: dict[str, datetime] = {}
        for index, (activity, timestamp) in enumerate(events):
            masked = index < next_eligible.get(activity, -1)
            if masked:
                masked_events += 1
                gap = (timestamp - last_emitted[activity]).total_seconds()
                if gap < 0:
                    raise ValueError("Negative same-route recurrence interval")
                masked_recurrence_seconds.append(gap)
            else:
                next_eligible[activity] = index + 3
                last_emitted[activity] = timestamp
            if index + 1 < len(events):
                prediction_count += 1
                masked_predictions += masked
                if index >= 2:
                    context = (events[index - 2][0], events[index - 1][0], activity)
                    context_counts[context][int(masked)][events[index + 1][0]] += 1
    expected_cases, expected_predictions = EXPECTED_TRAINING[dataset]
    if case_count != expected_cases or prediction_count != expected_predictions:
        raise ValueError("Training split counts differ from frozen manifests")
    eligible_contexts = []
    support = 0
    weighted_tv_sum = 0.0
    weight_sum = 0
    differing_majority = 0
    for emitted, masked in context_counts.values():
        emitted_count, masked_count = sum(emitted.values()), sum(masked.values())
        if min(emitted_count, masked_count) < SCREEN["minimum_per_status_context_support"]:
            continue
        weight = min(emitted_count, masked_count)
        eligible_contexts.append((emitted_count, masked_count))
        support += emitted_count + masked_count
        weighted_tv_sum += weight * _total_variation(emitted, masked)
        weight_sum += weight
        differing_majority += (emitted.most_common(1)[0][0] != masked.most_common(1)[0][0])
    masked_fraction = masked_predictions / prediction_count
    long_gap_fraction = (sum(gap > 86400 for gap in masked_recurrence_seconds)
                         / len(masked_recurrence_seconds) if masked_recurrence_seconds else 0.0)
    weighted_tv = weighted_tv_sum / weight_sum if weight_sum else 0.0
    checks = {
        "mask_rate_in_range": (SCREEN["minimum_masked_prediction_fraction"]
                               <= masked_fraction
                               <= SCREEN["maximum_masked_prediction_fraction"]),
        "enough_eligible_contexts": len(eligible_contexts) >= SCREEN["minimum_eligible_contexts"],
        "enough_eligible_support": support >= SCREEN["minimum_eligible_prediction_support"],
        "conditional_tv_at_least_0_10": weighted_tv >= SCREEN["minimum_weighted_conditional_tv"],
        "masked_long_gap_fraction_at_most_0_05": (
            long_gap_fraction <= SCREEN["maximum_masked_recurrence_over_one_day_fraction"]),
    }
    return {
        "dataset": dataset,
        "training_cases": case_count,
        "training_events": event_count,
        "training_predictions": prediction_count,
        "masked_events": masked_events,
        "masked_predictions": masked_predictions,
        "masked_prediction_fraction": masked_fraction,
        "masked_recurrence_median_seconds": (median(masked_recurrence_seconds)
                                             if masked_recurrence_seconds else None),
        "masked_recurrence_over_one_day_fraction": long_gap_fraction,
        "eligible_context_count": len(eligible_contexts),
        "eligible_prediction_support": support,
        "eligible_contexts_with_different_majority": differing_majority,
        "weighted_conditional_tv": weighted_tv,
        "screen_checks": checks,
        "screen_passed": all(checks.values()),
    }


def run_registered_audit(*, expected_protocol_sha256: str) -> dict:
    protocol = load_protocol(expected_protocol_sha256)
    output = Path(ensure_allowed_output_path(protocol["output_path"]))
    if output.exists():
        raise ValueError("Applicability audit already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".lock").open("xb") as lock:
        lock.write((expected_protocol_sha256 + "\n").encode("ascii"))
    datasets = [audit_training_dataset(dataset) for dataset in protocol["datasets"]]
    result = {
        "schema": "sara-real-event-refractory-applicability-result-v1",
        "protocol_sha256": expected_protocol_sha256,
        "datasets": datasets,
        "cross_dataset_screen_passed": all(row["screen_passed"] for row in datasets),
        "candidate_scoring_authorized": False,
        "development_or_test_outcomes_read": False,
    }
    payload = _canonical_bytes(result)
    with output.open("xb") as stream:
        stream.write(payload)
    return result
