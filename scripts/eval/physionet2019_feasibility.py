#!/usr/bin/env python3
"""Audit label/time support without interpreting clinical feature columns."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "physionet2019_feasibility_protocol_v1.json"))
ACQUISITION = Path(workspace_path("evaluation", "physionet2019_acquisition_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "physionet2019_feasibility_v1.json"))
PROTOCOL_SHA256 = "36023484674c6941ff63857c40ef93de50f84d9ad1509d0708c8486a4d116d52"


def patient_summary(path: Path, allowed_columns: list[str]) -> tuple[list[int], list[int]]:
    """Interpret only the two explicitly allowed metadata columns."""
    with path.open() as stream:
        header = stream.readline().rstrip("\n").split("|")
        if len(header) != len(set(header)) or any(name not in header for name in allowed_columns):
            raise ValueError("Patient header is invalid")
        indices = [header.index(name) for name in allowed_columns]
        hours = []
        labels = []
        for raw_line in stream:
            cells = raw_line.rstrip("\n").split("|")
            if len(cells) != len(header):
                raise ValueError("Patient row width changed")
            hour_text, label_text = (cells[index] for index in indices)
            hours.append(int(float(hour_text)))
            labels.append(int(label_text))
    return hours, labels


def validate_sequence(hours: list[int], labels: list[int]) -> None:
    if not hours or len(hours) != len(labels):
        raise ValueError("Patient sequence is empty or misaligned")
    if hours != list(range(1, len(hours) + 1)):
        raise ValueError("ICULOS is not a continuous one-based sequence")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("SepsisLabel is not binary")
    if any(current < previous for previous, current in zip(labels, labels[1:])):
        raise ValueError("SepsisLabel is not non-decreasing")


def assigned_split(set_name: str, relative_path: str, protocol: dict) -> str:
    if set_name == "training_setB":
        return "heldout"
    value = int(sha256(f"{protocol['hash_namespace']}|{relative_path}".encode()).hexdigest(), 16) % 5
    return "development" if value in protocol["split"]["training_setA"]["development_remainders"] else "training"


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("PhysioNet feasibility result already exists")
    if sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("PhysioNet feasibility protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(ACQUISITION.read_bytes()).hexdigest() != protocol["acquisition_result_sha256"]:
        raise ValueError("Pinned acquisition result changed")
    root = Path(raw_data_path("r2_physionet2019"))
    excluded = set(protocol["excluded_preprotocol_inspection"])
    decision = protocol["decision_icu_hour"]
    horizon_end = decision + protocol["future_horizon_hours"]
    counts = {name: Counter() for name in ("training", "development", "heldout")}
    exclusions = Counter()
    audited = Counter()
    for set_name in ("training_setA", "training_setB"):
        for path in sorted((root / set_name).glob("*.psv")):
            relative = f"{set_name}/{path.name}"
            if relative in excluded:
                exclusions["preprotocol_inspection"] += 1
                continue
            split = assigned_split(set_name, relative, protocol)
            hours, labels = patient_summary(path, protocol["allowed_columns"])
            try:
                validate_sequence(hours, labels)
            except ValueError as error:
                reason = str(error)
                if reason == "ICULOS is not a continuous one-based sequence":
                    exclusions[f"{split}_invalid_iculos_sequence"] += 1
                    continue
                raise
            audited[set_name] += 1
            first_positive = next((hour for hour, label in zip(hours, labels) if label == 1), None)
            if first_positive is not None and first_positive <= decision:
                exclusions[f"{split}_already_positive"] += 1
            elif hours[-1] < horizon_end:
                exclusions[f"{split}_insufficient_followup"] += 1
            elif first_positive is not None and first_positive <= horizon_end:
                counts[split]["positive"] += 1
            else:
                counts[split]["negative"] += 1
    checks = {split: all(counts[split][label] >= minimum for label in ("positive", "negative"))
              for split, minimum in protocol["minimum_per_class"].items()}
    result = {"schema": "sara-physionet2019-feasibility-result-v1",
              "protocol_sha256": PROTOCOL_SHA256,
              "acquisition_result_sha256": protocol["acquisition_result_sha256"],
              "audited_patients": dict(sorted(audited.items())),
              "case_counts": {split: dict(sorted(value.items())) for split, value in counts.items()},
              "exclusions": dict(sorted(exclusions.items())), "checks": checks,
              "passed": all(checks.values()), "allowed_columns_interpreted": protocol["allowed_columns"],
              "all_other_columns_interpreted_or_stored": False,
              "clinical_features_materialized": False, "candidate_scored": False,
              "calendar_chronological_split": False,
              "preprotocol_inspected_patient_excluded": True}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
