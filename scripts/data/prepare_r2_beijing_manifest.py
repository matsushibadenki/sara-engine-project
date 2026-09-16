#!/usr/bin/env python3
"""Build train/development metadata and bounded baseline evidence for R2."""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, raw_data_path, workspace_path


INNER_SHA256 = "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
TRAIN_START = datetime(2013, 3, 1, 0)
TRAIN_END = datetime(2015, 12, 31, 23)
DEV_START = datetime(2016, 1, 1, 0)
DEV_END = datetime(2016, 6, 30, 23)
TEST_START = datetime(2016, 7, 1, 0)
TEST_END = datetime(2017, 2, 28, 23)


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp(row: dict[str, str]) -> datetime:
    return datetime(int(row["year"]), int(row["month"]), int(row["day"]), int(row["hour"]))


def _period(value: datetime) -> str | None:
    if TRAIN_START <= value <= TRAIN_END:
        return "training"
    if DEV_START <= value <= DEV_END:
        return "development"
    if TEST_START <= value <= TEST_END:
        return "frozen_test"
    return None


def _metrics(rows: list[tuple[int, float]]) -> dict:
    positives = sum(label for label, _ in rows)
    negatives = len(rows) - positives
    true_positive = sum(label == 1 and probability >= 0.5 for label, probability in rows)
    true_negative = sum(label == 0 and probability < 0.5 for label, probability in rows)
    return {
        "count": len(rows),
        "positive_rate": positives / len(rows),
        "accuracy": sum((probability >= 0.5) == bool(label) for label, probability in rows) / len(rows),
        "balanced_accuracy": 0.5 * (true_positive / positives + true_negative / negatives),
        "brier": sum((probability - label) ** 2 for label, probability in rows) / len(rows),
    }


def build() -> tuple[dict, dict]:
    archive_path = Path(raw_data_path("r2_beijing_air_quality", "PRSA2017_Data_20130301-20170228.zip"))
    if _digest_bytes(archive_path.read_bytes()) != INNER_SHA256:
        raise ValueError("R2 inner archive hash mismatch")

    source_members = []
    pairs = {"training": [], "development": []}
    split_counts = {"training": 0, "development": 0, "frozen_test": 0}
    excluded = {"missing_pm25": 0, "cross_split": 0, "outside_split": 0}
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        for name in names:
            payload = archive.read(name)
            source_members.append({"name": name, "bytes": len(payload), "sha256": _digest_bytes(payload)})
            reader = csv.DictReader(line.decode("utf-8-sig") for line in payload.splitlines(keepends=True))
            station_rows = list(reader)
            for index in range(len(station_rows) - 1):
                current = station_rows[index]
                following = station_rows[index + 1]
                current_time = _timestamp(current)
                following_time = _timestamp(following)
                current_period = _period(current_time)
                following_period = _period(following_time)
                if current_period is None or following_period is None:
                    excluded["outside_split"] += 1
                    continue
                if current_period != following_period:
                    excluded["cross_split"] += 1
                    continue
                if current["PM2.5"] == "NA" or following["PM2.5"] == "NA":
                    excluded["missing_pm25"] += 1
                    continue
                split_counts[current_period] += 1
                if current_period == "frozen_test":
                    continue
                previous_pm25 = None
                if index > 0 and station_rows[index - 1]["PM2.5"] != "NA":
                    previous_pm25 = float(station_rows[index - 1]["PM2.5"])
                current_pm25 = float(current["PM2.5"])
                label = int(float(following["PM2.5"]) > current_pm25)
                direction = "missing" if previous_pm25 is None else "up" if current_pm25 > previous_pm25 else "down_or_equal"
                pairs[current_period].append({
                    "station": current["station"], "hour": int(current["hour"]),
                    "direction": direction, "label": label,
                })

    train = pairs["training"]
    development = pairs["development"]
    positive = sum(row["label"] for row in train)
    majority_probability = positive / len(train)
    transition_counts: dict[tuple[str, int, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in train:
        counts = transition_counts[(row["station"], row["hour"], row["direction"])]
        counts[row["label"]] += 1

    constant_rows = [(row["label"], majority_probability) for row in development]
    persistence_rows = [(row["label"], 1.0 if row["direction"] == "up" else 0.0) for row in development]
    transition_rows = []
    online_transition_rows = []
    online_counts = {key: list(value) for key, value in transition_counts.items()}
    for row in development:
        key = (row["station"], row["hour"], row["direction"])
        negative, positive = transition_counts[key]
        probability = (positive + 1.0) / (negative + positive + 2.0)
        transition_rows.append((row["label"], probability))
        online_negative, online_positive = online_counts.get(key, [0, 0])
        online_probability = (online_positive + 1.0) / (online_negative + online_positive + 2.0)
        online_transition_rows.append((row["label"], online_probability))
        online_counts.setdefault(key, [0, 0])[row["label"]] += 1

    manifest = {
        "schema": "sara-r2-beijing-processed-manifest-v1",
        "source_inner_sha256": INNER_SHA256,
        "source_members": source_members,
        "chronological_split": {
            "training": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
            "development": [DEV_START.isoformat(), DEV_END.isoformat()],
            "frozen_test": [TEST_START.isoformat(), TEST_END.isoformat()],
        },
        "eligible_pair_counts": split_counts,
        "excluded_pair_counts": excluded,
        "baseline_fit_scope": "training only",
        "development_label_access": True,
        "frozen_test_period_metrics_computed": False,
        "transition_table_capacity": 12 * 24 * 3,
        "transition_table_occupied": len(transition_counts),
    }
    baselines = {
        "schema": "sara-r2-beijing-development-baselines-v1",
        "source_inner_sha256": INNER_SHA256,
        "training_pairs": len(train),
        "development_pairs": len(development),
        "training_positive_rate": majority_probability,
        "arms": {
            "training_majority": _metrics(constant_rows),
            "current_change_persistence": _metrics(persistence_rows),
            "station_hour_direction_transition": _metrics(transition_rows),
            "online_station_hour_direction_transition": _metrics(online_transition_rows),
        },
        "frozen_test_period_metrics_computed": False,
    }
    return manifest, baselines


def main() -> int:
    manifest, baselines = build()
    manifest_path = Path(ensure_parent_directory(processed_data_path("r2_beijing", "source_manifest_v1.json")))
    baseline_path = Path(ensure_parent_directory(workspace_path("evaluation", "r2_beijing_development_baselines_v1.json")))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    baseline_path.write_text(json.dumps(baselines, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "baselines": str(baseline_path),
                      "pairs": manifest["eligible_pair_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
