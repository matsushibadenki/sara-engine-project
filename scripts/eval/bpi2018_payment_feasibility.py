#!/usr/bin/env python3
"""Assess future payment-outcome support using BPI 2018 event metadata only."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_bpi2018", "BPI_Challenge_2018.xes.gz"))
SOURCE_AUDIT = Path(workspace_path("evaluation", "bpi2018_source_audit_v1.json"))
PROTOCOL = Path(processed_data_path("benchmark_fixtures", "bpi2018_payment_feasibility_protocol_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "bpi2018_payment_feasibility_v1.json"))
PROTOCOL_SHA256 = "654efe4feba4d02d9a34087b40d824079de4743d15626f309d68f871f718d707"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("BPI 2018 payment feasibility already exists")
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("BPI 2018 payment protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if digest(SOURCE) != protocol["source_sha256"] or digest(SOURCE_AUDIT) != protocol["source_audit_sha256"]:
        raise ValueError("Pinned source or audit changed")
    summaries = []
    global_end = None
    with gzip.open(SOURCE, "rb") as stream:
        context = ET.iterparse(stream, events=("start", "end"))
        _, root = next(context)
        for event_type, trace in context:
            if event_type != "end" or tag(trace) != "trace":
                continue
            times = []
            payment_times = []
            for event in trace:
                if tag(event) != "event":
                    continue
                activity = None
                timestamp = None
                for attribute in event:
                    key = attribute.attrib.get("key")
                    if tag(attribute) == "string" and key == "concept:name":
                        activity = attribute.attrib.get("value")
                    elif tag(attribute) == "date" and key == "time:timestamp":
                        timestamp = parse_time(attribute.attrib["value"])
                if timestamp is not None:
                    times.append(timestamp)
                    global_end = timestamp if global_end is None else max(global_end, timestamp)
                    if activity == "finish payment":
                        payment_times.append(timestamp)
            unique = sorted(set(times))
            if unique:
                summaries.append((unique[0], unique[2] if len(unique) >= 3 else None,
                                  min(payment_times) if payment_times else None, unique[-1]))
            root.clear()
    year_to_split = {year: split for split, years in protocol["opening_year_splits"].items() for year in years}
    counts = {split: Counter() for split in protocol["opening_year_splits"]}
    exclusions = Counter()
    individual_coverage = {split: Counter() for split in counts}
    horizon = timedelta(days=protocol["outcome_rule"]["horizon_days"])
    for opened, cutoff, payment, last_event in summaries:
        split = year_to_split.get(opened.year)
        if split is None:
            exclusions["outside_split_years"] += 1
            continue
        if cutoff is None:
            exclusions["insufficient_distinct_times"] += 1
            continue
        if payment is not None and payment <= cutoff:
            exclusions["payment_at_or_before_cutoff"] += 1
            continue
        end = cutoff + horizon
        if payment is not None and payment <= end:
            counts[split]["positive"] += 1
        elif global_end >= end:
            counts[split]["negative"] += 1
            individual_coverage[split]["negative_last_event_at_or_after_horizon" if last_event >= end
                                       else "negative_last_event_before_horizon"] += 1
        else:
            exclusions["global_followup_insufficient"] += 1
    checks = {split: all(counts[split][label] >= minimum for label in ("positive", "negative"))
              for split, minimum in protocol["minimum_per_class"].items()}
    result = {"schema": "sara-bpi2018-payment-feasibility-result-v1",
              "protocol_sha256": PROTOCOL_SHA256, "source_sha256": protocol["source_sha256"],
              "case_summaries": len(summaries), "global_observation_end_utc": global_end.isoformat(),
              "case_counts": {split: dict(count) for split, count in counts.items()},
              "negative_individual_coverage": {split: dict(count) for split, count in individual_coverage.items()},
              "exclusions": dict(exclusions), "checks": checks, "passed": all(checks.values()),
              "descriptive_attribute_values_read": False, "candidate_scored": False,
              "heldout_features_materialized": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
