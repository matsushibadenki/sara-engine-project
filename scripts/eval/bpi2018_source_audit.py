#!/usr/bin/env python3
"""Audit BPI 2018 case/activity/time metadata from the pinned XES archive."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_bpi2018", "BPI_Challenge_2018.xes.gz"))
PROTOCOL = Path(processed_data_path("benchmark_fixtures", "bpi2018_source_audit_protocol_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "bpi2018_source_audit_v1.json"))
PROTOCOL_SHA256 = "2702eb51879b267c8172dbd24d9b1fa01dbc62b23b21dd48d9c1f66d343564c3"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("BPI 2018 event timestamp lacks a timezone")
    return parsed.astimezone(timezone.utc)


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("BPI 2018 source audit already exists")
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("BPI 2018 source audit protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if digest(SOURCE) != protocol["source_sha256"]:
        raise ValueError("Pinned BPI 2018 source changed")
    seen_cases: set[str] = set()
    first_months = Counter()
    last_months = Counter()
    activities = Counter()
    last_activities = Counter()
    counts = Counter()
    with gzip.open(SOURCE, "rb") as stream:
        context = ET.iterparse(stream, events=("start", "end"))
        _, root = next(context)
        for event_type, trace in context:
            if event_type != "end" or local_tag(trace) != "trace":
                continue
            case_id = None
            events = []
            for child in trace:
                tag = local_tag(child)
                if tag == "string" and child.attrib.get("key") == "concept:name":
                    case_id = child.attrib.get("value")
                elif tag == "event":
                    activity = None
                    timestamp = None
                    for attribute in child:
                        key = attribute.attrib.get("key")
                        if local_tag(attribute) == "string" and key == "concept:name":
                            activity = attribute.attrib.get("value")
                        elif local_tag(attribute) == "date" and key == "time:timestamp":
                            timestamp = _time(attribute.attrib["value"])
                    counts["event_count"] += 1
                    if activity is None:
                        counts["missing_activity_values"] += 1
                    if timestamp is None:
                        counts["missing_timestamps"] += 1
                    if activity is not None and timestamp is not None:
                        activities[activity] += 1
                        events.append((timestamp, len(events), activity))
            if not case_id or case_id in seen_cases:
                counts["missing_or_duplicate_case_ids"] += 1
            else:
                seen_cases.add(case_id)
            if events:
                events.sort()
                first_months[events[0][0].strftime("%Y-%m")] += 1
                last_months[events[-1][0].strftime("%Y-%m")] += 1
                last_activities[events[-1][2]] += 1
                unique_times = {event[0] for event in events}
                counts["cases_with_three_distinct_timestamps"] += len(unique_times) >= 3
                counts["same_timestamp_events"] += len(events) - len(unique_times)
            counts["trace_count"] += 1
            root.clear()
    limits = protocol["required_minimums"]
    checks = {
        "independent_cases": len(seen_cases) >= limits["independent_cases"],
        "timestamped_events": sum(activities.values()) >= limits["timestamped_events"],
        "three_distinct_timestamps": counts["cases_with_three_distinct_timestamps"] >= limits["cases_with_three_distinct_timestamps"],
        "opening_years": len({month[:4] for month in first_months}) >= limits["opening_years"],
        "complete_case_identity": counts["missing_or_duplicate_case_ids"] == 0,
        "complete_activity_and_time": counts["missing_activity_values"] == counts["missing_timestamps"] == 0,
    }
    result = {"schema": "sara-bpi2018-source-audit-result-v1", "protocol_sha256": PROTOCOL_SHA256,
              "source_sha256": protocol["source_sha256"], "trace_count": counts["trace_count"],
              "independent_cases": len(seen_cases), "event_count": counts["event_count"],
              "activity_counts": dict(sorted(activities.items())),
              "last_activity_counts": dict(sorted(last_activities.items())),
              "first_month_counts": dict(sorted(first_months.items())),
              "last_month_counts": dict(sorted(last_months.items())),
              "cases_with_three_distinct_timestamps": counts["cases_with_three_distinct_timestamps"],
              "same_timestamp_events": counts["same_timestamp_events"],
              "missing_or_duplicate_case_ids": counts["missing_or_duplicate_case_ids"],
              "missing_activity_values": counts["missing_activity_values"],
              "missing_timestamps": counts["missing_timestamps"],
              "checks": checks, "passed": all(checks.values()),
              "other_attribute_values_read": False, "candidate_scored": False,
              "outcome_task_defined": False, "heldout_selected": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"cases": len(seen_cases), "events": counts["event_count"],
                      "activities": result["activity_counts"], "last_activities": result["last_activity_counts"],
                      "checks": checks}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
