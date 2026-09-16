#!/usr/bin/env python3
"""Audit the hash-pinned BPI Challenge 2012 irregular event log."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from sara_engine.utils.project_paths import ensure_parent_directory, raw_data_path, workspace_path


SOURCE_MD5 = "74c7ba9aba85bfcb181a22c9d565e5b5"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attributes(element: ET.Element) -> dict[str, str]:
    result = {}
    for child in element:
        tag = _local(child.tag)
        if tag in ("string", "date", "int", "float", "boolean") and "key" in child.attrib:
            result[child.attrib["key"]] = child.attrib.get("value", "")
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def build_audit() -> dict:
    path = Path(raw_data_path("r2_bpi2012", "BPI_Challenge_2012.xes.gz"))
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != SOURCE_MD5:
        raise ValueError("BPI 2012 source MD5 mismatch")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    traces = 0
    events = 0
    activities = Counter()
    lifecycles = Counter()
    trace_lengths = []
    gaps = []
    non_monotonic = 0
    starts = []
    ends = []
    with gzip.open(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if _local(element.tag) != "trace":
                continue
            event_rows = []
            for child in element:
                if _local(child.tag) != "event":
                    continue
                attrs = _attributes(child)
                timestamp = datetime.fromisoformat(attrs["time:timestamp"])
                event_rows.append(timestamp)
                activities[attrs["concept:name"]] += 1
                lifecycles[attrs.get("lifecycle:transition", "missing")] += 1
            if event_rows:
                starts.append(event_rows[0]); ends.append(event_rows[-1])
                for left, right in zip(event_rows, event_rows[1:]):
                    delta = (right - left).total_seconds()
                    gaps.append(delta)
                    non_monotonic += int(delta < 0)
            traces += 1
            events += len(event_rows)
            trace_lengths.append(len(event_rows))
            element.clear()
    return {
        "schema": "sara-r2-bpi2012-source-audit-v1",
        "source": {
            "title": "BPI Challenge 2012",
            "publisher": "Eindhoven University of Technology / 4TU.ResearchData",
            "doi": "10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f",
            "landing_page": "https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204/1",
            "file_id": 24027287,
            "md5": md5,
            "sha256": sha256,
            "terms": "4TU General Terms of Use"
        },
        "trace_count": traces,
        "event_count": events,
        "activity_count": len(activities),
        "lifecycle_count": len(lifecycles),
        "time_start": min(starts).isoformat(),
        "time_end": max(ends).isoformat(),
        "case_start_percentiles": {
            "p70": _percentile(starts, 0.70).isoformat(),
            "p85": _percentile(starts, 0.85).isoformat(),
        },
        "trace_length": {
            "minimum": min(trace_lengths), "median": _percentile(trace_lengths, 0.5),
            "p90": _percentile(trace_lengths, 0.9), "maximum": max(trace_lengths),
        },
        "inter_event_seconds": {
            "count": len(gaps), "minimum": min(gaps), "median": _percentile(gaps, 0.5),
            "p90": _percentile(gaps, 0.9), "p99": _percentile(gaps, 0.99), "maximum": max(gaps),
            "distinct_values": len(set(gaps)), "negative_count": non_monotonic,
            "zero_count": sum(value == 0 for value in gaps),
        },
        "activities": dict(sorted(activities.items())),
        "lifecycles": dict(sorted(lifecycles.items())),
        "candidate_task": {
            "prediction": "next activity within the same case",
            "available_timing": "elapsed time since the preceding event and bounded recent gap history",
            "outcome": "the next recorded activity",
            "timing_ablation": "replace observed gap buckets with a constant while preserving activity order and counts",
            "case_isolation_required": True,
        },
    }


def main() -> int:
    output = Path(ensure_parent_directory(workspace_path("evaluation", "r2_bpi2012_source_audit.json")))
    audit = build_audit()
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "traces": audit["trace_count"], "events": audit["event_count"],
                      "activities": audit["activity_count"], "gap_values": audit["inter_event_seconds"]["distinct_values"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
