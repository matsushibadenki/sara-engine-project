#!/usr/bin/env python3
"""Audit the hash-pinned Sepsis Cases real event log."""

from __future__ import annotations
from collections import Counter
from datetime import datetime
import gzip, hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET
from sara_engine.utils.project_paths import ensure_parent_directory, raw_data_path, workspace_path

SOURCE_MD5 = "b5671166ac71eb20680d3c74616c43d2"

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

def _attributes(element: ET.Element) -> dict[str, str]:
    return {child.attrib["key"]: child.attrib.get("value", "") for child in element
            if _local(child.tag) in ("string", "date", "int", "float", "boolean") and "key" in child.attrib}

def _percentile(values, fraction: float):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]

def build_audit() -> dict:
    path = Path(raw_data_path("r2_sepsis", "Sepsis Cases - Event Log.xes.gz"))
    raw = path.read_bytes(); md5 = hashlib.md5(raw).hexdigest()
    if md5 != SOURCE_MD5:
        raise ValueError("Sepsis source MD5 mismatch")
    traces = events = non_monotonic = 0
    activities = Counter(); lifecycles = Counter(); lengths = []; gaps = []; starts = []; ends = []
    with gzip.open(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if _local(element.tag) != "trace":
                continue
            timestamps = []
            for child in element:
                if _local(child.tag) != "event":
                    continue
                attrs = _attributes(child); timestamps.append(datetime.fromisoformat(attrs["time:timestamp"]))
                activities[attrs["concept:name"]] += 1
                lifecycles[attrs.get("lifecycle:transition", "missing")] += 1
            if timestamps:
                starts.append(timestamps[0]); ends.append(timestamps[-1])
                for left, right in zip(timestamps, timestamps[1:]):
                    gap = (right - left).total_seconds(); gaps.append(gap); non_monotonic += int(gap < 0)
            traces += 1; events += len(timestamps); lengths.append(len(timestamps)); element.clear()
    return {"schema": "sara-r2-sepsis-source-audit-v1", "source": {
        "title": "Sepsis Cases - Event Log", "publisher": "Eindhoven University of Technology / 4TU.ResearchData",
        "doi": "10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460",
        "landing_page": "https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639",
        "file_id": 24061976, "md5": md5, "sha256": hashlib.sha256(raw).hexdigest(),
        "terms": "4TU General Terms of Use", "timestamp_note": "Absolute timestamps randomized; within-trace intervals preserved"},
        "trace_count": traces, "event_count": events, "activity_count": len(activities),
        "lifecycle_count": len(lifecycles), "time_start": min(starts).isoformat(), "time_end": max(ends).isoformat(),
        "case_start_percentiles": {"p70": _percentile(starts, .70).isoformat(), "p85": _percentile(starts, .85).isoformat()},
        "trace_length": {"minimum": min(lengths), "median": _percentile(lengths, .5), "p90": _percentile(lengths, .9), "maximum": max(lengths)},
        "inter_event_seconds": {"count": len(gaps), "minimum": min(gaps), "median": _percentile(gaps, .5),
            "p90": _percentile(gaps, .9), "p99": _percentile(gaps, .99), "maximum": max(gaps),
            "distinct_values": len(set(gaps)), "negative_count": non_monotonic, "zero_count": sum(value == 0 for value in gaps)},
        "activities": dict(sorted(activities.items())), "lifecycles": dict(sorted(lifecycles.items())),
        "candidate_task": {"prediction": "next activity within the same hospital case",
            "available_timing": "within-trace elapsed time preserved by the publisher",
            "timing_ablation": "replace positive gap buckets with one constant while preserving event order",
            "case_isolation_required": True}}

def main() -> int:
    output = Path(ensure_parent_directory(workspace_path("evaluation", "r2_sepsis_source_audit.json")))
    audit = build_audit(); output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "traces": audit["trace_count"], "events": audit["event_count"],
        "activities": audit["activity_count"], "gap_values": audit["inter_event_seconds"]["distinct_values"]}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
