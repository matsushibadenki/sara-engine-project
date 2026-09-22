#!/usr/bin/env python3
"""Check prospective Helpdesk outcome support from activity/time metadata only."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_helpdesk", "helpdesk_anonymized.csv"))
SOURCE_AUDIT = Path(workspace_path("evaluation", "helpdesk_source_audit_v1.json"))
PROTOCOL = Path(processed_data_path("benchmark_fixtures", "helpdesk_prospective_feasibility_protocol_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "helpdesk_prospective_feasibility_v1.json"))
PROTOCOL_SHA256 = "d96369420df88d11453219810c91f05ebaacfd46304522c6c642fb011ff97119"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Helpdesk feasibility result already exists")
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Helpdesk feasibility protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if digest(SOURCE) != protocol["source_sha256"] or digest(SOURCE_AUDIT) != protocol["source_audit_sha256"]:
        raise ValueError("Pinned Helpdesk source or audit changed")
    histories: dict[str, list[tuple[datetime, int, str]]] = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = {name: header.index(name) for name in protocol["allowed_columns"]}
        for ordinal, row in enumerate(reader):
            case_id = row[indices["Case ID"]]
            activity = row[indices["Activity"]]
            timestamp = datetime.strptime(row[indices["Complete Timestamp"]], "%Y/%m/%d %H:%M:%S.%f")
            histories[case_id].append((timestamp, ordinal, activity))
    year_to_split = {year: split for split, years in protocol["opening_year_splits"].items() for year in years}
    counts = {split: {"positive": 0, "negative": 0} for split in protocol["opening_year_splits"]}
    exclusions = Counter()
    decision_to_close_days = {split: {"positive": [], "negative": []} for split in counts}
    horizon = timedelta(days=protocol["outcome_rule"]["horizon_days"])
    for history in histories.values():
        history.sort()
        opened = history[0][0]
        split = year_to_split.get(opened.year)
        if split is None:
            exclusions["outside_split_years"] += 1
            continue
        closed = sorted({stamp for stamp, _, activity in history if activity == "Closed"})
        if len(closed) != 1:
            exclusions["missing_or_conflicting_closure"] += 1
            continue
        updates = sorted({stamp for stamp, _, activity in history
                          if opened < stamp < closed[0] and activity not in {"Closed", "Resolve ticket"}})
        if len(updates) < 2:
            exclusions["insufficient_preterminal_updates"] += 1
            continue
        cutoff = updates[1]
        if closed[0] <= cutoff:
            exclusions["invalid_outcome_order"] += 1
            continue
        label = "positive" if closed[0] <= cutoff + horizon else "negative"
        counts[split][label] += 1
        decision_to_close_days[split][label].append((closed[0] - cutoff).total_seconds() / 86400)
    checks = {split: all(counts[split][label] >= minimum for label in ("positive", "negative"))
              for split, minimum in protocol["minimum_per_class"].items()}
    result = {
        "schema": "sara-helpdesk-prospective-feasibility-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": protocol["source_sha256"],
        "case_counts": counts,
        "exclusions": dict(exclusions),
        "lead_days_min_max": {split: {label: [min(values), max(values)] if values else None
                                      for label, values in groups.items()}
                              for split, groups in decision_to_close_days.items()},
        "checks": checks,
        "passed": all(checks.values()),
        "candidate_scored": False,
        "descriptive_features_read": False,
        "heldout_features_materialized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
