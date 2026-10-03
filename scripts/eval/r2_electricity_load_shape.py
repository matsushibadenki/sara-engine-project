#!/usr/bin/env python3
"""Audit frozen electricity archive structure without parsing load cells."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "r2_electricity_load_source_protocol_v1.json"))
PROTOCOL_HASH = "9828a9e359c5ebfb3e6fc88f1a399b562e31a99c3d27b07c5805773101509434"
ARCHIVE_HASH = "f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1"


def file_hash(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_lines(lines, *, client_count: int, timestamp_count: int, interval_minutes: int):
    header = next(lines).rstrip("\r\n").split(";")
    clients = [cell.strip('"') for cell in header[1:]]
    count = malformed = duplicate_or_regressive = gaps = 0
    previous = first = last = None
    years = Counter()
    for line in lines:
        count += 1
        # Count separators and inspect only the timestamp prefix. Load cells stay opaque.
        if line.count(";") != client_count:
            malformed += 1
            continue
        prefix = line.partition(";")[0].strip('"')
        try:
            current = datetime.strptime(prefix, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            malformed += 1
            continue
        if previous is not None:
            duplicate_or_regressive += int(current <= previous)
            gaps += int(current - previous != timedelta(minutes=interval_minutes))
        first = current if first is None else first
        last = previous = current
        years[str(current.year)] += 1
    checks = {
        "client_columns": len(clients) == client_count,
        "unique_client_columns": len(set(clients)) == len(clients) and all(clients),
        "timestamp_count": count == timestamp_count,
        "row_shape_and_timestamp_format": malformed == 0,
        "strict_timestamp_order": duplicate_or_regressive == 0,
        "regular_interval": gaps == 0,
    }
    return {
        "schema": "sara-r2-electricity-load-shape-v1",
        "passed": all(checks.values()), "checks": checks,
        "row_count": count, "client_count": len(clients),
        "timestamp_counts_by_year": dict(sorted(years.items())),
        "first_timestamp": first.isoformat() if first else None,
        "last_timestamp": last.isoformat() if last else None,
        "malformed_row_count": malformed,
        "duplicate_or_regressive_count": duplicate_or_regressive,
        "interval_violation_count": gaps,
        "load_cells_parsed": 0, "candidate_scored": False,
        "outcome_support_checked": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=workspace_path("evaluation", "r2_electricity_load_shape_v1.json"))
    args = parser.parse_args()
    output = Path(ensure_parent_directory(args.report_path))
    if file_hash(PROTOCOL) != PROTOCOL_HASH:
        raise ValueError("Frozen source protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    source = protocol["source"]
    archive = Path(project_path(source["archive_path"]))
    if archive.stat().st_size > source["maximum_archive_bytes"] or file_hash(archive) != ARCHIVE_HASH:
        raise ValueError("Frozen archive identity changed")
    contract = protocol["catalog_contract"]
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open(source["expected_primary_member"]) as member:
            with io.TextIOWrapper(member, encoding="utf-8-sig") as lines:
                report = audit_lines(iter(lines), client_count=contract["client_count"],
                                     timestamp_count=contract["timestamp_count"],
                                     interval_minutes=contract["interval_minutes"])
    report["protocol_sha256"] = PROTOCOL_HASH
    report["archive_sha256"] = ARCHIVE_HASH
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
