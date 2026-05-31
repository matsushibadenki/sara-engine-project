#!/usr/bin/env python3
"""Audit ROADMAP closure markers without creating unmanaged artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_CLOSURE_MARKERS = (
    "release-critical path",
    "observed-only",
    "long-term research backlog",
    "roadmap completion audit",
    "research product completion gate",
)


def _extract_closure_section_lines(lines: List[str]) -> List[Dict[str, Any]]:
    section_lines: List[Dict[str, Any]] = []
    in_section = False
    for index, line in enumerate(lines):
        lower_line = line.lower()
        if "roadmap closure audit" in lower_line:
            in_section = True
        elif in_section and line.startswith("* **"):
            break
        if in_section:
            section_lines.append({"line": index + 1, "text": line.strip()})
    return section_lines


def audit_roadmap_text(text: str) -> Dict[str, Any]:
    lines = text.splitlines()
    closure_section_lines = _extract_closure_section_lines(lines)
    closure_present = bool(closure_section_lines)
    closure_done_lines = [
        entry
        for entry in closure_section_lines
        if "DONE:" in str(entry.get("text", ""))
    ]
    closure_done_text = "\n".join(
        str(entry.get("text", "")) for entry in closure_done_lines
    ).lower()
    missing_markers = [
        marker
        for marker in REQUIRED_CLOSURE_MARKERS
        if marker.lower() not in closure_done_text
    ]
    unchecked_lines = [
        {"line": index + 1, "text": line.strip()}
        for index, line in enumerate(lines)
        if "[ ]" in line
    ]
    long_term_lines = [
        {"line": index + 1, "text": line.strip()}
        for index, line in enumerate(lines)
        if "中長期" in line or "long-term" in line.lower()
    ]
    candidate_lines = [
        {"line": index + 1, "text": line.strip()}
        for index, line in enumerate(lines)
        if "候補" in line or "candidate" in line.lower()
    ]
    passed = bool(closure_present and not missing_markers and not unchecked_lines)
    return {
        "schema": "sara-roadmap-completion-audit-v1",
        "passed": passed,
        "closure_present": closure_present,
        "closure_done_count": len(closure_done_lines),
        "closure_done_markers": closure_done_lines,
        "missing_markers": missing_markers,
        "unchecked_marker_count": len(unchecked_lines),
        "unchecked_markers": unchecked_lines[:20],
        "long_term_backlog_line_count": len(long_term_lines),
        "candidate_line_count": len(candidate_lines),
        "status": "complete" if passed else "needs_review",
    }


def audit_roadmap_path(path: Path) -> Dict[str, Any]:
    return audit_roadmap_text(path.read_text(encoding="utf-8"))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit ROADMAP closure status.")
    parser.add_argument(
        "--roadmap-path",
        default="doc/ROADMAP.md",
        help="Path to ROADMAP.md.",
    )
    args = parser.parse_args(argv)
    report = audit_roadmap_path(Path(args.roadmap_path))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
