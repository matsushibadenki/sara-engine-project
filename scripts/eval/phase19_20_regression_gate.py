#!/usr/bin/env python3
"""Check the Phase 19/20 cross-treebank evidence gate without changing data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_COMPARISON = workspace_path("evaluation", "phase19_20_cross_treebank_comparison.json")
DEFAULT_SPLIT_AUDIT = workspace_path("evaluation", "audit_ud_split_isolation.json")
DEFAULT_OUTPUT = workspace_path("evaluation", "phase19_20_regression_gate.json")


def _read(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_report(*, comparison_path: str = DEFAULT_COMPARISON, split_audit_path: str = DEFAULT_SPLIT_AUDIT) -> Dict[str, Any]:
    comparison = _read(comparison_path)
    split_audit = _read(split_audit_path)
    structural = comparison["structural"]
    raw = comparison["raw_text"]
    checks = {
        "comparison_is_observed_only": comparison["observed_only"] is True,
        "split_isolation_passed": split_audit["passed"] is True,
        "gsd_test_structural_bounded": structural["GSD_test"]["bounded"] is True,
        "gsd_dev_structural_bounded": structural["GSD_dev"]["bounded"] is True,
        "pud_test_structural_bounded": structural["PUD_test"]["bounded"] is True,
        "gsd_test_control_zero": structural["GSD_test"]["control_recall"] == 0.0,
        "gsd_dev_control_zero": structural["GSD_dev"]["control_recall"] == 0.0,
        "pud_test_control_zero": structural["PUD_test"]["control_recall"] == 0.0,
        "gsd_test_raw_promotion_blocked": raw["GSD_test"]["bounded"] is False,
        "pud_test_raw_promotion_blocked": raw["PUD_test"]["bounded"] is False,
    }
    return {
        "schema": "sara-phase19-20-regression-gate-v1",
        "phase": "19/20",
        "observed_only": True,
        "comparison_path": os.path.abspath(comparison_path),
        "split_audit_path": os.path.abspath(split_audit_path),
        "checks": checks,
        "passed": all(checks.values()),
        "decision": "preserve_observed_only_status_and_raw_text_block" if all(checks.values()) else "investigate_regression",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-path", default=DEFAULT_COMPARISON)
    parser.add_argument("--split-audit-path", default=DEFAULT_SPLIT_AUDIT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report(comparison_path=args.comparison_path, split_audit_path=args.split_audit_path)
    with open(ensure_parent_directory(args.output), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"passed": report["passed"], "output": os.path.abspath(args.output)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
