#!/usr/bin/env python3
"""Build a cross-treebank comparison from preserved evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_OUTPUT = workspace_path("evaluation", "phase19_20_cross_treebank_comparison.json")


def _read(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_report() -> Dict[str, Any]:
    structural = {
        "GSD_test": _read("workspace/evaluation/semantic_echo_role_labelled_benchmark_test_large.json"),
        "GSD_dev": _read("workspace/evaluation/semantic_echo_role_labelled_benchmark_dev_large.json"),
        "PUD_test": _read("workspace/evaluation/semantic_echo_role_labelled_benchmark_pud_test.json"),
    }
    raw = {
        "GSD_test": _read("workspace/evaluation/semantic_echo_ud_text_benchmark_test.json"),
        "PUD_test": _read("workspace/evaluation/semantic_echo_ud_text_benchmark_pud_test.json"),
    }
    structural_rows = {
        name: {
            "case_count": report["metrics"]["case_count"],
            "role_binding_recall": report["metrics"]["role_binding_recall"],
            "control_recall": report["metrics"]["control_role_binding_recall"],
            "bounded": bool(report["metrics"]["bounded_execution"]),
            "replay_deterministic": bool(report["metrics"]["replay_determinism"]),
            "by_language": report["metrics"]["by_language"],
        }
        for name, report in structural.items()
    }
    raw_rows = {
        name: {
            "case_count": report["variants"]["baseline"]["case_count"],
            "baseline_accuracy": report["variants"]["baseline"]["accuracy"],
            "gap_cap_18_accuracy": report["variants"]["gap_cap_18"]["accuracy"],
            "baseline_state_bytes": report["variants"]["baseline"]["max_state_bytes"],
            "gap_cap_18_state_bytes": report["variants"]["gap_cap_18"]["max_state_bytes"],
            "bounded": bool(report["bounded_execution"]),
            "by_language_baseline": report["variants"]["baseline"]["by_language"],
            "by_language_gap_cap_18": report["variants"]["gap_cap_18"]["by_language"],
        }
        for name, report in raw.items()
    }
    return {
        "schema": "sara-phase19-20-cross-treebank-comparison-v1",
        "phase": "19/20",
        "observed_only": True,
        "structural": structural_rows,
        "raw_text": raw_rows,
        "decision": {
            "structural_evidence": "observed_only_supported",
            "raw_text_promotion": "blocked",
            "reason": "Raw-text bounded-state and language-balanced quality gates are not jointly satisfied.",
        },
        "source_reports": {
            "structural": list(structural),
            "raw_text": list(raw),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report()
    with open(ensure_parent_directory(args.output), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(Path(args.output).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
