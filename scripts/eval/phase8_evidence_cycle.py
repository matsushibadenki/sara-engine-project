#!/usr/bin/env python3
"""Run the managed Phase 8 external-validity, comparison, and completion cycle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


DEFAULT_CORPUS_PATH = processed_data_path("corpus.txt")
DEFAULT_EXTERNAL_PATH = workspace_path("evaluation", "real_data_external_validity.json")
DEFAULT_EXTERNAL_SUMMARY_PATH = workspace_path("evaluation", "real_data_external_validity_summary.txt")
DEFAULT_EXTERNAL_HISTORY_PATH = workspace_path("evaluation", "real_data_external_validity_history.json")
DEFAULT_LADDER_PATH = workspace_path("evaluation", "real_data_external_validity_ladder.json")
DEFAULT_LADDER_SUMMARY_PATH = workspace_path("evaluation", "real_data_external_validity_ladder_summary.txt")
DEFAULT_COMPARISON_PATH = workspace_path("evaluation", "sara_ann_comparison_report.json")
DEFAULT_COMPARISON_SUMMARY_PATH = workspace_path("evaluation", "sara_ann_comparison_report.txt")
DEFAULT_GATE_PATH = workspace_path("evaluation", "phase8_completion_gate.json")
DEFAULT_GATE_SUMMARY_PATH = workspace_path("evaluation", "phase8_completion_gate_summary.txt")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase8_evidence_cycle.json")


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run(command: Sequence[str]) -> int:
    result = subprocess.run(list(command), cwd=PROJECT_ROOT)
    return int(result.returncode)


def build_cycle_report(
    *,
    external_validity: Mapping[str, Any],
    ladder: Mapping[str, Any],
    comparison: Mapping[str, Any],
    gate: Mapping[str, Any],
    return_codes: Mapping[str, int],
) -> Dict[str, Any]:
    return {
        "schema": "sara-phase8-evidence-cycle-v1",
        "passed": bool(gate.get("phase8_complete", False)),
        "implementation_ready": bool(gate.get("implementation_ready", False)),
        "status": str(gate.get("status", "implementation_repair_required")),
        "stages": {
            "external_validity_passed": bool(external_validity.get("passed", False)),
            "ladder_passed": bool(ladder.get("passed", False)),
            "comparison_passed": bool(comparison.get("passed", False)),
            "phase8_complete": bool(gate.get("phase8_complete", False)),
        },
        "return_codes": {str(key): int(value) for key, value in return_codes.items()},
        "next_action": str(gate.get("next_action", "")),
        "policy_notes": [
            "A comparison stage may return non-zero while producing a valid partial report; the completion gate is the promotion authority.",
            "Phase 6 physical-energy evidence remains a separate claim tier.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--pretrained-embedding-model", default="")
    parser.add_argument("--cross-encoder-model", default="")
    parser.add_argument("--external-report-path", default=DEFAULT_EXTERNAL_PATH)
    parser.add_argument("--external-summary-path", default=DEFAULT_EXTERNAL_SUMMARY_PATH)
    parser.add_argument("--external-history-path", default=DEFAULT_EXTERNAL_HISTORY_PATH)
    parser.add_argument("--ladder-report-path", default=DEFAULT_LADDER_PATH)
    parser.add_argument("--ladder-summary-path", default=DEFAULT_LADDER_SUMMARY_PATH)
    parser.add_argument("--comparison-path", default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--comparison-summary-path", default=DEFAULT_COMPARISON_SUMMARY_PATH)
    parser.add_argument("--gate-path", default=DEFAULT_GATE_PATH)
    parser.add_argument("--gate-summary-path", default=DEFAULT_GATE_SUMMARY_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--max-docs", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--no-history-update", action="store_true")
    args = parser.parse_args(argv)

    base = [sys.executable, "scripts/eval/real_data_external_validity.py", "--corpus", args.corpus, "--max-docs", str(args.max_docs), "--max-cases", str(args.max_cases), "--report-path", args.external_report_path, "--summary-path", args.external_summary_path, "--history-path", args.external_history_path]
    if args.pretrained_embedding_model:
        base.extend(["--pretrained-embedding-model", args.pretrained_embedding_model])
    if args.cross_encoder_model:
        base.extend(["--cross-encoder-model", args.cross_encoder_model])
    if args.no_history_update:
        base.append("--no-history-update")
    return_codes = {"external_validity": _run(base)}

    ladder = [sys.executable, "scripts/eval/real_data_external_validity_ladder.py", "--corpus", args.corpus, "--report-path", args.ladder_report_path, "--summary-path", args.ladder_summary_path]
    if args.pretrained_embedding_model:
        ladder.extend(["--pretrained-embedding-model", args.pretrained_embedding_model])
    if args.cross_encoder_model:
        ladder.extend(["--cross-encoder-model", args.cross_encoder_model])
    if args.no_history_update:
        ladder.append("--no-history-update")
    return_codes["ladder"] = _run(ladder)

    comparison = [sys.executable, "scripts/eval/sara_ann_comparison_report.py", "--external-validity-report-path", args.external_report_path, "--external-ladder-report-path", args.ladder_report_path, "--report-path", args.comparison_path, "--summary-path", args.comparison_summary_path]
    return_codes["comparison"] = _run(comparison)

    gate = [sys.executable, "scripts/eval/phase8_completion_gate.py", "--comparison-path", args.comparison_path, "--report-path", args.gate_path, "--summary-path", args.gate_summary_path]
    return_codes["completion_gate"] = _run(gate)
    cycle = build_cycle_report(
        external_validity=_load_json(args.external_report_path),
        ladder=_load_json(args.ladder_report_path),
        comparison=_load_json(args.comparison_path),
        gate=_load_json(args.gate_path),
        return_codes=return_codes,
    )
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(cycle, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if cycle["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
