#!/usr/bin/env python3
"""Execute the frozen evaluator-isolated Phase 30 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.phase30_benchmark import evaluate_frozen_decisions, freeze_decisions  # noqa: E402
from sara_engine.utils.project_paths import processed_data_path, workspace_path  # noqa: E402


DEFAULT_INPUTS = processed_data_path("benchmark_fixtures", "phase30_temporal_effective_interaction_cases.jsonl")
DEFAULT_KEY = processed_data_path("benchmark_fixtures", "phase30_temporal_effective_interaction_evaluator_key.jsonl")
DEFAULT_FIXTURE_MANIFEST = workspace_path("evaluation", "phase30_temporal_effective_interaction_fixture_freeze.json")
DEFAULT_PREREGISTRATION = workspace_path("evaluation", "phase30_temporal_effective_interaction_preregistration.json")
DEFAULT_DECISIONS = workspace_path("evaluation", "phase30_temporal_effective_interaction_decisions.json")
DEFAULT_REPORT = workspace_path("evaluation", "phase30_temporal_effective_interaction_benchmark.json")


def _jsonl(path: str):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=DEFAULT_INPUTS)
    parser.add_argument("--evaluator-key", default=DEFAULT_KEY)
    parser.add_argument("--fixture-manifest", default=DEFAULT_FIXTURE_MANIFEST)
    parser.add_argument("--preregistration", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--output", default=DEFAULT_REPORT)
    args = parser.parse_args()

    inputs = _jsonl(args.inputs)
    decisions, decision_identity = freeze_decisions(inputs)
    decisions_path = Path(args.decisions)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(json.dumps({"identity": decision_identity, "decisions": decisions}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evaluator_keys = _jsonl(args.evaluator_key)
    fixture_manifest = json.loads(Path(args.fixture_manifest).read_text(encoding="utf-8"))
    preregistration = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    report = evaluate_frozen_decisions(inputs, evaluator_keys, fixture_manifest, preregistration, decisions, decision_identity)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("report_digest", "threshold_gate_passed", "budget_gate_passed", "comparative_gate_passed", "mechanism_gate_passed", "promotion_ready")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
