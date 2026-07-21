#!/usr/bin/env python3
"""Prepare a larger internal experiment without running it before evidence gates pass."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402

DEFAULT_PROMOTION_GATE = workspace_path("evaluation", "next_level_promotion_gate.json")
DEFAULT_EXTERNAL_GATE = workspace_path("evaluation", "continual_horizon_external_gate.json")
DEFAULT_OUTPUT = workspace_path("evaluation", "scale_up_experiment_readiness.json")


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def build_readiness(
    promotion_gate: Mapping[str, Any],
    external_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers = []
    if not bool(promotion_gate.get("promotion_allowed", False)):
        blockers.append("next_level_promotion_gate_blocked")
    if not bool(external_gate.get("promotion_allowed", False)):
        blockers.append("independent_horizon_coverage_incomplete")
    plan = {
        "profiles": ["frozen_control", "event_memory", "structural_feedback_event_memory"],
        "episode_buckets": [1000, 10000],
        "domains": 4,
        "replicates_per_condition": 5,
        "equal_state_budget": 128,
        "equal_event_budget_per_episode": 256,
        "metrics": [
            "revision_uptake_latency",
            "retained_useful_recall",
            "catastrophic_interference",
            "abstention_integrity",
            "state_growth",
            "event_cost",
            "latency",
            "provenance_completeness",
        ],
        "execution_policy": {
            "cpu_only": True,
            "gpu_required": False,
            "external_device_required": False,
            "physical_energy_claim": False,
            "network_collection": False,
        },
    }
    return {
        "schema": "sara-scale-up-experiment-readiness-v1",
        "ready_to_execute": not blockers,
        "observed_only": True,
        "blockers": blockers,
        "plan": plan,
        "required_before_execution": [
            "complete independent 10/30/100 horizon coverage",
            "complete human promotion review",
            "freeze fixture, source, and environment fingerprints",
            "record pre-registered thresholds before execution",
        ],
        "policy": "planning_only; no large run or promotion is performed by this command",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-gate", default=DEFAULT_PROMOTION_GATE)
    parser.add_argument("--external-gate", default=DEFAULT_EXTERNAL_GATE)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_readiness(_read_json(args.promotion_gate), _read_json(args.external_gate))
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
