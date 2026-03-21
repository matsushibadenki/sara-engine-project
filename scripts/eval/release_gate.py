# Directory Path: scripts/eval/release_gate.py
# English Title: Release Gate Validator
# Purpose/Content: Validates release readiness from soak reports and packaging metadata, then exits non-zero when required gates are not satisfied.

import argparse
import json
import os
import re
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "release", "release_soak_report.json")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _int_value(container: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(container.get(key, default))
    except (TypeError, ValueError):
        return default


def validate_release_report(report: Dict[str, object]) -> List[str]:
    errors: List[str] = []

    agent = report.get("agent", {})
    inference = report.get("inference", {})
    criteria = report.get("criteria", {})

    min_agent_turns = 24
    min_inference_iterations = 32
    min_pattern_count = 1
    min_duration_seconds = 5.0

    if isinstance(criteria, dict):
        min_agent_turns = _int_value(criteria, "min_agent_turns", min_agent_turns)
        min_inference_iterations = _int_value(criteria, "min_inference_iterations", min_inference_iterations)
        min_pattern_count = _int_value(criteria, "min_pattern_count", min_pattern_count)
        min_duration_raw = criteria.get("min_duration_seconds", min_duration_seconds)
        try:
            if isinstance(min_duration_raw, (int, float, str)):
                min_duration_seconds = float(min_duration_raw)
        except (TypeError, ValueError):
            min_duration_seconds = 5.0

    actual_duration_raw = report.get("duration_seconds", 0.0)
    try:
        if isinstance(actual_duration_raw, (int, float, str)):
            actual_duration_seconds = float(actual_duration_raw)
        else:
            actual_duration_seconds = 0.0
    except (TypeError, ValueError):
        actual_duration_seconds = 0.0

    if actual_duration_seconds < min_duration_seconds:
        errors.append(f"Soak duration is below the minimum required window ({min_duration_seconds} seconds).")

    if not isinstance(agent, dict) or not agent.get("history_bounded", False):
        errors.append("Agent history is not bounded.")
    if not isinstance(agent, dict) or int(agent.get("issue_count", 0)) > 0:
        errors.append("Agent soak recorded runtime issues.")
    if not isinstance(agent, dict) or _int_value(agent, "turns", 0) < min_agent_turns:
        errors.append(f"Agent soak did not reach the minimum turn count ({min_agent_turns}).")
    if not isinstance(inference, dict) or not inference.get("roundtrip_ok", False):
        errors.append("Inference memory round-trip failed.")
    if not isinstance(inference, dict) or not inference.get("tuple_keys_only", False):
        errors.append("Inference memory uses non-tuple keys.")
    if not isinstance(inference, dict) or _int_value(inference, "iterations", 0) < min_inference_iterations:
        errors.append(
            f"Inference soak did not reach the minimum iteration count ({min_inference_iterations})."
        )
    if isinstance(inference, dict) and _int_value(inference, "pattern_count", 0) < min_pattern_count:
        errors.append(
            f"Inference soak did not produce the minimum memory patterns ({min_pattern_count})."
        )

    return errors


def validate_packaging_metadata(project_root: str) -> List[str]:
    errors: List[str] = []
    pyproject = _read_text(os.path.join(project_root, "pyproject.toml"))
    cargo = _read_text(os.path.join(project_root, "Cargo.toml"))

    pyproject_version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    cargo_version = re.search(r'^version = "([^"]+)"', cargo, re.MULTILINE)

    if pyproject_version is None or cargo_version is None:
        errors.append("Could not read version from pyproject.toml or Cargo.toml.")
    elif pyproject_version.group(1) != cargo_version.group(1):
        errors.append("pyproject.toml and Cargo.toml versions do not match.")

    if 'sara-chat = "sara_engine.cli:chat"' not in pyproject:
        errors.append("Missing sara-chat console script entry.")
    if 'sara-train = "sara_engine.cli:train"' not in pyproject:
        errors.append("Missing sara-train console script entry.")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate release readiness from soak report and metadata.")
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="Managed path to the release soak report JSON.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.report_path):
        print(f"Release gate failed: soak report not found at {args.report_path}")
        raise SystemExit(1)

    with open(args.report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    errors = []
    errors.extend(validate_release_report(report))
    errors.extend(validate_packaging_metadata(PROJECT_ROOT))

    if errors:
        print("Release gate failed:")
        for item in errors:
            print(f"- {item}")
        raise SystemExit(1)

    print("Release gate passed.")


if __name__ == "__main__":
    main()
