# Directory Path: scripts/eval/release_gate.py
# English Title: Release Gate Validator
# Purpose/Content: Validates release readiness from soak reports and packaging metadata, then exits non-zero when required gates are not satisfied.

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "release", "release_soak_report.json")
DEFAULT_ACCURACY_REPORT_PATH = os.path.join(
    PROJECT_ROOT, "workspace", "evaluation", "phase3_accuracy_suite.json"
)


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
    embedded_accuracy = report.get("accuracy")
    release_metadata = report.get("release_metadata")

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

    require_phase3_accuracy = False
    if isinstance(criteria, dict):
        require_phase3_accuracy = bool(criteria.get("require_phase3_accuracy", False))
    if require_phase3_accuracy:
        if not isinstance(embedded_accuracy, dict):
            errors.append("Release soak report requires embedded Phase 3 accuracy results.")
        else:
            errors.extend(validate_phase3_accuracy_report(embedded_accuracy))

    if isinstance(release_metadata, dict):
        if not bool(release_metadata.get("versions_match", False)):
            errors.append("Embedded release metadata reports mismatched package versions.")
        if not bool(release_metadata.get("has_expected_console_scripts", False)):
            errors.append("Embedded release metadata reports missing console scripts.")
        if not str(release_metadata.get("release_notes_heading", "")).strip():
            errors.append("Embedded release metadata is missing a release notes heading.")

    return errors


def _float_value(container: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(container.get(key, default))
    except (TypeError, ValueError):
        return default


def validate_phase3_accuracy_report(report: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    if not isinstance(report, dict):
        return ["Phase 3 accuracy report is not a valid object."]

    if report.get("suite_name") != "Phase3AccuracySuite":
        errors.append("Phase 3 accuracy report has an unexpected suite name.")

    if not bool(report.get("passed", False)):
        errors.append("Phase 3 accuracy suite did not pass.")

    overall_score = _float_value(report, "overall_score", 0.0)
    if overall_score <= 0.0:
        errors.append("Phase 3 accuracy suite overall score is missing or invalid.")

    component_reports = report.get("component_reports", {})
    required_components = {"agent_dialogue", "sara_inference", "spiking_llm"}
    if not isinstance(component_reports, dict):
        errors.append("Phase 3 accuracy suite is missing component reports.")
        return errors

    missing_components = sorted(required_components.difference(component_reports.keys()))
    if missing_components:
        errors.append(
            "Phase 3 accuracy suite is missing required components: "
            + ", ".join(missing_components)
            + "."
        )

    for component_name in sorted(required_components.intersection(component_reports.keys())):
        component = component_reports.get(component_name, {})
        if not isinstance(component, dict):
            errors.append(f"Phase 3 component '{component_name}' is not a valid object.")
            continue
        if not bool(component.get("passed", False)):
            errors.append(f"Phase 3 component '{component_name}' did not pass.")

    trend = report.get("trend", {})
    if isinstance(trend, dict) and bool(trend.get("has_previous", False)):
        regression_count = _int_value(trend, "regression_count", 0)
        if regression_count > 0:
            errors.append(
                f"Phase 3 accuracy suite detected {regression_count} metric regression(s) versus the previous run."
            )

    focus_summary = report.get("focus_summary", {})
    required_focus = {"few_shot", "continual"}
    if not isinstance(focus_summary, dict):
        errors.append("Phase 3 accuracy suite is missing focus summary data.")
        return errors

    missing_focus = sorted(required_focus.difference(focus_summary.keys()))
    if missing_focus:
        errors.append(
            "Phase 3 accuracy suite is missing focus summaries: "
            + ", ".join(missing_focus)
            + "."
        )

    for focus_name in sorted(required_focus.intersection(focus_summary.keys())):
        focus_report = focus_summary.get(focus_name, {})
        if not isinstance(focus_report, dict):
            errors.append(f"Phase 3 focus summary '{focus_name}' is not a valid object.")
            continue
        if not bool(focus_report.get("passed", False)):
            errors.append(f"Phase 3 focus summary '{focus_name}' did not pass.")

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
    parser.add_argument(
        "--accuracy-report-path",
        default=DEFAULT_ACCURACY_REPORT_PATH,
        help="Managed path to the Phase 3 accuracy suite JSON.",
    )
    parser.add_argument(
        "--skip-accuracy-gate",
        action="store_true",
        help="Skip validation of the Phase 3 accuracy suite report.",
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
    if not args.skip_accuracy_gate:
        if not os.path.exists(args.accuracy_report_path):
            errors.append(
                f"Phase 3 accuracy report not found at {args.accuracy_report_path}"
            )
        else:
            with open(args.accuracy_report_path, "r", encoding="utf-8") as handle:
                accuracy_report: Optional[Dict[str, Any]] = json.load(handle)
            errors.extend(validate_phase3_accuracy_report(accuracy_report))

    if errors:
        print("Release gate failed:")
        for item in errors:
            print(f"- {item}")
        raise SystemExit(1)

    print("Release gate passed.")


if __name__ == "__main__":
    main()
