# Directory Path: scripts/eval/phase3_completion_gate.py
# English Title: Phase 3 Completion Gate
# Purpose/Content: Validates Phase 3 completion status from a managed phase3 accuracy report and exits non-zero when completion criteria are not satisfied.

import argparse
import json
import os
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "evaluation", "phase3_accuracy_suite.json")


def _load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Phase 3 report is not a JSON object.")
    return payload


def validate_phase3_completion(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    completion = report.get("phase3_completion", {})
    if not isinstance(completion, dict):
        return ["phase3_completion block is missing."]

    if not bool(completion.get("passed", False)):
        errors.append("phase3_completion gate did not pass.")
    completion_score = float(completion.get("completion_score", 0.0) or 0.0)
    if completion_score < 1.0:
        errors.append(
            f"phase3_completion score is below the required complete threshold (value={completion_score:.3f}, required>=1.000)."
        )
    failed_checks = completion.get("failed_checks", [])
    if isinstance(failed_checks, list) and failed_checks:
        errors.append("phase3_completion has failed checks: " + ", ".join(str(item) for item in failed_checks))
    checks = completion.get("checks", {})
    if isinstance(checks, dict):
        missing_or_failed = sorted(name for name, passed in checks.items() if not bool(passed))
        if missing_or_failed:
            errors.append("phase3_completion check map contains failed checks: " + ", ".join(missing_or_failed))

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 3 completion gate from phase3_accuracy_suite report.")
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="Managed path to phase3 accuracy suite report JSON.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.report_path):
        print(f"Phase 3 completion gate failed: report not found at {args.report_path}")
        raise SystemExit(1)

    try:
        report = _load_report(args.report_path)
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"Phase 3 completion gate failed: {exc}")
        raise SystemExit(1)

    errors = validate_phase3_completion(report)
    if errors:
        print("Phase 3 completion gate failed:")
        for item in errors:
            print(f"- {item}")
        raise SystemExit(1)

    completion = report.get("phase3_completion", {}) if isinstance(report.get("phase3_completion"), dict) else {}
    print("Phase 3 completion gate passed.")
    print(f"completion_score={float(completion.get('completion_score', 0.0)):.3f}")


if __name__ == "__main__":
    main()
