# Directory Path: scripts/eval/phase4_operational_cycle.py
# English Title: Phase4 Operational Cycle Runner
# Purpose/Content: Runs periodic release/extended operational readiness cycles and persists a machine-readable report for Phase4 continuous-learning operations.

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.dirname(__file__)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", "cache"))


def _load_project_paths_helpers():
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "utils", "project_paths.py")
    spec = importlib.util.spec_from_file_location("project_paths_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load project paths helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ensure_parent = getattr(module, "ensure_parent_directory", None)
    workspace = getattr(module, "workspace_path", None)
    if not callable(ensure_parent) or not callable(workspace):
        raise RuntimeError("project_paths helper is missing required callables.")
    return ensure_parent, workspace


ensure_parent_directory, workspace_path = _load_project_paths_helpers()


DEFAULT_CYCLE_REPORT_PATH = workspace_path("release", "phase4_operational_cycle_report.json")
DEFAULT_CYCLE_SUMMARY_PATH = workspace_path("release", "phase4_operational_cycle_summary.txt")


def _build_operational_command(
    profile: str,
    *,
    include_accuracy: bool = False,
    runbook_action_limit: int = 2,
    runbook_max_actions: int = 50,
    runbook_max_per_source: int = 0,
    runbook_drop_rate_threshold: float = 0.9,
    v1_actions_max_age_seconds: float = 86400.0,
) -> List[str]:
    command: List[str] = [
        sys.executable,
        os.path.join("scripts", "eval", "operational_readiness.py"),
        "--refresh-artifacts",
        "--soak-profile",
        str(profile),
        "--append-iterative-next-actions",
        "--append-runbook-actions",
        "--append-runbook-actions-min-priority",
        "medium",
        "--append-runbook-actions-max",
        str(max(int(runbook_action_limit), 0)),
        "--runbook-max-actions",
        str(max(int(runbook_max_actions), 1)),
        "--runbook-max-per-source",
        str(max(int(runbook_max_per_source), 0)),
        "--runbook-drop-rate-threshold",
        f"{max(float(runbook_drop_rate_threshold), 0.0):.3f}",
        "--v1-actions-max-age-seconds",
        f"{max(float(v1_actions_max_age_seconds), 0.0):.1f}",
    ]
    if str(profile).strip().lower() == "extended":
        command.append("--strict-production")
        include_accuracy = True
    if include_accuracy:
        command.append("--include-accuracy")
    return command


def _run_command(command: List[str]) -> Dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration = time.time() - started
    return {
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "passed": completed.returncode == 0,
        "duration_seconds": float(duration),
        "stdout_tail": str(completed.stdout or "")[-3000:],
        "stderr_tail": str(completed.stderr or "")[-3000:],
    }


def _summarize_cycle(report: Dict[str, Any]) -> str:
    runs = report.get("runs", []) if isinstance(report.get("runs"), list) else []
    lines = [
        "SARA Engine Phase4 Operational Cycle Summary",
        f"- status: {'PASS' if bool(report.get('passed', False)) else 'FAIL'}",
        f"- cycle_count: {len(runs)}",
        f"- dry_run: {bool(report.get('dry_run', False))}",
    ]
    for item in runs:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- cycle: "
            f"profile={str(item.get('profile', 'unknown'))} "
            f"status={'PASS' if bool(item.get('passed', False)) else 'FAIL'} "
            f"duration_seconds={float(item.get('duration_seconds', 0.0) or 0.0):.2f}"
        )
    return "\n".join(lines)


def run_phase4_operational_cycle(
    *,
    profiles: List[str],
    include_accuracy: bool = False,
    runbook_action_limit: int = 2,
    runbook_max_actions: int = 50,
    runbook_max_per_source: int = 0,
    runbook_drop_rate_threshold: float = 0.9,
    v1_actions_max_age_seconds: float = 86400.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    cycle_passed = True
    for profile in profiles:
        command = _build_operational_command(
            profile,
            include_accuracy=bool(include_accuracy),
            runbook_action_limit=int(runbook_action_limit),
            runbook_max_actions=int(runbook_max_actions),
            runbook_max_per_source=int(runbook_max_per_source),
            runbook_drop_rate_threshold=float(runbook_drop_rate_threshold),
            v1_actions_max_age_seconds=float(v1_actions_max_age_seconds),
        )
        if dry_run:
            result = {
                "profile": str(profile),
                "command": " ".join(command),
                "returncode": 0,
                "passed": True,
                "duration_seconds": 0.0,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        else:
            result = {"profile": str(profile), **_run_command(command)}
        runs.append(result)
        if not result.get("passed", False):
            cycle_passed = False
    return {
        "suite_name": "Phase4OperationalCycle",
        "passed": bool(cycle_passed),
        "dry_run": bool(dry_run),
        "profiles": [str(item) for item in profiles],
        "runs": runs,
        "generated_at": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run periodic Phase4 operational cycles for release/extended profiles.")
    parser.add_argument(
        "--profiles",
        default="release,extended",
        help="Comma-separated soak profiles to run in order.",
    )
    parser.add_argument(
        "--include-accuracy",
        action="store_true",
        help="Force include-accuracy on non-extended profile runs too.",
    )
    parser.add_argument(
        "--runbook-action-limit",
        type=int,
        default=2,
        help="Maximum runbook manifest actions appended per cycle run.",
    )
    parser.add_argument(
        "--runbook-max-actions",
        type=int,
        default=50,
        help="Global cap for operational runbook action manifest generation.",
    )
    parser.add_argument(
        "--runbook-max-per-source",
        type=int,
        default=0,
        help="Per-source cap for operational runbook action manifest generation.",
    )
    parser.add_argument(
        "--runbook-drop-rate-threshold",
        type=float,
        default=0.9,
        help="Drop-rate warning threshold for operational runbook checklist.",
    )
    parser.add_argument(
        "--v1-actions-max-age-seconds",
        type=float,
        default=86400.0,
        help="Maximum allowed age for imported v1 recovery actions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute commands; only emit planned command report.",
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_CYCLE_REPORT_PATH,
        help="Managed output path for cycle report JSON.",
    )
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_CYCLE_SUMMARY_PATH,
        help="Managed output path for cycle summary text.",
    )
    args = parser.parse_args()

    profiles = [token.strip() for token in str(args.profiles).split(",") if token.strip()]
    if not profiles:
        print("No profile provided.")
        return 1

    report = run_phase4_operational_cycle(
        profiles=profiles,
        include_accuracy=bool(args.include_accuracy),
        runbook_action_limit=int(args.runbook_action_limit),
        runbook_max_actions=int(args.runbook_max_actions),
        runbook_max_per_source=int(args.runbook_max_per_source),
        runbook_drop_rate_threshold=float(args.runbook_drop_rate_threshold),
        v1_actions_max_age_seconds=float(args.v1_actions_max_age_seconds),
        dry_run=bool(args.dry_run),
    )
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(_summarize_cycle(report))

    if not report.get("passed", False):
        print("Phase4 operational cycle failed.")
        print(f"Saved report: {report_path}")
        print(f"Saved summary: {summary_path}")
        return 1
    print("Phase4 operational cycle completed.")
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
