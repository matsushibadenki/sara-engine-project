# Directory Path: scripts/eval/release_soak.py
# English Title: Release Soak Runner
# Purpose/Content: Runs a lightweight wall-clock soak test for agent dialogue and inference memory loops, then saves a managed report for release validation.

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SCRIPT_PATH not in sys.path:
    sys.path.insert(0, SCRIPT_PATH)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Keep optional plotting/font caches inside the managed workspace to avoid
# noisy warnings on restricted environments during release validation.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", "cache"))


from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path
from phase3_accuracy_suite import run_phase3_accuracy_suite
from release_gate import validate_packaging_metadata, validate_release_report


SOAK_PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "duration_seconds": 1.0,
        "max_agent_turns": 8,
        "min_agent_turns": 4,
        "max_inference_iterations": 12,
        "min_inference_iterations": 6,
        "shipping_ready": False,
    },
    "release": {
        "duration_seconds": 5.0,
        "max_agent_turns": 120,
        "min_agent_turns": 24,
        "max_inference_iterations": 256,
        "min_inference_iterations": 32,
        "shipping_ready": False,
    },
    "extended": {
        "duration_seconds": 30.0,
        "max_agent_turns": 360,
        "min_agent_turns": 60,
        "max_inference_iterations": 768,
        "min_inference_iterations": 96,
        "shipping_ready": True,
    },
}


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def collect_release_metadata(project_root: str = PROJECT_ROOT) -> Dict[str, Any]:
    pyproject_path = os.path.join(project_root, "pyproject.toml")
    cargo_path = os.path.join(project_root, "Cargo.toml")
    notes_path = os.path.join(project_root, "doc", "RELEASE_NOTES.md")

    pyproject = _read_text(pyproject_path)
    cargo = _read_text(cargo_path)
    notes = _read_text(notes_path)

    pyproject_version_match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    cargo_version_match = re.search(r'^version = "([^"]+)"', cargo, re.MULTILINE)
    current_release_heading = re.search(r"^##\s+(.+)$", notes, re.MULTILINE)
    note_sections = re.findall(r"^###\s+(.+)$", notes, re.MULTILINE)

    pyproject_version = pyproject_version_match.group(1) if pyproject_version_match else ""
    cargo_version = cargo_version_match.group(1) if cargo_version_match else ""
    console_scripts = []
    if 'sara-chat = "sara_engine.cli:chat"' in pyproject:
        console_scripts.append("sara-chat")
    if 'sara-train = "sara_engine.cli:train"' in pyproject:
        console_scripts.append("sara-train")

    return {
        "pyproject_version": pyproject_version,
        "cargo_version": cargo_version,
        "versions_match": bool(pyproject_version and pyproject_version == cargo_version),
        "console_scripts": console_scripts,
        "has_expected_console_scripts": set(console_scripts) == {"sara-chat", "sara-train"},
        "release_notes_heading": current_release_heading.group(1) if current_release_heading else "",
        "release_note_sections": note_sections,
        "release_notes_path": notes_path,
    }


def resolve_soak_profile(
    profile_name: str,
    duration_seconds: Optional[float],
    max_agent_turns: Optional[int],
    min_agent_turns: Optional[int],
    max_inference_iterations: Optional[int],
    min_inference_iterations: Optional[int],
) -> Dict[str, Any]:
    if profile_name not in SOAK_PROFILES:
        raise ValueError(f"Unknown soak profile: {profile_name}")

    baseline = dict(SOAK_PROFILES[profile_name])
    profile = dict(baseline)
    if duration_seconds is not None:
        profile["duration_seconds"] = duration_seconds
    if max_agent_turns is not None:
        profile["max_agent_turns"] = max_agent_turns
    if min_agent_turns is not None:
        profile["min_agent_turns"] = min_agent_turns
    if max_inference_iterations is not None:
        profile["max_inference_iterations"] = max_inference_iterations
    if min_inference_iterations is not None:
        profile["min_inference_iterations"] = min_inference_iterations
    profile["profile_name"] = profile_name
    profile["shipping_ready"] = bool(
        baseline["shipping_ready"]
        and profile["duration_seconds"] >= baseline["duration_seconds"]
        and profile["max_agent_turns"] >= baseline["max_agent_turns"]
        and profile["min_agent_turns"] >= baseline["min_agent_turns"]
        and profile["max_inference_iterations"] >= baseline["max_inference_iterations"]
        and profile["min_inference_iterations"] >= baseline["min_inference_iterations"]
    )
    return profile


def run_agent_soak(duration_seconds: float, max_turns: int, min_turns: int) -> Dict[str, Any]:
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    agent.register_tool("<CALC>", lambda _: "5")

    start = time.time()
    turns = 0
    while turns < max_turns and (time.time() - start) < duration_seconds:
        agent.chat(f"Python の補足知識 {turns} は 可読性 を高めます。", teaching_mode=True)
        agent.chat(f"この要点を教えて <CALC> {turns}", teaching_mode=False)
        turns += 1

    return {
        "turns": turns,
        "elapsed_seconds": time.time() - start,
        "history_size": len(agent.dialogue_history),
        "history_limit": agent.max_history_turns * 2,
        "issue_count": len(agent.get_recent_issues(limit=100)),
        "active_terms": agent.topic_tracker.active_terms(limit=5),
        "history_bounded": len(agent.dialogue_history) <= agent.max_history_turns * 2,
        "min_turns_required": min_turns,
        "meets_min_turns": turns >= min_turns,
    }


def run_inference_soak(duration_seconds: float, max_iterations: int, min_iterations: int) -> Dict[str, Any]:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = model_path("tests", "release_soak_runtime.msgpack")
    engine.direct_map = {}
    engine.refractory_buffer = []
    engine.lif_network = None

    start = time.time()
    iterations = 0
    while iterations < max_iterations and (time.time() - start) < duration_seconds:
        base = iterations % 256
        engine.learn_sequence([base, base + 1, base + 2, base + 3])
        iterations += 1

    ensure_parent_directory(engine.model_path)
    engine.save_pretrained(engine.model_path)

    reloaded = SaraInference.__new__(SaraInference)
    reloaded.model_path = engine.model_path
    reloaded.direct_map = {}
    reloaded.refractory_buffer = []
    reloaded.lif_network = None
    reloaded._load_memory()

    return {
        "iterations": iterations,
        "elapsed_seconds": time.time() - start,
        "pattern_count": len(engine.direct_map),
        "roundtrip_ok": reloaded.direct_map == engine.direct_map,
        "tuple_keys_only": all(isinstance(key, tuple) for key in engine.direct_map.keys()),
        "min_iterations_required": min_iterations,
        "meets_min_iterations": iterations >= min_iterations,
    }


def run_accuracy_soak(
    history_path: Optional[str] = None,
    history_limit: int = 50,
) -> Dict[str, Any]:
    report = run_phase3_accuracy_suite(
        history_path=history_path,
        persist_history=bool(history_path),
        history_limit=history_limit,
    )
    return {
        "suite_name": report.get("suite_name", "Phase3AccuracySuite"),
        "overall_score": float(report.get("overall_score", 0.0)),
        "passed": bool(report.get("passed", False)),
        "trend": report.get("trend", {}),
        "component_reports": report.get("component_reports", {}),
        "focus_summary": report.get("focus_summary", {}),
        "history_length": int(report.get("history_length", 0)) if history_path else 0,
    }


def _status_label(passed: bool) -> str:
    return "PASS" if passed else "WARN"


def _agent_status(agent: Dict[str, Any]) -> bool:
    return bool(
        agent.get("meets_min_turns", False)
        and agent.get("history_bounded", False)
        and int(agent.get("issue_count", 0)) == 0
    )


def _inference_status(inference: Dict[str, Any]) -> bool:
    return bool(
        inference.get("meets_min_iterations", False)
        and inference.get("roundtrip_ok", False)
        and inference.get("tuple_keys_only", False)
        and int(inference.get("pattern_count", 0)) >= 1
    )


def _metadata_status(metadata: Dict[str, Any]) -> bool:
    return bool(
        metadata.get("versions_match", False)
        and metadata.get("has_expected_console_scripts", False)
        and str(metadata.get("release_notes_heading", "")).strip()
    )


def _accuracy_status(accuracy: Dict[str, Any]) -> bool:
    trend = accuracy.get("trend", {}) if isinstance(accuracy.get("trend"), dict) else {}
    return bool(
        accuracy.get("passed", False)
        and int(trend.get("regression_count", 0)) == 0
    )


def collect_release_gate_feedback(
    report: Dict[str, Any],
    project_root: str = PROJECT_ROOT,
) -> Dict[str, Any]:
    errors = []
    errors.extend(validate_release_report(report))
    errors.extend(validate_packaging_metadata(project_root))
    return {
        "passed": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
    }


def format_release_summary(report: Dict[str, Any]) -> str:
    criteria = report.get("criteria", {}) if isinstance(report.get("criteria"), dict) else {}
    agent = report.get("agent", {}) if isinstance(report.get("agent"), dict) else {}
    inference = report.get("inference", {}) if isinstance(report.get("inference"), dict) else {}
    accuracy = report.get("accuracy", {}) if isinstance(report.get("accuracy"), dict) else {}
    metadata = report.get("release_metadata", {}) if isinstance(report.get("release_metadata"), dict) else {}
    gate = report.get("release_gate", {}) if isinstance(report.get("release_gate"), dict) else {}
    accuracy_required = bool(criteria.get("require_phase3_accuracy", False))

    agent_ok = _agent_status(agent)
    inference_ok = _inference_status(inference)
    metadata_ok = _metadata_status(metadata)
    accuracy_ok = _accuracy_status(accuracy) if accuracy else (not accuracy_required)
    gate_ok = bool(gate.get("passed", False)) if gate else False
    overall_ok = agent_ok and inference_ok and metadata_ok and accuracy_ok and gate_ok

    lines = [
        "SARA Engine Release Soak Summary",
        f"overall_status: {_status_label(overall_ok)}",
        f"profile: {criteria.get('profile_name', 'unknown')}",
        f"duration_seconds: {report.get('duration_seconds', 0.0)}",
        f"shipping_ready_profile: {criteria.get('shipping_ready', False)}",
        "",
        "Agent",
        f"- status: {_status_label(agent_ok)}",
        f"- turns: {agent.get('turns', 0)} / min {agent.get('min_turns_required', 0)}",
        f"- history_bounded: {agent.get('history_bounded', False)}",
        f"- issue_count: {agent.get('issue_count', 0)}",
        "",
        "Inference",
        f"- status: {_status_label(inference_ok)}",
        f"- iterations: {inference.get('iterations', 0)} / min {inference.get('min_iterations_required', 0)}",
        f"- roundtrip_ok: {inference.get('roundtrip_ok', False)}",
        f"- tuple_keys_only: {inference.get('tuple_keys_only', False)}",
        f"- pattern_count: {inference.get('pattern_count', 0)}",
        "",
        "Release Metadata",
        f"- status: {_status_label(metadata_ok)}",
        f"- version: {metadata.get('pyproject_version', '')}",
        f"- versions_match: {metadata.get('versions_match', False)}",
        f"- console_scripts: {', '.join(metadata.get('console_scripts', []))}",
        f"- release_notes_heading: {metadata.get('release_notes_heading', '')}",
    ]

    if accuracy:
        trend = accuracy.get("trend", {}) if isinstance(accuracy.get("trend"), dict) else {}
        focus_summary = accuracy.get("focus_summary", {}) if isinstance(accuracy.get("focus_summary"), dict) else {}
        lines.extend(
            [
                "",
                "Accuracy",
                f"- status: {_status_label(accuracy_ok)}",
                f"- suite_name: {accuracy.get('suite_name', '')}",
                f"- passed: {accuracy.get('passed', False)}",
                f"- overall_score: {accuracy.get('overall_score', 0.0):.3f}",
                f"- regression_count: {trend.get('regression_count', 0)}",
            ]
        )
        if focus_summary:
            few_shot = focus_summary.get("few_shot", {}) if isinstance(focus_summary.get("few_shot"), dict) else {}
            continual = focus_summary.get("continual", {}) if isinstance(focus_summary.get("continual"), dict) else {}
            lines.extend(
                [
                    "",
                    "Phase 3 Focus",
                    f"- few_shot_status: {_status_label(bool(few_shot.get('passed', False)))}",
                    f"- few_shot_score: {float(few_shot.get('score', 0.0)):.3f}",
                    f"- continual_status: {_status_label(bool(continual.get('passed', False)))}",
                    f"- continual_score: {float(continual.get('score', 0.0)):.3f}",
                ]
            )
    elif accuracy_required:
        lines.extend(
            [
                "",
                "Accuracy",
                f"- status: {_status_label(False)}",
                "- suite_name: missing",
                "- passed: False",
                "- overall_score: 0.000",
                "- regression_count: 0",
            ]
        )
        lines.extend(
            [
                "",
                "Phase 3 Focus",
                f"- few_shot_status: {_status_label(False)}",
                "- few_shot_score: 0.000",
                f"- continual_status: {_status_label(False)}",
                "- continual_score: 0.000",
            ]
        )

    lines.extend(
        [
            "",
            "Gate",
            f"- status: {_status_label(gate_ok)}",
            f"- error_count: {gate.get('error_count', 0) if gate else 0}",
        ]
    )
    if gate and isinstance(gate.get("errors"), list) and gate["errors"]:
        for error in gate["errors"]:
            lines.append(f"- error: {error}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight release soak checks.")
    parser.add_argument(
        "--profile",
        choices=sorted(SOAK_PROFILES.keys()),
        default="release",
        help="Named soak profile with SNN-friendly default thresholds.",
    )
    parser.add_argument("--duration-seconds", type=float, default=None, help="Wall-clock budget per soak section.")
    parser.add_argument("--max-agent-turns", type=int, default=None, help="Maximum agent turns to execute.")
    parser.add_argument(
        "--min-agent-turns",
        type=int,
        default=None,
        help="Minimum agent turns required for the soak report to satisfy the release gate.",
    )
    parser.add_argument(
        "--max-inference-iterations",
        type=int,
        default=None,
        help="Maximum inference learning iterations.",
    )
    parser.add_argument(
        "--min-inference-iterations",
        type=int,
        default=None,
        help="Minimum inference iterations required for the soak report to satisfy the release gate.",
    )
    parser.add_argument(
        "--report-path",
        default=workspace_path("release", "release_soak_report.json"),
        help="Managed output path for the soak report.",
    )
    parser.add_argument(
        "--summary-path",
        default=workspace_path("release", "release_soak_summary.txt"),
        help="Managed output path for the human-readable release summary.",
    )
    parser.add_argument(
        "--include-accuracy",
        action="store_true",
        help="Run the Phase 3 accuracy suite and embed its summary into the release soak report.",
    )
    parser.add_argument(
        "--accuracy-history-path",
        default=workspace_path("evaluation", "phase3_accuracy_history.json"),
        help="Managed output path for Phase 3 accuracy history snapshots.",
    )
    parser.add_argument(
        "--accuracy-history-limit",
        type=int,
        default=50,
        help="Maximum number of embedded Phase 3 accuracy history snapshots to keep.",
    )
    args = parser.parse_args()

    settings = resolve_soak_profile(
        profile_name=args.profile,
        duration_seconds=args.duration_seconds,
        max_agent_turns=args.max_agent_turns,
        min_agent_turns=args.min_agent_turns,
        max_inference_iterations=args.max_inference_iterations,
        min_inference_iterations=args.min_inference_iterations,
    )

    if settings["duration_seconds"] <= 0:
        raise ValueError("--duration-seconds must be greater than 0.")
    if settings["min_agent_turns"] < 1:
        raise ValueError("--min-agent-turns must be at least 1.")
    if settings["min_inference_iterations"] < 1:
        raise ValueError("--min-inference-iterations must be at least 1.")
    if settings["min_agent_turns"] > settings["max_agent_turns"]:
        raise ValueError("--min-agent-turns cannot exceed --max-agent-turns.")
    if settings["min_inference_iterations"] > settings["max_inference_iterations"]:
        raise ValueError("--min-inference-iterations cannot exceed --max-inference-iterations.")

    report = {
        "agent": run_agent_soak(
            settings["duration_seconds"],
            settings["max_agent_turns"],
            settings["min_agent_turns"],
        ),
        "inference": run_inference_soak(
            settings["duration_seconds"],
            settings["max_inference_iterations"],
            settings["min_inference_iterations"],
        ),
        "duration_seconds": settings["duration_seconds"],
        "criteria": {
            "profile_name": settings["profile_name"],
            "min_duration_seconds": settings["duration_seconds"],
            "max_agent_turns": settings["max_agent_turns"],
            "min_agent_turns": settings["min_agent_turns"],
            "max_inference_iterations": settings["max_inference_iterations"],
            "min_inference_iterations": settings["min_inference_iterations"],
            "require_zero_agent_issues": True,
            "require_bounded_history": True,
            "require_roundtrip_ok": True,
            "require_tuple_keys_only": True,
            "min_pattern_count": 1,
            "shipping_ready": settings["shipping_ready"],
        },
        "release_metadata": collect_release_metadata(),
    }
    if args.include_accuracy:
        report["accuracy"] = run_accuracy_soak(
            history_path=args.accuracy_history_path,
            history_limit=args.accuracy_history_limit,
        )
        report["criteria"]["require_phase3_accuracy"] = True
    else:
        report["criteria"]["require_phase3_accuracy"] = False
    report["release_gate"] = collect_release_gate_feedback(report)

    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    summary_path = ensure_parent_directory(args.summary_path)
    summary_text = format_release_summary(report)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary_text)

    print("Release soak completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
