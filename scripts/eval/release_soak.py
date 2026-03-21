# Directory Path: scripts/eval/release_soak.py
# English Title: Release Soak Runner
# Purpose/Content: Runs a lightweight wall-clock soak test for agent dialogue and inference memory loops, then saves a managed report for release validation.

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Keep optional plotting/font caches inside the managed workspace to avoid
# noisy warnings on restricted environments during release validation.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", "cache"))


from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path


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
    }

    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("Release soak completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
