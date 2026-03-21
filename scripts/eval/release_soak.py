# Directory Path: scripts/eval/release_soak.py
# English Title: Release Soak Runner
# Purpose/Content: Runs a lightweight wall-clock soak test for agent dialogue and inference memory loops, then saves a managed report for release validation.

import argparse
import json
import os
import sys
import time
from typing import Any, Dict


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


def run_agent_soak(duration_seconds: float, max_turns: int) -> Dict[str, Any]:
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
    }


def run_inference_soak(duration_seconds: float, max_iterations: int) -> Dict[str, Any]:
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight release soak checks.")
    parser.add_argument("--duration-seconds", type=float, default=5.0, help="Wall-clock budget per soak section.")
    parser.add_argument("--max-agent-turns", type=int, default=120, help="Maximum agent turns to execute.")
    parser.add_argument("--max-inference-iterations", type=int, default=256, help="Maximum inference learning iterations.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("release", "release_soak_report.json"),
        help="Managed output path for the soak report.",
    )
    args = parser.parse_args()

    report = {
        "agent": run_agent_soak(args.duration_seconds, args.max_agent_turns),
        "inference": run_inference_soak(args.duration_seconds, args.max_inference_iterations),
        "duration_seconds": args.duration_seconds,
    }

    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("Release soak completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
