# Directory Path: scripts/eval/agent_dialogue_benchmark.py
# English Title: Agent Dialogue Benchmark
# Purpose/Content: Runs a lightweight Phase 3 benchmark for SaraAgent dialogue quality and writes a managed JSON report under workspace/.

import argparse
import json
import os
import sys
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", "cache"))


from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.evaluation.evaluator import AgentDialogueEvaluator
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def build_benchmark_agent() -> SaraAgent:
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    agent.chat(
        "Pythonのリスト内包表記とは、既存のリストから新しいリストを短く簡潔に生成するための構文のことです。",
        teaching_mode=True,
    )
    agent.chat(
        "そのメリットは、コードの行数が減って可読性が上がり、処理速度も速くなることです。",
        teaching_mode=True,
    )
    agent.chat(
        "Pythonの関数は再利用可能な処理のまとまりで、引数を受け取り結果を返せます。",
        teaching_mode=True,
    )
    return agent


def run_agent_dialogue_benchmark() -> Dict[str, Any]:
    agent = build_benchmark_agent()
    evaluator = AgentDialogueEvaluator()

    def _respond(text: str) -> str:
        response = agent.chat(text, teaching_mode=False)
        if isinstance(response, str):
            return response
        return "".join(str(chunk) for chunk in response)

    test_cases = [
        AgentDialogueEvaluator.TestCase(
            user_input="Pythonの関数とは？",
            expected_keywords=["Python", "関数", "引数"],
            should_fallback=False,
            description="Known domain query should stay grounded.",
        ),
        AgentDialogueEvaluator.TestCase(
            user_input="それを書くメリットは何ですか？",
            expected_keywords=["可読性", "処理速度", "メリット"],
            should_fallback=False,
            description="Demonstrative follow-up should recover contextual memory.",
        ),
        AgentDialogueEvaluator.TestCase(
            user_input="量子うどん文明の慣習を教えて",
            expected_keywords=[],
            should_fallback=True,
            description="Unknown topic should trigger a guided fallback.",
        ),
        AgentDialogueEvaluator.TestCase(
            user_input="関数の引数は何ですか？",
            expected_keywords=["関数", "引数", "入力値"],
            should_fallback=False,
            description="Known support detail should remain grounded.",
        ),
    ]

    result = evaluator.evaluate(
        test_cases=test_cases,
        respond_fn=_respond,
        diagnostics_fn=lambda: agent.get_recent_retrieval_diagnostics(limit=3),
    )

    metrics = {metric.name: metric.value for metric in result.metrics}
    thresholds = {
        "response_keyword_recall": 0.60,
        "fallback_control": 0.90,
        "retrieval_grounding": 0.35,
    }
    threshold_results = {
        name: metrics.get(name, 0.0) >= threshold
        for name, threshold in thresholds.items()
    }
    return {
        "evaluator_name": result.evaluator_name,
        "overall_score": result.overall_score,
        "metrics": metrics,
        "details": result.details,
        "thresholds": thresholds,
        "threshold_results": threshold_results,
        "passed": all(threshold_results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight SaraAgent dialogue benchmark.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "agent_dialogue_benchmark.json"),
        help="Managed output path for the benchmark report.",
    )
    args = parser.parse_args()

    report = run_agent_dialogue_benchmark()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("Agent dialogue benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
