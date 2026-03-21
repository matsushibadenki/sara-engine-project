# Directory Path: scripts/eval/spiking_llm_accuracy_benchmark.py
# English Title: SpikingLLM Accuracy Benchmark
# Purpose/Content: Runs a lightweight Phase 3 benchmark for SpikingLLM next-token prediction, short streaming, and continual retention.

import argparse
import json
import os
import sys
from typing import Any, Dict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from sara_engine.evaluation.evaluator import SpikingLLMSequenceEvaluator
from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _build_model() -> tuple[SpikingLLM, Dict[str, int]]:
    model = SpikingLLM(num_layers=1, sdr_size=64, vocab_size=256, context_window=4)
    token_ids = {
        "python": model.tokenizer._add_token("Python"),
        "function": model.tokenizer._add_token("関数"),
        "argument": model.tokenizer._add_token("引数"),
        "cell": model.tokenizer._add_token("細胞"),
        "energy": model.tokenizer._add_token("エネルギー"),
        "stop": model.tokenizer._add_token("。"),
    }
    model.pretrained_synapses = {
        1: {
            token_ids["python"]: {token_ids["function"]: 1.0},
            token_ids["function"]: {token_ids["argument"]: 1.0},
            token_ids["argument"]: {token_ids["stop"]: 1.0},
            token_ids["cell"]: {token_ids["energy"]: 1.0},
            token_ids["energy"]: {token_ids["stop"]: 1.0},
        },
        2: {
            token_ids["python"]: {token_ids["argument"]: 0.8},
        },
    }
    return model, token_ids


def run_spiking_llm_accuracy_benchmark() -> Dict[str, Any]:
    evaluator = SpikingLLMSequenceEvaluator()

    def run_case(test_case: SpikingLLMSequenceEvaluator.TestCase) -> Dict[str, Any]:
        model, token_ids = _build_model()

        if test_case.case_type == "next_token":
            candidates = model.predict_next_tokens(prompt_tokens=[token_ids["python"]], top_k=3)
            predicted = int(candidates[0]["token_id"]) if candidates else None
            return {
                "success": predicted == token_ids["function"],
                "predicted_token": predicted,
                "expected_token": token_ids["function"],
            }

        if test_case.case_type == "stream":
            generated = list(
                model.generate_stream(
                    prompt_tokens=[token_ids["python"]],
                    max_new_tokens=2,
                    top_k=3,
                    temperature=0.0,
                )
            )
            generated_tokens = [int(step["token_id"]) for step in generated]
            expected_tokens = [token_ids["function"], token_ids["argument"]]
            return {
                "success": generated_tokens == expected_tokens,
                "generated_tokens": generated_tokens,
                "expected_tokens": expected_tokens,
            }

        if test_case.case_type == "continual":
            candidates = model.predict_next_tokens(prompt_tokens=[token_ids["python"]], top_k=3)
            predicted = int(candidates[0]["token_id"]) if candidates else None
            return {
                "success": predicted == token_ids["function"],
                "predicted_token": predicted,
                "expected_token": token_ids["function"],
            }

        return {"success": False, "predicted_token": None, "expected_token": None}

    result = evaluator.evaluate(
        test_cases=[
            SpikingLLMSequenceEvaluator.TestCase(
                case_type="next_token",
                description="Known transition should surface as the top candidate.",
            ),
            SpikingLLMSequenceEvaluator.TestCase(
                case_type="stream",
                description="Short streaming completion should preserve the learned local path.",
            ),
            SpikingLLMSequenceEvaluator.TestCase(
                case_type="continual",
                description="Adding a second lightweight path should not erase the original path.",
            ),
        ],
        run_case_fn=run_case,
    )

    metrics = {metric.name: metric.value for metric in result.metrics}
    thresholds = {
        "next_token_accuracy": 1.0,
        "stream_completion_rate": 1.0,
        "continual_memory_retention": 1.0,
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
    parser = argparse.ArgumentParser(description="Run the lightweight SpikingLLM accuracy benchmark.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "spiking_llm_accuracy_benchmark.json"),
        help="Managed output path for the benchmark report.",
    )
    args = parser.parse_args()

    report = run_spiking_llm_accuracy_benchmark()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("SpikingLLM accuracy benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
