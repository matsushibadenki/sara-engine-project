# Directory Path: scripts/eval/inference_accuracy_benchmark.py
# English Title: SaraInference Accuracy Benchmark
# Purpose/Content: Runs a lightweight Phase 3 benchmark for SaraInference one-shot, few-shot, fuzzy retrieval, and continual retention under CPU-only constraints.

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from sara_engine.evaluation.evaluator import InferenceSequenceEvaluator
from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


class _DummyTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> Dict[str, Any]:
        token_ids = [ord(char) for char in text]
        return {"input_ids": [[token for token in token_ids]]}

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _build_engine() -> SaraInference:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.tokenizer = _DummyTokenizer()
    engine.direct_map = {}
    engine.context_index = {}
    engine.refractory_buffer = []
    engine.lif_network = None
    return engine


def _predict_next_token(engine: SaraInference, context_tokens: list[int]) -> Optional[int]:
    key = engine._find_best_matching_key(context_tokens)
    if key is None or key not in engine.direct_map:
        return None
    predicted = engine._sample_next_token(
        key,
        top_k=1,
        temperature=0.0,
        refractory_penalty=1.0,
    )
    return int(predicted) if predicted is not None else None


def run_inference_accuracy_benchmark() -> Dict[str, Any]:
    evaluator = InferenceSequenceEvaluator()

    def run_case(test_case: InferenceSequenceEvaluator.TestCase) -> Dict[str, Any]:
        if test_case.case_type == "one_shot":
            engine = _build_engine()
            engine.learn_sequence([10, 20, 30])
            predicted = _predict_next_token(engine, [10, 20])
            return {
                "success": predicted == 30,
                "predicted_token": predicted,
                "expected_token": 30,
            }

        if test_case.case_type == "fuzzy":
            engine = _build_engine()
            engine.learn_sequence([101, 102, 103, 999])
            predicted = _predict_next_token(engine, [101, 103])
            return {
                "success": predicted == 999,
                "predicted_token": predicted,
                "expected_token": 999,
            }

        if test_case.case_type == "few_shot":
            engine = _build_engine()
            for _ in range(3):
                engine.learn_sequence([70, 80, 90])
            engine.learn_sequence([70, 81, 91])
            predicted = _predict_next_token(engine, [70, 80])
            return {
                "success": predicted == 90,
                "predicted_token": predicted,
                "expected_token": 90,
            }

        if test_case.case_type == "continual":
            engine = _build_engine()
            engine.learn_sequence([1, 2, 3])
            engine.learn_sequence([4, 5, 6])
            predicted = _predict_next_token(engine, [1, 2])
            return {
                "success": predicted == 3,
                "predicted_token": predicted,
                "expected_token": 3,
            }

        if test_case.case_type == "long_continual":
            engine = _build_engine()
            engine.learn_sequence([1, 2, 3])
            for offset in range(10, 22):
                engine.learn_sequence([offset, offset + 1, offset + 2])
            predicted = _predict_next_token(engine, [1, 2])
            return {
                "success": predicted == 3,
                "predicted_token": predicted,
                "expected_token": 3,
            }

        return {"success": False, "predicted_token": None, "expected_token": None}

    result = evaluator.evaluate(
        test_cases=[
            InferenceSequenceEvaluator.TestCase(
                case_type="one_shot",
                description="One-shot sequence memory should recover the next token.",
            ),
            InferenceSequenceEvaluator.TestCase(
                case_type="few_shot",
                description="Repeated local exposure should stabilize a preferred continuation.",
            ),
            InferenceSequenceEvaluator.TestCase(
                case_type="fuzzy",
                description="Nearby context should recover a compatible continuation.",
            ),
            InferenceSequenceEvaluator.TestCase(
                case_type="continual",
                description="Continual learning should preserve an earlier lightweight memory path.",
            ),
            InferenceSequenceEvaluator.TestCase(
                case_type="long_continual",
                description="A longer continual-learning sequence should still retain the earliest path.",
            ),
        ],
        run_case_fn=run_case,
    )

    metrics = {metric.name: metric.value for metric in result.metrics}
    thresholds = {
        "one_shot_accuracy": 1.0,
        "few_shot_accuracy": 1.0,
        "fuzzy_retrieval_accuracy": 1.0,
        "continual_retention": 1.0,
        "long_horizon_retention": 1.0,
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
    parser = argparse.ArgumentParser(description="Run the lightweight SaraInference accuracy benchmark.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "inference_accuracy_benchmark.json"),
        help="Managed output path for the benchmark report.",
    )
    args = parser.parse_args()

    report = run_inference_accuracy_benchmark()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("SaraInference accuracy benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
