# ディレクトリパス: tests/test_evaluation.py
# ファイル名: test_evaluation.py
# ファイルの目的や内容: 評価基盤の単体テスト

from sara_engine.evaluation.evaluator import (
    AgentDialogueEvaluator,
    EvalMetric,
    EvalResult,
    InferenceSequenceEvaluator,
    RAGEvaluator,
    SARABenchmark,
    SafetyEvaluator,
    SpikingLLMSequenceEvaluator,
    ToolEvaluator,
)
from typing import Any, Dict, Iterator
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))


# --- RAGEvaluator テスト ---

class TestRAGEvaluator:

    def test_perfect_recall(self) -> None:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            queries=["q1"],
            retrieved_docs=[["doc1", "doc2"]],
            relevant_docs=[["doc1"]],
        )
        recall = next(m for m in result.metrics if m.name == "recall_at_k")
        assert recall.value == 1.0

    def test_zero_recall(self) -> None:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            queries=["q1"],
            retrieved_docs=[["doc_wrong"]],
            relevant_docs=[["doc1"]],
        )
        recall = next(m for m in result.metrics if m.name == "recall_at_k")
        assert recall.value == 0.0

    def test_mrr_first_position(self) -> None:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            queries=["q1"],
            retrieved_docs=[["doc1", "doc2"]],
            relevant_docs=[["doc1"]],
        )
        mrr = next(m for m in result.metrics if m.name == "mrr")
        assert mrr.value == 1.0

    def test_mrr_second_position(self) -> None:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            queries=["q1"],
            retrieved_docs=[["wrong", "doc1"]],
            relevant_docs=[["doc1"]],
        )
        mrr = next(m for m in result.metrics if m.name == "mrr")
        assert mrr.value == 0.5

    def test_precision_at_k(self) -> None:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            queries=["q1"],
            retrieved_docs=[["doc1", "wrong1", "wrong2"]],
            relevant_docs=[["doc1"]],
        )
        precision = next(
            m for m in result.metrics if m.name == "precision_at_k")
        assert abs(precision.value - 1.0 / 3.0) < 0.01

    def test_response_quality(self) -> None:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            queries=["q1"],
            retrieved_docs=[["doc1"]],
            relevant_docs=[["doc1"]],
            generated_responses=["SNNは計算モデル"],
            reference_responses=["SNNは計算モデル"],
        )
        quality = next(m for m in result.metrics if m.name ==
                       "response_quality")
        assert quality.value > 0.5

    def test_overall_score(self) -> None:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            queries=["q1"],
            retrieved_docs=[["doc1"]],
            relevant_docs=[["doc1"]],
        )
        assert 0.0 <= result.overall_score <= 1.0

    def test_eval_result_summary(self) -> None:
        result = EvalResult(
            evaluator_name="Test",
            metrics=[EvalMetric(name="test_metric",
                                value=0.5, description="テスト")],
            overall_score=0.5,
        )
        summary = result.summary()
        assert "Test" in summary
        assert "0.500" in summary


# --- ToolEvaluator テスト ---

class TestToolEvaluator:

    def test_all_success(self) -> None:
        evaluator = ToolEvaluator()
        test_cases = [
            ToolEvaluator.TestCase(
                tool_name="calc",
                params={"a": 1, "b": 2},
                expected_output="3",
                description="足し算",
            ),
        ]

        def execute_fn(name: str, params: dict) -> str:
            return "3"

        result = evaluator.evaluate(test_cases, execute_fn)
        success = next(m for m in result.metrics if m.name == "success_rate")
        assert success.value == 1.0

    def test_failure_case(self) -> None:
        evaluator = ToolEvaluator()
        test_cases = [
            ToolEvaluator.TestCase(
                tool_name="fail",
                params={},
                expected_output="ok",
            ),
        ]

        def execute_fn(name: str, params: dict) -> str:
            raise RuntimeError("ツール実行エラー")

        result = evaluator.evaluate(test_cases, execute_fn)
        success = next(m for m in result.metrics if m.name == "success_rate")
        assert success.value == 0.0

    def test_accuracy_check(self) -> None:
        evaluator = ToolEvaluator()
        test_cases = [
            ToolEvaluator.TestCase(
                tool_name="echo",
                params={"text": "hello"},
                expected_output="hello",
            ),
        ]

        def execute_fn(name: str, params: dict) -> str:
            return "hello"

        result = evaluator.evaluate(test_cases, execute_fn)
        accuracy = next(m for m in result.metrics if m.name == "accuracy")
        assert accuracy.value == 1.0


# --- SafetyEvaluator テスト ---

class TestSafetyEvaluator:

    def test_perfect_safety(self) -> None:
        evaluator = SafetyEvaluator()

        class MockResult:
            def __init__(self, safe: bool) -> None:
                self.is_safe = safe

        def check_fn(text: str) -> MockResult:
            if "unsafe" in text:
                return MockResult(False)
            return MockResult(True)

        result = evaluator.evaluate(
            safe_inputs=["safe text1", "safe text2"],
            unsafe_inputs=["unsafe1", "unsafe2"],
            check_fn=check_fn,
        )
        safe_rate = next(
            m for m in result.metrics if m.name == "safe_pass_rate")
        block_rate = next(
            m for m in result.metrics if m.name == "unsafe_block_rate")
        assert safe_rate.value == 1.0
        assert block_rate.value == 1.0

    def test_f1_score_computed(self) -> None:
        evaluator = SafetyEvaluator()

        class MockResult:
            def __init__(self, safe: bool) -> None:
                self.is_safe = safe

        def check_fn(text: str) -> MockResult:
            return MockResult(True)  # 全て安全と判定(偽陰性あり)

        result = evaluator.evaluate(
            safe_inputs=["safe"],
            unsafe_inputs=["unsafe"],
            check_fn=check_fn,
        )
        f1 = next(m for m in result.metrics if m.name == "f1_score")
        assert 0.0 <= f1.value <= 1.0


class TestAgentDialogueEvaluator:

    def test_dialogue_evaluator_scores_keyword_recall_and_grounding(self) -> None:
        evaluator = AgentDialogueEvaluator()
        responses = iter(
            [
                "Pythonの関数は再利用でき、引数を受け取れます。",
                "[MoE Router: general (Fallback)]\n >> 関連知識は十分に取り出せませんでした。",
            ]
        )
        diagnostics = iter(
            [
                [
                    {
                        "keyword_score": 12.0,
                        "current_keyword_coverage": 1.0,
                        "context_keyword_coverage": 0.5,
                        "metadata_keyword_coverage": 0.5,
                    }
                ],
                [],
            ]
        )

        result = evaluator.evaluate(
            test_cases=[
                AgentDialogueEvaluator.TestCase(
                    user_input="Pythonの関数とは？",
                    expected_keywords=["Python", "関数", "引数"],
                    should_fallback=False,
                ),
                AgentDialogueEvaluator.TestCase(
                    user_input="未知の話題について教えて",
                    expected_keywords=[],
                    should_fallback=True,
                ),
            ],
            respond_fn=lambda _text: next(responses),
            diagnostics_fn=lambda: next(diagnostics),
        )

        keyword_recall = next(m for m in result.metrics if m.name == "response_keyword_recall")
        fallback_control = next(m for m in result.metrics if m.name == "fallback_control")
        grounding = next(m for m in result.metrics if m.name == "retrieval_grounding")

        assert keyword_recall.value > 0.8
        assert fallback_control.value == 1.0
        assert grounding.value >= 0.4

    def test_dialogue_evaluator_penalizes_wrong_fallback_behavior(self) -> None:
        evaluator = AgentDialogueEvaluator()

        result = evaluator.evaluate(
            test_cases=[
                AgentDialogueEvaluator.TestCase(
                    user_input="Pythonの関数とは？",
                    expected_keywords=["Python"],
                    should_fallback=False,
                )
            ],
            respond_fn=lambda _text: "[MoE Router: general (Fallback)]\n >> 関連知識は十分に取り出せませんでした。",
            diagnostics_fn=lambda: [],
        )

        fallback_control = next(m for m in result.metrics if m.name == "fallback_control")
        assert fallback_control.value == 0.0


class TestInferenceSequenceEvaluator:

    def test_sequence_evaluator_tracks_one_shot_fuzzy_and_retention(self) -> None:
        evaluator = InferenceSequenceEvaluator()
        outcomes: Iterator[Dict[str, Any]] = iter(
            [
                {"success": True, "predicted_token": 30, "expected_token": 30},
                {"success": True, "predicted_token": 999, "expected_token": 999},
                {"success": True, "predicted_token": 3, "expected_token": 3},
            ]
        )

        result = evaluator.evaluate(
            test_cases=[
                InferenceSequenceEvaluator.TestCase(case_type="one_shot"),
                InferenceSequenceEvaluator.TestCase(case_type="fuzzy"),
                InferenceSequenceEvaluator.TestCase(case_type="continual"),
            ],
            run_case_fn=lambda _case: next(outcomes),
        )

        one_shot = next(m for m in result.metrics if m.name == "one_shot_accuracy")
        fuzzy = next(m for m in result.metrics if m.name == "fuzzy_retrieval_accuracy")
        retention = next(m for m in result.metrics if m.name == "continual_retention")

        assert one_shot.value == 1.0
        assert fuzzy.value == 1.0
        assert retention.value == 1.0


class TestSpikingLLMSequenceEvaluator:

    def test_spiking_llm_evaluator_tracks_next_token_stream_and_retention(self) -> None:
        evaluator = SpikingLLMSequenceEvaluator()
        outcomes: Iterator[Dict[str, Any]] = iter(
            [
                {"success": True, "predicted_token": 2, "expected_token": 2},
                {"success": True, "generated_tokens": [2, 3], "expected_tokens": [2, 3]},
                {"success": True, "predicted_token": 2, "expected_token": 2},
            ]
        )

        result = evaluator.evaluate(
            test_cases=[
                SpikingLLMSequenceEvaluator.TestCase(case_type="next_token"),
                SpikingLLMSequenceEvaluator.TestCase(case_type="stream"),
                SpikingLLMSequenceEvaluator.TestCase(case_type="continual"),
            ],
            run_case_fn=lambda _case: next(outcomes),
        )

        next_token = next(m for m in result.metrics if m.name == "next_token_accuracy")
        stream = next(m for m in result.metrics if m.name == "stream_completion_rate")
        retention = next(m for m in result.metrics if m.name == "continual_memory_retention")

        assert next_token.value == 1.0
        assert stream.value == 1.0
        assert retention.value == 1.0


# --- SARABenchmark テスト ---

class TestSARABenchmark:

    def test_run_and_report(self) -> None:
        benchmark = SARABenchmark()
        rag_eval = RAGEvaluator()
        benchmark.add_evaluator(
            "RAG",
            rag_eval.evaluate,
            {
                "queries": ["q1"],
                "retrieved_docs": [["doc1"]],
                "relevant_docs": [["doc1"]],
            },
        )

        results = benchmark.run()
        assert len(results) == 1
        assert results[0].evaluator_name == "RAGEvaluator"

        report = benchmark.report()
        assert "ベンチマークレポート" in report
        assert "全体平均スコア" in report

    def test_empty_benchmark(self) -> None:
        benchmark = SARABenchmark()
        report = benchmark.report()
        assert "未実行" in report

    def test_evaluator_error_handling(self) -> None:
        benchmark = SARABenchmark()

        def failing_eval() -> EvalResult:
            raise RuntimeError("評価エラー")

        benchmark.add_evaluator("failing", failing_eval)
        results = benchmark.run()
        assert len(results) == 1
        assert results[0].overall_score == 0.0
