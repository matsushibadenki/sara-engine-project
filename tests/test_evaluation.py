# ディレクトリパス: tests/test_evaluation.py
# ファイル名: test_evaluation.py
# ファイルの目的や内容: 評価基盤の単体テスト

from sara_engine.evaluation.evaluator import (
    EvalMetric,
    EvalResult,
    RAGEvaluator,
    SARABenchmark,
    SafetyEvaluator,
    ToolEvaluator,
)
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
