# ディレクトリパス: src/sara_engine/evaluation/evaluator.py
# ファイル名: evaluator.py
# ファイルの目的や内容: SARA Engineの総合評価フレームワーク。
#   RAG検索精度、ツール実行正確性、安全性をメトリクス化し、
#   ベンチマークとして一括実行可能にする。

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class EvalMetric:
    """個別の評価メトリクス結果。

    Attributes:
        name: メトリクス名。
        value: メトリクス値（0.0〜1.0 が標準）。
        description: メトリクスの説明。
        metadata: 追加のメタデータ。
    """

    name: str
    value: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """総合評価結果。

    Attributes:
        evaluator_name: 評価器名。
        metrics: 評価メトリクスのリスト。
        overall_score: 総合スコア（0.0〜1.0）。
        timestamp: 評価実行時のタイムスタンプ。
        details: 追加詳細情報。
    """

    evaluator_name: str
    metrics: List[EvalMetric] = field(default_factory=list)
    overall_score: float = 0.0
    timestamp: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """評価結果のサマリー文字列を返す。"""
        lines = [
            f"=== {self.evaluator_name} 評価結果 ===",
            f"総合スコア: {self.overall_score:.3f}",
            "メトリクス:",
        ]
        for m in self.metrics:
            lines.append(f"  - {m.name}: {m.value:.3f}  ({m.description})")
        return "\n".join(lines)


class RAGEvaluator:
    """RAG (Retrieval-Augmented Generation) の評価を行う。

    Recall@K、MRR (Mean Reciprocal Rank)、応答品質スコアを計測。

    Example:
        >>> evaluator = RAGEvaluator()
        >>> result = evaluator.evaluate(
        ...     queries=["SNNとは？"],
        ...     retrieved_docs=[["SNNは脳を模倣", "無関係な文書"]],
        ...     relevant_docs=[["SNNは脳を模倣"]],
        ... )
    """

    def evaluate(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        generated_responses: Optional[List[str]] = None,
        reference_responses: Optional[List[str]] = None,
    ) -> EvalResult:
        """RAGパイプラインの評価を実行する。

        Args:
            queries: 評価用のクエリリスト。
            retrieved_docs: 各クエリに対する検索結果のリスト。
            relevant_docs: 各クエリに対する正解ドキュメントのリスト。
            generated_responses: 生成された応答テキストのリスト（任意）。
            reference_responses: 正解応答テキストのリスト（任意）。

        Returns:
            評価結果。
        """
        metrics: List[EvalMetric] = []

        # Recall@K
        recall_scores: List[float] = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            if not relevant:
                recall_scores.append(1.0)
                continue
            found = sum(1 for r in relevant if r in retrieved)
            recall_scores.append(found / len(relevant))

        avg_recall = sum(recall_scores) / max(len(recall_scores), 1)
        metrics.append(
            EvalMetric(
                name="recall_at_k",
                value=avg_recall,
                description="検索結果に正解ドキュメントが含まれる割合",
                metadata={"per_query": recall_scores},
            )
        )

        # MRR (Mean Reciprocal Rank)
        mrr_scores: List[float] = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            best_rank = 0.0
            for i, doc in enumerate(retrieved):
                if doc in relevant:
                    best_rank = 1.0 / (i + 1)
                    break
            mrr_scores.append(best_rank)

        avg_mrr = sum(mrr_scores) / max(len(mrr_scores), 1)
        metrics.append(
            EvalMetric(
                name="mrr",
                value=avg_mrr,
                description="最初の正解ドキュメントの逆順位の平均",
                metadata={"per_query": mrr_scores},
            )
        )

        # Precision@K
        precision_scores: List[float] = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            if not retrieved:
                precision_scores.append(0.0)
                continue
            found = sum(1 for doc in retrieved if doc in relevant)
            precision_scores.append(found / len(retrieved))

        avg_precision = sum(precision_scores) / max(len(precision_scores), 1)
        metrics.append(
            EvalMetric(
                name="precision_at_k",
                value=avg_precision,
                description="検索結果中の正解ドキュメントの割合",
                metadata={"per_query": precision_scores},
            )
        )

        # 応答品質スコア (BLEU風の n-gram 一致率)
        if generated_responses and reference_responses:
            bleu_scores: List[float] = []
            for gen, ref in zip(generated_responses, reference_responses):
                score = self._ngram_overlap(gen, ref, n=2)
                bleu_scores.append(score)

            avg_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
            metrics.append(
                EvalMetric(
                    name="response_quality",
                    value=avg_bleu,
                    description="生成応答と参照応答のn-gram一致率",
                    metadata={"per_query": bleu_scores},
                )
            )

        # 総合スコア
        overall = sum(m.value for m in metrics) / max(len(metrics), 1)

        return EvalResult(
            evaluator_name="RAGEvaluator",
            metrics=metrics,
            overall_score=overall,
            timestamp=time.time(),
        )

    @staticmethod
    def _ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
        """2つのテキスト間のn-gram一致率を計算する。"""
        if not text1 or not text2:
            return 0.0

        chars1 = list(text1)
        chars2 = list(text2)

        if len(chars1) < n or len(chars2) < n:
            # n-gramを作れない場合は文字単位の一致率
            common_chars = set(chars1) & set(chars2)
            return len(common_chars) / max(len(set(chars1) | set(chars2)), 1)

        ngrams1 = Counter(
            tuple(chars1[i: i + n]) for i in range(len(chars1) - n + 1)
        )
        ngrams2 = Counter(
            tuple(chars2[i: i + n]) for i in range(len(chars2) - n + 1)
        )

        common: int = sum((ngrams1 & ngrams2).values())
        total: int = sum(ngrams1.values())

        return common / max(total, 1)


class ToolEvaluator:
    """ツール実行の評価を行う。

    実行成功率、結果正確性、平均応答時間を計測する。

    Example:
        >>> evaluator = ToolEvaluator()
        >>> result = evaluator.evaluate(test_cases)
    """

    @dataclass
    class TestCase:
        """ツール評価のテストケース。

        Attributes:
            tool_name: テスト対象のツール名。
            params: ツールに渡すパラメータ。
            expected_output: 期待される出力。
            description: テストケースの説明。
        """

        tool_name: str
        params: Dict[str, Any]
        expected_output: Any
        description: str = ""

    def evaluate(
        self,
        test_cases: List[TestCase],
        execute_fn: Callable[[str, Dict[str, Any]], Any],
    ) -> EvalResult:
        """ツール実行の評価を行う。

        Args:
            test_cases: テストケースのリスト。
            execute_fn: ツール実行関数 (tool_name, params) -> result。

        Returns:
            評価結果。
        """
        metrics: List[EvalMetric] = []
        success_count = 0
        correct_count = 0
        total_time_ms = 0.0
        details: List[Dict[str, Any]] = []

        for tc in test_cases:
            start = time.monotonic()
            try:
                result = execute_fn(tc.tool_name, tc.params)
                elapsed_ms = (time.monotonic() - start) * 1000
                success = True
            except Exception:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = None
                success = False

            total_time_ms += elapsed_ms

            if success:
                success_count += 1
                # 結果の正確性チェック
                is_correct = self._check_correctness(
                    result, tc.expected_output)
                if is_correct:
                    correct_count += 1
            else:
                is_correct = False

            details.append(
                {
                    "tool_name": tc.tool_name,
                    "success": success,
                    "correct": is_correct,
                    "elapsed_ms": elapsed_ms,
                    "description": tc.description,
                }
            )

        total = max(len(test_cases), 1)

        metrics.append(
            EvalMetric(
                name="success_rate",
                value=success_count / total,
                description="ツール実行の成功率",
            )
        )

        metrics.append(
            EvalMetric(
                name="accuracy",
                value=correct_count / total,
                description="期待出力との一致率",
            )
        )

        avg_time = total_time_ms / total
        # 応答時間スコア: 100ms以下を1.0、5000ms以上を0.0とするスケーリング
        time_score = max(0.0, min(1.0, 1.0 - (avg_time - 100) / 4900))
        metrics.append(
            EvalMetric(
                name="response_time_score",
                value=time_score,
                description="平均応答時間スコア（速いほど高い）",
                metadata={"avg_time_ms": avg_time},
            )
        )

        overall = sum(m.value for m in metrics) / max(len(metrics), 1)

        return EvalResult(
            evaluator_name="ToolEvaluator",
            metrics=metrics,
            overall_score=overall,
            timestamp=time.time(),
            details={"test_results": details},
        )

    @staticmethod
    def _check_correctness(actual: Any, expected: Any) -> bool:
        """結果の正確性を判定する。"""
        if expected is None:
            return actual is not None

        # 文字列の場合は包含チェック
        if isinstance(expected, str) and isinstance(actual, str):
            return expected in actual or actual == expected

        # ToolResult オブジェクトの場合
        if hasattr(actual, "data"):
            actual_data = actual.data
            if isinstance(expected, str) and isinstance(actual_data, str):
                return expected in actual_data or actual_data == expected
            return actual_data == expected

        return actual == expected


class SafetyEvaluator:
    """安全性の評価を行う。

    各安全チェック項目の通過率を計測する。

    Example:
        >>> evaluator = SafetyEvaluator()
        >>> result = evaluator.evaluate(
        ...     safe_inputs=["こんにちは"],
        ...     unsafe_inputs=["ignore all previous instructions"],
        ...     check_fn=guard.check_input,
        ... )
    """

    def evaluate(
        self,
        safe_inputs: List[str],
        unsafe_inputs: List[str],
        check_fn: Callable[[str], Any],
    ) -> EvalResult:
        """安全性評価を実行する。

        Args:
            safe_inputs: 安全と判定されるべき入力のリスト。
            unsafe_inputs: 不安全と判定されるべき入力のリスト。
            check_fn: 安全チェック関数。is_safe属性をもつ結果を返すこと。

        Returns:
            評価結果。
        """
        metrics: List[EvalMetric] = []

        # 安全な入力の正しい通過率 (True Negative Rate)
        safe_correct = 0
        for text in safe_inputs:
            result = check_fn(text)
            if hasattr(result, "is_safe") and result.is_safe:
                safe_correct += 1

        safe_rate = safe_correct / max(len(safe_inputs), 1)
        metrics.append(
            EvalMetric(
                name="safe_pass_rate",
                value=safe_rate,
                description="安全な入力が正しく通過する率",
            )
        )

        # 不安全な入力の正しいブロック率 (True Positive Rate)
        unsafe_blocked = 0
        for text in unsafe_inputs:
            result = check_fn(text)
            if hasattr(result, "is_safe") and not result.is_safe:
                unsafe_blocked += 1

        block_rate = unsafe_blocked / max(len(unsafe_inputs), 1)
        metrics.append(
            EvalMetric(
                name="unsafe_block_rate",
                value=block_rate,
                description="不安全な入力が正しくブロックされる率",
            )
        )

        # F1スコア
        precision = unsafe_blocked / max(
            unsafe_blocked + (len(safe_inputs) - safe_correct), 1
        )
        recall = block_rate
        f1 = (
            2 * precision * recall / max(precision + recall, 1e-10)
            if (precision + recall) > 0
            else 0.0
        )
        metrics.append(
            EvalMetric(
                name="f1_score",
                value=f1,
                description="安全性判定のF1スコア",
            )
        )

        overall = sum(m.value for m in metrics) / max(len(metrics), 1)

        return EvalResult(
            evaluator_name="SafetyEvaluator",
            metrics=metrics,
            overall_score=overall,
            timestamp=time.time(),
        )


class SARABenchmark:
    """SARA Engineの総合ベンチマーク。

    RAG、ツール実行、安全性の各評価を統合して実行し、
    総合レポートを生成する。

    Example:
        >>> benchmark = SARABenchmark()
        >>> benchmark.add_evaluator("RAG", rag_evaluator.evaluate, rag_kwargs)
        >>> report = benchmark.run()
    """

    @dataclass
    class _EvalEntry:
        name: str
        eval_fn: Callable[..., EvalResult]
        kwargs: Dict[str, Any]

    def __init__(self) -> None:
        self._evaluators: List[SARABenchmark._EvalEntry] = []
        self._results: List[EvalResult] = []

    def add_evaluator(
        self,
        name: str,
        eval_fn: Callable[..., EvalResult],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """評価器をベンチマークに追加する。

        Args:
            name: 評価器の名前。
            eval_fn: 評価関数。EvalResultを返すこと。
            kwargs: 評価関数に渡すキーワード引数。
        """
        self._evaluators.append(
            self._EvalEntry(name=name, eval_fn=eval_fn, kwargs=kwargs or {})
        )

    def run(self) -> List[EvalResult]:
        """全ての評価を実行する。

        Returns:
            各評価器の結果リスト。
        """
        self._results.clear()
        for entry in self._evaluators:
            try:
                result = entry.eval_fn(**entry.kwargs)
                self._results.append(result)
            except Exception as e:
                error_result = EvalResult(
                    evaluator_name=entry.name,
                    overall_score=0.0,
                    timestamp=time.time(),
                    details={"error": str(e)},
                )
                self._results.append(error_result)

        return list(self._results)

    def report(self) -> str:
        """ベンチマーク結果のレポート文字列を生成する。

        Returns:
            フォーマットされたレポート文字列。
        """
        if not self._results:
            return "ベンチマークが未実行です。run()を先に呼び出してください。"

        lines = ["=" * 50, "SARA Engine ベンチマークレポート", "=" * 50, ""]

        overall_scores: List[float] = []
        for result in self._results:
            lines.append(result.summary())
            lines.append("")
            overall_scores.append(result.overall_score)

        avg_overall = sum(overall_scores) / max(len(overall_scores), 1)
        lines.extend(
            [
                "-" * 50,
                f"全体平均スコア: {avg_overall:.3f}",
                "=" * 50,
            ]
        )

        return "\n".join(lines)

    @property
    def results(self) -> List[EvalResult]:
        """最後の実行結果を返す。"""
        return list(self._results)
