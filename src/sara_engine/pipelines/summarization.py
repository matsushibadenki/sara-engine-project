# ディレクトリパス: src/sara_engine/pipelines/summarization.py
# ファイルの日本語タイトル: テキスト要約パイプライン
# ファイルの目的や内容: Transformersのpipeline('summarization')をSNNで再現し、テキストの要約機能を提供する。
import re
from typing import Any, List

class SummarizationPipeline:
    """Pipeline for text summarization using a causal Spiking Neural Network."""
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text: str, max_length: int = 100, min_length: int = 10, **kwargs) -> str:
        """
        Generates a summary based on the provided text.
        This uses the SNN model to identify key concepts and generate a condensed version.
        No backpropagation or matrix multiplications are used.
        """
        # SNNモデルは文脈を保持しつつ次のトークンを予測する性質を持つため、
        # "TL;DR:" などのプロンプトを追加して要約タスクを誘導する。
        prompt = f"{text}\n\nTL;DR:\n"
        
        # SNNの特性上、生成長はスパイクダイナミクスに依存するが、
        # API互換性のために引数を受け入れ、生成長をある程度制御する。
        summary = self.model.generate(prompt=prompt, max_new_tokens=max_length)
        
        # プロンプト部分を除去して要約部分のみを返す
        if summary.startswith(prompt):
            summary = summary[len(prompt):].strip()
            
        if len(summary) < min_length:
            summary = self._extractive_fallback(text, max_length=max_length, min_length=min_length)
             
        return summary

    def _extractive_fallback(self, text: str, max_length: int, min_length: int) -> str:
        """Builds a deterministic low-energy summary when generative recall is too weak."""

        normalized = " ".join(str(text).split())
        if not normalized:
            return ""

        sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", normalized) if part.strip()]
        if not sentences:
            sentences = [normalized]

        keyword_scores = self._keyword_scores(normalized)
        ranked = sorted(
            enumerate(sentences),
            key=lambda item: (-self._sentence_score(item[1], keyword_scores), item[0]),
        )

        selected: List[tuple[int, str]] = []
        total_length = 0
        for index, sentence in ranked:
            projected = total_length + len(sentence) + (1 if selected else 0)
            if selected and projected > max_length:
                continue
            selected.append((index, sentence))
            total_length = projected
            if total_length >= min_length:
                break

        if not selected:
            return normalized[:max_length].strip()

        ordered = [sentence for _index, sentence in sorted(selected, key=lambda item: item[0])]
        return " ".join(ordered)[:max_length].strip()

    def _keyword_scores(self, text: str) -> dict[str, int]:
        tokens = [token.lower() for token in re.findall(r"[\w一-龯ぁ-んァ-ヶ]+", text)]
        scores: dict[str, int] = {}
        for token in tokens:
            if len(token) <= 1:
                continue
            scores[token] = scores.get(token, 0) + 1
        return scores

    def _sentence_score(self, sentence: str, keyword_scores: dict[str, int]) -> int:
        tokens = [token.lower() for token in re.findall(r"[\w一-龯ぁ-んァ-ヶ]+", sentence)]
        return sum(keyword_scores.get(token, 0) for token in tokens)

    def learn(self, text: str, summary: str) -> None:
        """
        Trains the SNN locally on the provided text-summary pair using STDP.
        """
        training_data = f"{text}\n\nTL;DR:\n{summary}\n"
        if hasattr(self.tokenizer, "encode"):
            token_ids = self.tokenizer.encode(training_data)
        else:
            token_ids = [ord(c) for c in training_data]
        self.model.learn_sequence(token_ids)

    def save_pretrained(self, save_directory: str) -> None:
        """Saves the SNN configuration and synaptic weights to disk."""
        self.model.save_pretrained(save_directory)
