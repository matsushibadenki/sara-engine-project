_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/feature_extraction.py",
    "//": "ファイルの日本語タイトル: 特徴抽出パイプライン",
    "//": "ファイルの目的や内容: mypy型エラーの修正。isinstanceをif文の条件に直接記述し、型の絞り込み（Type Narrowing）を確実に機能させる。"
}

from typing import Union, List, Any
from .base import Pipeline

class FeatureExtractionPipeline(Pipeline):
    def __init__(self, model: Any, tokenizer: Any, **kwargs: Any):
        super().__init__(model, tokenizer, **kwargs)

    def __call__(self, text_inputs: Union[str, List[str]], **kwargs: Any) -> Any:
        # mypyのType Narrowingを効かせるため、if文の条件式に直接isinstanceを記述します
        if isinstance(text_inputs, str):
            inputs: List[str] = [text_inputs]
            is_single_input = True
        else:
            inputs = text_inputs
            is_single_input = False

        results = []
        for text in inputs:
            if self.tokenizer is not None and hasattr(self.model, 'forward'):
                token_ids = self.tokenizer.encode(text)
                embedding = self.model.forward(token_ids)
                results.append(embedding)
            else:
                results.append([])

        if is_single_input:
            return results[0]
        return results