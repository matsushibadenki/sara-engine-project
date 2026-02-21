_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/feature_extraction.py",
    "//": "ファイルの日本語タイトル: 特徴抽出パイプライン",
    "//": "ファイルの目的や内容: mypy型エラーの修正。入力リストの型判定を厳密化。"
}

from typing import Union, List, Any, cast
from .base import Pipeline

class FeatureExtractionPipeline(Pipeline):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)

    def __call__(self, text_inputs: Union[str, List[str]], **kwargs) -> Any:
        is_single_input = isinstance(text_inputs, str)
        inputs: List[str] = [text_inputs] if is_single_input else cast(List[str], text_inputs)

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