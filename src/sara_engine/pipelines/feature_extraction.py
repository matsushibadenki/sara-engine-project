_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/feature_extraction.py",
    "//": "ファイルの日本語タイトル: 特徴抽出パイプライン",
    "//": "ファイルの目的や内容: 入力テキストからSNNのリザーバー発火パターン（埋め込みベクトル）を抽出する。"
}

from typing import Union, List
from .base import Pipeline

class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline using SNN.
    Outputs the final membrane potentials of the reservoir as a dense vector.
    """
    def __call__(self, text_inputs: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        results = []
        for text in text_inputs:
            if self.tokenizer is not None and hasattr(self.model, 'forward'):
                token_ids = self.tokenizer.encode(text)
                # Extract SNN features (Returns a dense list of floats representing spike potentials)
                features = self.model.forward(token_ids)
                results.append(features)
            else:
                results.append([])

        if len(results) == 1:
            return results[0]
        return results