_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/text_classification.py",
    "//": "ファイルの日本語タイトル: テキスト分類パイプライン",
    "//": "ファイルの目的や内容: SNNモデルを利用してテキストの分類（感情分析など）を行うパイプライン。"
}

from typing import Union, List, Dict
from .base import Pipeline

class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using SNN Sequence Classifier.
    Evaluates spike rates to classify text sequences.
    """
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        # 簡易的なラベルマッピング（本来はconfigから読み込む設計に拡張可能）
        self.id2label = kwargs.get("id2label", {0: "LABEL_0", 1: "LABEL_1"})

    def __call__(self, text_inputs: Union[str, List[str]], **kwargs) -> Union[List[Dict[str, float]], List[List[Dict[str, float]]]]:
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        results = []
        for text in text_inputs:
            if self.tokenizer is not None and hasattr(self.model, 'forward'):
                token_ids = self.tokenizer.encode(text)
                
                # SNN Forward pass (No learning during inference)
                predicted_class_id = self.model.forward(token_ids, learning=False)
                label = self.id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")
                
                # スパイクベースなので厳密な確率(Softmax)は出ないが、ダミーの信頼度スコアを付与
                results.append({"label": label, "score": 1.0})
            else:
                results.append({"label": "UNKNOWN", "score": 0.0})

        if len(results) == 1:
            return results
        return results