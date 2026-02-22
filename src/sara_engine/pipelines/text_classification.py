_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/text_classification.py",
    "//": "ファイルの日本語タイトル: テキスト分類パイプライン",
    "//": "ファイルの目的や内容: SNNを用いたテキスト分類。局所学習(STDP)用のlearnメソッドを追加。"
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
                
                # SNN does not output strict probabilities (like Softmax), providing dummy score
                results.append({"label": label, "score": 1.0})
            else:
                results.append({"label": "UNKNOWN", "score": 0.0})

        if len(results) == 1:
            return results
        return results

    def learn(self, text: str, label_id: int) -> None:
        """
        Trains the SNN classifier locally on the provided sequence using STDP.
        """
        if self.tokenizer is not None and hasattr(self.model, 'forward'):
            token_ids = self.tokenizer.encode(text)
            self.model.forward(token_ids, learning=True, target_class=label_id)
        else:
            raise ValueError("Model or tokenizer is missing necessary methods for learning.")