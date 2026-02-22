_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/audio_classification.py",
    "//": "ファイルの日本語タイトル: 音声分類パイプライン",
    "//": "ファイルの目的や内容: 生の音声波形(float配列)をSNNに入力し、カテゴリを推論・学習する。"
}

from typing import Union, List, Dict
from .base import Pipeline

class AudioClassificationPipeline(Pipeline):
    """
    Audio classification pipeline using pure SNN.
    Accepts raw audio waveforms (List of floats).
    """
    def __init__(self, model, feature_extractor=None, **kwargs):
        super().__init__(model, tokenizer=feature_extractor, **kwargs)
        self.id2label = kwargs.get("id2label", {0: "CLASS_0", 1: "CLASS_1"})

    def __call__(self, waveforms: Union[List[float], List[List[float]]], **kwargs) -> Union[Dict[str, float], List[Dict[str, float]]]:
        if isinstance(waveforms[0], (int, float)):
            waveforms = [waveforms]

        results = []
        for wave in waveforms:
            predicted_class_id = self.model.forward(wave, learning=False)
            label = self.id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")
            results.append({"label": label, "score": 1.0})

        if len(results) == 1:
            return results[0]
        return results

    def learn(self, waveform: List[float], label_id: int) -> None:
        """
        Trains the SNN classifier locally on the provided waveform using STDP.
        """
        self.model.forward(waveform, learning=True, target_class=label_id)