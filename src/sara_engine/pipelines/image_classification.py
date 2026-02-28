_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/image_classification.py",
    "//": "ファイルの日本語タイトル: 画像分類パイプライン",
    "//": "ファイルの目的や内容: 画像を1次元ベクトルとしてSNNに入力し、カテゴリを推論・学習する。"
}

from typing import Union, List, Dict, Any
from .base import Pipeline

class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using pure SNN.
    Accepts flattened pixel intensities [0.0 - 1.0].
    """
    def __init__(self, model: Any, feature_extractor: Any = None, **kwargs: Any):
        # 視覚タスクではtokenizerの代わりにfeature_extractorを使用する
        super().__init__(model, tokenizer=feature_extractor, **kwargs)
        self.id2label = kwargs.get("id2label", {0: "CLASS_0", 1: "CLASS_1"})

    def __call__(self, images: Union[List[float], List[List[float]]], **kwargs: Any) -> Union[Dict[str, float], List[Dict[str, float]]]:
        # mypyエラー対策: 型を明示した新しい変数を使用する
        image_list: List[List[float]]
        if images and isinstance(images[0], (int, float)):
            image_list = [images] # type: ignore
        else:
            image_list = images # type: ignore

        results = []
        for img in image_list:
            predicted_class_id = self.model.forward(img, learning=False)
            label = self.id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")
            # BP不使用のため、疑似的な確信度として1.0を返す
            results.append({"label": label, "score": 1.0})

        if len(results) == 1:
            return results[0]
        return results

    def learn(self, image: List[float], label_id: int) -> None:
        """
        Trains the SNN classifier locally on the provided image using STDP.
        """
        self.model.forward(image, learning=True, target_class=label_id)