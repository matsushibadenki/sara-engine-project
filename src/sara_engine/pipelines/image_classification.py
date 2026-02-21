_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/image_classification.py",
    "//": "ファイルの日本語タイトル: 画像分類パイプライン",
    "//": "ファイルの目的や内容: mypy型エラーの修正。Union型の要素アクセスと引数の型互換性を解決。"
}

from typing import Union, List, Dict, Any, cast
from .base import Pipeline

class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using Spiking Neural Networks.
    Accepts flattened 1D lists of pixel intensities (0.0 to 1.0) or 2D lists.
    """
    def __init__(self, model, tokenizer=None, **kwargs):
        super().__init__(model, tokenizer=tokenizer, **kwargs)
        self.id2label: Dict[int, str] = kwargs.get("id2label", {0: "LABEL_0", 1: "LABEL_1"})

    def _flatten_image(self, image: Union[List[float], List[List[float]]]) -> List[float]:
        if not image:
            return []
        if isinstance(image[0], list):
            # 2D list to 1D list
            return [pixel for row in cast(List[List[float]], image) for pixel in row]
        return cast(List[float], image)

    def __call__(self, images: Any, **kwargs) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Args:
            images: Single image (1D/2D) or a list of images.
        """
        # Batch determination logic
        is_batch = True
        
        # Check if single 1D image
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], (int, float)):
            is_batch = False
            flat_images = [cast(List[float], images)]
        # Check if single 2D image
        elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], list) and \
             len(images[0]) > 0 and isinstance(images[0][0], (int, float)):
            # Distinguish between [List[float]] (Batch of 1D) and List[List[float]] (Single 2D)
            # This is a heuristic: if we expect batch, use kwargs.
            if kwargs.get("single_image", True):
                is_batch = False
                flat_images = [self._flatten_image(cast(List[List[float]], images))]
            else:
                flat_images = [cast(List[float], img) for img in images]
        else:
            flat_images = [self._flatten_image(img) for img in images]

        results = []
        for pixels in flat_images:
            if hasattr(self.model, 'forward'):
                predicted_class_id = self.model.forward(pixels, learning=False)
                label = self.id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")
                results.append({"label": label, "score": 1.0})
            else:
                results.append({"label": "UNKNOWN", "score": 0.0})

        if not is_batch:
            return results
        return results