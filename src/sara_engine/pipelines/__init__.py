_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/__init__.py",
    "//": "ファイルの日本語タイトル: パイプライン初期化モジュール",
    "//": "ファイルの目的や内容: TokenClassificationPipelineの登録。"
}

from typing import Any
from .text_generation import TextGenerationPipeline
from .text_classification import TextClassificationPipeline
from .feature_extraction import FeatureExtractionPipeline
from .image_classification import ImageClassificationPipeline
from .audio_classification import AudioClassificationPipeline
from .token_classification import TokenClassificationPipeline

def pipeline(task: str, model: Any = None, tokenizer: Any = None, feature_extractor: Any = None, **kwargs) -> Any:
    if task == "text-generation":
        return TextGenerationPipeline(model=model, tokenizer=tokenizer, **kwargs)
    elif task == "text-classification":
        return TextClassificationPipeline(model=model, tokenizer=tokenizer, **kwargs)
    elif task == "feature-extraction":
        return FeatureExtractionPipeline(model=model, tokenizer=tokenizer, **kwargs)
    elif task == "image-classification":
        return ImageClassificationPipeline(model=model, feature_extractor=feature_extractor, **kwargs)
    elif task == "audio-classification":
        return AudioClassificationPipeline(model=model, feature_extractor=feature_extractor, **kwargs)
    elif task == "token-classification":
        return TokenClassificationPipeline(model=model, tokenizer=tokenizer, **kwargs)
    else:
        raise NotImplementedError(f"The task '{task}' is not yet supported in the SARA engine.")