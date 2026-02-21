_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/__init__.py",
    "//": "ファイルの日本語タイトル: パイプライン初期化モジュール",
    "//": "ファイルの目的や内容: 各種パイプラインの登録とファクトリ関数。Autoクラスの呼び出しを相対インポートに修正。"
}

from .base import Pipeline
from .text_generation import TextGenerationPipeline
from .text_classification import TextClassificationPipeline
from .feature_extraction import FeatureExtractionPipeline
from .image_classification import ImageClassificationPipeline

SUPPORTED_TASKS = {
    "text-generation": TextGenerationPipeline,
    "text-classification": TextClassificationPipeline,
    "feature-extraction": FeatureExtractionPipeline,
    "image-classification": ImageClassificationPipeline,
}

def pipeline(task: str, model=None, tokenizer=None, **kwargs) -> Pipeline:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Task '{task}' is not supported. Available tasks: {list(SUPPORTED_TASKS.keys())}")
    
    if isinstance(model, str):
        if task == "text-classification":
            from ..auto import AutoSNNModelForSequenceClassification
            model = AutoSNNModelForSequenceClassification.from_pretrained(model)
        elif task == "feature-extraction":
            from ..auto import AutoSNNModelForFeatureExtraction
            model = AutoSNNModelForFeatureExtraction.from_pretrained(model)
        elif task == "image-classification":
            from ..auto import AutoSNNModelForImageClassification
            model = AutoSNNModelForImageClassification.from_pretrained(model)
        else:
            from ..auto import AutoSNNModelForCausalLM
            model = AutoSNNModelForCausalLM.from_pretrained(model)
            
    pipeline_class = SUPPORTED_TASKS[task]
    return pipeline_class(model=model, tokenizer=tokenizer, **kwargs)