_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/__init__.py",
    "//": "ファイルの日本語タイトル: パイプライン初期化",
    "//": "ファイルの目的や内容: SARA Engineの各種タスクパイプラインを統合管理し、Hugging Faceライクなpipeline()関数を提供する。"
}

from typing import Any
from .text_generation import TextGenerationPipeline
from .text_classification import TextClassificationPipeline
from .token_classification import TokenClassificationPipeline
# summarization 等他のパイプラインが存在する場合はここに追加

def pipeline(task: str, model: Any = None, tokenizer: Any = None, **kwargs) -> Any:
    """
    Utility factory method to build a pipeline for SNN tasks.
    Modeled after the Hugging Face Transformers pipeline() function.
    
    Args:
        task (str): The task defining which pipeline will be returned.
        model (Any, optional): The SNN model to be used.
        tokenizer (Any, optional): The tokenizer to be used.
        **kwargs: Additional keyword arguments.
    """
    if task == "text-generation":
        if model is None or tokenizer is None:
            raise ValueError("Both 'model' and 'tokenizer' must be provided for the 'text-generation' pipeline.")
        return TextGenerationPipeline(model=model, tokenizer=tokenizer)
        
    elif task == "text-classification" or task == "sentiment-analysis":
        if model is None or tokenizer is None:
            raise ValueError("Both 'model' and 'tokenizer' must be provided for the 'text-classification' pipeline.")
        return TextClassificationPipeline(model=model, tokenizer=tokenizer, **kwargs)
        
    elif task == "token-classification" or task == "ner":
        if model is None or tokenizer is None:
            raise ValueError("Both 'model' and 'tokenizer' must be provided for the 'token-classification' pipeline.")
        return TokenClassificationPipeline(model=model, tokenizer=tokenizer, **kwargs)
        
    else:
        raise NotImplementedError(f"The task '{task}' is not yet supported in the SARA engine.")