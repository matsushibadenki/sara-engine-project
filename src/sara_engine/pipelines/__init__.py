# ディレクトリパス: src/sara_engine/pipelines/__init__.py
# ファイルの日本語タイトル: パイプライン初期化
# ファイルの目的や内容: SARA Engineの各種タスクパイプラインを統合管理。モデル名(文字列)が渡された場合、自動的にAutoSpikingLMやAutoSpikingAgentをロードするスマートなファクトリ機能を追加。
from typing import Any
from .text_generation import TextGenerationPipeline
from .text_classification import TextClassificationPipeline
from .token_classification import TokenClassificationPipeline
from .agent_chat import AgentChatPipeline
from .feature_extraction import FeatureExtractionPipeline
from .summarization import SummarizationPipeline
from .image_classification import ImageClassificationPipeline
from .audio_classification import AudioClassificationPipeline

def pipeline(task: str, model: Any = None, tokenizer: Any = None, **kwargs) -> Any:
    """
    Utility factory method to build a pipeline for SNN tasks.
    Modeled closely after the Hugging Face Transformers pipeline() function.
    
    Args:
        task (str): The task defining which pipeline will be returned (e.g., 'text-generation', 'agent').
        model (Any or str, optional): The SNN model instance, or a path/name to load it via Auto classes.
        tokenizer (Any, optional): The tokenizer to be used.
        **kwargs: Additional keyword arguments.
    """
    
    # モデル名（文字列）が指定された場合の自動ロード処理
    if isinstance(model, str) or model is None:
        model_path = model if model else "sara-default-model"
        
        if task == "text-generation" or task == "summarization":
            from ..auto import AutoSpikingLM, AutoTokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoSpikingLM.from_pretrained(model_path)
            
        elif task in ["agent", "agent-chat", "conversational"]:
            from ..auto import AutoSpikingAgent
            model = AutoSpikingAgent.from_pretrained(model_path)
            
        elif task == "text-classification" or task == "sentiment-analysis":
            from ..auto import AutoSNNModelForSequenceClassification, AutoTokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoSNNModelForSequenceClassification.from_pretrained(model_path)
            
        elif task == "token-classification" or task == "ner":
            from ..auto import AutoSNNModelForTokenClassification, AutoTokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoSNNModelForTokenClassification.from_pretrained(model_path)
            
        elif task == "feature-extraction":
            from ..auto import AutoSNNModelForFeatureExtraction, AutoTokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoSNNModelForFeatureExtraction.from_pretrained(model_path)
            
        elif task == "image-classification":
            from ..auto import AutoSNNModelForImageClassification
            model = AutoSNNModelForImageClassification.from_pretrained(model_path)
            
        elif task == "audio-classification":
            from ..auto import AutoSNNModelForAudioClassification
            model = AutoSNNModelForAudioClassification.from_pretrained(model_path)

    # パイプラインのディスパッチ
    if task == "text-generation":
        if model is None or tokenizer is None:
            raise ValueError("Both 'model' and 'tokenizer' must be provided (or autoloaded) for text-generation.")
        return TextGenerationPipeline(model=model, tokenizer=tokenizer)
        
    elif task == "summarization":
        if model is None or tokenizer is None:
            raise ValueError("Both 'model' and 'tokenizer' must be provided (or autoloaded) for summarization.")
        return SummarizationPipeline(model=model, tokenizer=tokenizer)
        
    elif task in ["agent", "agent-chat", "conversational"]:
        if model is None:
            raise ValueError("A 'model' (Agent instance) must be provided for the agent pipeline.")
        return AgentChatPipeline(agent=model)
        
    elif task == "text-classification" or task == "sentiment-analysis":
        if model is None or tokenizer is None:
            raise ValueError("Both 'model' and 'tokenizer' must be provided for text-classification.")
        return TextClassificationPipeline(model=model, tokenizer=tokenizer, **kwargs)
        
    elif task == "token-classification" or task == "ner":
        if model is None or tokenizer is None:
            raise ValueError("Both 'model' and 'tokenizer' must be provided for token-classification.")
        return TokenClassificationPipeline(model=model, tokenizer=tokenizer, **kwargs)
        
    elif task == "feature-extraction":
        if model is None or tokenizer is None:
            raise ValueError("Both 'model' and 'tokenizer' must be provided (or autoloaded) for feature-extraction.")
        return FeatureExtractionPipeline(model=model, tokenizer=tokenizer, **kwargs)
        
    elif task == "image-classification":
        if model is None:
            raise ValueError("A 'model' must be provided (or autoloaded) for image-classification.")
        return ImageClassificationPipeline(model=model, feature_extractor=tokenizer, **kwargs)
        
    elif task == "audio-classification":
        if model is None:
            raise ValueError("A 'model' must be provided (or autoloaded) for audio-classification.")
        return AudioClassificationPipeline(model=model, feature_extractor=tokenizer, **kwargs)
        
    else:
        raise NotImplementedError(f"The task '{task}' is not yet supported in the SARA engine.")
