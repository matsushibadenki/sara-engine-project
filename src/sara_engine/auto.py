{
    "//": "ディレクトリパス: src/sara_engine/auto.py",
    "//": "ファイルの日本語タイトル: 自動読み込みモジュール",
    "//": "ファイルの目的や内容: Hugging Faceライクな Auto クラスを提供。強いAI(StrongSpikingLM)の自動読み込みとパイプラインにも対応。"
}

import os
from typing import Any, Optional
from .models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
from .models.spiking_sequence_classifier import SpikingSequenceClassifier, SNNSequenceClassifierConfig
from .models.spiking_feature_extractor import SpikingFeatureExtractor, SNNFeatureExtractorConfig
from .models.spiking_image_classifier import SpikingImageClassifier, SNNImageClassifierConfig
from .models.spiking_audio_classifier import SpikingAudioClassifier, SNNAudioClassifierConfig
from .models.spiking_token_classifier import SpikingTokenClassifier, SNNTokenClassifierConfig
from .models.hierarchical_snn import HierarchicalSNN
from .utils.tokenizer import SaraTokenizer
from .models.spiking_llm import SpikingLLM
from .agent.sara_agent import SaraAgent
from .models.strong_spiking_lm import StrongSpikingLM, StrongSpikingLMConfig

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, save_directory: str):
        if save_directory.endswith(".json") and os.path.exists(save_directory):
            tokenizer_path = save_directory
        else:
            tokenizer_path = os.path.join(save_directory, "tokenizer.json")

        if os.path.exists(tokenizer_path):
            return SaraTokenizer(model_path=tokenizer_path)
        else:
            print(f"Warning: Tokenizer not found at {tokenizer_path}, initializing a fresh SaraTokenizer (BPE Subword fallback).")
            return SaraTokenizer(model_path=tokenizer_path)

class AutoModelForCausalSNN:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            print(f"Initializing a new SpikingTransformerModel. No config at {pretrained_model_name_or_path}")
            config = SNNTransformerConfig()
            return SpikingTransformerModel(config)
        return SpikingTransformerModel.from_pretrained(pretrained_model_name_or_path)

# 互換性維持のためのエイリアス
AutoSNNModel = AutoModelForCausalSNN

class AutoSNNModelForSequenceClassification:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            config = SNNSequenceClassifierConfig(num_classes=2)
            return SpikingSequenceClassifier(config)
        return SpikingSequenceClassifier.from_pretrained(pretrained_model_name_or_path)

class AutoSNNModelForFeatureExtraction:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            config = SNNFeatureExtractorConfig()
            return SpikingFeatureExtractor(config)
        return SpikingFeatureExtractor.from_pretrained(pretrained_model_name_or_path)

class AutoSNNModelForHierarchicalExtraction:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            print(f"Initializing a new HierarchicalSNN. No config at {pretrained_model_name_or_path}")
            configs = [
                {"embed_dim": 128},
                {"embed_dim": 64},
                {"embed_dim": 32}
            ]
            return HierarchicalSNN(layer_configs=configs)
        
        if hasattr(HierarchicalSNN, 'from_pretrained'):
            return HierarchicalSNN.from_pretrained(pretrained_model_name_or_path)
        else:
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            model = HierarchicalSNN(layer_configs=config.get("layers", []))
            weight_path = os.path.join(pretrained_model_name_or_path, "sara_model.json")
            if os.path.exists(weight_path) and hasattr(model, 'load_pretrained'):
                model.load_pretrained(weight_path)
            return model

class AutoSNNModelForImageClassification:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            config = SNNImageClassifierConfig(input_size=64, num_classes=2)
            return SpikingImageClassifier(config)
        return SpikingImageClassifier.from_pretrained(pretrained_model_name_or_path)

class AutoSNNModelForAudioClassification:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            config = SNNAudioClassifierConfig(num_classes=2)
            return SpikingAudioClassifier(config)
        return SpikingAudioClassifier.from_pretrained(pretrained_model_name_or_path)

class AutoSNNModelForTokenClassification:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            print(f"Initializing a new SpikingTokenClassifier. No config at {pretrained_model_name_or_path}")
            config = SNNTokenClassifierConfig(num_classes=3)
            return SpikingTokenClassifier(config)
        return SpikingTokenClassifier.from_pretrained(pretrained_model_name_or_path)

class AutoSpikingLM:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if not os.path.exists(pretrained_model_name_or_path):
            print(f"Warning: LLM path not found {pretrained_model_name_or_path}, creating a fresh one.")
            return SpikingLLM(vocab_size=256, embed_dim=64, num_layers=2)
        return SpikingLLM.from_pretrained(pretrained_model_name_or_path)

class AutoSpikingAgent:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if not os.path.exists(pretrained_model_name_or_path):
            print(f"Warning: Agent path not found {pretrained_model_name_or_path}, creating a fresh one.")
            return SaraAgent()
        return SaraAgent.from_pretrained(pretrained_model_name_or_path)

class AutoStrongSpikingLM:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            print(f"Initializing a new StrongSpikingLM. No config at {pretrained_model_name_or_path}")
            return StrongSpikingLM(StrongSpikingLMConfig())
        return StrongSpikingLM.from_pretrained(pretrained_model_name_or_path)

class SaraPipeline:
    """タスク指向のパイプラインAPI。TransformersのpipelineをSNNで模倣する。"""
    def __init__(self, task: str, model: Any, tokenizer: Any = None):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, input_data: Any, **kwargs) -> Any:
        if self.task in ["text-generation", "strong-text-generation"]:
            if hasattr(self.model, "generate"):
                return self.model.generate(input_data, **kwargs)
        elif self.task == "feature-extraction" or self.task == "hierarchical-extraction":
            if hasattr(self.model, "forward"):
                return self.model.forward(input_data, learning=False)
        elif self.task == "text-classification":
            if hasattr(self.model, "classify"):
                return self.model.classify(input_data, **kwargs)
        
        if hasattr(self.model, "__call__"):
            return self.model(input_data, **kwargs)
        return None

def pipeline(task: str, model_path: Optional[str] = None, model: Optional[Any] = None, tokenizer: Optional[Any] = None) -> SaraPipeline:
    """
    Transformersライクなpipeline関数。
    """
    if model is None:
        if model_path is None:
            raise ValueError("model_path or model must be provided.")
            
        if task == "text-generation":
            model = AutoSpikingLM.from_pretrained(model_path)
        elif task == "strong-text-generation":
            model = AutoStrongSpikingLM.from_pretrained(model_path)
        elif task == "feature-extraction":
            model = AutoSNNModelForFeatureExtraction.from_pretrained(model_path)
        elif task == "hierarchical-extraction":
            model = AutoSNNModelForHierarchicalExtraction.from_pretrained(model_path)
        elif task == "text-classification":
            model = AutoSNNModelForSequenceClassification.from_pretrained(model_path)
        elif task == "image-classification":
            model = AutoSNNModelForImageClassification.from_pretrained(model_path)
        else:
            model = AutoModelForCausalSNN.from_pretrained(model_path)
            
    if tokenizer is None and model_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    return SaraPipeline(task, model, tokenizer)