_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/auto.py",
    "//": "ファイルの日本語タイトル: 自動読み込みモジュール（Autoクラス）",
    "//": "ファイルの目的や内容: Autoクラス群の定義。絶対インポートによるModuleNotFoundErrorを防ぐため、相対インポートに修正。"
}

import os
from .models.snn_transformer import SpikingTransformerModel
from .models.spiking_sequence_classifier import SpikingSequenceClassifier
from .models.spiking_feature_extractor import SpikingFeatureExtractor
from .models.spiking_image_classifier import SpikingImageClassifier
from .encoders.spike_tokenizer import SpikeTokenizer

class AutoSNNModelForCausalLM:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if not os.path.exists(pretrained_model_name_or_path):
            raise FileNotFoundError(f"Model directory not found: {pretrained_model_name_or_path}")
        print(f"Loading Causal LM SNN weights from {pretrained_model_name_or_path}...")
        return SpikingTransformerModel.from_pretrained(pretrained_model_name_or_path)

class AutoSNNModelForSequenceClassification:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if not os.path.exists(pretrained_model_name_or_path):
            raise FileNotFoundError(f"Model directory not found: {pretrained_model_name_or_path}")
        print(f"Loading Sequence Classifier SNN weights from {pretrained_model_name_or_path}...")
        return SpikingSequenceClassifier.from_pretrained(pretrained_model_name_or_path)

class AutoSNNModelForFeatureExtraction:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if not os.path.exists(pretrained_model_name_or_path):
            raise FileNotFoundError(f"Model directory not found: {pretrained_model_name_or_path}")
        print(f"Loading Feature Extractor SNN from {pretrained_model_name_or_path}...")
        return SpikingFeatureExtractor.from_pretrained(pretrained_model_name_or_path)

class AutoSNNModelForImageClassification:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if not os.path.exists(pretrained_model_name_or_path):
            raise FileNotFoundError(f"Model directory not found: {pretrained_model_name_or_path}")
        print(f"Loading Image Classifier SNN from {pretrained_model_name_or_path}...")
        return SpikingImageClassifier.from_pretrained(pretrained_model_name_or_path)

class AutoSpikeTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if pretrained_model_name_or_path.endswith(".json") and os.path.exists(pretrained_model_name_or_path):
            tokenizer_path = pretrained_model_name_or_path
        else:
            tokenizer_path = os.path.join(pretrained_model_name_or_path, "tokenizer.json")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"tokenizer.json not found in {pretrained_model_name_or_path}")
        
        print(f"Loading SpikeTokenizer from {tokenizer_path}...")
        tokenizer = SpikeTokenizer()
        tokenizer.load(tokenizer_path)
        return tokenizer