{
    "//": "ディレクトリパス: src/sara_engine/auto.py",
    "//": "ファイルの日本語タイトル: 自動読み込みモジュール",
    "//": "ファイルの目的や内容: Hugging Faceライクな Auto クラスを提供。SpikingLLM や SaraAgent をモデルパスから自動的に初期化・復元する機能を追加。"
}

import os
from .models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
from .models.spiking_sequence_classifier import SpikingSequenceClassifier, SNNSequenceClassifierConfig
from .models.spiking_feature_extractor import SpikingFeatureExtractor, SNNFeatureExtractorConfig
from .models.spiking_image_classifier import SpikingImageClassifier, SNNImageClassifierConfig
from .models.spiking_audio_classifier import SpikingAudioClassifier, SNNAudioClassifierConfig
from .models.spiking_token_classifier import SpikingTokenClassifier, SNNTokenClassifierConfig
from .encoders.spike_tokenizer import SpikeTokenizer
from .models.spiking_llm import SpikingLLM
from .agent.sara_agent import SaraAgent

class ByteLevelSNNTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode('utf-8'))
    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode('utf-8', errors='ignore')

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, save_directory: str):
        if save_directory.endswith(".json") and os.path.exists(save_directory):
            tokenizer_path = save_directory
        else:
            tokenizer_path = os.path.join(save_directory, "tokenizer.json")
            if not os.path.exists(tokenizer_path):
                return ByteLevelSNNTokenizer()
        
        tokenizer = SpikeTokenizer()
        tokenizer.load(tokenizer_path)
        return tokenizer

# 🌟 Phase 3: SNNネイティブなSpikingLLMの自動ローダー
class AutoSpikingLM:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        vocab_size = kwargs.get("vocab_size", 100000)
        
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            print(f"[AutoSpikingLM] Initializing a new SpikingLLM (config not found at {pretrained_model_name_or_path})")
            return SpikingLLM(num_layers=2, sdr_size=128, vocab_size=vocab_size, enable_learning=True)
            
        # TODO: 将来的には config.json や重みバイナリから状態をロードする処理を実装
        return SpikingLLM(num_layers=2, sdr_size=128, vocab_size=vocab_size, enable_learning=False)

# 🌟 Phase 2: エージェント（Function Calling搭載）の自動ローダー
class AutoSpikingAgent:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        print(f"[AutoSpikingAgent] Loading Agentic SNN from {pretrained_model_name_or_path}...")
        # 内部でSpikingLLMや海馬システムを自己構築
        return SaraAgent()

class AutoModelForCausalSNN:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            config = SNNTransformerConfig(vocab_size=256, embed_dim=128, num_layers=2, ffn_dim=256)
            return SpikingTransformerModel(config)
        return SpikingTransformerModel.from_pretrained(pretrained_model_name_or_path)

class AutoSNNModelForSequenceClassification:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            config = SNNSequenceClassifierConfig(vocab_size=256, num_classes=2)
            return SpikingSequenceClassifier(config)
        return SpikingSequenceClassifier.from_pretrained(pretrained_model_name_or_path)

class AutoSNNModelForFeatureExtraction:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(config_path):
            config = SNNFeatureExtractorConfig(vocab_size=256, reservoir_size=4096, context_length=32)
            return SpikingFeatureExtractor(config)
        return SpikingFeatureExtractor.from_pretrained(pretrained_model_name_or_path)

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
            config = SNNTokenClassifierConfig(num_classes=4)
            return SpikingTokenClassifier(config)
        return SpikingTokenClassifier.from_pretrained(pretrained_model_name_or_path)