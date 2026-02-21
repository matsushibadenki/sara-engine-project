_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/__init__.py",
    "//": "ファイルの日本語タイトル: パッケージ初期化モジュール",
    "//": "ファイルの目的や内容: 外部から主要なクラスへ簡単にアクセスできるようにする。従来のモデルに加えて、TransformersライクなAutoクラスとpipelineを公開。"
}

__version__ = "0.2.0"

# --- New Hugging Face Transformers-like API ---
from .auto import (
    AutoSNNModelForCausalLM,
    AutoSNNModelForSequenceClassification,
    AutoSNNModelForFeatureExtraction,
    AutoSNNModelForImageClassification,
    AutoSpikeTokenizer
)
from .pipelines import pipeline

# --- Legacy & Core Models ---
from .models.gpt import SaraGPT
from .models.rlm import StatefulRLMAgent
from .memory.sdr import SDREncoder
from .memory.ltm import SparseMemoryStore
from .core.layers import DynamicLiquidLayer
from .utils.visualizer import SaraVisualizer
from .encoders.audio import AudioSpikeEncoder
from .encoders.vision import ImageSpikeEncoder

__all__ = [
    # Pipelines & Auto Classes
    "pipeline",
    "AutoSNNModelForCausalLM",
    "AutoSNNModelForSequenceClassification",
    "AutoSNNModelForFeatureExtraction",
    "AutoSNNModelForImageClassification",
    "AutoSpikeTokenizer",
    
    # Legacy Core
    "SaraGPT",
    "StatefulRLMAgent",
    "SDREncoder",
    "SparseMemoryStore",
    "DynamicLiquidLayer",
    "SaraVisualizer",
    "AudioSpikeEncoder",
    "ImageSpikeEncoder",
]