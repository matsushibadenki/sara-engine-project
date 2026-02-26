from .encoders.vision import ImageSpikeEncoder
from .encoders.audio import AudioSpikeEncoder
from .utils.visualizer import SaraVisualizer
from .memory.ltm import SparseMemoryStore
from .memory.sdr import SDREncoder
from .models.rlm import StatefulRLMAgent
from .models.gpt import SaraGPT
from .core.data_loader import SpikeStreamDataLoader, TextToSpikeEncoder, SemanticSpikeEncoder  # 追加
from .core.layers import DynamicLiquidLayer, SpikeNormalization, SpikeFeedForward
from .core.spike_attention import SpikeSelfAttention
from .core.transformer import SpikeTransformerBlock, SpikeTransformerModel
from .pipelines import pipeline
from .auto import (
    AutoTokenizer,
    AutoModelForCausalSNN,
    AutoSNNModelForSequenceClassification,
    AutoSNNModelForFeatureExtraction,
    AutoSNNModelForImageClassification
)
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/__init__.py",
    "//": "ファイルの日本語タイトル: パッケージ初期化モジュール",
    "//": "ファイルの目的や内容: SpikeStreamDataLoaderなどの新規モジュールを公開し、アクセス性を向上させる。"
}

__version__ = "0.2.3"

# --- New Hugging Face Transformers-like API ---

# --- Core Components ---

# --- Legacy & Memory ---

__all__ = [
    "pipeline",
    "AutoTokenizer",
    "AutoModelForCausalSNN",
    "AutoSNNModelForSequenceClassification",
    "AutoSNNModelForFeatureExtraction",
    "AutoSNNModelForImageClassification",
    "SpikeTransformerBlock",
    "SpikeTransformerModel",
    "SpikeSelfAttention",
    "SpikeNormalization",
    "SpikeFeedForward",
    "SpikeStreamDataLoader",
    "TextToSpikeEncoder",
    "SemanticSpikeEncoder",
    "SaraGPT",
    "StatefulRLMAgent",
    "SDREncoder",
    "SparseMemoryStore",
    "DynamicLiquidLayer",
    "SaraVisualizer",
    "AudioSpikeEncoder",
    "ImageSpikeEncoder",
]
