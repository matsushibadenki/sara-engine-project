_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/__init__.py",
    "//": "ファイルの日本語タイトル: パッケージ初期化モジュール",
    "//": "ファイルの目的や内容: SpikeStreamDataLoaderなどの新規モジュールを公開し、アクセス性を向上させる。"
}

__version__ = "0.2.1"

# --- New Hugging Face Transformers-like API ---
from .auto import (
    AutoTokenizer,
    AutoModelForCausalSNN,
    AutoSNNModelForSequenceClassification,
    AutoSNNModelForFeatureExtraction,
    AutoSNNModelForImageClassification
)
from .pipelines import pipeline

# --- Core Components ---
from .core.transformer import SpikeTransformerBlock, SpikeTransformerModel
from .core.spike_attention import SpikeSelfAttention
from .core.layers import DynamicLiquidLayer, SpikeNormalization, SpikeFeedForward
from .core.data_loader import SpikeStreamDataLoader, TextToSpikeEncoder, SemanticSpikeEncoder # 追加

# --- Legacy & Memory ---
from .models.gpt import SaraGPT
from .models.rlm import StatefulRLMAgent
from .memory.sdr import SDREncoder
from .memory.ltm import SparseMemoryStore
from .utils.visualizer import SaraVisualizer
from .encoders.audio import AudioSpikeEncoder
from .encoders.vision import ImageSpikeEncoder

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