_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/__init__.py",
    "//": "ファイルの日本語タイトル: パッケージ初期化モジュール",
    "//": "ファイルの目的や内容: ユーザーがSARA Engineを利用する際の最上位APIエントリーポイント。pipelineとAutoクラス群を最優先で公開。"
}

__version__ = "0.3.0"  # PyPIリリースに向けたバージョンアップ

# --- Hugging Face Transformers-like API (Main Public Interface) ---
from .pipelines import pipeline
from .auto import (
    AutoTokenizer,
    AutoSpikingLM,            # 追加
    AutoSpikingAgent,         # 追加
    AutoModelForCausalSNN,
    AutoSNNModelForSequenceClassification,
    AutoSNNModelForFeatureExtraction,
    AutoSNNModelForImageClassification,
    AutoSNNModelForTokenClassification
)

# --- Core Components ---
from .models.spiking_llm import SpikingLLM
from .agent.sara_agent import SaraAgent
from .inference import SaraInference
from .core.data_loader import SpikeStreamDataLoader, TextToSpikeEncoder, SemanticSpikeEncoder
from .core.layers import DynamicLiquidLayer, SpikeNormalization, SpikeFeedForward
from .core.spike_attention import SpikeSelfAttention
from .core.transformer import SpikeTransformerBlock, SpikeTransformerModel

# --- Legacy & Memory ---
from .encoders.vision import ImageSpikeEncoder
from .encoders.audio import AudioSpikeEncoder
from .utils.visualizer import SaraVisualizer
from .memory.ltm import SparseMemoryStore
from .memory.sdr import SDREncoder
from .models.rlm import StatefulRLMAgent
from .models.gpt import SaraGPT

__all__ = [
    # Transformers-like API
    "pipeline",
    "AutoTokenizer",
    "AutoSpikingLM",
    "AutoSpikingAgent",
    "AutoModelForCausalSNN",
    "AutoSNNModelForSequenceClassification",
    "AutoSNNModelForFeatureExtraction",
    "AutoSNNModelForImageClassification",
    "AutoSNNModelForTokenClassification",
    
    # Core & Agent
    "SpikingLLM",
    "SaraAgent",
    "SaraInference",
    
    # Neural Components
    "SpikeTransformerBlock",
    "SpikeTransformerModel",
    "SpikeSelfAttention",
    "SpikeNormalization",
    "SpikeFeedForward",
    "DynamicLiquidLayer",
    
    # Encoders & Data Loaders
    "SpikeStreamDataLoader",
    "TextToSpikeEncoder",
    "SemanticSpikeEncoder",
    "SDREncoder",
    "AudioSpikeEncoder",
    "ImageSpikeEncoder",
    
    # Legacy & Utils
    "SaraGPT",
    "StatefulRLMAgent",
    "SparseMemoryStore",
    "SaraVisualizer",
]