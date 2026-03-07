from .models.gpt import SaraGPT
from .models.rlm import StatefulRLMAgent
from .memory.sdr import SDREncoder
from .memory.ltm import SparseMemoryStore
from .utils.visualizer import SaraVisualizer
from .encoders.audio import AudioSpikeEncoder
from .encoders.vision import ImageSpikeEncoder
from .core.transformer import SpikeTransformerBlock, SpikeTransformerModel
from .core.spike_attention import SpikeSelfAttention
from .core.layers import DynamicLiquidLayer, SpikeNormalization, SpikeFeedForward
from .core.data_loader import SpikeStreamDataLoader, TextToSpikeEncoder, SemanticSpikeEncoder
from .inference import SaraInference
from .agent.sara_agent import SaraAgent
from .models.spiking_llm import SpikingLLM
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
from .pipelines import pipeline
from .rag import SNNRAGPipeline
from .tools import ToolRegistry, ToolResult, ToolDefinition, ToolParameter, tool, register_builtin_tools
from .safety import SafetyGuard, SafetyLevel, SafetyCheckResult
from .evaluation import SARABenchmark, RAGEvaluator, ToolEvaluator, SafetyEvaluator, EvalResult, EvalMetric
# ディレクトリパス: src/sara_engine/__init__.py
# ファイルの日本語タイトル: パッケージ初期化モジュール
# ファイルの目的や内容: ユーザーがSARA Engineを利用する際の最上位APIエントリーポイント。pipelineとAutoクラス群を最優先で公開。RAG/ツール/安全制御/評価基盤を統合。
__version__ = "0.4.0"  # RAG/ツール/安全制御/評価基盤の強化

# --- Hugging Face Transformers-like API (Main Public Interface) ---

# --- Core Components ---

# --- Legacy & Memory ---

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

    # RAG
    "SNNRAGPipeline",

    # ツール実行基盤
    "ToolRegistry",
    "ToolResult",
    "ToolDefinition",
    "ToolParameter",
    "tool",
    "register_builtin_tools",

    # 安全制御
    "SafetyGuard",
    "SafetyLevel",
    "SafetyCheckResult",

    # 評価基盤
    "SARABenchmark",
    "RAGEvaluator",
    "ToolEvaluator",
    "SafetyEvaluator",
    "EvalResult",
    "EvalMetric",

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
