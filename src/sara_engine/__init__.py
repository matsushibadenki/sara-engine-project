from .evaluation import SARABenchmark, RAGEvaluator, ToolEvaluator, SafetyEvaluator, EvalResult, EvalMetric
from .safety import SafetyGuard, SafetyLevel, SafetyCheckResult
from .tools import ToolRegistry, ToolResult, ToolDefinition, ToolParameter, tool, register_builtin_tools
from .rag import SNNRAGPipeline
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
from .models.spiking_llm import SpikingLLM
from .agent.sara_agent import SaraAgent
from .inference import SaraInference
from .core.data_loader import SpikeStreamDataLoader, TextToSpikeEncoder, SemanticSpikeEncoder
from .core.layers import DynamicLiquidLayer, SpikeNormalization, SpikeFeedForward
from .core.spike_attention import SpikeSelfAttention
from .core.transformer import SpikeTransformerBlock, SpikeTransformerModel
from .encoders.vision import ImageSpikeEncoder
from .encoders.audio import AudioSpikeEncoder
from .utils.visualizer import SaraVisualizer
from .memory.ltm import SparseMemoryStore
from .memory.sdr import SDREncoder
from .models.rlm import StatefulRLMAgent
from .models.gpt import SaraGPT
import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", ".."))
_WORKSPACE_CACHE_DIR = os.path.join(_PROJECT_ROOT, "workspace", "cache")
_MPL_CACHE_DIR = os.path.join(_WORKSPACE_CACHE_DIR, "matplotlib")
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _WORKSPACE_CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)

# ディレクトリパス: src/sara_engine/__init__.py
# ファイルの日本語タイトル: パッケージ初期化モジュール
# ファイルの目的や内容: ユーザーがSARA Engineを利用する際の最上位APIエントリーポイント。pipelineとAutoクラス群を最優先で公開。RAG/ツール/安全制御/評価基盤を統合。
__version__ = "0.4.1"  # RAG/ツール/安全制御/評価基盤の強化

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
