"""Top-level public API for sara_engine.

This module intentionally avoids eager imports of heavy submodules.
Public symbols are resolved lazily via ``__getattr__`` to keep import-time
side effects and optional dependency coupling minimal.
"""

from __future__ import annotations

import importlib
import os
from typing import Dict, Tuple

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", ".."))
_WORKSPACE_CACHE_DIR = os.path.join(_PROJECT_ROOT, "workspace", "cache")
_MPL_CACHE_DIR = os.path.join(_WORKSPACE_CACHE_DIR, "matplotlib")
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _WORKSPACE_CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)

__version__ = "1.0.0"

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Transformers-like API
    "pipeline": ("sara_engine.pipelines", "pipeline"),
    "AutoTokenizer": ("sara_engine.auto", "AutoTokenizer"),
    "AutoSpikingLM": ("sara_engine.auto", "AutoSpikingLM"),
    "AutoSpikingAgent": ("sara_engine.auto", "AutoSpikingAgent"),
    "AutoModelForCausalSNN": ("sara_engine.auto", "AutoModelForCausalSNN"),
    "AutoSNNModelForSequenceClassification": ("sara_engine.auto", "AutoSNNModelForSequenceClassification"),
    "AutoSNNModelForFeatureExtraction": ("sara_engine.auto", "AutoSNNModelForFeatureExtraction"),
    "AutoSNNModelForImageClassification": ("sara_engine.auto", "AutoSNNModelForImageClassification"),
    "AutoSNNModelForTokenClassification": ("sara_engine.auto", "AutoSNNModelForTokenClassification"),
    # RAG
    "SNNRAGPipeline": ("sara_engine.rag", "SNNRAGPipeline"),
    # Tooling
    "ToolRegistry": ("sara_engine.tools", "ToolRegistry"),
    "ToolResult": ("sara_engine.tools", "ToolResult"),
    "ToolDefinition": ("sara_engine.tools", "ToolDefinition"),
    "ToolParameter": ("sara_engine.tools", "ToolParameter"),
    "tool": ("sara_engine.tools", "tool"),
    "register_builtin_tools": ("sara_engine.tools", "register_builtin_tools"),
    # Safety
    "SafetyGuard": ("sara_engine.safety", "SafetyGuard"),
    "SafetyLevel": ("sara_engine.safety", "SafetyLevel"),
    "SafetyCheckResult": ("sara_engine.safety", "SafetyCheckResult"),
    # Evaluation
    "SARABenchmark": ("sara_engine.evaluation", "SARABenchmark"),
    "RAGEvaluator": ("sara_engine.evaluation", "RAGEvaluator"),
    "ToolEvaluator": ("sara_engine.evaluation", "ToolEvaluator"),
    "SafetyEvaluator": ("sara_engine.evaluation", "SafetyEvaluator"),
    "EvalResult": ("sara_engine.evaluation", "EvalResult"),
    "EvalMetric": ("sara_engine.evaluation", "EvalMetric"),
    # Core & Agent
    "SpikingLLM": ("sara_engine.models.spiking_llm", "SpikingLLM"),
    "SaraAgent": ("sara_engine.agent.sara_agent", "SaraAgent"),
    "SaraInference": ("sara_engine.inference", "SaraInference"),
    # Neural components
    "SpikeTransformerBlock": ("sara_engine.core.transformer", "SpikeTransformerBlock"),
    "SpikeTransformerModel": ("sara_engine.core.transformer", "SpikeTransformerModel"),
    "SpikeSelfAttention": ("sara_engine.core.spike_attention", "SpikeSelfAttention"),
    "SpikeNormalization": ("sara_engine.core.layers", "SpikeNormalization"),
    "SpikeFeedForward": ("sara_engine.core.layers", "SpikeFeedForward"),
    "DynamicLiquidLayer": ("sara_engine.core.layers", "DynamicLiquidLayer"),
    # Encoders & data loaders
    "SpikeStreamDataLoader": ("sara_engine.core.data_loader", "SpikeStreamDataLoader"),
    "TextToSpikeEncoder": ("sara_engine.core.data_loader", "TextToSpikeEncoder"),
    "SemanticSpikeEncoder": ("sara_engine.core.data_loader", "SemanticSpikeEncoder"),
    "SDREncoder": ("sara_engine.memory.sdr", "SDREncoder"),
    "AudioSpikeEncoder": ("sara_engine.encoders.audio", "AudioSpikeEncoder"),
    "ImageSpikeEncoder": ("sara_engine.encoders.vision", "ImageSpikeEncoder"),
    "FluidFieldDynamics": ("sara_engine.dynamics", "FluidFieldDynamics"),
    # Legacy & utils
    "SaraGPT": ("sara_engine.models.gpt", "SaraGPT"),
    "StatefulRLMAgent": ("sara_engine.models.rlm", "StatefulRLMAgent"),
    "SparseMemoryStore": ("sara_engine.memory.ltm", "SparseMemoryStore"),
    "SaraVisualizer": ("sara_engine.utils.visualizer", "SaraVisualizer"),
}

__all__ = list(_LAZY_EXPORTS.keys()) + ["__version__"]


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'sara_engine' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

