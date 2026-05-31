# Directory Path: scripts/eval/parameter_efficiency_benchmark.py
# English Title: Parameter Efficiency Benchmark
# Purpose/Content: Estimates sparse active parameter efficiency for representative CPU-first runtime components and writes a managed report.

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SCRIPT_PATH not in sys.path:
    sys.path.insert(0, SCRIPT_PATH)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from agent_dialogue_benchmark import run_agent_dialogue_benchmark
from inference_accuracy_benchmark import run_inference_accuracy_benchmark
from spiking_llm_accuracy_benchmark import run_spiking_llm_accuracy_benchmark
from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.dynamics.fluid_field import FluidFieldDynamics
from sara_engine.inference import SaraInference
from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.utils.future_state_runtime import LightweightFutureStateRuntime
from sara_engine.utils.turboquant import create_turboquant_engine
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _sum_row_lengths(rows: Iterable[Dict[Any, Any]]) -> int:
    return sum(len(row) for row in rows if isinstance(row, dict))


def _count_inference_parameters(engine: SaraInference) -> Dict[str, int]:
    active_parameter_count = _sum_row_lengths(engine.direct_map.values())
    return {
        "active_parameter_count": int(active_parameter_count),
        "nonzero_synapse_count": int(active_parameter_count),
        "state_unit_count": int(len(engine.direct_map)),
    }


def _count_llm_parameters(model: SpikingLLM) -> Dict[str, int]:
    pretrained_count = sum(
        len(post_dict)
        for delay_dict in model.pretrained_synapses.values()
        for post_dict in delay_dict.values()
        if isinstance(post_dict, dict)
    )
    direct_count = _sum_row_lengths(model._direct_map.values())
    lm_head_count = _sum_row_lengths(model.lm_head_w)
    active_parameter_count = pretrained_count + direct_count + lm_head_count
    return {
        "active_parameter_count": int(active_parameter_count),
        "nonzero_synapse_count": int(active_parameter_count),
        "state_unit_count": int(len(model.pretrained_synapses) + len(model._direct_map) + len(model.lm_head_w)),
    }


def _count_agent_parameters(agent: SaraAgent) -> Dict[str, int]:
    llm_counts = _count_llm_parameters(agent.llm)
    episodic_synapse_count = _sum_row_lengths(agent.episodic_snn.synapses.values())
    episodic_neuron_count = len(agent.episodic_snn.neurons)
    router_weights = getattr(agent.cortex, "router_weights", [])
    router_synapse_count = _sum_row_lengths(router_weights if isinstance(router_weights, list) else [])
    active_parameter_count = (
        llm_counts["active_parameter_count"]
        + episodic_synapse_count
        + episodic_neuron_count
        + router_synapse_count
    )
    return {
        "active_parameter_count": int(active_parameter_count),
        "nonzero_synapse_count": int(
            llm_counts["nonzero_synapse_count"] + episodic_synapse_count + router_synapse_count
        ),
        "state_unit_count": int(
            llm_counts["state_unit_count"] + episodic_neuron_count + len(router_weights)
        ),
    }


def _path_size_bytes(path: str) -> int:
    if os.path.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for filename in files:
                total += os.path.getsize(os.path.join(root, filename))
        return total
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0


def _score_ratio(value: float, target: float) -> float:
    if target <= 0.0:
        return 0.0
    return max(0.0, min(1.0, value / target))


def _build_inference_case() -> Dict[str, Any]:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = workspace_path("tests", "parameter_efficiency_inference.msgpack")
    engine.direct_map = {}
    engine.context_index = {}
    engine.retrieval_diagnostics = []
    engine.refractory_buffer = []
    engine.session_memory = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.context_encoding = "stable_v1"
    engine.quantization_enabled = True
    engine._future_state_runtime = LightweightFutureStateRuntime()
    engine._fluid_field_dynamics = FluidFieldDynamics()
    engine.lif_network = None
    engine._turboquant_engine = create_turboquant_engine()
    for offset in range(24):
        engine.learn_sequence([offset, offset + 1, offset + 2, offset + 3])
    engine.save_pretrained(engine.model_path)
    counts = _count_inference_parameters(engine)
    return {
        "component": "sara_inference",
        "quality_score": float(run_inference_accuracy_benchmark().get("overall_score", 0.0)),
        "artifact_size_bytes": int(_path_size_bytes(engine.model_path)),
        "quantization_enabled": bool(engine.quantization_enabled),
        **counts,
    }


def _build_llm_case() -> Dict[str, Any]:
    model = SpikingLLM(
        num_layers=1,
        sdr_size=32,
        vocab_size=256,
        enable_learning=True,
        enable_turboquant=True,
    )
    model.fit("release checks should stay stable and lightweight during local planning")
    save_dir = workspace_path("tests", "parameter_efficiency_spiking_llm")
    model.save_pretrained(save_dir)
    counts = _count_llm_parameters(model)
    return {
        "component": "spiking_llm",
        "quality_score": float(run_spiking_llm_accuracy_benchmark().get("overall_score", 0.0)),
        "artifact_size_bytes": int(_path_size_bytes(save_dir)),
        "quantization_enabled": bool(model.quantization_enabled),
        **counts,
    }


def _build_agent_case() -> Dict[str, Any]:
    agent = SaraAgent(
        input_size=128,
        hidden_size=128,
        compartments=["general", "python_expert"],
    )
    agent.register_tool("<CALC>", lambda _: "5")
    agent.chat("Python の関数は再利用可能な処理です。", teaching_mode=True)
    agent.chat("この要点を教えて <CALC>", teaching_mode=False)
    save_dir = workspace_path("tests", "parameter_efficiency_sara_agent")
    agent.save_agent(save_dir)
    counts = _count_agent_parameters(agent)
    return {
        "component": "sara_agent",
        "quality_score": float(run_agent_dialogue_benchmark().get("overall_score", 0.0)),
        "artifact_size_bytes": int(_path_size_bytes(save_dir)),
        "quantization_enabled": bool(getattr(agent.llm, "quantization_enabled", False)),
        **counts,
    }


def run_parameter_efficiency_benchmark() -> Dict[str, Any]:
    cases = [
        _build_inference_case(),
        _build_llm_case(),
        _build_agent_case(),
    ]
    for case in cases:
        active_parameter_count = max(int(case.get("active_parameter_count", 0)), 1)
        artifact_size_bytes = max(int(case.get("artifact_size_bytes", 0)), 1)
        quality_score = float(case.get("quality_score", 0.0))
        case["quality_per_kparam"] = float(quality_score / (active_parameter_count / 1000.0))
        case["quality_per_mb"] = float(quality_score / (artifact_size_bytes / (1024.0 * 1024.0)))

    average_quality_per_kparam = sum(float(case["quality_per_kparam"]) for case in cases) / max(len(cases), 1)
    average_quality_per_mb = sum(float(case["quality_per_mb"]) for case in cases) / max(len(cases), 1)
    average_active_parameters = sum(int(case["active_parameter_count"]) for case in cases) / max(len(cases), 1)
    average_artifact_size_mb = (
        sum(int(case["artifact_size_bytes"]) for case in cases) / max(len(cases), 1)
    ) / (1024.0 * 1024.0)

    metrics = {
        "quality_per_kparam_score": _score_ratio(average_quality_per_kparam, 1.0),
        "quality_per_mb_score": _score_ratio(average_quality_per_mb, 4.0),
        "bounded_parameter_footprint_score": 1.0 if average_active_parameters <= 20000.0 else 0.0,
        "bounded_artifact_footprint_score": 1.0 if average_artifact_size_mb <= 2.0 else 0.0,
        "average_quality_per_kparam": float(average_quality_per_kparam),
        "average_quality_per_mb": float(average_quality_per_mb),
    }
    thresholds = {
        "quality_per_kparam_score": 0.5,
        "quality_per_mb_score": 0.5,
        "bounded_parameter_footprint_score": 1.0,
        "bounded_artifact_footprint_score": 1.0,
    }
    threshold_results = {
        name: float(metrics.get(name, 0.0)) >= threshold
        for name, threshold in thresholds.items()
    }
    return {
        "evaluator_name": "ParameterEfficiencyBenchmark",
        "overall_score": sum(float(metrics[name]) for name in thresholds) / max(len(thresholds), 1),
        "metrics": metrics,
        "details": {
            "test_results": cases,
            "average_active_parameters": float(average_active_parameters),
            "average_artifact_size_mb": float(average_artifact_size_mb),
        },
        "thresholds": thresholds,
        "threshold_results": threshold_results,
        "passed": all(threshold_results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight sparse parameter-efficiency benchmark.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "parameter_efficiency_benchmark.json"),
        help="Managed output path for the benchmark report.",
    )
    args = parser.parse_args()

    report = run_parameter_efficiency_benchmark()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("Parameter-efficiency benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
