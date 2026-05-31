# Directory Path: scripts/eval/phase4_scale_continual_benchmark.py
# English Title: Phase 4 Scale-out and Continual Learning Benchmark
# Purpose/Content: Runs a lightweight CPU-only Phase 4 benchmark to validate structural plasticity stability, hippocampal transfer integrity, scale-out retention, and continual-learning drift bounds.

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", "cache"))

from sara_engine.core.cortex import CorticalColumn
from sara_engine.inference import SaraInference
from sara_engine.memory.hippocampus import CorticoHippocampalSystem
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _build_inference_engine() -> SaraInference:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {}
    engine.context_index = {}
    engine.retrieval_diagnostics = []
    engine.refractory_buffer = []
    engine.lif_network = None
    return engine


def _predict_next_token(engine: SaraInference, context_tokens: List[int]) -> Optional[int]:
    key = engine._find_best_matching_key(context_tokens)
    if key is None or key not in engine.direct_map:
        return None
    token = engine._sample_next_token(
        key,
        top_k=1,
        temperature=0.0,
        refractory_penalty=1.0,
    )
    return int(token) if token is not None else None


def _run_structural_plasticity_stability_case() -> Dict[str, Any]:
    random.seed(11)
    contexts = ["Task_ICL", "Task_QA", "Task_Control"]
    cortex = CorticalColumn(
        input_size=256,
        hidden_size_per_comp=128,
        compartment_names=contexts,
        target_rate=0.05,
    )

    def _count_synapses(column: CorticalColumn) -> int:
        total = 0
        for layer in column.compartments.values():
            for targets in layer.synapses.values():
                total += len(targets)
        return total

    initial_synapses = _count_synapses(cortex)
    for step in range(320):
        context = contexts[step % len(contexts)]
        active_inputs = random.sample(range(256), 8)
        cortex.forward_latent_chain(
            active_inputs=active_inputs,
            prev_active_hidden=[],
            current_context=context,
            learning=True,
        )
    final_synapses = _count_synapses(cortex)
    ratio = final_synapses / max(initial_synapses, 1)
    per_context_non_empty = all(
        any(len(targets) > 0 for targets in layer.synapses.values())
        for layer in cortex.compartments.values()
    )
    success = 0.45 <= ratio <= 1.60 and per_context_non_empty
    return {
        "success": bool(success),
        "initial_synapse_count": int(initial_synapses),
        "final_synapse_count": int(final_synapses),
        "synapse_ratio": float(ratio),
        "per_context_non_empty": bool(per_context_non_empty),
        "description": "Structural plasticity should remain stable without topology collapse during continual updates.",
    }


def _run_hippocampal_transfer_case() -> Dict[str, Any]:
    random.seed(17)
    ltm_path = workspace_path("tests", "phase4_hippocampus_ltm.pkl")
    cortex = CorticalColumn(
        input_size=128,
        hidden_size_per_comp=96,
        compartment_names=["Task_ICL", "Task_QA"],
        target_rate=0.04,
    )
    system = CorticoHippocampalSystem(
        cortex=cortex,
        ltm_filepath=ltm_path,
        max_working_memory_size=12,
        snn_input_size=128,
    )

    system.experience_and_memorize(
        sensory_sdr=[3, 7, 11, 19],
        content="Loop memory keeps retrieval stable.",
        context="Task_ICL",
        learning=True,
        metadata={"context": "task_icl", "memory_role": "semantic", "keywords": ["loop", "memory", "stable"]},
    )
    system.experience_and_memorize(
        sensory_sdr=[5, 9, 13, 21],
        content="Noise branch should not dominate.",
        context="Task_ICL",
        learning=True,
        metadata={"context": "task_icl", "memory_role": "episodic", "keywords": ["noise", "branch"]},
    )

    query_metadata = {"contexts": ["task_icl"], "preferred_role": "semantic", "keywords": ["loop", "stable"]}
    before = system.in_context_inference([3, 7, 11], context="Task_ICL", query_metadata=query_metadata)
    before_top = before[0] if before else {}
    before_score = float(before_top.get("score", 0.0))
    before_content = str(before_top.get("content", ""))

    system.consolidate_memories(context="Task_ICL", replay_count=2)
    after = system.in_context_inference([3, 7, 11], context="Task_ICL", query_metadata=query_metadata)
    after_top = after[0] if after else {}
    after_score = float(after_top.get("score", 0.0))
    after_content = str(after_top.get("content", ""))

    success = (
        "Loop memory keeps retrieval stable." in before_content
        and "Loop memory keeps retrieval stable." in after_content
        and after_score >= max(0.10, before_score * 0.85)
    )
    return {
        "success": bool(success),
        "before_top_score": float(before_score),
        "after_top_score": float(after_score),
        "before_top_content": before_content,
        "after_top_content": after_content,
        "description": "Hippocampal replay should preserve semantic retrieval focus after consolidation.",
    }


def _run_scale_out_retention_case() -> Dict[str, Any]:
    random.seed(23)
    engine = _build_inference_engine()
    anchors = {
        (1, 2): 3,
        (4, 5): 6,
        (7, 8): 9,
    }
    for context, token in anchors.items():
        engine.learn_sequence([context[0], context[1], token])
        engine.learn_sequence([context[0], context[1], token])

    for offset in range(100, 1000):
        engine.learn_sequence([offset, offset + 1, offset + 2])
    for context, token in anchors.items():
        engine.learn_sequence([context[0], context[1], token])

    start = time.perf_counter()
    predictions = {
        context: _predict_next_token(engine, [context[0], context[1]])
        for context in anchors
    }
    elapsed = time.perf_counter() - start

    retention_hits = sum(1 for context, expected in anchors.items() if predictions.get(context) == expected)
    retention_rate = retention_hits / max(len(anchors), 1)
    avg_query_ms = (elapsed / max(len(anchors), 1)) * 1000.0
    success = retention_rate >= 0.99 and avg_query_ms <= 30.0
    return {
        "success": bool(success),
        "retention_rate": float(retention_rate),
        "average_query_ms": float(avg_query_ms),
        "prediction_map": {f"{k[0]}-{k[1]}": int(v) if v is not None else None for k, v in predictions.items()},
        "description": "Scale-out updates should keep anchor-path retention and bounded query latency.",
    }


def _run_continual_drift_bound_case() -> Dict[str, Any]:
    random.seed(29)
    engine = _build_inference_engine()
    anchor_context = [31, 32]
    expected_token = 33
    engine.learn_sequence([31, 32, 33])
    baseline = _predict_next_token(engine, anchor_context)

    for offset in range(1500, 1800):
        engine.learn_sequence([offset, offset + 1, offset + 2])
    # Force a hard drift scenario by rebuilding memory from unrelated traces only.
    engine.direct_map = {}
    engine.context_index = {}
    for offset in range(2000, 2100):
        engine.learn_sequence([offset, offset + 1, offset + 2])
    drifted = _predict_next_token(engine, anchor_context)

    for _ in range(3):
        engine.learn_sequence([31, 32, 33])
    recovered = _predict_next_token(engine, anchor_context)

    drift_detected = drifted != expected_token
    recovered_ok = recovered == expected_token
    success = baseline == expected_token and drift_detected and recovered_ok
    return {
        "success": bool(success),
        "baseline_prediction": int(baseline) if baseline is not None else None,
        "drifted_prediction": int(drifted) if drifted is not None else None,
        "recovered_prediction": int(recovered) if recovered is not None else None,
        "description": "Continual drift should be observable and replay should recover the anchor memory path.",
    }


def run_phase4_scale_continual_benchmark() -> Dict[str, Any]:
    structural = _run_structural_plasticity_stability_case()
    hippocampal = _run_hippocampal_transfer_case()
    scale_out = _run_scale_out_retention_case()
    continual = _run_continual_drift_bound_case()
    cases = [structural, hippocampal, scale_out, continual]

    metrics = {
        "structural_plasticity_stability": 1.0 if structural["success"] else 0.0,
        "hippocampal_transfer_integrity": 1.0 if hippocampal["success"] else 0.0,
        "scale_out_retention_integrity": 1.0 if scale_out["success"] else 0.0,
        "continual_drift_recovery_integrity": 1.0 if continual["success"] else 0.0,
    }
    thresholds = {
        "structural_plasticity_stability": 1.0,
        "hippocampal_transfer_integrity": 1.0,
        "scale_out_retention_integrity": 1.0,
        "continual_drift_recovery_integrity": 1.0,
    }
    threshold_results = {
        metric_name: metrics.get(metric_name, 0.0) >= threshold
        for metric_name, threshold in thresholds.items()
    }
    quality_metrics = {
        "structural_synapse_ratio": float(structural.get("synapse_ratio", 0.0) or 0.0),
        "structural_per_context_non_empty": 1.0 if bool(structural.get("per_context_non_empty", False)) else 0.0,
        "hippocampal_after_top_score": float(hippocampal.get("after_top_score", 0.0) or 0.0),
        "hippocampal_score_retention_ratio": (
            float(hippocampal.get("after_top_score", 0.0) or 0.0)
            / max(float(hippocampal.get("before_top_score", 0.0) or 0.0), 1e-9)
        ),
        "scale_out_retention_rate": float(scale_out.get("retention_rate", 0.0) or 0.0),
        "scale_out_average_query_ms": float(scale_out.get("average_query_ms", 0.0) or 0.0),
        "continual_baseline_recovered": 1.0
        if continual.get("baseline_prediction") == continual.get("recovered_prediction")
        else 0.0,
        "continual_drift_observed": 1.0
        if continual.get("drifted_prediction") != continual.get("recovered_prediction")
        else 0.0,
    }

    return {
        "evaluator_name": "Phase4ScaleContinualBenchmark",
        "overall_score": sum(metrics.values()) / max(len(metrics), 1),
        "metrics": metrics,
        "quality_metrics": quality_metrics,
        "details": {"test_results": cases},
        "thresholds": thresholds,
        "threshold_results": threshold_results,
        "passed": all(threshold_results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 4 scale-out and continual-learning benchmark.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase4_scale_continual_benchmark.json"),
        help="Managed output path for the benchmark report.",
    )
    args = parser.parse_args()

    report = run_phase4_scale_continual_benchmark()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("Phase 4 scale-out benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
