import json
import msgpack
import os
from typing import Any, Dict, Optional

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _build_inference_runtime(model_path: str) -> SaraInference:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = model_path
    engine.direct_map = {}
    engine.context_index = {}
    engine.retrieval_diagnostics = []
    engine.refractory_buffer = []
    engine.session_memory = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.lif_network = None
    engine._load_memory()
    return engine


def _read_quantization_state(model_path: str) -> Dict[str, Any]:
    with open(model_path, "rb") as handle:
        payload = msgpack.unpack(handle, raw=False)
    if not isinstance(payload, dict):
        return {"enabled": False, "metadata": {}}

    metadata = payload.get("quantization", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "enabled": isinstance(payload.get("quantized_direct_map"), dict),
        "metadata": metadata,
    }


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def inspect_inference_memory(
    model_path: str,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_model_path = os.path.abspath(model_path)
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

    engine = _build_inference_runtime(resolved_model_path)
    quantization = _read_quantization_state(resolved_model_path)

    branching_counts = [len(values) for values in engine.direct_map.values()]
    diagnostics = engine.get_recent_retrieval_diagnostics(limit=10)
    stability_values = [
        _safe_float(item.get("stability_score", 0.0), 0.0)
        for item in diagnostics
        if isinstance(item, dict)
    ]
    contexts_cover_patterns = len(engine.context_index) >= len(engine.direct_map) and len(engine.direct_map) > 0
    diagnostics_schema_ok = all(
        isinstance(item, dict)
        and "source" in item
        and "content_preview" in item
        and "stability_score" in item
        for item in diagnostics
    )
    supports_fuzzy_retrieval = len(engine.context_index) > 0
    artifact_generation = "indexed" if supports_fuzzy_retrieval else "legacy_direct_map_only"
    session_memory_keys = sorted(engine.session_memory.keys())
    predictor_state = getattr(engine, "predictor_state", {})
    adaptation_state = getattr(engine, "adaptation_state", {})
    predictor_state_keys = sorted(
        key for key, value in predictor_state.items() if value not in ("", 0.0, False)
    )
    adaptation_state_keys = sorted(
        key for key, value in adaptation_state.items() if value not in ("", 0.0, False)
    )
    diagnostic_memory_hits = sorted(
        {
            str(item.get("memory_hit", "retrieval"))
            for item in diagnostics
            if isinstance(item, dict)
        }
    )
    next_step_ready = bool(
        engine.session_memory.get("goal")
        and engine.session_memory.get("task")
    )
    conversational_readiness = {
        "profile_memory_ready": bool(
            engine.session_memory.get("name")
            or engine.session_memory.get("location")
            or engine.session_memory.get("preference")
        ),
        "next_step_ready": next_step_ready,
        "predictor_state_ready": bool(
            predictor_state.get("action")
            and predictor_state.get("target_state")
        ),
        "predictive_branching_ready": bool(
            isinstance(predictor_state.get("branch_candidates"), list)
            and len(predictor_state.get("branch_candidates", [])) >= 2
            and predictor_state.get("preferred_branch")
        ),
        "predictive_simulation_ready": bool(
            isinstance(predictor_state.get("simulated_branch_candidates"), list)
            and len(predictor_state.get("simulated_branch_candidates", [])) >= 2
            and predictor_state.get("best_simulated_branch")
        ),
        "meta_adaptation_ready": str(adaptation_state.get("response_mode", "")) == "directive",
        "session_memory_observable": "session_memory" in diagnostic_memory_hits,
        "operator_trace_ready": bool(str(predictor_state.get("transition_operator", "")).strip()),
        "speculative_trace_ready": bool(
            isinstance(predictor_state.get("speculative_trace"), dict)
            and str(predictor_state.get("speculative_trace", {}).get("predicted_operator", "")).strip()
            and str(predictor_state.get("speculative_trace", {}).get("verified_operator", "")).strip()
        ),
        "fluid_trace_ready": bool(
            isinstance(predictor_state.get("fluid_trace"), dict)
            and bool(predictor_state.get("fluid_trace", {}).get("bounded", False))
            and _safe_float(predictor_state.get("fluid_trace", {}).get("support_score", 0.0), 0.0) > 0.0
        ),
    }

    recommendations = []
    if len(engine.direct_map) <= 0:
        recommendations.append("Model memory is empty. Re-run training or online learning before deployment.")
    if not supports_fuzzy_retrieval:
        recommendations.append(
            "This artifact does not contain context_index entries, so fuzzy retrieval hygiene is limited. "
            "Re-save the model with the current runtime after additional learning to persist indexed contexts."
        )
    if getattr(engine, "context_encoding", "unknown") == "legacy_python_hash":
        recommendations.append(
            "This artifact uses legacy_python_hash context encoding, which is not stable across processes. "
            "Prefer rebuilding with replay data into stable_v1 encoding for practical deployment."
        )
    if not diagnostics:
        recommendations.append(
            "No retrieval diagnostics are stored yet. Run recent inference sessions with the current runtime and save again for richer observability."
        )
    if quantization["enabled"] is False:
        recommendations.append(
            "Quantization is disabled for this artifact. Consider TurboQuant for smaller managed model outputs if accuracy remains acceptable."
        )
    if not conversational_readiness["session_memory_observable"]:
        recommendations.append(
            "Session-memory hits are not visible in recent diagnostics yet. Run a few memory-based dialogue turns and save again for stronger conversational observability."
        )
    if not conversational_readiness["next_step_ready"]:
        recommendations.append(
            "Next-step guidance is not fully ready yet because goal/task session memory is incomplete. Capture both a goal and a current task in a recent session."
        )
    if not conversational_readiness["predictor_state_ready"]:
        recommendations.append(
            "Predictor state is not populated yet. Capture a goal and task so the lightweight future-state predictor can initialize."
        )
    if not conversational_readiness["operator_trace_ready"]:
        recommendations.append(
            "Operator trace is not populated yet. Run a recent next-step exchange so the runtime can persist a transition_operator."
        )
    if not conversational_readiness["speculative_trace_ready"]:
        recommendations.append(
            "Speculative trace is not populated yet. Run a recent next-step exchange so draft/verify operator observability can be saved."
        )
    if not conversational_readiness["fluid_trace_ready"]:
        recommendations.append(
            "Fluid trace is not populated yet. Run a recent next-step exchange so the supplementary fluid dynamics can persist bounded support signals."
        )
    if not conversational_readiness["predictive_branching_ready"]:
        recommendations.append(
            "Predictive branching is not populated yet. Run a short goal/task planning exchange with the current runtime and save again to persist branch candidates."
        )
    if not conversational_readiness["predictive_simulation_ready"]:
        recommendations.append(
            "Predictive simulation is not populated yet. Run a short goal/task planning exchange with ranking or choice prompts and save again to persist simulated branch comparisons."
        )
    if not conversational_readiness["meta_adaptation_ready"]:
        recommendations.append(
            "Meta-adaptation is not active yet. Ask for the next step across a short planning exchange so the response mode can adapt from guided to directive."
        )

    report = {
        "model_path": resolved_model_path,
        "artifact_generation": artifact_generation,
        "context_encoding": getattr(engine, "context_encoding", "unknown"),
        "pattern_count": len(engine.direct_map),
        "context_count": len(engine.context_index),
        "avg_branching_factor": (
            sum(branching_counts) / max(len(branching_counts), 1)
            if branching_counts
            else 0.0
        ),
        "max_branching_factor": max(branching_counts) if branching_counts else 0,
        "quantization_enabled": bool(quantization["enabled"]),
        "quantization_metadata": quantization["metadata"],
        "retrieval_diagnostics_count": len(diagnostics),
        "retrieval_stability_average": (
            sum(stability_values) / max(len(stability_values), 1)
            if stability_values
            else 0.0
        ),
        "latest_retrieval_diagnostics": diagnostics[:3],
        "session_memory_keys": session_memory_keys,
        "session_memory_snapshot": dict(engine.session_memory),
        "predictor_state_keys": predictor_state_keys,
        "predictor_state_snapshot": dict(predictor_state),
        "adaptation_state_keys": adaptation_state_keys,
        "adaptation_state_snapshot": dict(adaptation_state),
        "future_state_runtime_state": dict(getattr(engine, "future_state_runtime_state", {})),
        "diagnostic_memory_hits": diagnostic_memory_hits,
        "conversational_readiness": conversational_readiness,
        "health_checks": {
            "has_patterns": len(engine.direct_map) > 0,
            "contexts_cover_patterns": contexts_cover_patterns,
            "supports_fuzzy_retrieval": supports_fuzzy_retrieval,
            "diagnostics_schema_ok": diagnostics_schema_ok,
        },
        "recommendations": recommendations,
    }

    if report_path:
        resolved_report_path = ensure_parent_directory(report_path)
        with open(resolved_report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        report["report_path"] = resolved_report_path

    return report


def default_memory_health_report_path() -> str:
    return workspace_path("reports", "memory_health_report.json")
