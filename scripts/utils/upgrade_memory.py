import json
import os
from typing import Any, Dict, Iterable, List, Optional

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path


def _build_runtime(model_path_value: str, enable_turboquant: bool = False) -> SaraInference:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = model_path_value
    engine.direct_map = {}
    engine.context_index = {}
    engine.retrieval_diagnostics = []
    engine.refractory_buffer = []
    engine.lif_network = None
    engine.quantization_enabled = bool(enable_turboquant)
    engine._load_memory()
    return engine


def _build_empty_runtime(output_path: str, enable_turboquant: bool = False) -> SaraInference:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = output_path
    engine.direct_map = {}
    engine.context_index = {}
    engine.retrieval_diagnostics = []
    engine.refractory_buffer = []
    engine.lif_network = None
    engine.context_encoding = "stable_v1"
    engine.quantization_enabled = bool(enable_turboquant)
    return engine


def _iter_token_sequences(data_path: str) -> Iterable[List[int]]:
    with open(data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            for key in ("tokens", "input_ids", "context_tokens"):
                value = payload.get(key)
                if isinstance(value, list) and all(isinstance(item, int) for item in value):
                    yield [int(item) for item in value]
                    break


def _reindex_contexts(engine: SaraInference, sequences: Iterable[List[int]]) -> Dict[str, int]:
    matched_contexts = 0
    scanned_contexts = 0

    for sequence in sequences:
        for idx in range(1, len(sequence)):
            max_window = min(8, idx)
            for window in range(max_window, 0, -1):
                context = sequence[idx - window:idx]
                scanned_contexts += 1
                sdr_key = engine._encode_context_sdr(context)
                if sdr_key in engine.direct_map:
                    if sdr_key not in engine.context_index:
                        matched_contexts += 1
                    engine._remember_context(sdr_key, context)

    return {
        "scanned_contexts": scanned_contexts,
        "matched_contexts": matched_contexts,
    }


def _rebuild_from_replay(
    output_path: str,
    sequences: Iterable[List[int]],
    enable_turboquant: bool,
) -> tuple[SaraInference, Dict[str, int]]:
    engine = _build_empty_runtime(output_path, enable_turboquant=enable_turboquant)
    example_count = 0
    scanned_contexts = 0
    for sequence in sequences:
        if not sequence:
            continue
        example_count += 1
        scanned_contexts += sum(min(8, idx) for idx in range(1, len(sequence)))
        engine.learn_sequence(sequence)
    return engine, {
        "scanned_contexts": scanned_contexts,
        "matched_contexts": len(engine.context_index),
        "rebuilt_from_replay": example_count,
    }


def upgrade_inference_memory(
    model_path_value: str,
    output_path: str,
    *,
    replay_data_path: Optional[str] = None,
    enable_turboquant: bool = False,
) -> Dict[str, Any]:
    resolved_model_path = os.path.abspath(model_path_value)
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

    resolved_output_path = ensure_parent_directory(output_path)
    engine = _build_runtime(resolved_model_path, enable_turboquant=enable_turboquant)

    before_context_count = len(engine.context_index)
    reindex_summary = {
        "scanned_contexts": 0,
        "matched_contexts": 0,
    }
    if replay_data_path:
        resolved_replay_path = os.path.abspath(replay_data_path)
        if not os.path.exists(resolved_replay_path):
            raise FileNotFoundError(f"Replay data not found: {resolved_replay_path}")
        reindex_summary = _reindex_contexts(engine, _iter_token_sequences(resolved_replay_path))
        if (
            reindex_summary["matched_contexts"] == 0
            and getattr(engine, "context_encoding", "stable_v1") == "legacy_python_hash"
        ):
            engine, reindex_summary = _rebuild_from_replay(
                resolved_output_path,
                _iter_token_sequences(resolved_replay_path),
                enable_turboquant=enable_turboquant,
            )
    else:
        resolved_replay_path = None

    engine.quantization_enabled = bool(enable_turboquant)
    engine.save_pretrained(resolved_output_path)

    after_context_count = len(engine.context_index)
    unresolved_patterns = max(0, len(engine.direct_map) - after_context_count)
    notes = []
    if replay_data_path is None:
        notes.append(
            "No replay data was provided, so legacy direct_map entries were preserved without reconstructing missing contexts."
        )
    if replay_data_path and after_context_count <= before_context_count:
        notes.append(
            "Replay data did not add new context_index coverage. Provide token sequences that overlap the legacy artifact contexts."
        )
    if unresolved_patterns > 0:
        notes.append(
            "Some direct_map patterns still have no indexed context, so fuzzy retrieval support remains partial."
        )

    return {
        "input_model_path": resolved_model_path,
        "output_model_path": resolved_output_path,
        "context_encoding": getattr(engine, "context_encoding", "stable_v1"),
        "pattern_count": len(engine.direct_map),
        "context_count_before": before_context_count,
        "context_count_after": after_context_count,
        "reindex_summary": reindex_summary,
        "unresolved_pattern_count": unresolved_patterns,
        "quantization_enabled": bool(enable_turboquant),
        "replay_data_path": resolved_replay_path,
        "notes": notes,
    }


def default_upgraded_model_path() -> str:
    return model_path("upgraded", "distilled_sara_llm_upgraded.msgpack")


def default_upgrade_report_path() -> str:
    return workspace_path("reports", "memory_upgrade_report.json")
