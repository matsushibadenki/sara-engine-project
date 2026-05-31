import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import model_path, workspace_path


def _load_upgrade_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "utils", "upgrade_memory.py")
    )
    spec = importlib.util.spec_from_file_location("upgrade_memory_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upgrade_inference_memory_preserves_legacy_artifact_without_replay_data():
    module = _load_upgrade_module()
    input_path = model_path("tests", "legacy_upgrade_input.msgpack")
    output_path = model_path("tests", "legacy_upgrade_output.msgpack")
    os.makedirs(os.path.dirname(input_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = input_path
    writer.direct_map = {
        (12345,): {7: 1.0, 8: 2.0},
    }
    writer.context_index = {}
    writer.retrieval_diagnostics = []
    writer.refractory_buffer = []
    writer.lif_network = None
    writer.context_encoding = "legacy_python_hash"
    writer.save_pretrained(input_path)

    report = module.upgrade_inference_memory(input_path, output_path)

    assert report["context_encoding"] == "legacy_python_hash"
    assert report["pattern_count"] == 1
    assert report["context_count_before"] == 0
    assert report["context_count_after"] == 0
    assert report["unresolved_pattern_count"] == 1
    assert any("No replay data" in note for note in report["notes"])
    assert os.path.exists(output_path)


def test_upgrade_inference_memory_reindexes_contexts_from_token_jsonl():
    module = _load_upgrade_module()
    input_path = model_path("tests", "indexed_upgrade_input.msgpack")
    output_path = model_path("tests", "indexed_upgrade_output.msgpack")
    replay_path = workspace_path("tests", "upgrade_replay.jsonl")
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    os.makedirs(os.path.dirname(replay_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = input_path
    writer.direct_map = {}
    writer.context_index = {}
    writer.retrieval_diagnostics = []
    writer.refractory_buffer = []
    writer.lif_network = None
    writer.learn_sequence([10, 20, 30])
    writer.context_index = {}
    writer.save_pretrained(input_path)

    with open(replay_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"tokens": [10, 20, 30]}) + "\n")

    report = module.upgrade_inference_memory(
        input_path,
        output_path,
        replay_data_path=replay_path,
        enable_turboquant=True,
    )

    assert report["context_count_before"] == 0
    assert report["context_count_after"] >= 1
    assert report["reindex_summary"]["matched_contexts"] >= 1
    assert report["context_encoding"] == "stable_v1"
    assert report["quantization_enabled"] is True

    upgraded = SaraInference.__new__(SaraInference)
    upgraded.model_path = output_path
    upgraded.direct_map = {}
    upgraded.context_index = {}
    upgraded.retrieval_diagnostics = []
    upgraded.refractory_buffer = []
    upgraded.lif_network = None
    upgraded._load_memory()

    assert upgraded.context_index
    assert upgraded.context_encoding == "stable_v1"
