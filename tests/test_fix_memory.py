import importlib.util
import os

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import model_path, workspace_path


def _load_fix_memory_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "utils", "fix_memory.py")
    )
    spec = importlib.util.spec_from_file_location("fix_memory_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_memory(path: str) -> None:
    writer = SaraInference.__new__(SaraInference)
    writer.model_path = path
    writer.direct_map = {
        (111,): {7: 1.0, 8: 2.0},
    }
    writer.context_index = {
        (111,): (1, 2, 3),
    }
    writer.retrieval_diagnostics = []
    writer.refractory_buffer = []
    writer.session_memory = {}
    writer.predictor_state = {}
    writer.adaptation_state = {}
    writer.future_state_runtime_state = {}
    writer.lif_network = None
    writer.quantization_enabled = False
    writer.save_pretrained(path)


def _read_memory(path: str) -> SaraInference:
    reader = SaraInference.__new__(SaraInference)
    reader.model_path = path
    reader.direct_map = {}
    reader.context_index = {}
    reader.retrieval_diagnostics = []
    reader.refractory_buffer = []
    reader.session_memory = {}
    reader.predictor_state = {}
    reader.adaptation_state = {}
    reader.future_state_runtime_state = {}
    reader.lif_network = None
    reader._load_memory()
    return reader


def test_fix_memory_removes_target_association_and_writes_report():
    module = _load_fix_memory_module()
    input_path = model_path("tests", "fix_memory_input.msgpack")
    output_path = model_path("tests", "fix_memory_output.msgpack")
    report_path = workspace_path("tests", "fix_memory_report.json")
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    _write_memory(input_path)

    report = module.fix_inference_memory(
        input_path,
        output_path,
        context_tokens=[1, 2, 3],
        wrong_token_id=7,
        report_path=report_path,
    )
    repaired = _read_memory(output_path)

    assert report["matched_token"] is True
    assert report["removed"] is True
    assert 7 not in repaired.direct_map[(111,)]
    assert repaired.direct_map[(111,)][8] == 2.0
    assert os.path.exists(report_path)


def test_fix_memory_dry_run_does_not_write_output():
    module = _load_fix_memory_module()
    input_path = model_path("tests", "fix_memory_dry_run_input.msgpack")
    output_path = model_path("tests", "fix_memory_dry_run_output.msgpack")
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    _write_memory(input_path)

    report = module.fix_inference_memory(
        input_path,
        output_path,
        context_tokens=[1, 2, 3],
        wrong_token_id=7,
        dry_run=True,
    )

    assert report["matched_token"] is True
    assert report["dry_run"] is True
    assert not os.path.exists(output_path)
