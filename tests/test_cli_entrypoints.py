import os
import sys
import types
from contextlib import nullcontext
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine import cli
from sara_engine.utils.project_paths import model_path, workspace_path


def test_chat_entrypoint_returns_error_for_missing_model(capsys):
    exit_code = cli.run_chat_cli(["--model", model_path("tests", "missing_model.msgpack")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Model file not found" in captured.out


def test_chat_entrypoint_runs_interactive_session(monkeypatch, capsys):
    fake_model_path = model_path("tests", "chat_entrypoint_model.msgpack")
    os.makedirs(os.path.dirname(fake_model_path), exist_ok=True)
    with open(fake_model_path, "wb") as handle:
        handle.write(b"stub")

    events: list[tuple[Any, ...]] = []

    class FakeInference:
        def __init__(self, model_path: str):
            events.append(("init", model_path))

        def reset_state(self):
            events.append(("reset", None))

        def generate(self, prompt: str, **kwargs):
            events.append(("generate", prompt, kwargs))
            return "memory reply"

        def format_recent_retrieval_diagnostics(self, limit: int = 3):
            events.append(("diagnostics", limit))
            return "Recent retrieval diagnostics:\n- source=inference_direct_map base=1.00 stability=1.00"

        def save_pretrained(self, model_path: str):
            events.append(("save", model_path))

    fake_inference_module = types.ModuleType("sara_engine.inference")
    setattr(fake_inference_module, "SaraInference", FakeInference)
    monkeypatch.setitem(sys.modules, "sara_engine.inference", fake_inference_module)

    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = cli.run_chat_cli(["--model", fake_model_path, "--max_length", "4"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert ("init", fake_model_path) in events
    assert any(event[0] == "generate" for event in events)
    assert "SARA: memory reply" in captured.out


def test_chat_entrypoint_can_show_diagnostics_and_save_on_exit(monkeypatch, capsys):
    fake_model_path = model_path("tests", "chat_entrypoint_save_model.msgpack")
    os.makedirs(os.path.dirname(fake_model_path), exist_ok=True)
    with open(fake_model_path, "wb") as handle:
        handle.write(b"stub")

    events: list[tuple[Any, ...]] = []

    class FakeInference:
        def __init__(self, model_path: str):
            events.append(("init", model_path))

        def reset_state(self):
            events.append(("reset", None))

        def generate(self, prompt: str, **kwargs):
            events.append(("generate", prompt, kwargs))
            return "memory reply"

        def format_recent_retrieval_diagnostics(self, limit: int = 3):
            events.append(("diagnostics", limit))
            return "Recent retrieval diagnostics:\n- source=inference_direct_map base=1.00 stability=1.00"

        def save_pretrained(self, model_path: str):
            events.append(("save", model_path))

    fake_inference_module = types.ModuleType("sara_engine.inference")
    setattr(fake_inference_module, "SaraInference", FakeInference)
    monkeypatch.setitem(sys.modules, "sara_engine.inference", fake_inference_module)

    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = cli.run_chat_cli(
        ["--model", fake_model_path, "--show-diagnostics", "--save-on-exit", "--max_length", "4"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert ("diagnostics", 3) in events
    assert ("save", fake_model_path) in events
    assert "Recent retrieval diagnostics:" in captured.out
    assert f"[INFO] Saved updated SARA state to {fake_model_path}" in captured.out


def test_train_entrypoint_rejects_output_outside_managed_directories(capsys):
    training_data_path = workspace_path("tests", "chat_train_fixture.jsonl")
    os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
    with open(training_data_path, "w", encoding="utf-8") as handle:
        handle.write('{"user":"hi","sara":"hello"}\n')

    exit_code = cli.run_train_cli(
        [
            training_data_path,
            "--model",
            "/tmp/outside_sara_engine_tests/model.msgpack",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Output path must be under one of" in captured.out


def test_train_entrypoint_runs_end_to_end_with_stubbed_runtime(monkeypatch, capsys):
    training_data_path = workspace_path("tests", "chat_train_success.jsonl")
    model_output_path = model_path("tests", "chat_train_success.msgpack")
    os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
    with open(training_data_path, "w", encoding="utf-8") as handle:
        handle.write('{"user":"hi","sara":"hello"}\n')

    events: list[tuple[Any, ...]] = []

    class FakeScalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class FakeRow:
        def __init__(self, probs):
            self.probs = probs

    class FakeBatch(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def __call__(self, _text, **_kwargs):
            return FakeBatch({"input_ids": [[10, 20, 30]]})

    class FakeTeacher:
        @classmethod
        def from_pretrained(cls, _name, **_kwargs):
            return cls()

        def eval(self):
            events.append(("teacher_eval", None))

        def __call__(self, **_inputs):
            return types.SimpleNamespace(logits=[[FakeRow([0.9, 0.6, 0.3, 0.2, 0.1])]])

    class FakeTorch:
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        class cuda:
            @staticmethod
            def is_available():
                return False

        float32 = "float32"

        @staticmethod
        def no_grad():
            return nullcontext()

        @staticmethod
        def softmax(logits, dim=-1):
            return logits

        @staticmethod
        def topk(row, k):
            values = [FakeScalar(v) for v in row.probs[:k]]
            indices = [FakeScalar(i + 100) for i in range(k)]
            return values, indices

    class FakeTqdm:
        @staticmethod
        def tqdm(items, desc=None):
            events.append(("tqdm", desc))
            return items

    class FakeStudent:
        def __init__(self):
            self._direct_map = {}
            self.saved_path = None

        def load_memory(self, path):
            events.append(("load_memory", path))
            return 0

        def save_memory(self, path):
            self.saved_path = path
            events.append(("save_memory", path, len(self._direct_map)))

        def _encode_to_sdr(self, context_tokens):
            return list(context_tokens)

        def _sdr_key(self, sdr):
            return tuple(sdr)

    monkeypatch.setattr(
        cli,
        "_load_training_runtime",
        lambda: (FakeTeacher, FakeTokenizer, FakeTorch, FakeTqdm),
    )
    fake_student = FakeStudent()
    monkeypatch.setattr(cli, "_build_student_model", lambda _args: fake_student)

    exit_code = cli.run_train_cli([training_data_path, "--model", model_output_path])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert ("teacher_eval", None) in events
    assert any(event[0] == "tqdm" for event in events)
    assert any(event[0] == "save_memory" for event in events)
    assert fake_student.saved_path == model_output_path
    assert fake_student._direct_map
    assert "Dialogue training completed successfully!" in captured.out


def test_train_entrypoint_reports_missing_training_dependency(monkeypatch, capsys):
    training_data_path = workspace_path("tests", "chat_train_missing_dep.jsonl")
    model_output_path = model_path("tests", "chat_train_missing_dep.msgpack")
    os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
    with open(training_data_path, "w", encoding="utf-8") as handle:
        handle.write('{"user":"hi","sara":"hello"}\n')

    def _raise_import_error():
        raise ImportError("transformers")

    monkeypatch.setattr(cli, "_load_training_runtime", _raise_import_error)

    exit_code = cli.run_train_cli([training_data_path, "--model", model_output_path])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Missing training dependency" in captured.out


def test_module_main_defaults_to_chat_entrypoint(monkeypatch):
    called: list[list[str]] = []

    def _fake_run_chat(argv: Any = None) -> int:
        called.append(list(argv or []))
        return 0

    monkeypatch.setattr(cli, "run_chat_cli", _fake_run_chat)

    exit_code = cli.main(["--model", "models/test.msgpack"])

    assert exit_code == 0
    assert called == [["--model", "models/test.msgpack"]]


def test_module_main_dispatches_train_subcommand(monkeypatch):
    called: list[list[str]] = []

    def _fake_run_train(argv: Any = None) -> int:
        called.append(list(argv or []))
        return 0

    monkeypatch.setattr(cli, "run_train_cli", _fake_run_train)

    exit_code = cli.main(["train", "data/raw/chat_data.jsonl", "--model", "models/test.msgpack"])

    assert exit_code == 0
    assert called == [["data/raw/chat_data.jsonl", "--model", "models/test.msgpack"]]
