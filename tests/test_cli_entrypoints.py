import os
import sys
import types

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

    events = []

    class FakeInference:
        def __init__(self, model_path: str):
            events.append(("init", model_path))

        def reset_state(self):
            events.append(("reset", None))

        def generate(self, prompt: str, **kwargs):
            events.append(("generate", prompt, kwargs))
            return "memory reply"

    fake_inference_module = types.ModuleType("sara_engine.inference")
    fake_inference_module.SaraInference = FakeInference
    monkeypatch.setitem(sys.modules, "sara_engine.inference", fake_inference_module)

    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = cli.run_chat_cli(["--model", fake_model_path, "--max_length", "4"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert ("init", fake_model_path) in events
    assert any(event[0] == "generate" for event in events)
    assert "SARA: memory reply" in captured.out


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
