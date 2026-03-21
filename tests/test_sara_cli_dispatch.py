import importlib.util
import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def _load_sara_cli_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "sara_cli.py")
    )
    spec = importlib.util.spec_from_file_location("sara_cli_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chat_distill_dispatches_to_agent_chat(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "chat-distill", "--model", "models/test_agent"])

    sara_cli.main()

    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[0] == sys.executable
    assert args[1] == "scripts/eval/chat_agent.py"
    assert "--model-dir" in args
    assert "models/test_agent" in args


def test_train_self_org_dispatches_to_training_script(monkeypatch):
    sara_cli = _load_sara_cli_module()
    mock_run = Mock()
    monkeypatch.setattr(sara_cli.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "train-self-org"])

    sara_cli.main()

    mock_run.assert_called_once_with([sys.executable, "scripts/train/train_self_organized.py"])


def test_prune_dispatches_to_memory_pruner(monkeypatch):
    sara_cli = _load_sara_cli_module()
    prune_mock = Mock()
    monkeypatch.setattr(sara_cli, "prune_model_memory", prune_mock)
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "prune", "--model", "models/test.msgpack", "--threshold", "12"])

    sara_cli.main()

    prune_mock.assert_called_once_with("models/test.msgpack", 12.0)


def test_db_status_without_database_prints_empty_notice(monkeypatch, capsys):
    sara_cli = _load_sara_cli_module()
    original_exists = sara_cli.os.path.exists
    monkeypatch.setattr(
        sara_cli.os.path,
        "exists",
        lambda path: False if path == "data/sara_corpus.db" else original_exists(path),
    )
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "db-status"])

    sara_cli.main()

    captured = capsys.readouterr()
    assert "DBが存在しません" in captured.out


def test_clean_removes_non_gitkeep_items(monkeypatch):
    sara_cli = _load_sara_cli_module()
    removed_files = []
    removed_dirs = []

    def fake_exists(path: str) -> bool:
        return path in {"data/interim", "data/processed", "data/interim/tmp.txt", "data/processed/subdir"}

    def fake_listdir(path: str):
        if path == "data/interim":
            return [".gitkeep", "tmp.txt"]
        if path == "data/processed":
            return ["subdir"]
        return []

    monkeypatch.setattr(sara_cli.os.path, "exists", fake_exists)
    monkeypatch.setattr(sara_cli.os, "listdir", fake_listdir)
    monkeypatch.setattr(sara_cli.os.path, "isdir", lambda path: path == "data/processed/subdir")
    monkeypatch.setattr(sara_cli.os, "remove", lambda path: removed_files.append(path))
    monkeypatch.setattr(sara_cli.shutil, "rmtree", lambda path: removed_dirs.append(path))
    monkeypatch.setattr(sys, "argv", ["sara_cli.py", "clean"])

    sara_cli.main()

    assert removed_files == ["data/interim/tmp.txt"]
    assert removed_dirs == ["data/processed/subdir"]
