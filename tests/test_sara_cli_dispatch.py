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
