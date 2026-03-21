import importlib.util
import os


def _load_chat_agent_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "chat_agent.py")
    )
    spec = importlib.util.spec_from_file_location("chat_agent_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calculator_handles_basic_arithmetic():
    module = _load_chat_agent_module()

    assert module.get_calculator("1 + 2 * 3 計算") == "7"
    assert module.get_calculator("(8 - 2) / 3 計算") == "2"


def test_calculator_rejects_invalid_or_unsafe_input():
    module = _load_chat_agent_module()

    assert module.get_calculator("1 / 0 計算") == "計算できませんでした"
    assert module.get_calculator("__import__('os') 計算") == "計算できませんでした"
    assert module.get_calculator("abs(1) 計算") == "計算できませんでした"
