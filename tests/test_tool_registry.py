# ディレクトリパス: tests/test_tool_registry.py
# ファイル名: test_tool_registry.py
# ファイルの目的や内容: ツール実行基盤の単体テスト

from sara_engine.tools.builtin_tools import calculator, get_datetime, text_search, register_builtin_tools
from sara_engine.tools.tool_registry import (
    PermissionLevel,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    tool,
)
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))


def test_register_and_execute_tool() -> None:
    """ツールの登録と実行テスト。"""
    registry = ToolRegistry()

    def greet(name: str) -> str:
        return f"こんにちは、{name}さん！"

    registry.register(ToolDefinition(
        name="greet",
        description="挨拶ツール",
        func=greet,
        parameters=[ToolParameter(
            name="name", param_type="str", required=True)],
    ))

    result = registry.execute("greet", {"name": "太郎"})
    assert result.success is True
    assert result.data == "こんにちは、太郎さん！"
    assert result.execution_time_ms >= 0


def test_execute_nonexistent_tool() -> None:
    """存在しないツールの実行テスト。"""
    registry = ToolRegistry()
    result = registry.execute("nonexistent")
    assert result.success is False
    assert "登録されていません" in result.error_message


def test_parameter_validation() -> None:
    """パラメータバリデーション：型不正テスト。"""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="add",
        description="加算",
        func=lambda a, b: a + b,
        parameters=[
            ToolParameter(name="a", param_type="int", required=True),
            ToolParameter(name="b", param_type="int", required=True),
        ],
    ))

    # 型が不正
    result = registry.execute("add", {"a": "not_int", "b": 2})
    assert result.success is False
    assert "型が不正" in result.error_message


def test_missing_required_parameter() -> None:
    """必須パラメータ欠落テスト。"""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="echo",
        description="エコー",
        func=lambda text: text,
        parameters=[ToolParameter(
            name="text", param_type="str", required=True)],
    ))

    result = registry.execute("echo", {})
    assert result.success is False
    assert "未指定" in result.error_message


def test_permission_check() -> None:
    """権限チェックテスト。"""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="admin_tool",
        description="管理ツール",
        func=lambda: "admin",
        permission_level=PermissionLevel.ADMIN,
    ))

    # 低い権限で実行
    result = registry.execute(
        "admin_tool", permission_level=PermissionLevel.READ_ONLY
    )
    assert result.success is False
    assert "権限不足" in result.error_message

    # 十分な権限で実行
    result = registry.execute(
        "admin_tool", permission_level=PermissionLevel.ADMIN
    )
    assert result.success is True


def test_tool_decorator() -> None:
    """@toolデコレータテスト。"""
    registry = ToolRegistry()

    @tool(registry, name="multiply", description="乗算ツール",
          parameters=[
              ToolParameter(name="a", param_type="int"),
              ToolParameter(name="b", param_type="int"),
          ])
    def multiply(a: int, b: int) -> int:
        return a * b

    assert "multiply" in registry.list_tool_names()
    result = registry.execute("multiply", {"a": 3, "b": 4})
    assert result.success is True
    assert result.data == 12


def test_duplicate_registration_raises() -> None:
    """重複登録エラーテスト。"""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="test", description="", func=lambda: None))

    try:
        registry.register(ToolDefinition(
            name="test", description="", func=lambda: None))
        assert False, "ValueError が発生すべき"
    except ValueError:
        pass


def test_unregister_tool() -> None:
    """ツール登録解除テスト。"""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="temp", description="", func=lambda: None))
    assert registry.unregister("temp") is True
    assert registry.unregister("temp") is False


def test_execution_log() -> None:
    """実行ログテスト。"""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="noop", description="", func=lambda: "ok"))
    registry.execute("noop")
    assert len(registry.execution_log) == 1
    registry.clear_log()
    assert len(registry.execution_log) == 0


# --- 組み込みツールのテスト ---

def test_calculator_basic_operations() -> None:
    """計算機ツール：基本演算テスト。"""
    assert calculator("2 + 3") == "5"
    assert calculator("10 - 4") == "6"
    assert calculator("3 * 7") == "21"
    assert calculator("15 / 3") == "5.0"
    assert calculator("2 ** 10") == "1024"


def test_calculator_safe_functions() -> None:
    """計算機ツール：安全な関数テスト。"""
    assert calculator("abs(-5)") == "5"
    assert calculator("max(1, 2, 3)") == "3"
    assert calculator("min(1, 2, 3)") == "1"


def test_calculator_zero_division() -> None:
    """計算機ツール：ゼロ除算テスト。"""
    result = calculator("1 / 0")
    assert "ゼロ除算" in result


def test_calculator_empty_expression() -> None:
    """計算機ツール：空の式テスト。"""
    result = calculator("")
    assert "エラー" in result


def test_calculator_rejects_dangerous_input() -> None:
    """計算機ツール：危険な入力の拒否テスト。"""
    result = calculator("__import__('os').system('ls')")
    assert "エラー" in result


def test_calculator_rejects_large_exponent() -> None:
    """計算機ツール：大きなべき乗の拒否テスト。"""
    result = calculator("2 ** 200")
    assert "エラー" in result


def test_datetime_returns_string() -> None:
    """日時ツール：戻り値テスト。"""
    result = get_datetime()
    assert isinstance(result, str)
    assert len(result) > 0


def test_text_search_found() -> None:
    """テキスト検索ツール：パターン発見テスト。"""
    result = text_search("Hello World Hello", "Hello")
    assert "2 箇所" in result


def test_text_search_not_found() -> None:
    """テキスト検索ツール：パターン未発見テスト。"""
    result = text_search("Hello World", "xyz")
    assert "見つかりませんでした" in result


def test_text_search_empty() -> None:
    """テキスト検索ツール：空入力テスト。"""
    result = text_search("", "test")
    assert "エラー" in result


def test_register_builtin_tools() -> None:
    """組み込みツール一括登録テスト。"""
    registry = ToolRegistry()
    register_builtin_tools(registry)
    tool_names = registry.list_tool_names()
    assert "calculator" in tool_names
    assert "datetime" in tool_names
    assert "text_search" in tool_names
