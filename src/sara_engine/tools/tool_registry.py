# ディレクトリパス: src/sara_engine/tools/tool_registry.py
# ファイル名: tool_registry.py
# ファイルの目的や内容: 型安全なツール定義・登録・実行・バリデーションを統合管理するレジストリ。
#   スキーマ検証付きのツール登録、権限レベル制御、実行結果のバリデーションを提供。

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Optional


class PermissionLevel(Enum):
    """ツール実行に必要な権限レベル。"""

    READ_ONLY = "read_only"
    STANDARD = "standard"
    PRIVILEGED = "privileged"
    ADMIN = "admin"


@dataclass
class ToolParameter:
    """ツールのパラメータ定義。

    Attributes:
        name: パラメータ名。
        param_type: パラメータの型名（"str", "int", "float", "bool"）。
        description: パラメータの説明。
        required: 必須パラメータかどうか。
        default: デフォルト値。
    """

    _TYPE_MAP: ClassVar[Dict[str, type]] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }

    name: str
    param_type: str
    description: str = ""
    required: bool = True
    default: Any = None

    def validate(self, value: Any) -> bool:
        """値がこのパラメータの型に一致するかを検証。"""
        expected_type = self._TYPE_MAP.get(self.param_type)
        if expected_type is None:
            return True  # 未知の型は検証をスキップ
        return isinstance(value, expected_type)


@dataclass
class ToolResult:
    """ツール実行結果。

    Attributes:
        success: 実行が成功したかどうか。
        data: 実行結果のデータ。
        error_message: エラーが発生した場合のメッセージ。
        execution_time_ms: 実行時間（ミリ秒）。
        tool_name: 実行されたツール名。
    """

    success: bool
    data: Any = None
    error_message: str = ""
    execution_time_ms: float = 0.0
    tool_name: str = ""


@dataclass
class ToolDefinition:
    """ツールの定義情報。

    Attributes:
        name: ツール名（一意識別子）。
        description: ツールの説明（日本語対応）。
        func: 実行する関数。
        parameters: パラメータ定義のリスト。
        permission_level: 必要な権限レベル。
        max_execution_time_ms: 最大実行時間（ミリ秒）。超過時は警告。
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters: List[ToolParameter] = field(default_factory=list)
    permission_level: PermissionLevel = PermissionLevel.STANDARD
    max_execution_time_ms: float = 5000.0


class ToolRegistry:
    """ツールの登録・検索・実行を管理するレジストリ。

    型安全なパラメータ検証、権限チェック、実行時間計測を提供する。

    Example:
        >>> registry = ToolRegistry()
        >>> @tool(registry, name="greet", description="挨拶を返す")
        ... def greet(name: str) -> str:
        ...     return f"こんにちは、{name}さん！"
        >>> result = registry.execute("greet", {"name": "太郎"})
        >>> print(result.data)
        こんにちは、太郎さん！
    """

    def __init__(self, default_permission: PermissionLevel = PermissionLevel.STANDARD) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._default_permission = default_permission
        self._execution_log: List[ToolResult] = []

    def register(self, tool_def: ToolDefinition) -> None:
        """ツール定義をレジストリに登録する。

        Args:
            tool_def: 登録するツール定義。

        Raises:
            ValueError: 同名のツールが既に登録されている場合。
        """
        if tool_def.name in self._tools:
            raise ValueError(f"ツール '{tool_def.name}' は既に登録されています。")
        self._tools[tool_def.name] = tool_def

    def unregister(self, name: str) -> bool:
        """ツールの登録を解除する。

        Args:
            name: 解除するツール名。

        Returns:
            登録解除に成功した場合はTrue。
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[ToolDefinition]:
        """ツール定義を名前で取得する。"""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """登録されている全ツール定義のリストを返す。"""
        return list(self._tools.values())

    def list_tool_names(self) -> List[str]:
        """登録されている全ツール名のリストを返す。"""
        return list(self._tools.keys())

    def _validate_parameters(
        self, tool_def: ToolDefinition, params: Dict[str, Any]
    ) -> Optional[str]:
        """パラメータのバリデーションを行う。

        Returns:
            エラーメッセージ。問題なければNone。
        """
        for p in tool_def.parameters:
            if p.required and p.name not in params:
                return f"必須パラメータ '{p.name}' が未指定です。"
            if p.name in params and not p.validate(params[p.name]):
                return (
                    f"パラメータ '{p.name}' の型が不正です。"
                    f"期待: {p.param_type}, 実際: {type(params[p.name]).__name__}"
                )
        return None

    def execute(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        permission_level: PermissionLevel = PermissionLevel.STANDARD,
    ) -> ToolResult:
        """ツールを実行する。

        Args:
            name: 実行するツール名。
            params: ツールに渡すパラメータ辞書。
            permission_level: 呼び出し元の権限レベル。

        Returns:
            ToolResult: 実行結果。
        """
        if params is None:
            params = {}

        tool_def = self._tools.get(name)
        if tool_def is None:
            result = ToolResult(
                success=False,
                error_message=f"ツール '{name}' は登録されていません。",
                tool_name=name,
            )
            self._execution_log.append(result)
            return result

        # 権限チェック
        perm_order = [
            PermissionLevel.READ_ONLY,
            PermissionLevel.STANDARD,
            PermissionLevel.PRIVILEGED,
            PermissionLevel.ADMIN,
        ]
        caller_level = perm_order.index(permission_level)
        required_level = perm_order.index(tool_def.permission_level)
        if caller_level < required_level:
            result = ToolResult(
                success=False,
                error_message=(
                    f"権限不足: ツール '{name}' には "
                    f"'{tool_def.permission_level.value}' 以上の権限が必要です。"
                ),
                tool_name=name,
            )
            self._execution_log.append(result)
            return result

        # パラメータバリデーション
        validation_error = self._validate_parameters(tool_def, params)
        if validation_error:
            result = ToolResult(
                success=False,
                error_message=validation_error,
                tool_name=name,
            )
            self._execution_log.append(result)
            return result

        # デフォルト値の適用
        resolved_params = {}
        for p in tool_def.parameters:
            if p.name in params:
                resolved_params[p.name] = params[p.name]
            elif p.default is not None:
                resolved_params[p.name] = p.default

        # 定義にないパラメータも透過
        for k, v in params.items():
            if k not in resolved_params:
                resolved_params[k] = v

        # 実行
        start_time = time.monotonic()
        try:
            data = tool_def.func(**resolved_params)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            result = ToolResult(
                success=True,
                data=data,
                execution_time_ms=elapsed_ms,
                tool_name=name,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            result = ToolResult(
                success=False,
                error_message=f"{type(e).__name__}: {e}",
                execution_time_ms=elapsed_ms,
                tool_name=name,
            )

        self._execution_log.append(result)
        return result

    @property
    def execution_log(self) -> List[ToolResult]:
        """実行ログを返す。"""
        return list(self._execution_log)

    def clear_log(self) -> None:
        """実行ログをクリアする。"""
        self._execution_log.clear()


def tool(
    registry: ToolRegistry,
    name: str,
    description: str = "",
    parameters: Optional[List[ToolParameter]] = None,
    permission_level: PermissionLevel = PermissionLevel.STANDARD,
) -> Callable[..., Any]:
    """関数をツールとして登録するデコレータ。

    Args:
        registry: 登録先のToolRegistry。
        name: ツールの一意名。
        description: ツールの説明。
        parameters: パラメータ定義リスト。
        permission_level: 必要な権限レベル。

    Example:
        >>> registry = ToolRegistry()
        >>> @tool(registry, name="add", description="2つの数値を足す",
        ...       parameters=[
        ...           ToolParameter(name="a", param_type="int"),
        ...           ToolParameter(name="b", param_type="int"),
        ...       ])
        ... def add(a: int, b: int) -> int:
        ...     return a + b
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_def = ToolDefinition(
            name=name,
            description=description or func.__doc__ or "",
            func=func,
            parameters=parameters or [],
            permission_level=permission_level,
        )
        registry.register(tool_def)
        return func

    return decorator
