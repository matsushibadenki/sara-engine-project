# ディレクトリパス: src/sara_engine/tools/builtin_tools.py
# ファイル名: builtin_tools.py
# ファイルの目的や内容: SARA Engine標準搭載の組み込みツール群。
#   安全な数式計算、日時取得、テキスト検索の3つを提供する。

from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from typing import Any, Callable, Dict

from .tool_registry import (
    PermissionLevel,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
)


# --- 安全な数式計算 ---

# 許可する演算子のマッピング (eval は使わず AST で安全に評価)
_SAFE_OPERATORS: Dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# 許可する関数一覧
_SAFE_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
}


def _safe_eval_node(node: ast.AST) -> Any:
    """ASTノードを安全に評価する。"""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"許可されていない定数型: {type(node.value).__name__}")
    elif isinstance(node, ast.BinOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"許可されていない演算子: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # べき乗の上限チェック
        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > 100:
                raise ValueError("べき乗の指数が大きすぎます（上限: 100）。")
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"許可されていない単項演算子: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            args = [_safe_eval_node(arg) for arg in node.args]
            return _SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError("許可されていない関数呼び出しです。")
    else:
        raise ValueError(f"許可されていないASTノード: {type(node).__name__}")


def calculator(expression: str) -> str:
    """安全な数式計算ツール。evalを使わずASTベースで評価する。

    Args:
        expression: 計算式（例: "2 + 3 * 4"）。

    Returns:
        計算結果の文字列。
    """
    expression = expression.strip()
    if not expression:
        return "エラー: 計算式が空です。"

    # 文字列長の制限
    if len(expression) > 200:
        return "エラー: 計算式が長すぎます（上限: 200文字）。"

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree)
        return str(result)
    except ZeroDivisionError:
        return "エラー: ゼロ除算です。"
    except (ValueError, TypeError, SyntaxError) as e:
        return f"エラー: {e}"


def get_datetime(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """現在の日時を取得するツール。

    Args:
        format_str: 出力フォーマット（strftime形式）。

    Returns:
        フォーマットされた日時文字列。
    """
    try:
        return datetime.now().strftime(format_str)
    except (ValueError, TypeError):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def text_search(text: str, pattern: str) -> str:
    """テキスト内のパターン検索ツール。

    Args:
        text: 検索対象テキスト。
        pattern: 検索パターン（プレーンテキスト）。

    Returns:
        検索結果の文字列。
    """
    if not text or not pattern:
        return "エラー: テキストまたはパターンが空です。"

    # 安全のため正規表現の特殊文字をエスケープ
    escaped_pattern = re.escape(pattern)
    matches = list(re.finditer(escaped_pattern, text))

    if not matches:
        return f"パターン '{pattern}' は見つかりませんでした。"

    result_parts = [f"パターン '{pattern}' が {len(matches)} 箇所で見つかりました:"]
    for i, match in enumerate(matches[:10]):  # 最大10件表示
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 20)
        context = text[start:end]
        result_parts.append(f"  [{i + 1}] 位置 {match.start()}: ...{context}...")

    return "\n".join(result_parts)


def register_builtin_tools(registry: ToolRegistry) -> None:
    """組み込みツールをレジストリに一括登録する。

    Args:
        registry: 登録先のToolRegistry。
    """
    # 計算機ツール
    registry.register(
        ToolDefinition(
            name="calculator",
            description="安全な数式計算。四則演算、べき乗、abs/round/min/max 等をサポート。",
            func=calculator,
            parameters=[
                ToolParameter(
                    name="expression",
                    param_type="str",
                    description="計算式（例: '2 + 3 * 4'）",
                    required=True,
                ),
            ],
            permission_level=PermissionLevel.READ_ONLY,
        )
    )

    # 日時取得ツール
    registry.register(
        ToolDefinition(
            name="datetime",
            description="現在の日時情報を取得する。",
            func=get_datetime,
            parameters=[
                ToolParameter(
                    name="format_str",
                    param_type="str",
                    description="日時フォーマット（strftime形式）",
                    required=False,
                    default="%Y-%m-%d %H:%M:%S",
                ),
            ],
            permission_level=PermissionLevel.READ_ONLY,
        )
    )

    # テキスト検索ツール
    registry.register(
        ToolDefinition(
            name="text_search",
            description="テキスト内でパターンを検索する。",
            func=text_search,
            parameters=[
                ToolParameter(
                    name="text",
                    param_type="str",
                    description="検索対象のテキスト",
                    required=True,
                ),
                ToolParameter(
                    name="pattern",
                    param_type="str",
                    description="検索パターン",
                    required=True,
                ),
            ],
            permission_level=PermissionLevel.READ_ONLY,
        )
    )
