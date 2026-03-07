# ディレクトリパス: src/sara_engine/safety/safety_guard.py
# ファイル名: safety_guard.py
# ファイルの目的や内容: 入力サニタイズ、コンテンツフィルタリング、ツール権限制御、
#   出力安全性チェックを統合した多層防御の安全制御ゲートウェイ。

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set


class SafetyLevel(IntEnum):
    """安全性レベル（数値が大きいほど深刻）。"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SafetyCheckResult:
    """安全性チェックの結果。

    Attributes:
        is_safe: 安全かどうか。
        level: 検出された問題の深刻度。
        reasons: 不安全と判定された理由のリスト。
        checker_name: チェックを実行したチェッカー名。
        original_text: チェック対象の元テキスト（サニタイズ前）。
        sanitized_text: サニタイズ後のテキスト。
    """

    is_safe: bool
    level: SafetyLevel = SafetyLevel.LOW
    reasons: List[str] = field(default_factory=list)
    checker_name: str = ""
    original_text: str = ""
    sanitized_text: str = ""


class InputSanitizer:
    """入力テキストのサニタイズを行う。

    制御文字の除去、長さ制限、NULLバイト除去等を行い、
    安全な入力テキストに変換する。
    """

    def __init__(self, max_length: int = 10000) -> None:
        self.max_length = max_length

    def sanitize(self, text: str) -> SafetyCheckResult:
        """入力テキストをサニタイズする。

        Args:
            text: サニタイズ対象のテキスト。

        Returns:
            サニタイズ結果。sanitized_text に安全化されたテキストが格納される。
        """
        reasons: List[str] = []
        level = SafetyLevel.LOW
        original = text

        # NULLバイト除去
        if "\x00" in text:
            text = text.replace("\x00", "")
            reasons.append("NULLバイトを除去しました。")
            level = max(level, SafetyLevel.MEDIUM)

        # 制御文字除去（タブ、改行は許可）
        control_pattern = r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]"
        if re.search(control_pattern, text):
            text = re.sub(control_pattern, "", text)
            reasons.append("制御文字を除去しました。")
            level = max(level, SafetyLevel.LOW)

        # 長さ制限
        if len(text) > self.max_length:
            text = text[: self.max_length]
            reasons.append(
                f"テキストが最大長 {self.max_length} 文字を超過したため切り詰めました。"
            )
            level = max(level, SafetyLevel.MEDIUM)

        # 連続空白の正規化
        text = re.sub(r" {10,}", " " * 5, text)

        is_safe = level <= SafetyLevel.MEDIUM
        return SafetyCheckResult(
            is_safe=is_safe,
            level=level,
            reasons=reasons,
            checker_name="InputSanitizer",
            original_text=original,
            sanitized_text=text,
        )


class ContentFilter:
    """コンテンツフィルタリングを行う。

    プロンプトインジェクション、スクリプトインジェクション、
    過剰な特殊文字パターン等を検出する。
    """

    def __init__(self) -> None:
        # プロンプトインジェクションパターン
        self._injection_patterns: List[str] = [
            r"(?i)ignore\s+(all\s+)?previous\s+instructions",
            r"(?i)ignore\s+(all\s+)?above",
            r"(?i)disregard\s+(all\s+)?previous",
            r"(?i)forget\s+(everything|all)",
            r"(?i)you\s+are\s+now\s+a",
            r"(?i)act\s+as\s+if\s+you\s+are",
            r"(?i)system\s*:\s*you\s+are",
            r"(?i)<\s*system\s*>",
            r"(?i)\[\s*system\s*\]",
        ]

        # スクリプトインジェクションパターン
        self._script_patterns: List[str] = [
            r"<\s*script[^>]*>",
            r"javascript\s*:",
            r"on\w+\s*=\s*[\"']",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            r"os\.system\s*\(",
            r"subprocess\.",
        ]

    def check(self, text: str) -> SafetyCheckResult:
        """テキストのコンテンツ安全性をチェックする。

        Args:
            text: チェック対象テキスト。

        Returns:
            チェック結果。
        """
        reasons: List[str] = []
        level = SafetyLevel.LOW

        # プロンプトインジェクション検出
        for pattern in self._injection_patterns:
            if re.search(pattern, text):
                reasons.append(
                    f"プロンプトインジェクションの可能性を検出: パターン '{pattern}'"
                )
                level = max(level, SafetyLevel.HIGH)

        # スクリプトインジェクション検出
        for pattern in self._script_patterns:
            if re.search(pattern, text):
                reasons.append(
                    f"スクリプトインジェクションの可能性を検出: パターン '{pattern}'"
                )
                level = max(level, SafetyLevel.CRITICAL)

        # 過剰な特殊文字の検出
        special_ratio = sum(1 for c in text if not c.isalnum(
        ) and c not in " \t\n。、！？!?.,") / max(len(text), 1)
        if special_ratio > 0.5 and len(text) > 20:
            reasons.append(
                f"特殊文字の比率が異常に高い ({special_ratio:.1%})。"
            )
            level = max(level, SafetyLevel.MEDIUM)

        is_safe = level < SafetyLevel.HIGH
        return SafetyCheckResult(
            is_safe=is_safe,
            level=level,
            reasons=reasons,
            checker_name="ContentFilter",
            original_text=text,
            sanitized_text=text,
        )


class ToolPermissionGuard:
    """ツール実行の権限制御を行う。

    許可リスト方式で、実行を許可するツールを管理する。
    ツールの実行回数制限（レートリミット）も提供。
    """

    def __init__(
        self,
        allowed_tools: Optional[Set[str]] = None,
        max_calls_per_tool: int = 100,
    ) -> None:
        self._allowed_tools: Set[str] = allowed_tools or set()
        self._allow_all = allowed_tools is None
        self._max_calls_per_tool = max_calls_per_tool
        self._call_counts: Dict[str, int] = {}

    def allow_tool(self, tool_name: str) -> None:
        """ツールを許可リストに追加する。"""
        self._allowed_tools.add(tool_name)
        self._allow_all = False

    def deny_tool(self, tool_name: str) -> None:
        """ツールを許可リストから削除する。"""
        self._allowed_tools.discard(tool_name)

    def check(self, tool_name: str) -> SafetyCheckResult:
        """ツール実行の権限をチェックする。

        Args:
            tool_name: チェック対象のツール名。

        Returns:
            チェック結果。
        """
        reasons: List[str] = []
        level = SafetyLevel.LOW

        # 許可リストチェック
        if not self._allow_all and tool_name not in self._allowed_tools:
            reasons.append(
                f"ツール '{tool_name}' は許可リストに含まれていません。"
            )
            level = SafetyLevel.HIGH

        # レートリミットチェック
        current_count = self._call_counts.get(tool_name, 0)
        if current_count >= self._max_calls_per_tool:
            reasons.append(
                f"ツール '{tool_name}' の実行回数が上限 "
                f"({self._max_calls_per_tool}) に達しました。"
            )
            level = max(level, SafetyLevel.HIGH)

        is_safe = level < SafetyLevel.HIGH
        if is_safe:
            self._call_counts[tool_name] = current_count + 1

        return SafetyCheckResult(
            is_safe=is_safe,
            level=level,
            reasons=reasons,
            checker_name="ToolPermissionGuard",
            original_text=tool_name,
            sanitized_text=tool_name,
        )

    def reset_counts(self) -> None:
        """実行カウントをリセットする。"""
        self._call_counts.clear()


class OutputSafetyChecker:
    """出力テキストの安全性をチェックする。

    PII（個人識別情報）の検出や、過度な長さのチェックを行う。
    """

    def __init__(self, max_output_length: int = 50000) -> None:
        self.max_output_length = max_output_length

        # PII検出パターン（日本語対応）
        self._pii_patterns: Dict[str, str] = {
            "メールアドレス": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "電話番号": r"(?:\d{2,4}[-\s]?\d{2,4}[-\s]?\d{3,4})",
            "クレジットカード番号": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "マイナンバー": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        }

    def check(self, text: str) -> SafetyCheckResult:
        """出力テキストの安全性をチェックする。

        Args:
            text: チェック対象の出力テキスト。

        Returns:
            チェック結果。
        """
        reasons: List[str] = []
        level = SafetyLevel.LOW

        # PII検出
        for pii_type, pattern in self._pii_patterns.items():
            if re.search(pattern, text):
                reasons.append(f"PIIの可能性を検出: {pii_type}")
                level = max(level, SafetyLevel.HIGH)

        # 出力長チェック
        if len(text) > self.max_output_length:
            reasons.append(
                f"出力が最大長 ({self.max_output_length} 文字) を超過しています。"
            )
            level = max(level, SafetyLevel.MEDIUM)

        is_safe = level < SafetyLevel.HIGH
        return SafetyCheckResult(
            is_safe=is_safe,
            level=level,
            reasons=reasons,
            checker_name="OutputSafetyChecker",
            original_text=text,
            sanitized_text=text,
        )


class SafetyGuard:
    """全安全チェックを統合する安全制御ゲートウェイ。

    入力サニタイズ → コンテンツフィルタ → (ツール権限) → 出力チェック の
    多層防御を提供する。

    Example:
        >>> guard = SafetyGuard()
        >>> result = guard.check_input("こんにちは、SARAエンジンです。")
        >>> if result.is_safe:
        ...     # 安全な入力として処理
        ...     pass
    """

    def __init__(
        self,
        max_input_length: int = 10000,
        max_output_length: int = 50000,
        allowed_tools: Optional[Set[str]] = None,
        max_calls_per_tool: int = 100,
    ) -> None:
        self.input_sanitizer = InputSanitizer(max_length=max_input_length)
        self.content_filter = ContentFilter()
        self.tool_guard = ToolPermissionGuard(
            allowed_tools=allowed_tools,
            max_calls_per_tool=max_calls_per_tool,
        )
        self.output_checker = OutputSafetyChecker(
            max_output_length=max_output_length
        )
        self._check_log: List[SafetyCheckResult] = []

    def check_input(self, text: str) -> SafetyCheckResult:
        """入力テキストの安全性を総合チェックする。

        サニタイズとコンテンツフィルタリングを順に実行する。

        Args:
            text: チェック対象の入力テキスト。

        Returns:
            統合されたチェック結果。sanitized_textにサニタイズ済みテキストが入る。
        """
        # 1. 入力サニタイズ
        sanitize_result = self.input_sanitizer.sanitize(text)
        self._check_log.append(sanitize_result)

        if not sanitize_result.is_safe:
            return sanitize_result

        # 2. コンテンツフィルタリング
        filter_result = self.content_filter.check(
            sanitize_result.sanitized_text)
        self._check_log.append(filter_result)

        # 結果を統合
        combined_reasons = sanitize_result.reasons + filter_result.reasons
        combined_level = max(sanitize_result.level, filter_result.level)
        combined_safe = sanitize_result.is_safe and filter_result.is_safe

        return SafetyCheckResult(
            is_safe=combined_safe,
            level=combined_level,
            reasons=combined_reasons,
            checker_name="SafetyGuard.check_input",
            original_text=text,
            sanitized_text=sanitize_result.sanitized_text,
        )

    def check_tool_execution(self, tool_name: str) -> SafetyCheckResult:
        """ツール実行の安全性をチェックする。

        Args:
            tool_name: チェック対象のツール名。

        Returns:
            チェック結果。
        """
        result = self.tool_guard.check(tool_name)
        self._check_log.append(result)
        return result

    def check_output(self, text: str) -> SafetyCheckResult:
        """出力テキストの安全性をチェックする。

        Args:
            text: チェック対象の出力テキスト。

        Returns:
            チェック結果。
        """
        result = self.output_checker.check(text)
        self._check_log.append(result)
        return result

    @property
    def check_log(self) -> List[SafetyCheckResult]:
        """チェックログを返す。"""
        return list(self._check_log)

    def clear_log(self) -> None:
        """チェックログをクリアする。"""
        self._check_log.clear()
