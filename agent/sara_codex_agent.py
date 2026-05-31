"""Codex-inspired local agent for SARA Engine training data.

The agent is intentionally CPU-only and retrieval-first. It does not call a
remote model; instead it plans a small action loop, searches project learning
data, runs safe read-only tools, and returns a grounded response.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import operator
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_PROJECT_PATHS_FILE = _SRC_DIR / "sara_engine" / "utils" / "project_paths.py"
_PROJECT_PATHS_SPEC = importlib.util.spec_from_file_location(
    "sara_engine_project_paths", _PROJECT_PATHS_FILE
)
if _PROJECT_PATHS_SPEC is None or _PROJECT_PATHS_SPEC.loader is None:
    raise ImportError(f"Cannot load project paths from {_PROJECT_PATHS_FILE}")
_PROJECT_PATHS = importlib.util.module_from_spec(_PROJECT_PATHS_SPEC)
_PROJECT_PATHS_SPEC.loader.exec_module(_PROJECT_PATHS)

ensure_parent_directory = _PROJECT_PATHS.ensure_parent_directory
project_path = _PROJECT_PATHS.project_path
resolve_project_relative = _PROJECT_PATHS.resolve_project_relative
workspace_path = _PROJECT_PATHS.workspace_path


class ApprovalMode(str, Enum):
    """Execution mode for tool calls."""

    SUGGEST = "suggest"
    AUTO = "auto"


@dataclass(frozen=True)
class TrainingRecord:
    """A normalized training-data item."""

    prompt: str
    response: str
    source: str
    kind: str

    @property
    def searchable_text(self) -> str:
        return f"{self.prompt}\n{self.response}".strip()


@dataclass(frozen=True)
class SearchHit:
    """A scored training-data retrieval result."""

    score: float
    prompt: str
    response: str
    source: str
    kind: str


@dataclass
class AgentStep:
    """A single step in the local agent loop."""

    name: str
    status: str
    detail: str


@dataclass
class AgentResult:
    """Structured result returned by the agent runtime."""

    task: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    tool_outputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "answer": self.answer,
            "hits": [asdict(hit) for hit in self.hits],
            "steps": [asdict(step) for step in self.steps],
            "tool_outputs": dict(self.tool_outputs),
        }


@dataclass
class SaraCodexAgentConfig:
    """Configuration for the root-level local agent."""

    data_paths: tuple[str, ...] = (
        project_path("data", "raw", "chat_data.jsonl"),
        project_path("data", "interim", "chat_data.jsonl"),
        project_path("data", "interim", "test_corpus.txt"),
        project_path("data", "corpus.txt"),
    )
    top_k: int = 5
    approval_mode: ApprovalMode = ApprovalMode.AUTO
    max_answer_chars: int = 1200


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}|[一-龥ぁ-んァ-ヴー]{2,}")
_SAFE_OPERATORS: dict[type[ast.AST], Callable[..., Any]] = {
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


def _tokenize(text: str) -> set[str]:
    tokens = {token.lower() for token in _TOKEN_RE.findall(text)}
    expanded = set(tokens)
    important_terms = (
        "特徴",
        "標準ライブラリ",
        "読みやす",
        "設計哲学",
        "インデント",
        "プログラミング",
        "ニューラル",
        "スパイキング",
        "エネルギー",
    )
    for token in tokens:
        stripped = token.lstrip("のはがをにでとやも")
        stripped = re.sub(r"(について|教えて|とは|ですか|ますか)$", "", stripped)
        if len(stripped) >= 2:
            expanded.add(stripped)
        if re.search(r"[一-龥ぁ-んァ-ヴー]", token):
            expanded.update(part for part in re.split(r"[のはがをにでとやも、。]+", token) if len(part) >= 2)
        for term in important_terms:
            if term in token:
                expanded.add(term)
    return expanded


def _compact(text: str, limit: int = 280) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _safe_eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported unary operator")
        return op(_safe_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported operator")
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 100:
            raise ValueError("exponent is too large")
        return op(_safe_eval_node(node.left), right)
    raise ValueError("unsupported expression")


def safe_calculate(expression: str) -> str:
    """Evaluate a small arithmetic expression without using eval."""

    cleaned = expression.strip()
    if not cleaned or len(cleaned) > 160:
        return "Calculation error: expression is empty or too long."
    try:
        tree = ast.parse(cleaned, mode="eval")
        return str(_safe_eval_node(tree))
    except Exception as exc:
        return f"Calculation error: {exc}"


class TrainingDataIndex:
    """Keyword index over SARA project learning data."""

    def __init__(self, records: Iterable[TrainingRecord]) -> None:
        self.records = list(records)
        self._record_tokens = [_tokenize(record.searchable_text) for record in self.records]

    @classmethod
    def from_paths(cls, paths: Iterable[str]) -> "TrainingDataIndex":
        records: list[TrainingRecord] = []
        for raw_path in paths:
            path = Path(resolve_project_relative(raw_path))
            if not path.exists() or not path.is_file():
                continue
            if path.suffix == ".jsonl":
                records.extend(_load_jsonl_records(path))
            else:
                records.extend(_load_text_records(path))
        return cls(records)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        hits: list[SearchHit] = []
        seen_responses: set[str] = set()
        lowered_query = query.lower()
        for record, tokens in zip(self.records, self._record_tokens):
            normalized_response = re.sub(r"\s+", " ", record.response).strip()
            if normalized_response in seen_responses:
                continue
            overlap = query_tokens & tokens
            if not overlap:
                continue
            seen_responses.add(normalized_response)
            record_text_lower = record.searchable_text.lower()
            exact_bonus = 2.0 if lowered_query in record.searchable_text.lower() else 0.0
            response_bonus = 0.6 if any(token in record.response.lower() for token in query_tokens) else 0.0
            intent_bonus = 0.0
            if "特徴" in query_tokens:
                for term in ("読みやす", "標準ライブラリ", "設計哲学", "インデント", "パラダイム"):
                    if term in record_text_lower:
                        intent_bonus += 1.4
            score = (len(overlap) * 3.0) + exact_bonus + response_bonus
            score += intent_bonus
            if record.kind == "corpus":
                score += 0.8
            score += min(1.0, len(record.response) / 240.0)
            hits.append(
                SearchHit(
                    score=score,
                    prompt=record.prompt,
                    response=record.response,
                    source=record.source,
                    kind=record.kind,
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, top_k)]


def _load_jsonl_records(path: Path) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(payload.get("prompt") or payload.get("user") or "").strip()
            response = str(
                payload.get("response") or payload.get("completion") or payload.get("sara") or ""
            ).strip()
            if _is_usable_pair(prompt, response):
                records.append(
                    TrainingRecord(
                        prompt=prompt,
                        response=response,
                        source=os.path.relpath(path, project_path()),
                        kind="chat_jsonl",
                    )
                )
    return records


def _load_text_records(path: Path) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = re.sub(r"\s+", " ", line).strip()
            if not _is_usable_text_record(text):
                continue
            records.append(
                TrainingRecord(
                    prompt=f"corpus line {line_number}",
                    response=text,
                    source=os.path.relpath(path, project_path()),
                    kind="corpus",
                )
            )
    return records


def _is_usable_pair(prompt: str, response: str) -> bool:
    if len(prompt) < 2 or len(response) < 2:
        return False
    noisy_markers = (
        "Wikipedia",
        "Category:",
        "http://",
        "https://",
        "オライリー",
        "閲覧",
        "参考文献",
        "SBクリエイティブ",
        "コロナ社",
    )
    if any(marker in response for marker in noisy_markers):
        return False
    return True


def _is_usable_text_record(text: str) -> bool:
    if len(text) < 12:
        return False
    noisy_markers = ("http://", "https://", "閲覧", "参考文献", "オライリー", "SBクリエイティブ")
    if any(marker in text for marker in noisy_markers):
        return False
    if text.endswith(("として", "ため、", "のため")):
        return False
    return True


class SaraCodexAgent:
    """Small local agent inspired by Codex's plan-act-observe loop."""

    def __init__(
        self,
        config: Optional[SaraCodexAgentConfig] = None,
        index: Optional[TrainingDataIndex] = None,
    ) -> None:
        self.config = config or SaraCodexAgentConfig()
        self.index = index or TrainingDataIndex.from_paths(self.config.data_paths)

    def run(self, task: str) -> AgentResult:
        task = task.strip()
        steps = [
            AgentStep("understand", "done", "Parsed the user task."),
            AgentStep("retrieve", "running", "Searching local SARA training data."),
        ]
        hits = [] if self._is_tool_only_task(task) else self.index.search(task, top_k=self.config.top_k)
        steps[-1] = AgentStep("retrieve", "done", f"Found {len(hits)} relevant records.")

        tool_outputs = self._run_tools(task, steps)
        answer = self._compose_answer(task, hits, tool_outputs)
        steps.append(AgentStep("respond", "done", "Composed a grounded local response."))
        return AgentResult(task=task, answer=answer, hits=hits, steps=steps, tool_outputs=tool_outputs)

    def save_trace(self, result: AgentResult, path: str = workspace_path("agent", "last_trace.json")) -> str:
        output_path = ensure_parent_directory(path)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
        return output_path

    def _run_tools(self, task: str, steps: list[AgentStep]) -> dict[str, str]:
        outputs: dict[str, str] = {}
        expression = self._extract_calculation(task)
        if expression:
            if self.config.approval_mode == ApprovalMode.SUGGEST:
                steps.append(AgentStep("tool:calculator", "suggested", expression))
            else:
                outputs["calculator"] = safe_calculate(expression)
                steps.append(AgentStep("tool:calculator", "done", expression))

        lowered = task.lower()
        if any(marker in lowered for marker in ("time", "date", "日時", "時刻", "今日")):
            if self.config.approval_mode == ApprovalMode.SUGGEST:
                steps.append(AgentStep("tool:datetime", "suggested", "Read current local datetime."))
            else:
                outputs["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                steps.append(AgentStep("tool:datetime", "done", "Read current local datetime."))
        return outputs

    def _extract_calculation(self, task: str) -> str:
        if not re.search(r"(計算|calculate|calc|=)", task, flags=re.IGNORECASE):
            return ""
        candidates = re.findall(r"[0-9\.\+\-\*/%\(\)\s]+", task)
        candidates = [candidate.strip() for candidate in candidates if re.search(r"\d", candidate)]
        return max(candidates, key=len) if candidates else ""

    def _is_tool_only_task(self, task: str) -> bool:
        expression = self._extract_calculation(task)
        if not expression:
            return False
        remainder = task.replace(expression, "")
        remainder = re.sub(r"(を)?(計算|calculate|calc|して|ください|=)", "", remainder, flags=re.IGNORECASE)
        return len(re.sub(r"\s+", "", remainder)) <= 2

    def _compose_answer(
        self,
        task: str,
        hits: list[SearchHit],
        tool_outputs: dict[str, str],
    ) -> str:
        parts: list[str] = []
        if tool_outputs:
            parts.append("Tool results:")
            for name, value in tool_outputs.items():
                parts.append(f"- {name}: {value}")

        if hits:
            parts.append("Training-data grounded answer:")
            for idx, hit in enumerate(hits[: self.config.top_k], start=1):
                parts.append(
                    f"{idx}. {_compact(hit.response)} "
                    f"(source: {hit.source}, score: {hit.score:.2f})"
                )
            parts.append(
                "Summary: The answer above is assembled from the closest local learning-data records."
            )
        else:
            if tool_outputs:
                parts.append("No additional local training-data context was needed for this tool-only task.")
            else:
                parts.append(
                    "No close local training-data match was found. Add relevant examples under "
                    "data/raw or data/interim, then run the agent again."
                )

        answer = "\n".join(parts)
        if len(answer) > self.config.max_answer_chars:
            return answer[: self.config.max_answer_chars - 1].rstrip() + "..."
        return answer


def format_result(result: AgentResult, show_trace: bool = False) -> str:
    """Render an agent result for CLI output."""

    lines = [result.answer]
    if show_trace:
        lines.append("")
        lines.append("Agent steps:")
        for step in result.steps:
            lines.append(f"- {step.name}: {step.status} - {step.detail}")
    return "\n".join(lines)
