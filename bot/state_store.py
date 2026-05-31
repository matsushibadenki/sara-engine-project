from __future__ import annotations

import json
import os
from datetime import datetime
from dataclasses import dataclass
from bot.io_utils import atomic_write_json


@dataclass
class BotState:
    visited_urls: list[str]
    processed_files: list[str]
    modality_counts: dict[str, int]
    failed_attempts: dict[str, int]
    content_hashes: list[str]
    semantic_hashes: list[int]
    language_counts: dict[str, int]
    domain_reputation: dict[str, float]
    replay_cursor: int
    last_eval_passed: bool


class StateStore:
    def __init__(self, state_path: str) -> None:
        self.state_path = state_path
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

    def _default_state(self) -> BotState:
        return BotState(
            visited_urls=[],
            processed_files=[],
            modality_counts={},
            failed_attempts={},
            content_hashes=[],
            semantic_hashes=[],
            language_counts={"jp": 0, "en": 0, "other": 0},
            domain_reputation={},
            replay_cursor=0,
            last_eval_passed=True,
        )

    def _corrupt_backup_path(self) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{self.state_path}.corrupt.{ts}.json"

    def _recover_from_corruption(self, raw_content: str = "") -> BotState:
        backup_path = self._corrupt_backup_path()
        try:
            if os.path.exists(self.state_path):
                os.replace(self.state_path, backup_path)
            elif raw_content:
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(raw_content)
        finally:
            default_state = self._default_state()
            self.save(default_state)
            return default_state

    def recovered_count(self) -> int:
        base = os.path.basename(self.state_path)
        dirname = os.path.dirname(self.state_path) or "."
        count = 0
        try:
            for name in os.listdir(dirname):
                if name.startswith(base + ".corrupt.") and name.endswith(".json"):
                    count += 1
        except OSError:
            count = 0
        return count

    def load(self) -> BotState:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if not isinstance(raw, dict):
                    raise ValueError("State payload must be an object.")
                return BotState(
                    visited_urls=list(raw.get("visited_urls", [])),
                    processed_files=list(raw.get("processed_files", [])),
                    modality_counts=dict(raw.get("modality_counts", {})),
                    failed_attempts=dict(raw.get("failed_attempts", {})),
                    content_hashes=list(raw.get("content_hashes", [])),
                    semantic_hashes=[int(x) for x in list(raw.get("semantic_hashes", [])) if isinstance(x, (int, float, str)) and str(x).strip()],
                    language_counts={
                        "jp": int(dict(raw.get("language_counts", {})).get("jp", 0) or 0),
                        "en": int(dict(raw.get("language_counts", {})).get("en", 0) or 0),
                        "other": int(dict(raw.get("language_counts", {})).get("other", 0) or 0),
                    },
                    domain_reputation={
                        str(k): float(v)
                        for k, v in dict(raw.get("domain_reputation", {})).items()
                    },
                    replay_cursor=int(raw.get("replay_cursor", 0) or 0),
                    last_eval_passed=bool(raw.get("last_eval_passed", True)),
                )
            except (json.JSONDecodeError, ValueError):
                raw_content = ""
                try:
                    with open(self.state_path, "r", encoding="utf-8", errors="ignore") as f:
                        raw_content = f.read()
                except Exception:
                    raw_content = ""
                return self._recover_from_corruption(raw_content=raw_content)
            except Exception:
                return self._default_state()
        return self._default_state()

    def save(self, state: BotState) -> None:
        atomic_write_json(
            self.state_path,
            {
                "visited_urls": state.visited_urls,
                "processed_files": state.processed_files,
                "modality_counts": state.modality_counts,
                "failed_attempts": state.failed_attempts,
                "content_hashes": state.content_hashes,
                "semantic_hashes": state.semantic_hashes,
                "language_counts": state.language_counts,
                "domain_reputation": state.domain_reputation,
                "replay_cursor": int(state.replay_cursor),
                "last_eval_passed": bool(state.last_eval_passed),
            },
        )
