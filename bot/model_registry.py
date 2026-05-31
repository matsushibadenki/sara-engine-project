from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from bot.io_utils import atomic_write_json


@dataclass
class PromotionResult:
    promoted: bool
    reason: str
    production_dir: str
    candidate_dir: str
    backup_dir: str | None


class ModelRegistry:
    def __init__(self, root_dir: str, metadata_path: str) -> None:
        self.root_dir = root_dir
        self.candidate_dir = os.path.join(root_dir, "candidate")
        self.production_dir = os.path.join(root_dir, "production")
        self.backup_root = os.path.join(root_dir, "backups")
        self.metadata_path = metadata_path
        os.makedirs(self.candidate_dir, exist_ok=True)
        os.makedirs(self.production_dir, exist_ok=True)
        os.makedirs(self.backup_root, exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    def _write_metadata(self, payload: dict[str, object]) -> None:
        atomic_write_json(self.metadata_path, payload)

    def _copy_tree_replace(self, src: str, dst: str) -> None:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def prune_backups(self, max_keep: int) -> int:
        if max_keep < 1:
            max_keep = 1
        try:
            backups = [
                os.path.join(self.backup_root, name)
                for name in os.listdir(self.backup_root)
                if name.startswith("production_")
            ]
        except OSError:
            return 0
        backups = [p for p in backups if os.path.isdir(p)]
        backups.sort(reverse=True)
        removed = 0
        for path in backups[max_keep:]:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        return removed

    def rollback_latest(self) -> bool:
        try:
            backups = [
                os.path.join(self.backup_root, name)
                for name in os.listdir(self.backup_root)
                if name.startswith("production_")
            ]
        except OSError:
            return False
        backups = [p for p in backups if os.path.isdir(p)]
        if not backups:
            return False
        backups.sort(reverse=True)
        latest = backups[0]
        self._copy_tree_replace(latest, self.production_dir)
        return True

    def promote_candidate(self, eval_report: dict[str, object]) -> PromotionResult:
        if not bool(eval_report.get("passed", False)):
            self._write_metadata(
                {
                    "last_attempt_at": datetime.utcnow().isoformat(),
                    "last_attempt_promoted": False,
                    "last_attempt_reason": "evaluation_failed",
                    "production_dir": self.production_dir,
                    "candidate_dir": self.candidate_dir,
                    "backup_dir": None,
                    "evaluation": eval_report,
                }
            )
            return PromotionResult(
                promoted=False,
                reason="evaluation_failed",
                production_dir=self.production_dir,
                candidate_dir=self.candidate_dir,
                backup_dir=None,
            )

        if not any(os.scandir(self.candidate_dir)):
            self._write_metadata(
                {
                    "last_attempt_at": datetime.utcnow().isoformat(),
                    "last_attempt_promoted": False,
                    "last_attempt_reason": "empty_candidate",
                    "production_dir": self.production_dir,
                    "candidate_dir": self.candidate_dir,
                    "backup_dir": None,
                    "evaluation": eval_report,
                }
            )
            return PromotionResult(
                promoted=False,
                reason="empty_candidate",
                production_dir=self.production_dir,
                candidate_dir=self.candidate_dir,
                backup_dir=None,
            )

        backup_dir = None
        if any(os.scandir(self.production_dir)):
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.backup_root, f"production_{ts}")
            shutil.copytree(self.production_dir, backup_dir)

        self._copy_tree_replace(self.candidate_dir, self.production_dir)

        self._write_metadata(
            {
                "last_promoted_at": datetime.utcnow().isoformat(),
                "last_attempt_at": datetime.utcnow().isoformat(),
                "last_attempt_promoted": True,
                "last_attempt_reason": "promoted",
                "production_dir": self.production_dir,
                "candidate_dir": self.candidate_dir,
                "backup_dir": backup_dir,
                "evaluation": eval_report,
            }
        )

        return PromotionResult(
            promoted=True,
            reason="promoted",
            production_dir=self.production_dir,
            candidate_dir=self.candidate_dir,
            backup_dir=backup_dir,
        )
