from __future__ import annotations

import json
import os
from datetime import datetime
from dataclasses import dataclass
from bot.io_utils import atomic_write_json


@dataclass
class QueueStats:
    pending: int
    total_enqueued: int
    recovered_count: int


class TrainingQueue:
    def __init__(self, queue_path: str) -> None:
        self.queue_path = queue_path
        os.makedirs(os.path.dirname(queue_path), exist_ok=True)
        if not os.path.exists(queue_path):
            self._save([])

    def _corrupt_backup_path(self) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{self.queue_path}.corrupt.{ts}.json"

    def _recover_from_corruption(self, raw_content: str = "") -> None:
        backup_path = self._corrupt_backup_path()
        try:
            if os.path.exists(self.queue_path):
                os.replace(self.queue_path, backup_path)
            elif raw_content:
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(raw_content)
        finally:
            self._save([])

    def _load(self) -> list[dict[str, object]]:
        try:
            with open(self.queue_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError("Queue payload must be a list.")
            return [item for item in payload if isinstance(item, dict)]
        except (json.JSONDecodeError, ValueError):
            raw = ""
            try:
                with open(self.queue_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
            except Exception:
                raw = ""
            self._recover_from_corruption(raw_content=raw)
            return []
        except FileNotFoundError:
            self._save([])
            return []

    def _save(self, items: list[dict[str, object]]) -> None:
        atomic_write_json(self.queue_path, items)

    def enqueue(self, item: dict[str, object]) -> None:
        items = self._load()
        if "priority" not in item:
            item["priority"] = 0.0
        if "created_at" not in item:
            item["created_at"] = datetime.utcnow().isoformat()
        items.append(item)
        self._save(items)

    def enqueue_learning_materials(self, manifest_items: list[dict[str, object]]) -> int:
        enqueued = 0
        for item in manifest_items:
            material_type = str(item.get("material_type", ""))
            stage = str(item.get("curriculum_stage", "medium"))
            queue_item = {
                "source": item.get("source", "autobot_dataset_builder"),
                "material_hash": item.get("material_hash", ""),
                "material_type": material_type,
                "curriculum_stage": stage,
                "priority": float(item.get("priority", 0.0) or 0.0),
                "path": str(item.get("path", "data/processed/autobot/curriculum_manifest.jsonl") or "data/processed/autobot/curriculum_manifest.jsonl"),
            }
            if stage == "repair":
                queue_item["priority"] = max(1.0, float(queue_item["priority"]))
            if stage == "replay" and float(queue_item["priority"]) < 0.75:
                continue
            self.enqueue(queue_item)
            enqueued += 1
        return enqueued

    def drain(self, limit: int) -> list[dict[str, object]]:
        items = self._load()
        def _key(obj: dict[str, object]) -> tuple[float, str]:
            pr = float(obj.get("priority", 0.0) or 0.0)
            created = str(obj.get("created_at", ""))
            return (-pr, created)

        ordered = sorted(items, key=_key)
        picked = ordered[:limit]
        remaining = ordered[limit:]
        self._save(remaining)
        return picked

    def drain_curriculum(
        self,
        limit: int,
        easy_ratio: float,
        medium_ratio: float,
        hard_ratio: float,
    ) -> list[dict[str, object]]:
        items = self._load()

        def _key(obj: dict[str, object]) -> tuple[float, str]:
            pr = float(obj.get("priority", 0.0) or 0.0)
            created = str(obj.get("created_at", ""))
            return (-pr, created)

        stages = {"easy": [], "medium": [], "hard": []}
        for it in items:
            stage = str(it.get("curriculum_stage", "medium")).strip().lower()
            if stage not in stages:
                stage = "medium"
            stages[stage].append(it)
        for k in stages:
            stages[k] = sorted(stages[k], key=_key)

        total_ratio = max(0.0001, float(easy_ratio) + float(medium_ratio) + float(hard_ratio))
        target_easy = int(round(limit * float(easy_ratio) / total_ratio))
        target_medium = int(round(limit * float(medium_ratio) / total_ratio))
        target_hard = max(0, limit - target_easy - target_medium)

        picked = []
        picked.extend(stages["easy"][:target_easy])
        picked.extend(stages["medium"][:target_medium])
        picked.extend(stages["hard"][:target_hard])

        if len(picked) < limit:
            selected_ids = {id(x) for x in picked}
            leftovers = []
            for k in ("easy", "medium", "hard"):
                leftovers.extend([x for x in stages[k] if id(x) not in selected_ids])
            leftovers = sorted(leftovers, key=_key)
            picked.extend(leftovers[: max(0, limit - len(picked))])

        selected_ids = {id(x) for x in picked}
        remaining = [x for x in items if id(x) not in selected_ids]
        self._save(remaining)
        return picked

    def stats(self) -> QueueStats:
        items = self._load()
        base = os.path.basename(self.queue_path)
        dirname = os.path.dirname(self.queue_path) or "."
        recovered_count = 0
        try:
            for name in os.listdir(dirname):
                if name.startswith(base + ".corrupt.") and name.endswith(".json"):
                    recovered_count += 1
        except OSError:
            recovered_count = 0
        return QueueStats(pending=len(items), total_enqueued=len(items), recovered_count=recovered_count)
