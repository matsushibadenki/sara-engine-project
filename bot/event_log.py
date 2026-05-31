from __future__ import annotations

import json
import os
from datetime import datetime


class EventLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def emit(self, event_type: str, payload: dict[str, object]) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
