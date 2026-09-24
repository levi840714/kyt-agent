import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLog:
    """每個案件一個只追加的 JSONL 檔。"""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def record(self, case_id: str, event: str, **data: Any) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(UTC).isoformat(), "case_id": case_id, "event": event, **data}
        with (self._directory / f"{case_id}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
