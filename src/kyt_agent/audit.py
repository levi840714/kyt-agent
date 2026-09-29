import hashlib
import json
import os
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


def redact_secrets(text: str) -> str:
    """遮蔽文字中出現的 *_API_KEY 環境變數值，避免錯誤訊息把 key 寫進 audit log。"""
    for name, value in os.environ.items():
        if name.endswith("_API_KEY") and len(value) >= 8:
            text = text.replace(value, "***")
    return text
