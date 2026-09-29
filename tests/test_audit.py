import hashlib
import json

from kyt_agent.audit import AuditLog, digest, redact_secrets
from tests.fakes import TARGET


def test_record_appends_json_lines(tmp_path):
    log = AuditLog(tmp_path / "audit")
    log.record("case-1", "case_opened", target=TARGET)
    log.record("case-1", "case_closed", status="approved")
    path = tmp_path / "audit" / "case-1.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [entry["event"] for entry in entries] == ["case_opened", "case_closed"]
    assert entries[0]["target"] == TARGET
    assert entries[0]["case_id"] == "case-1"
    assert "ts" in entries[0]


def test_digest_is_sha256():
    assert digest("abc") == hashlib.sha256(b"abc").hexdigest()


def test_redact_secrets_masks_api_key_values(monkeypatch):
    monkeypatch.setenv("ETHERSCAN_API_KEY", "abcdef123456")
    monkeypatch.setenv("EMPTY_API_KEY", "")
    assert redact_secrets("url?apikey=abcdef123456&x=1") == "url?apikey=***&x=1"
    assert redact_secrets("no secrets here") == "no secrets here"
