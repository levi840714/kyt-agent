from kyt_agent.models import Review
from kyt_agent.report import finalize, to_markdown, unknown_evidence, write_case_files
from tests.fakes import TARGET, draft, tx_hash


def test_unknown_evidence_is_case_insensitive():
    assert (
        unknown_evidence(draft("LOW", [tx_hash(1).upper().replace("0X", "0x")]), {tx_hash(1)}) == []
    )
    assert unknown_evidence(draft("LOW", ["0xfake"]), {tx_hash(1)}) == [0]


def test_finalize_applies_floor_without_lowering():
    raised = finalize(draft("LOW"), floor="HIGH", unverified=[], version=1)
    assert (raised.risk_level, raised.llm_risk_level, raised.risk_floor) == ("HIGH", "LOW", "HIGH")
    kept = finalize(draft("SEVERE"), floor="MEDIUM", unverified=[], version=2)
    assert (kept.risk_level, kept.version) == ("SEVERE", 2)


def test_markdown_marks_unverified_findings_and_reviews():
    report = finalize(draft("HIGH", ["0xfake"]), floor="LOW", unverified=[0], version=1)
    text = to_markdown(report, TARGET, [Review(decision="approve", reviewer="tester", round=1)])
    assert TARGET in text
    assert "（未驗證）" in text
    assert "tester：approve" in text


def test_write_case_files(tmp_path):
    report = finalize(draft("LOW"), floor="LOW", unverified=[], version=1)
    write_case_files(tmp_path / "case", TARGET, report, [])
    assert (tmp_path / "case" / "report.json").exists()
    assert (
        (tmp_path / "case" / "report.md").read_text(encoding="utf-8").startswith("# 地址風險報告")
    )
