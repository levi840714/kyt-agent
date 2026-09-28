from kyt_agent.labels import LabelStore
from kyt_agent.models import Review
from kyt_agent.report import (
    FundPath,
    PathHop,
    finalize,
    to_markdown,
    unknown_evidence,
    write_case_files,
)
from tests.fakes import MIXER, TARGET, UNKNOWN, draft, label, tx_hash

NO_LABELS = LabelStore([])


def test_unknown_evidence_is_case_insensitive():
    assert (
        unknown_evidence(draft("LOW", [tx_hash(1).upper().replace("0X", "0x")]), {tx_hash(1)}) == []
    )
    assert unknown_evidence(draft("LOW", ["0xfake"]), {tx_hash(1)}) == [0]


def test_finalize_applies_floor_without_lowering():
    raised = finalize(draft("LOW"), floor="HIGH", unverified=[], version=1, labels=NO_LABELS)
    assert (raised.risk_level, raised.llm_risk_level, raised.risk_floor) == ("HIGH", "LOW", "HIGH")
    kept = finalize(draft("SEVERE"), floor="MEDIUM", unverified=[], version=2, labels=NO_LABELS)
    assert (kept.risk_level, kept.version) == ("SEVERE", 2)


def test_finalize_replaces_fund_path_labels_with_known_labels():
    hops = [
        PathHop(address=TARGET, label="交易所"),
        PathHop(address=MIXER.upper().replace("0X", "0x"), label="編造的標籤"),
        PathHop(address=UNKNOWN, label="Lazarus"),
    ]
    llm_draft = draft("LOW").model_copy(update={"fund_paths": [FundPath(hops=hops, note="n")]})
    labels = LabelStore([label(MIXER, "mixer", "Tornado Cash")])
    report = finalize(llm_draft, floor="LOW", unverified=[], version=1, labels=labels)
    assert [hop.label for hop in report.fund_paths[0].hops] == [None, "mixer: Tornado Cash", None]
    assert llm_draft.fund_paths[0].hops[1].label == "編造的標籤"


def test_markdown_marks_unverified_findings_and_reviews():
    report = finalize(
        draft("HIGH", ["0xfake"]), floor="LOW", unverified=[0], version=1, labels=NO_LABELS
    )
    text = to_markdown(report, TARGET, [Review(decision="approve", reviewer="tester", round=1)])
    assert TARGET in text
    assert "（未驗證）" in text
    assert "tester：approve" in text


def test_write_case_files(tmp_path):
    report = finalize(draft("LOW"), floor="LOW", unverified=[], version=1, labels=NO_LABELS)
    write_case_files(tmp_path / "case", TARGET, report, [])
    assert (tmp_path / "case" / "report.json").exists()
    assert (
        (tmp_path / "case" / "report.md").read_text(encoding="utf-8").startswith("# 地址風險報告")
    )
