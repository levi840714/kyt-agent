from kyt_agent.labels import LabelStore
from kyt_agent.models import Evidence, Review
from kyt_agent.report import (
    Finding,
    FundPath,
    PathHop,
    ReportDraft,
    finalize,
    to_markdown,
    unknown_evidence,
    write_case_files,
)
from tests.fakes import MIXER, TARGET, UNKNOWN, draft, label, tx_hash

NO_LABELS = LabelStore([])


EVIDENCE = {
    "T1": Evidence(id="T1", kind="tx", summary="s", ref=tx_hash(1)),
    "T2": Evidence(id="T2", kind="tx", summary="s", ref=tx_hash(2)),
    "T5": Evidence(id="T5", kind="tx", summary="s", ref=tx_hash(0xABC)),
    f"label:{MIXER}": Evidence(id=f"label:{MIXER}", kind="label", summary="s", ref=MIXER),
}


def test_unknown_evidence_checks_aliases_and_label_ids_case_insensitively():
    cited = ["t1", " T2 ", f"LABEL:{MIXER.upper().replace('0X', '0x')}"]
    assert unknown_evidence(draft("LOW", cited), EVIDENCE) == []
    assert unknown_evidence(draft("LOW", ["T3"]), EVIDENCE) == [0]
    assert unknown_evidence(draft("LOW", ["0xfake"]), EVIDENCE) == [0]


def test_unknown_evidence_accepts_registered_hash_only():
    cited = tx_hash(0xABC).upper().replace("0X", "0x")
    assert unknown_evidence(draft("LOW", [cited, "T2"]), EVIDENCE) == []
    assert unknown_evidence(draft("LOW", [tx_hash(9)]), EVIDENCE) == [0]
    assert unknown_evidence(draft("LOW", [MIXER]), EVIDENCE) == [0]


def test_finalize_replaces_tx_aliases_with_real_hashes():
    llm_draft = draft("HIGH", ["t1", f"label:{MIXER}", "T2"])
    report = finalize(
        llm_draft, evidence=EVIDENCE, floor="LOW", unverified=[], version=1, labels=NO_LABELS
    )
    assert report.findings[0].evidence == [tx_hash(1), f"label:{MIXER}", tx_hash(2)]
    assert llm_draft.findings[0].evidence[0] == "t1"


def test_finalize_keeps_cited_registered_hash():
    cited = tx_hash(0xABC).upper().replace("0X", "0x")
    report = finalize(
        draft("HIGH", [cited]),
        evidence=EVIDENCE,
        floor="LOW",
        unverified=[],
        version=1,
        labels=NO_LABELS,
    )
    assert report.findings[0].evidence == [tx_hash(0xABC)]


def test_finalize_resolves_registered_aliases_in_free_text():
    llm_draft = ReportDraft(
        risk_level="HIGH",
        summary="經由T1轉入混幣器，t2 亦同；T9、T10 與 ST1 未登記",
        findings=[Finding(claim="T2 由混幣器轉入", evidence=["T2"])],
        fund_paths=[FundPath(hops=[PathHop(address=TARGET)], note="見 (T1)")],
        recommendation="凍結 T1 相關資金",
        limitations=["未查詢 T2 的細節", "T9 無資料"],
    )
    report = finalize(
        llm_draft, evidence=EVIDENCE, floor="LOW", unverified=[], version=1, labels=NO_LABELS
    )
    assert report.summary == (
        f"經由{tx_hash(1)}轉入混幣器，{tx_hash(2)} 亦同；T9、T10 與 ST1 未登記"
    )
    assert report.findings[0].claim == f"{tx_hash(2)} 由混幣器轉入"
    assert report.fund_paths[0].note == f"見 ({tx_hash(1)})"
    assert report.recommendation == f"凍結 {tx_hash(1)} 相關資金"
    assert report.limitations == [f"未查詢 {tx_hash(2)} 的細節", "T9 無資料"]
    assert llm_draft.summary.startswith("經由T1")


def test_finalize_keeps_unknown_references_as_cited():
    report = finalize(
        draft("HIGH", ["T9"]),
        evidence=EVIDENCE,
        floor="LOW",
        unverified=[0],
        version=1,
        labels=NO_LABELS,
    )
    assert report.findings[0].evidence == ["T9"]


def test_finalize_applies_floor_without_lowering():
    raised = finalize(
        draft("LOW"), evidence={}, floor="HIGH", unverified=[], version=1, labels=NO_LABELS
    )
    assert (raised.risk_level, raised.llm_risk_level, raised.risk_floor) == ("HIGH", "LOW", "HIGH")
    kept = finalize(
        draft("SEVERE"), evidence={}, floor="MEDIUM", unverified=[], version=2, labels=NO_LABELS
    )
    assert (kept.risk_level, kept.version) == ("SEVERE", 2)


def test_finalize_replaces_fund_path_labels_with_known_labels():
    hops = [
        PathHop(address=TARGET, label="交易所"),
        PathHop(address=MIXER.upper().replace("0X", "0x"), label="編造的標籤"),
        PathHop(address=UNKNOWN, label="Lazarus"),
    ]
    llm_draft = draft("LOW").model_copy(update={"fund_paths": [FundPath(hops=hops, note="n")]})
    labels = LabelStore([label(MIXER, "mixer", "Tornado Cash")])
    report = finalize(llm_draft, evidence={}, floor="LOW", unverified=[], version=1, labels=labels)
    assert [hop.label for hop in report.fund_paths[0].hops] == [None, "mixer: Tornado Cash", None]
    assert llm_draft.fund_paths[0].hops[1].label == "編造的標籤"


def test_markdown_marks_unverified_findings_and_reviews():
    report = finalize(
        draft("HIGH", ["0xfake"]),
        evidence={},
        floor="LOW",
        unverified=[0],
        version=1,
        labels=NO_LABELS,
    )
    text = to_markdown(report, TARGET, [Review(decision="approve", reviewer="tester", round=1)])
    assert TARGET in text
    assert "（未驗證）" in text
    assert "tester：approve" in text


def test_write_case_files(tmp_path):
    report = finalize(
        draft("LOW"), evidence={}, floor="LOW", unverified=[], version=1, labels=NO_LABELS
    )
    write_case_files(tmp_path / "case", TARGET, report, [])
    assert (tmp_path / "case" / "report.json").exists()
    assert (
        (tmp_path / "case" / "report.md").read_text(encoding="utf-8").startswith("# 地址風險報告")
    )
