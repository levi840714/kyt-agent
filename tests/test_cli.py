from typing import Any

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from kyt_agent import cli, render
from kyt_agent.evaluation.metrics import CaseResult, summarize
from kyt_agent.graph.nodes import ReportError
from kyt_agent.labels import LabelStore
from kyt_agent.report import finalize
from tests.fakes import SANCTIONED, TARGET, UNKNOWN, draft, label

runner = CliRunner()


def test_investigate_rejects_malformed_address():
    result = runner.invoke(cli.app, ["investigate", "0x123"])
    assert result.exit_code != 0


def test_sync_ofac_writes_csv(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        cli, "fetch_ofac_labels", lambda: [label(SANCTIONED, "sanctioned", "OFAC SDN")]
    )
    result = runner.invoke(cli.app, ["labels", "sync-ofac"])
    assert result.exit_code == 0, result.output
    assert SANCTIONED in (tmp_path / "labels" / "ofac.csv").read_text(encoding="utf-8")


def test_eval_reports_missing_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    result = runner.invoke(cli.app, ["eval"])
    assert result.exit_code == 1
    assert "找不到" in result.output
    assert "dataset.jsonl" in result.output


def test_show_report_renders_level_and_unverified_findings():
    console = Console(record=True, width=160)
    report = finalize(
        draft("LOW", ["0xfake"]),
        evidence={},
        floor="HIGH",
        unverified=[0],
        version=1,
        labels=LabelStore([]),
    )
    render.show_report(console, {"target": TARGET, "report": report.model_dump(mode="json")})
    text = console.export_text()
    assert "HIGH" in text
    assert "未驗證" in text


def test_show_report_renders_error_prominently():
    console = Console(record=True, width=160)
    report = finalize(
        draft("LOW"), evidence={}, floor="LOW", unverified=[], version=2, labels=LabelStore([])
    )
    render.show_report(
        console,
        {
            "target": TARGET,
            "report": report.model_dump(mode="json"),
            "error": "decision 欄位不可為空",
        },
    )
    text = console.export_text()
    assert "輸入有誤" in text
    assert "decision 欄位不可為空" in text


def test_ask_decision_requires_comment_for_reject(monkeypatch):
    answers = iter(["r", "", "理由"])
    monkeypatch.setattr(render.Prompt, "ask", lambda *args, **kwargs: next(answers))
    decision = render.ask_decision(Console(), allow_more=True)
    assert (decision["decision"], decision["comment"]) == ("reject", "理由")
    assert decision["reviewer"]


def test_ask_decision_omits_request_more_when_not_allowed(monkeypatch):
    seen_choices: list[list[str]] = []

    def fake_ask(prompt, *args, **kwargs):
        if "choices" in kwargs:
            seen_choices.append(kwargs["choices"])
            return "a"
        return ""

    monkeypatch.setattr(render.Prompt, "ask", fake_ask)
    decision = render.ask_decision(Console(), allow_more=False)
    assert decision["decision"] == "approve"
    assert "m" not in seen_choices[0]


class _FakeConn:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeSaver:
    def __init__(self) -> None:
        self.conn = _FakeConn()


class _FakeSnapshot:
    def __init__(self, values: dict[str, Any], interrupts: tuple = ()) -> None:
        self.values = values
        self.interrupts = interrupts
        self.next = () if not interrupts else ("review",)


class _FakeGraph:
    def __init__(self, snapshot: _FakeSnapshot, *, fail: bool = False) -> None:
        self._snapshot = snapshot
        self._fail = fail

    def get_state(self, config: Any) -> _FakeSnapshot:
        return self._snapshot

    def stream(self, graph_input: Any, config: Any, stream_mode: str):
        if self._fail:
            raise ReportError("報告格式解析失敗")
        return iter(())


def _patch_checkpointer(monkeypatch, saver: _FakeSaver) -> None:
    monkeypatch.setattr(cli, "open_checkpointer", lambda path: saver)
    monkeypatch.setattr(cli, "make_deps", lambda settings: object())


def test_run_case_closes_checkpointer_connection_on_success(monkeypatch, settings):
    saver = _FakeSaver()
    _patch_checkpointer(monkeypatch, saver)
    snapshot = _FakeSnapshot({"case_id": "case-1", "status": "approved"})
    monkeypatch.setattr(cli, "build_graph", lambda deps, checkpointer: _FakeGraph(snapshot))

    cli._run_case(settings, "case-1", {"case_id": "case-1"})

    assert saver.conn.closed


def test_run_case_closes_checkpointer_connection_on_report_error(monkeypatch, settings):
    saver = _FakeSaver()
    _patch_checkpointer(monkeypatch, saver)
    snapshot = _FakeSnapshot({"case_id": "case-1", "status": "investigating"})
    monkeypatch.setattr(
        cli, "build_graph", lambda deps, checkpointer: _FakeGraph(snapshot, fail=True)
    )

    with pytest.raises(typer.Exit):
        cli._run_case(settings, "case-1", {"case_id": "case-1"})

    assert saver.conn.closed


def test_eval_summary_shows_category_rates_and_clean_tokens():
    results = [
        CaseResult(
            address=TARGET,
            expected="risky",
            category="indirect_exposure",
            predicted="HIGH",
            baseline="LOW",
            input_tokens=100,
        ),
        CaseResult(
            address=UNKNOWN,
            expected="clean",
            category="indirect_minor",
            predicted="MEDIUM",
            baseline="LOW",
            input_tokens=300,
        ),
    ]
    console = Console(record=True, width=160)
    render.show_eval_summary(console, summarize("fake:model", results))
    text = console.export_text()
    assert "乾淨地址平均 token" in text
    assert "indirect_exposure" in text
    assert "indirect_minor" in text
    assert "indirect_minor" in render.eval_row(results[1])
