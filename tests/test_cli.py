import json
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


def test_eval_fill_missing_is_passed_to_runner(monkeypatch, tmp_path):
    (tmp_path / "eval").mkdir()
    row = {"address": TARGET, "expected": "clean", "category": "exchange_user", "source": "t"}
    (tmp_path / "eval" / "dataset.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("ETHERSCAN_API_KEY", "test-key")
    seen: dict[str, Any] = {}

    def fake_run_eval(settings, cases, model, on_result, *, fill_missing, repeat):
        seen["fill_missing"] = fill_missing
        return summarize("fake:model", [], snapshots_filled=2), []

    monkeypatch.setattr(cli, "run_eval", fake_run_eval)
    result = runner.invoke(cli.app, ["eval", "--fill-missing"])
    assert result.exit_code == 0, result.output
    assert seen["fill_missing"] is True
    assert "補錄快照 2 個檔案" in result.output


def _write_dataset(root):
    (root / "eval").mkdir()
    row = {"address": TARGET, "expected": "clean", "category": "exchange_user", "source": "t"}
    (root / "eval" / "dataset.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_eval_hints_when_results_contain_snapshot_miss(monkeypatch, tmp_path):
    _write_dataset(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("VAR_DIR", str(tmp_path / "var"))
    failed = CaseResult(
        address=TARGET,
        expected="clean",
        category="exchange_user",
        predicted=None,
        baseline=None,
        error="SnapshotMissError: transfers",
    )
    monkeypatch.setattr(
        cli, "run_eval", lambda *args, **kwargs: (summarize("fake:model", [failed]), [failed])
    )
    result = runner.invoke(cli.app, ["eval"])
    assert result.exit_code == 0, result.output
    assert "缺少快照，請先執行 kyt eval --record 或加上 --fill-missing" in result.output


def test_eval_omits_snapshot_hint_when_no_miss(monkeypatch, tmp_path):
    _write_dataset(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setattr(cli, "run_eval", lambda *args, **kwargs: (summarize("fake:model", []), []))
    result = runner.invoke(cli.app, ["eval"])
    assert "缺少快照" not in result.output


def test_eval_fill_missing_requires_etherscan_key(monkeypatch, tmp_path):
    _write_dataset(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ETHERSCAN_API_KEY", "")
    result = runner.invoke(cli.app, ["eval", "--fill-missing"])
    assert result.exit_code == 1
    assert "--fill-missing 需要 ETHERSCAN_API_KEY" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_eval_rejects_record_with_fill_missing(monkeypatch, tmp_path):
    _write_dataset(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ETHERSCAN_API_KEY", "test-key")
    result = runner.invoke(cli.app, ["eval", "--record", "--fill-missing"])
    assert result.exit_code == 1
    assert "不可同時使用" in result.output


def test_eval_row_shows_dash_for_missing_baseline():
    result = CaseResult(
        address=TARGET, expected="clean", category="exchange_user", predicted="LOW", baseline=None
    )
    assert "規則 -" in render.eval_row(result)


def test_eval_repeat_is_passed_to_runner(monkeypatch, tmp_path):
    _write_dataset(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("VAR_DIR", str(tmp_path / "var"))
    seen: dict[str, Any] = {}

    def fake_run_eval(settings, cases, model, on_result, *, fill_missing, repeat):
        seen["repeat"] = repeat
        return summarize("fake:model", [], repeat=repeat), []

    monkeypatch.setattr(cli, "run_eval", fake_run_eval)
    result = runner.invoke(cli.app, ["eval", "--repeat", "3"])
    assert result.exit_code == 0, result.output
    assert seen["repeat"] == 3


@pytest.mark.parametrize("value", ["0", "-1"])
def test_eval_rejects_repeat_below_one(monkeypatch, tmp_path, value):
    _write_dataset(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    called: list[bool] = []
    monkeypatch.setattr(cli, "run_eval", lambda *args, **kwargs: called.append(True))
    result = runner.invoke(cli.app, ["eval", "--repeat", value])
    assert result.exit_code == 2
    assert called == []


def _repeated(address, level, run, tokens, expanded=()):
    return CaseResult(
        address=address,
        expected="risky",
        category="indirect_exposure",
        predicted=level,
        baseline="LOW",
        input_tokens=tokens,
        run=run,
        expanded=list(expanded),
    )


def test_eval_summary_shows_consistency_section_when_repeated():
    results = [
        _repeated(TARGET, "HIGH", 1, 1200, [TARGET]),
        _repeated(TARGET, "MEDIUM", 2, 900, [TARGET, UNKNOWN]),
        _repeated(TARGET, "HIGH", 3, 1500, [TARGET]),
        _repeated(UNKNOWN, "HIGH", 1, 500),
        _repeated(UNKNOWN, "HIGH", 2, 500),
        _repeated(UNKNOWN, "HIGH", 3, 500),
    ]
    console = Console(record=True, width=160)
    render.show_eval_summary(console, summarize("fake:model", results, repeat=3))
    text = console.export_text()
    assert "一致性" in text
    assert "決策一致率" in text and "50%" in text
    assert "等級一致率" in text
    assert "不穩定案例" in text
    row = next(line for line in text.splitlines() if "0x1111…1111" in line)
    for cell in ("indirect_exposure", "HIGH / MEDIUM / HIGH", "2/3", "900–1,500", "是"):
        assert cell in row
    assert "0x4444…4444" not in text


def test_eval_summary_omits_consistency_section_for_single_run():
    console = Console(record=True, width=160)
    render.show_eval_summary(console, summarize("fake:model", [_repeated(TARGET, "HIGH", 1, 1)]))
    assert "一致性" not in console.export_text()


def test_eval_row_shows_run_number_when_repeated():
    result = _repeated(TARGET, "HIGH", 2, 10)
    assert "[2/3]" in render.eval_row(result, repeat=3)
    assert "[2/" not in render.eval_row(result)
