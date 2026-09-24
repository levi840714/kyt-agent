import json

import pytest
from langgraph.types import Command

from kyt_agent.graph.build import build_graph, case_config, open_checkpointer
from kyt_agent.graph.nodes import ReportError
from kyt_agent.graph.state import initial_state
from kyt_agent.labels import LabelStore
from tests.fakes import EXCHANGE, MIXER, TARGET, UNKNOWN, StubChain, draft, label, transfer, tx_hash
from tests.graph_fakes import ai_text, ai_tool_call, fake_deps


def audit_events(settings, case_id):
    path = settings.var_dir / "audit" / f"{case_id}.jsonl"
    return [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_case_with_supplement_round_survives_restart(settings):
    chain = StubChain(
        transfers={TARGET: [transfer(1, TARGET, MIXER), transfer(2, EXCHANGE, TARGET)]}
    )
    labels = LabelStore([label(MIXER, "mixer", "Tornado Cash")])
    script = [
        ai_tool_call("get_counterparties", "c1", address=TARGET),
        ai_text("初步結論"),
        ai_tool_call("lookup_address", "c2", address=MIXER),
        ai_text("補查完成"),
    ]
    drafts = [draft("MEDIUM", [tx_hash(1)]), draft("HIGH", [f"label:{MIXER}"])]
    deps = fake_deps(settings, chain, labels, script, drafts)
    database = settings.var_dir / "checkpoints.sqlite"
    config = case_config("case-1")

    graph = build_graph(deps, open_checkpointer(database))
    graph.invoke(initial_state("case-1", TARGET, settings), config)
    first = graph.get_state(config).interrupts[0].value
    assert (first["report"]["version"], first["report"]["risk_level"]) == (1, "MEDIUM")
    assert first["allow_more"]

    # 以新的 graph 實例接續，模擬關掉終端機後 resume
    resumed = build_graph(deps, open_checkpointer(database))
    more = {"decision": "request_more", "comment": "請確認混幣器", "reviewer": "tester"}
    resumed.invoke(Command(resume=more), config)
    second = resumed.get_state(config).interrupts[0].value
    assert (second["report"]["version"], second["report"]["risk_level"]) == (2, "HIGH")

    approve = {"decision": "approve", "comment": "", "reviewer": "tester"}
    final = resumed.invoke(Command(resume=approve), config)
    assert final["status"] == "approved"
    assert [review.decision for review in final["reviews"]] == ["request_more", "approve"]
    assert final["tool_call_limit"] == settings.max_tool_calls + settings.supplement_tool_calls
    assert (settings.var_dir / "cases" / "case-1" / "report.md").exists()
    events = audit_events(settings, "case-1")
    assert (events[0], events[-1]) == ("case_opened", "case_closed")
    assert events.count("human_decision") == 2
    assert events.count("tool_call") == 2


def test_sanctioned_target_skips_agent_and_floors_to_severe(settings):
    labels = LabelStore([label(TARGET, "sanctioned", "OFAC SDN")])
    deps = fake_deps(
        settings, StubChain(), labels, drafts=[draft("LOW", [f"label:{TARGET}"])], auto_approve=True
    )
    final = build_graph(deps).invoke(
        initial_state("case-2", TARGET, settings), case_config("case-2")
    )
    assert (final["report"].risk_level, final["report"].llm_risk_level) == ("SEVERE", "LOW")
    assert final["tool_calls"] == 0
    assert final["report"].unverified_findings == []


def test_budget_exhaustion_forces_report(settings):
    tight = settings.model_copy(update={"max_tool_calls": 1})
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    script = [
        ai_tool_call("get_counterparties", "c1", address=TARGET),
        ai_tool_call("get_counterparties", "c2", address=UNKNOWN),
    ]
    deps = fake_deps(tight, chain, LabelStore([]), script, [draft("LOW")], auto_approve=True)
    final = build_graph(deps).invoke(initial_state("case-3", TARGET, tight), case_config("case-3"))
    assert final["tool_calls"] == 1
    assert "工具呼叫次數已達上限" in final["budget_note"]
    assert final["messages"][-1].content.startswith("未執行")
    assert "budget_exhausted" in audit_events(tight, "case-3")


@pytest.mark.parametrize(("second_evidence", "unverified"), [([tx_hash(1)], []), (["0xfake"], [0])])
def test_invalid_evidence_is_retried_once(settings, second_evidence, unverified):
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    script = [ai_tool_call("get_counterparties", "c1", address=TARGET), ai_text("完成")]
    drafts = [draft("LOW", ["0xfake"]), draft("LOW", second_evidence)]
    deps = fake_deps(settings, chain, LabelStore([]), script, drafts, auto_approve=True)
    final = build_graph(deps).invoke(
        initial_state("case-4", TARGET, settings), case_config("case-4")
    )
    assert final["report"].unverified_findings == unverified


def test_unparseable_report_raises(settings):
    deps = fake_deps(
        settings, StubChain(), LabelStore([]), [ai_text("完成")], [None, None], auto_approve=True
    )
    with pytest.raises(ReportError):
        build_graph(deps).invoke(initial_state("case-5", TARGET, settings), case_config("case-5"))
