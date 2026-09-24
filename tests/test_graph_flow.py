import json
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from kyt_agent.graph.build import build_graph, case_config, open_checkpointer
from kyt_agent.graph.nodes import ReportError
from kyt_agent.graph.state import initial_state
from kyt_agent.labels import LabelStore
from kyt_agent.models import RuleHit
from tests.fakes import EXCHANGE, MIXER, TARGET, UNKNOWN, StubChain, draft, label, transfer, tx_hash
from tests.graph_fakes import AGENT_USAGE, ai_text, ai_tool_call, fake_deps, scripted_drafter


def audit_events(settings, case_id):
    path = settings.var_dir / "audit" / f"{case_id}.jsonl"
    return [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def open_db(settings):
    savers = []

    def open_saver():
        savers.append(open_checkpointer(settings.var_dir / "checkpoints.sqlite"))
        return savers[-1]

    yield open_saver
    for saver in savers:
        saver.conn.close()


APPROVE = {"decision": "approve", "comment": "", "reviewer": "tester"}


def test_full_case_with_supplement_round_survives_restart(settings, open_db):
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
    config = case_config("case-1")

    graph = build_graph(deps, open_db())
    graph.invoke(initial_state("case-1", TARGET, settings), config)
    first = graph.get_state(config).interrupts[0].value
    assert (first["report"]["version"], first["report"]["risk_level"]) == (1, "MEDIUM")
    assert first["allow_more"]

    # 以新的 graph 實例接續，模擬關掉終端機後 resume
    resumed = build_graph(deps, open_db())
    more = {"decision": "request_more", "comment": "請確認混幣器", "reviewer": "tester"}
    resumed.invoke(Command(resume=more), config)
    second = resumed.get_state(config).interrupts[0].value
    assert (second["report"]["version"], second["report"]["risk_level"]) == (2, "HIGH")
    remaining = settings.max_tool_calls + settings.supplement_tool_calls - 1
    supplement = [m for m in resumed.get_state(config).values["messages"] if "要求補查" in m.text]
    assert f"{remaining} 次工具呼叫" in supplement[0].text

    final = resumed.invoke(Command(resume=APPROVE), config)
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


def test_sanctioned_case_survives_restart(settings, open_db):
    labels = LabelStore([label(TARGET, "sanctioned", "OFAC SDN")])
    deps = fake_deps(settings, StubChain(), labels, drafts=[draft("LOW", [f"label:{TARGET}"])])
    config = case_config("case-2b")
    build_graph(deps, open_db()).invoke(initial_state("case-2b", TARGET, settings), config)

    resumed = build_graph(deps, open_db())
    assert resumed.get_state(config).interrupts[0].value["report"]["risk_level"] == "SEVERE"
    final = resumed.invoke(Command(resume=APPROVE), config)
    assert final["status"] == "approved"
    assert all(isinstance(hit, RuleHit) for hit in final["rule_hits"])
    assert final["rule_hits"]


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


def test_parse_error_is_fed_back_before_retry(settings):
    seen = []
    drafter = scripted_drafter([None, draft("LOW")])

    def recording(messages):
        seen.append(list(messages))
        return drafter.invoke(messages)

    deps = fake_deps(settings, StubChain(), LabelStore([]), [ai_text("完成")], auto_approve=True)
    deps = replace(deps, drafter=RunnableLambda(recording))
    final = build_graph(deps).invoke(
        initial_state("case-6", TARGET, settings), case_config("case-6")
    )
    assert final["report"].version == 1
    assert len(seen[1]) == len(seen[0]) + 1
    assert isinstance(seen[1][-1], HumanMessage)
    assert "invalid json" in seen[1][-1].text
    assert not any("無法解析" in message.text for message in final["messages"])


def test_parallel_tool_calls_beyond_limit_are_skipped(settings):
    tight = settings.model_copy(update={"max_tool_calls": 1})
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_counterparties", "args": {"address": TARGET}, "id": "c1"},
            {"name": "lookup_address", "args": {"address": UNKNOWN}, "id": "c2"},
        ],
        usage_metadata=AGENT_USAGE,
    )
    script = [both, ai_text("完成")]
    deps = fake_deps(tight, chain, LabelStore([]), script, [draft("LOW")], auto_approve=True)
    final = build_graph(deps).invoke(initial_state("case-7", TARGET, tight), case_config("case-7"))
    assert final["tool_calls"] == 1
    replies = {m.tool_call_id: m.text for m in final["messages"] if m.type == "tool"}
    assert not replies["c1"].startswith("未執行")
    assert replies["c2"].startswith("未執行")


def test_supplement_not_offered_after_token_budget_exhausted(settings, open_db):
    tight = settings.model_copy(update={"max_tokens": 100})
    deps = fake_deps(tight, StubChain(), LabelStore([]), [ai_text("完成")], [draft("LOW")])
    config = case_config("case-8")
    graph = build_graph(deps, open_db())
    graph.invoke(initial_state("case-8", TARGET, tight), config)
    assert graph.get_state(config).interrupts[0].value["allow_more"] is False

    more = {"decision": "request_more", "comment": "再查", "reviewer": "tester"}
    graph.invoke(Command(resume=more), config)
    assert "error" in graph.get_state(config).interrupts[0].value
    assert graph.invoke(Command(resume=APPROVE), config)["status"] == "approved"


def test_invalid_review_input_reprompts_instead_of_failing(settings, open_db):
    deps = fake_deps(settings, StubChain(), LabelStore([]), [ai_text("完成")], [draft("LOW")])
    config = case_config("case-9")
    graph = build_graph(deps, open_db())
    graph.invoke(initial_state("case-9", TARGET, settings), config)

    graph.invoke(Command(resume={"decision": "reject", "comment": "", "reviewer": "t"}), config)
    pending = graph.get_state(config).interrupts
    assert len(pending) == 1
    assert "意見" in pending[0].value["error"]
    assert pending[0].value["report"]["version"] == 1

    final = graph.invoke(Command(resume=APPROVE), config)
    assert final["status"] == "approved"
    assert [review.decision for review in final["reviews"]] == ["approve"]
    assert audit_events(settings, "case-9").count("human_decision") == 1
