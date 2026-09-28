import json
import re
from dataclasses import replace
from decimal import Decimal

import ormsgpack
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from kyt_agent.chain.client import TransactionDetail
from kyt_agent.graph.build import CHECKPOINT_TYPES, build_graph, case_config, open_checkpointer
from kyt_agent.graph.nodes import CaseNodes, ReportError
from kyt_agent.graph.state import initial_state
from kyt_agent.labels import LabelStore
from kyt_agent.models import AddressNode, Evidence, RuleHit
from tests.fakes import (
    EXCHANGE,
    MIXER,
    TARGET,
    UNKNOWN,
    StubChain,
    draft,
    label,
    transfer,
    tx_hash,
    tx_refs,
)
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
    drafts = [draft("MEDIUM", ["T1"]), draft("HIGH", ["t1", f"label:{MIXER}"])]
    deps = fake_deps(settings, chain, labels, script, drafts)
    config = case_config("case-1")

    graph = build_graph(deps, open_db())
    graph.invoke(initial_state("case-1", TARGET, settings), config)
    first = graph.get_state(config).interrupts[0].value
    first_report = first["report"]
    assert (first_report["version"], first_report["llm_risk_level"]) == (1, "MEDIUM")
    assert first_report["risk_level"] == "HIGH"
    assert first_report["findings"][0]["evidence"] == [tx_hash(1)]
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
    case_dir = settings.var_dir / "cases" / "case-1"
    saved = json.loads((case_dir / "report.json").read_text(encoding="utf-8"))
    assert saved["findings"][0]["evidence"] == [tx_hash(1), f"label:{MIXER}"]
    markdown = (case_dir / "report.md").read_text(encoding="utf-8")
    assert f"證據：{tx_hash(1)}, label:{MIXER}" in markdown
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


@pytest.mark.parametrize(
    ("second_evidence", "unverified", "saved"),
    [
        (["T1"], [], [tx_hash(1)]),
        ([tx_hash(1)], [], [tx_hash(1)]),
        (["0xfake"], [0], ["0xfake"]),
    ],
)
def test_invalid_evidence_is_retried_once(settings, second_evidence, unverified, saved):
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    script = [ai_tool_call("get_counterparties", "c1", address=TARGET), ai_text("完成")]
    drafts = [draft("LOW", ["0xfake"]), draft("LOW", second_evidence)]
    deps = fake_deps(settings, chain, LabelStore([]), script, drafts, auto_approve=True)
    final = build_graph(deps).invoke(
        initial_state("case-4", TARGET, settings), case_config("case-4")
    )
    assert final["report"].unverified_findings == unverified
    assert final["report"].findings[0].evidence == saved


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


def test_tools_node_tolerates_checkpoint_without_wrap_up_fields(settings):
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    deps = fake_deps(settings, chain, LabelStore([]), auto_approve=True)
    state = initial_state("case-13", TARGET, settings)
    # 模擬從 v1.1 之前的 checkpoint resume：state 尚未帶有這兩個欄位
    del state["calls_without_risk"]
    del state["wrap_up_hinted"]
    state["messages"] = [
        *state["messages"],
        ai_tool_call("get_counterparties", "c1", address=TARGET),
    ]
    result = CaseNodes(deps).tools(state)
    assert (result["calls_without_risk"], result["wrap_up_hinted"]) == (1, False)


def wrap_up_hints(messages):
    return [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage) and "足以結案" in message.text
    ]


def test_wrap_up_hint_follows_tool_messages_and_fires_once(settings):
    hinting = settings.model_copy(update={"wrap_up_hint_after": 2})
    chain = StubChain(
        transfers={
            TARGET: [transfer(1, TARGET, UNKNOWN)],
            UNKNOWN: [transfer(2, UNKNOWN, EXCHANGE)],
        }
    )
    script = [
        ai_tool_call("get_counterparties", "c1", address=TARGET),
        ai_tool_call("get_counterparties", "c2", address=UNKNOWN),
        ai_tool_call("lookup_address", "c3", address=UNKNOWN),
        ai_text("完成"),
    ]
    deps = fake_deps(hinting, chain, LabelStore([]), script, [draft("LOW")], auto_approve=True)
    final = build_graph(deps).invoke(
        initial_state("case-10", TARGET, hinting), case_config("case-10")
    )
    hints = wrap_up_hints(final["messages"])
    assert len(hints) == 1
    index, hint = hints[0]
    assert "已連續 2 次查詢未發現風險跡象" in hint.text
    previous = final["messages"][index - 1]
    assert isinstance(previous, ToolMessage)
    assert previous.tool_call_id == "c2"
    assert (final["calls_without_risk"], final["wrap_up_hinted"]) == (3, True)
    assert "wrap_up_hint" in audit_events(hinting, "case-10")


def test_newly_found_risky_address_resets_counter(settings):
    hinting = settings.model_copy(update={"wrap_up_hint_after": 2})
    chain = StubChain(
        transfers={TARGET: [transfer(1, TARGET, UNKNOWN)], UNKNOWN: [transfer(2, MIXER, UNKNOWN)]}
    )
    script = [
        ai_tool_call("get_counterparties", "c1", address=TARGET),
        ai_tool_call("get_counterparties", "c2", address=UNKNOWN),
        ai_tool_call("lookup_address", "c3", address=UNKNOWN),
        ai_text("完成"),
    ]
    labels = LabelStore([label(MIXER, "mixer")])
    deps = fake_deps(hinting, chain, labels, script, [draft("LOW")], auto_approve=True)
    final = build_graph(deps).invoke(
        initial_state("case-11", TARGET, hinting), case_config("case-11")
    )
    assert wrap_up_hints(final["messages"]) == []
    assert (final["calls_without_risk"], final["wrap_up_hinted"]) == (1, False)


def test_request_more_resets_wrap_up_hint(settings, open_db):
    hinting = settings.model_copy(update={"wrap_up_hint_after": 1})
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    script = [
        ai_tool_call("get_counterparties", "c1", address=TARGET),
        ai_text("初步結論"),
        ai_tool_call("lookup_address", "c2", address=UNKNOWN),
        ai_text("補查完成"),
    ]
    deps = fake_deps(hinting, chain, LabelStore([]), script, [draft("LOW"), draft("LOW")])
    config = case_config("case-12")
    graph = build_graph(deps, open_db())
    graph.invoke(initial_state("case-12", TARGET, hinting), config)
    more = {"decision": "request_more", "comment": "請確認對手", "reviewer": "tester"}
    graph.invoke(Command(resume=more), config)
    final = graph.invoke(Command(resume=APPROVE), config)
    hints = wrap_up_hints(final["messages"])
    assert len(hints) == 2
    assert [final["messages"][index - 1].tool_call_id for index, _ in hints] == ["c1", "c2"]


def test_wrap_up_hint_crossed_on_first_call_fires_once_after_last_tool_message(settings):
    hinting = settings.model_copy(update={"wrap_up_hint_after": 1})
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
    deps = fake_deps(hinting, chain, LabelStore([]), script, [draft("LOW")], auto_approve=True)
    final = build_graph(deps).invoke(
        initial_state("case-14", TARGET, hinting), case_config("case-14")
    )
    hints = wrap_up_hints(final["messages"])
    assert len(hints) == 1
    index, _ = hints[0]
    previous = final["messages"][index - 1]
    assert isinstance(previous, ToolMessage)
    assert previous.tool_call_id == "c2"


def test_budget_skipped_calls_do_not_increase_streak(settings):
    tight = settings.model_copy(update={"max_tool_calls": 1, "wrap_up_hint_after": 5})
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_counterparties", "args": {"address": TARGET}, "id": "c1"},
            {"name": "get_counterparties", "args": {"address": UNKNOWN}, "id": "c2"},
        ],
        usage_metadata=AGENT_USAGE,
    )
    script = [both, ai_text("完成")]
    deps = fake_deps(tight, chain, LabelStore([]), script, [draft("LOW")], auto_approve=True)
    final = build_graph(deps).invoke(
        initial_state("case-15", TARGET, tight), case_config("case-15")
    )
    assert final["tool_calls"] == 1
    assert final["calls_without_risk"] == 1


def test_reparented_already_known_risky_node_does_not_reset_streak(settings):
    chain = StubChain(
        transfers={TARGET: [transfer(1, EXCHANGE, TARGET), transfer(2, TARGET, MIXER)]}
    )
    labels = LabelStore([label(MIXER, "mixer")])
    deps = fake_deps(settings, chain, labels, auto_approve=True)
    state = initial_state("case-16", TARGET, settings)
    state["nodes"] = {
        **state["nodes"],
        TARGET: AddressNode(address=TARGET, depth=0),
        EXCHANGE: AddressNode(address=EXCHANGE, depth=1, parent=TARGET),
        MIXER: AddressNode(address=MIXER, depth=2, parent=EXCHANGE, label=label(MIXER, "mixer")),
    }
    state["calls_without_risk"] = 2
    state["messages"] = [
        *state["messages"],
        ai_tool_call("get_counterparties", "c1", address=TARGET),
    ]
    result = CaseNodes(deps).tools(state)
    assert result["calls_without_risk"] == 3
    assert (result["nodes"][MIXER].depth, result["nodes"][MIXER].parent) == (1, TARGET)


def test_same_batch_calls_get_distinct_aliases(settings):
    chain = StubChain(
        transfers={
            TARGET: [transfer(1, TARGET, UNKNOWN)],
            UNKNOWN: [transfer(1, TARGET, UNKNOWN), transfer(2, UNKNOWN, EXCHANGE)],
        }
    )
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_counterparties", "args": {"address": TARGET}, "id": "c1"},
            {"name": "get_counterparties", "args": {"address": UNKNOWN}, "id": "c2"},
        ],
        usage_metadata=AGENT_USAGE,
    )
    deps = fake_deps(settings, chain, LabelStore([]), auto_approve=True)
    state = initial_state("case-17", TARGET, settings)
    state["nodes"] = {TARGET: AddressNode(address=TARGET, depth=0)}
    state["messages"] = [*state["messages"], both]
    result = CaseNodes(deps).tools(state)
    assert tx_refs(result["evidence"]) == {tx_hash(1): "T1", tx_hash(2): "T2"}
    replies = {m.tool_call_id: m.text for m in result["messages"] if m.type == "tool"}
    assert "例：T1" in replies["c1"]
    assert "例：T2" in replies["c2"]
    assert "例：T1" in replies["c2"]


def test_evidence_from_old_checkpoint_deserializes_with_ref_fallback():
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    old = {"id": tx_hash(1), "kind": "tx", "summary": "s"}
    payload = ("kyt_agent.models", "Evidence", old, "model_validate_json")
    blob = ormsgpack.packb(ormsgpack.Ext(5, ormsgpack.packb(payload)))
    restored = serde.loads_typed(("msgpack", blob))
    assert isinstance(restored, Evidence)
    assert (restored.id, restored.ref) == (tx_hash(1), tx_hash(1))


def test_audit_log_maps_every_alias_to_a_hash(settings):
    chain = StubChain(
        transfers={
            TARGET: [transfer(1, TARGET, UNKNOWN), transfer(2, EXCHANGE, TARGET)],
            UNKNOWN: [transfer(1, TARGET, UNKNOWN), transfer(3, UNKNOWN, MIXER)],
        },
        transactions={
            tx_hash(3): TransactionDetail(
                tx_hash=tx_hash(3),
                sender=UNKNOWN,
                recipient=MIXER,
                value_eth=Decimal("1"),
                block_number=1,
                method_id="0x",
            )
        },
    )
    script = [
        ai_tool_call("get_counterparties", "c1", address=TARGET),
        ai_tool_call("get_counterparties", "c2", address=UNKNOWN),
        ai_tool_call("get_transaction", "c3", tx="t3"),
        ai_text("完成"),
    ]
    deps = fake_deps(settings, chain, LabelStore([]), script, [draft("LOW")], auto_approve=True)
    build_graph(deps).invoke(initial_state("case-18", TARGET, settings), case_config("case-18"))

    path = settings.var_dir / "audit" / "case-18.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    calls = [entry for entry in entries if entry["event"] == "tool_call"]
    mapping = {alias: ref for entry in calls for alias, ref in entry["evidence"].items()}
    # 兩個交易對手筆數相同時依地址排序，EXCHANGE 在前
    assert mapping == {"T1": tx_hash(2), "T2": tx_hash(1), "T3": tx_hash(3)}
    assert calls[1]["evidence"] == {"T3": tx_hash(3)}
    seen = {alias for entry in calls for alias in re.findall(r"T\d+", entry["result_preview"])}
    assert seen and seen <= mapping.keys()
    assert calls[2]["resolved_tx"] == tx_hash(3)
