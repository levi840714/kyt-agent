from kyt_agent.graph.budget import exhausted_reason
from kyt_agent.graph.prompts import report_request
from kyt_agent.graph.state import initial_state
from tests.fakes import MIXED_CASE, TARGET


def test_initial_state(settings):
    state = initial_state("case-1", MIXED_CASE, settings)
    assert state["target"] == MIXED_CASE.lower()
    assert state["tool_call_limit"] == settings.max_tool_calls
    assert (state["review_round"], state["status"]) == (1, "investigating")
    assert MIXED_CASE.lower() in state["messages"][1].content
    assert str(settings.max_depth) in state["messages"][1].content


def test_exhausted_reason(settings):
    state = initial_state("case-1", TARGET, settings)
    assert exhausted_reason(state, settings) is None
    used_up = {**state, "tool_calls": settings.max_tool_calls}
    assert "工具呼叫次數已達上限" in exhausted_reason(used_up, settings)
    token_heavy = {**state, "input_tokens": settings.max_tokens}
    assert "token" in exhausted_reason(token_heavy, settings)


def test_report_request_lists_label_evidence_and_budget_note():
    text = report_request(
        label_evidence=["label:0xabc：0xabc 為 mixer"],
        rule_hits=[],
        budget_note="工具呼叫次數已達上限（25）",
    )
    assert "label:0xabc" in text
    assert "預算限制" in text
