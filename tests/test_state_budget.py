from kyt_agent.graph.budget import exhausted_reason
from kyt_agent.graph.prompts import INVESTIGATION_SYSTEM, REPORT_INSTRUCTION, report_request
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


def test_prompts_restrict_judgement_to_tool_facts_and_explain_flags():
    for prompt in (INVESTIGATION_SYSTEM, REPORT_INSTRUCTION):
        assert "只能依據工具結果與標籤庫" in prompt
        assert "不得稱為受制裁" in prompt
        assert "偽冒代幣" in prompt
        assert "粉塵" in prompt


def test_report_instruction_defines_indirect_high_via_intermediary():
    assert "不符合 HIGH 條件者" in REPORT_INSTRUCTION
    assert "經由中間地址間接接觸" in REPORT_INSTRUCTION
    assert "超過一半" in REPORT_INSTRUCTION
