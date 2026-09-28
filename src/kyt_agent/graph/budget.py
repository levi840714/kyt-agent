from kyt_agent.config import Settings
from kyt_agent.graph.state import CaseState


def exhausted_reason(state: CaseState, settings: Settings) -> str | None:
    if state["tool_calls"] >= state["tool_call_limit"]:
        return f"工具呼叫次數已達上限（{state['tool_call_limit']}）"
    if state["input_tokens"] + state["output_tokens"] >= settings.max_tokens:
        return f"LLM token 用量已達上限（{settings.max_tokens}）"
    return None
