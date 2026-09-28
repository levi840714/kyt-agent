from collections.abc import Iterable
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from kyt_agent.audit import AuditLog
from kyt_agent.chain.client import ChainClient
from kyt_agent.config import Settings
from kyt_agent.graph.deps import Deps
from kyt_agent.labels import LabelStore
from kyt_agent.report import ReportDraft

AGENT_USAGE = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
REPORT_USAGE = {"input_tokens": 200, "output_tokens": 50, "total_tokens": 250}


class ScriptedModel(GenericFakeChatModel):
    """依序回傳預設訊息，忽略 bind_tools。"""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def ai_tool_call(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
        usage_metadata=AGENT_USAGE,
    )


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text, usage_metadata=AGENT_USAGE)


def scripted_drafter(drafts: Iterable[ReportDraft | None]) -> Runnable[Any, dict[str, Any]]:
    queue = iter(drafts)

    def respond(_: Any) -> dict[str, Any]:
        raw = AIMessage(content="", usage_metadata=REPORT_USAGE)
        parsed = next(queue)
        error = None if parsed is not None else ValueError("invalid json")
        return {"raw": raw, "parsed": parsed, "parsing_error": error}

    return RunnableLambda(respond)


def fake_deps(
    settings: Settings,
    chain: ChainClient,
    labels: LabelStore,
    script: Iterable[AIMessage] = (),
    drafts: Iterable[ReportDraft | None] = (),
    auto_approve: bool = False,
) -> Deps:
    return Deps(
        settings=settings,
        chain=chain,
        labels=labels,
        audit=AuditLog(settings.var_dir / "audit"),
        agent_model=ScriptedModel(messages=iter(list(script))),
        drafter=scripted_drafter(drafts),
        model_name="fake:model",
        auto_approve=auto_approve,
    )
