from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from kyt_agent.audit import AuditLog
from kyt_agent.chain.client import ChainClient
from kyt_agent.chain.factory import make_chain_client
from kyt_agent.config import Settings
from kyt_agent.graph.tools import tool_schemas
from kyt_agent.labels import LabelStore
from kyt_agent.report import ReportDraft
from kyt_agent.tokens import TokenRegistry


@dataclass(frozen=True)
class Deps:
    settings: Settings
    chain: ChainClient
    labels: LabelStore
    tokens: TokenRegistry
    audit: AuditLog
    agent_model: Runnable[Any, AIMessage]
    drafter: Runnable[Any, Any]
    model_name: str
    auto_approve: bool = False


# 網路卡住時讓單次 LLM 呼叫逾時失敗，避免整個案件無限期等待
LLM_MAX_RETRIES = 2


def make_deps(settings: Settings, *, model: str | None = None, auto_approve: bool = False) -> Deps:
    model_name = model or settings.llm_model
    llm = init_chat_model(model_name, timeout=settings.llm_timeout, max_retries=LLM_MAX_RETRIES)
    return Deps(
        settings=settings,
        chain=make_chain_client(settings),
        labels=LabelStore.from_dir(settings.data_dir / "labels"),
        tokens=TokenRegistry.from_csv(settings.data_dir / "tokens.csv"),
        audit=AuditLog(settings.var_dir / "audit"),
        agent_model=llm.bind_tools(tool_schemas()),
        drafter=llm.with_structured_output(ReportDraft, include_raw=True),
        model_name=model_name,
        auto_approve=auto_approve,
    )
