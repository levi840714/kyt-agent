import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

from kyt_agent.config import Settings
from kyt_agent.graph import prompts
from kyt_agent.models import AddressNode, Evidence, Review, RuleHit
from kyt_agent.report import RiskReport

CaseStatus = Literal["investigating", "approved", "rejected"]


def merge_dicts[V](left: dict[str, V], right: dict[str, V]) -> dict[str, V]:
    return {**left, **right}


class CaseState(TypedDict):
    case_id: str
    target: str
    messages: Annotated[list[AnyMessage], add_messages]
    nodes: Annotated[dict[str, AddressNode], merge_dicts]
    evidence: Annotated[dict[str, Evidence], merge_dicts]
    rule_hits: list[RuleHit]
    tool_calls: int
    tool_call_limit: int
    input_tokens: int
    output_tokens: int
    snapshot_misses: int
    budget_note: str | None
    report: RiskReport | None
    review_round: int
    reviews: Annotated[list[Review], operator.add]
    status: CaseStatus


def initial_state(case_id: str, target: str, settings: Settings) -> CaseState:
    target = target.lower()
    return CaseState(
        case_id=case_id,
        target=target,
        messages=[
            SystemMessage(prompts.INVESTIGATION_SYSTEM),
            HumanMessage(prompts.target_message(target, settings)),
        ],
        nodes={},
        evidence={},
        rule_hits=[],
        tool_calls=0,
        tool_call_limit=settings.max_tool_calls,
        input_tokens=0,
        output_tokens=0,
        snapshot_misses=0,
        budget_note=None,
        report=None,
        review_round=1,
        reviews=[],
        status="investigating",
    )
