import sqlite3
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kyt_agent.graph.deps import Deps
from kyt_agent.graph.nodes import (
    CaseNodes,
    route_after_agent,
    route_after_guard,
    route_after_review,
    route_after_screen,
)
from kyt_agent.graph.state import CaseState

RECURSION_LIMIT = 500
CHECKPOINT_TYPES = [
    *(
        ("kyt_agent.models", name)
        for name in ("AddressNode", "Evidence", "Label", "Review", "RuleHit")
    ),
    *(("kyt_agent.report", name) for name in ("Finding", "FundPath", "PathHop", "RiskReport")),
]


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    nodes = CaseNodes(deps)
    graph = StateGraph(CaseState)
    graph.add_node("screen", nodes.screen)
    graph.add_node("agent", nodes.agent)
    graph.add_node("budget_guard", nodes.budget_guard)
    graph.add_node("tools", nodes.tools)
    graph.add_node("report", nodes.report)
    graph.add_node("review", nodes.review)
    graph.add_edge(START, "screen")
    graph.add_conditional_edges("screen", route_after_screen, ["agent", "report"])
    graph.add_conditional_edges("agent", route_after_agent, ["budget_guard", "report"])
    graph.add_conditional_edges("budget_guard", route_after_guard, ["tools", "report"])
    graph.add_edge("tools", "agent")
    graph.add_edge("report", "review")
    graph.add_conditional_edges("review", route_after_review, ["agent", END])
    return graph.compile(checkpointer=checkpointer)


def open_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(
        connection, serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    )


def case_config(case_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": case_id}, "recursion_limit": RECURSION_LIMIT}
