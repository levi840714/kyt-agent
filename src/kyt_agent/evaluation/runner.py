import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from kyt_agent.chain.factory import make_chain_client
from kyt_agent.config import Settings
from kyt_agent.evaluation.crawler import baseline_level, crawl
from kyt_agent.evaluation.dataset import EvalCase
from kyt_agent.evaluation.metrics import CaseResult, Summary, summarize
from kyt_agent.graph.build import build_graph, case_config
from kyt_agent.graph.deps import Deps, make_deps
from kyt_agent.graph.state import initial_state
from kyt_agent.labels import LabelStore

EVAL_DEPTH = 2


def record_snapshots(
    settings: Settings, cases: list[EvalCase], on_recorded: Callable[[EvalCase, int], None]
) -> None:
    chain = make_chain_client(settings.model_copy(update={"chain_mode": "record"}))
    labels = LabelStore.from_dir(settings.data_dir / "labels")
    for case in cases:
        on_recorded(
            case, crawl(chain, labels, case.address, EVAL_DEPTH, settings.top_counterparties)
        )


def run_eval(
    settings: Settings,
    cases: list[EvalCase],
    model: str | None,
    on_result: Callable[[CaseResult], None],
) -> tuple[Summary, list[CaseResult]]:
    replay = settings.model_copy(update={"chain_mode": "replay", "max_depth": EVAL_DEPTH})
    deps = make_deps(replay, model=model, auto_approve=True)
    graph = build_graph(deps)
    results: list[CaseResult] = []
    for case in cases:
        result = run_case(graph, deps, case)
        on_result(result)
        results.append(result)
    return summarize(deps.model_name, results), results


def run_case(graph: CompiledStateGraph, deps: Deps, case: EvalCase) -> CaseResult:
    baseline = baseline_level(deps.chain, deps.labels, case.address)
    case_id = f"eval-{uuid.uuid4().hex[:12]}"
    try:
        final = graph.invoke(
            initial_state(case_id, case.address, deps.settings), case_config(case_id)
        )
    except Exception as error:  # 單一案例失敗不應中斷整批 eval
        return CaseResult(
            address=case.address,
            expected=case.expected,
            category=case.category,
            predicted=None,
            baseline=baseline,
            error=f"{type(error).__name__}: {error}",
        )
    return CaseResult(
        address=case.address,
        expected=case.expected,
        category=case.category,
        predicted=final["report"].risk_level,
        baseline=baseline,
        tool_calls=final["tool_calls"],
        input_tokens=final["input_tokens"],
        output_tokens=final["output_tokens"],
        snapshot_misses=final["snapshot_misses"],
    )


def write_eval_result(directory: Path, summary: Summary, results: list[CaseResult]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model = summary.model.replace(":", "_").replace("/", "_")
    path = directory / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{model}.json"
    body = {"summary": summary.model_dump(), "results": [r.model_dump() for r in results]}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
