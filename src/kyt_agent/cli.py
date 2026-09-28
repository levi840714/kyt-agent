import re
from contextlib import closing
from datetime import UTC, datetime
from typing import Any

import typer
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from rich.console import Console

from kyt_agent import render
from kyt_agent.audit import AuditLog
from kyt_agent.chain.snapshot import SnapshotMissError
from kyt_agent.config import PROJECT_ROOT, Settings
from kyt_agent.evaluation.dataset import load_dataset
from kyt_agent.evaluation.runner import record_snapshots, run_eval, write_eval_result
from kyt_agent.graph.build import build_graph, case_config, open_checkpointer
from kyt_agent.graph.deps import make_deps
from kyt_agent.graph.nodes import ReportError
from kyt_agent.graph.state import initial_state
from kyt_agent.labels import fetch_ofac_labels, write_labels_csv

ADDRESS_PATTERN = re.compile(r"^0x[0-9a-fA-F]{40}$")

app = typer.Typer(help="鏈上地址風險調查 Agent（簡易版 KYT）", no_args_is_help=True)
labels_app = typer.Typer(help="標籤庫管理", no_args_is_help=True)
app.add_typer(labels_app, name="labels")
console = Console()


@app.callback()
def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


@app.command()
def investigate(address: str) -> None:
    """開新案件調查指定地址。"""
    if not ADDRESS_PATTERN.match(address):
        raise typer.BadParameter("地址須為 0x 開頭的 40 位十六進位字串")
    settings = Settings()
    case_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{address[2:8].lower()}"
    console.print(f"案件編號：[bold]{case_id}[/]")
    _run_case(settings, case_id, initial_state(case_id, address, settings))


@app.command()
def resume(case_id: str) -> None:
    """接續中斷的案件。"""
    _run_case(Settings(), case_id, None)


@labels_app.command("sync-ofac")
def sync_ofac() -> None:
    """從公開清單更新 OFAC 制裁地址。"""
    settings = Settings()
    labels = fetch_ofac_labels()
    path = settings.data_dir / "labels" / "ofac.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_labels_csv(path, labels)
    console.print(f"已寫入 {len(labels)} 筆 OFAC 制裁地址至 {path}")


@app.command("eval")
def evaluate(
    record: bool = typer.Option(False, "--record", help="錄製 Etherscan 快照"),
    model: str | None = typer.Option(
        None, "--model", help="覆蓋 LLM_MODEL，格式 <provider>:<model>"
    ),
) -> None:
    """錄製快照，或以快照重播執行 eval。"""
    settings = Settings()
    dataset_path = PROJECT_ROOT / "eval" / "dataset.jsonl"
    try:
        cases = load_dataset(dataset_path)
    except FileNotFoundError as error:
        console.print(f"[red]找不到 eval 資料集：{dataset_path}[/]")
        raise typer.Exit(1) from error
    if record:
        record_snapshots(
            settings,
            cases,
            lambda case, count: console.print(f"已錄製 {case.address}：{count} 個地址"),
        )
        return
    try:
        summary, results = run_eval(
            settings, cases, model, lambda result: console.print(render.eval_row(result))
        )
    except SnapshotMissError as error:
        console.print(f"[red]缺少快照 {error}，請先執行 kyt eval --record[/]")
        raise typer.Exit(1) from error
    path = write_eval_result(settings.var_dir / "eval", summary, results)
    render.show_eval_summary(console, summary)
    console.print(f"結果已寫入 {path}")


def _run_case(settings: Settings, case_id: str, graph_input: Any) -> None:
    saver = open_checkpointer(settings.var_dir / "checkpoints.sqlite")
    with closing(saver.conn):
        graph = build_graph(make_deps(settings), saver)
        config = case_config(case_id)
        if graph_input is None:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise typer.BadParameter(f"找不到案件 {case_id}")
            if not snapshot.next:
                console.print(f"案件已結案：{snapshot.values['status']}")
                return
        try:
            _drive(graph, config, graph_input)
        except ReportError as error:
            AuditLog(settings.var_dir / "audit").record(case_id, "case_error", error=str(error))
            console.print(f"[red]案件失敗：{error}[/]")
            raise typer.Exit(1) from error


def _drive(graph: CompiledStateGraph, config: RunnableConfig, graph_input: Any) -> None:
    if graph_input is not None or not graph.get_state(config).interrupts:
        _stream(graph, config, graph_input)
    while (snapshot := graph.get_state(config)).interrupts:
        payload = snapshot.interrupts[0].value
        render.show_report(console, payload)
        decision = render.ask_decision(console, allow_more=payload["allow_more"])
        _stream(graph, config, Command(resume=decision))
    render.show_closed(console, snapshot.values)


def _stream(graph: CompiledStateGraph, config: RunnableConfig, graph_input: Any) -> None:
    with console.status("調查中…"):
        for update in graph.stream(graph_input, config, stream_mode="updates"):
            render.show_progress(console, update)
