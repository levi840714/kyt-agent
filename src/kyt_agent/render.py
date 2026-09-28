import getpass
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

from kyt_agent.evaluation.metrics import FLAGGED, CaseResult, Summary
from kyt_agent.report import RiskReport

RISK_STYLE = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "SEVERE": "bold white on red"}


def show_progress(console: Console, update: dict[str, Any]) -> None:
    for node, data in update.items():
        if node == "agent":
            for call in data["messages"][-1].tool_calls:
                args = ", ".join(f"{key}={value}" for key, value in call["args"].items())
                console.print(f"  [cyan]→ {call['name']}[/]({escape(args)})")
        elif node == "budget_guard" and data.get("budget_note"):
            console.print(f"  [yellow]預算用盡：{escape(data['budget_note'])}[/]")
        elif node == "report":
            console.print(f"  [green]報告第 {data['report'].version} 版已產生[/]")


def show_report(console: Console, payload: dict[str, Any]) -> None:
    error = payload.get("error")
    if error:
        console.print(f"[bold red]輸入有誤：{escape(error)}[/]")
    report = RiskReport.model_validate(payload["report"])
    style = RISK_STYLE[report.risk_level]
    title = f"{payload['target']}｜風險 [{style}]{report.risk_level}[/]｜第 {report.version} 版"
    console.print(
        Panel(
            escape(report.summary),
            title=title,
            subtitle=f"LLM {report.llm_risk_level}／規則下限 {report.risk_floor}",
        )
    )
    findings = Table("#", "發現", "證據", title="發現")
    for index, finding in enumerate(report.findings):
        claim = escape(finding.claim)
        if index in report.unverified_findings:
            claim = f"[red]（未驗證）[/]{claim}"
        findings.add_row(str(index + 1), claim, "\n".join(finding.evidence))
    console.print(findings)
    tree = Tree("資金路徑")
    for path in report.fund_paths:
        branch = tree.add(escape(path.note))
        for hop in path.hops:
            branch = branch.add(f"{hop.address} [dim]{escape(hop.label or '未知')}[/]")
    console.print(tree)
    console.print(Panel(escape(report.recommendation), title="建議"))
    if report.limitations:
        limitations = "\n".join(f"• {escape(item)}" for item in report.limitations)
        console.print(Panel(limitations, title="調查限制"))


def ask_decision(console: Console, *, allow_more: bool) -> dict[str, str]:
    choices = {"a": "approve", "r": "reject"}
    if allow_more:
        choices["m"] = "request_more"
    hint = "a=核准 r=駁回" + (" m=要求補查" if allow_more else "")
    decision = choices[Prompt.ask(f"審核決策（{hint}）", choices=list(choices), console=console)]
    required = decision != "approve"
    comment = Prompt.ask(
        "意見（必填）" if required else "意見（選填）", default="", console=console
    )
    while required and not comment.strip():
        comment = Prompt.ask("意見（必填）", console=console)
    return {"decision": decision, "comment": comment, "reviewer": getpass.getuser()}


def show_closed(console: Console, values: dict[str, Any]) -> None:
    case_id = values["case_id"]
    console.print(
        f"[bold]案件 {case_id} 結案：{values['status']}[/]，報告存於 var/cases/{case_id}/"
    )


def eval_row(result: CaseResult) -> str:
    if result.error:
        return f"[red]✗[/] {result.address} 錯誤：{escape(result.error)}"
    mark = "✓" if (result.predicted in FLAGGED) == (result.expected == "risky") else "✗"
    return (
        f"{mark} {result.address} {result.category}｜預期 {result.expected}"
        f"｜agent {result.predicted}"
        f"｜規則 {result.baseline or '-'}｜工具 {result.tool_calls} 次"
        f"｜token {result.input_tokens + result.output_tokens}"
    )


def show_eval_summary(console: Console, summary: Summary) -> None:
    table = Table("指標", "Agent", "純規則", title=f"Eval：{summary.model}")
    table.add_row("召回率", _pct(summary.recall), _pct(summary.baseline_recall))
    table.add_row(
        "誤報率", _pct(summary.false_positive_rate), _pct(summary.baseline_false_positive_rate)
    )
    table.add_row("平均工具呼叫", f"{summary.avg_tool_calls:.1f}", "-")
    table.add_row("平均 token", f"{summary.avg_tokens:,.0f}", "-")
    table.add_row("乾淨地址平均 token", f"{summary.avg_clean_tokens:,.0f}", "-")
    cost = "-" if summary.estimated_cost_usd is None else f"${summary.estimated_cost_usd:.4f}"
    table.add_row("估算成本", cost, "-")
    table.add_row("錯誤 / 快照缺漏", f"{summary.errors} / {summary.snapshot_misses}", "-")
    console.print(table)
    categories = Table(
        "類別",
        "預期",
        "筆數",
        "Agent 攔下率",
        "純規則攔下率",
        "平均 token",
        title="依類別",
        caption="攔下率：陽性類別即召回率，陰性類別即誤報率",
    )
    for name, metrics in summary.by_category.items():
        categories.add_row(
            name,
            metrics.expected,
            str(metrics.cases),
            _pct(metrics.flag_rate),
            _pct(metrics.baseline_flag_rate),
            f"{metrics.avg_tokens:,.0f}",
        )
    console.print(categories)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"
