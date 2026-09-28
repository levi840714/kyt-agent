from pathlib import Path

from pydantic import BaseModel, Field

from kyt_agent.labels import LabelStore
from kyt_agent.models import Review, RiskLevel, max_risk


class Finding(BaseModel):
    claim: str = Field(description="一項具體的風險發現")
    evidence: list[str] = Field(
        min_length=1, description="支持此發現的證據 ID：tx hash 或 label:<address>"
    )


class PathHop(BaseModel):
    address: str
    label: str | None = Field(default=None, description="已知標籤，未知則留空")


class FundPath(BaseModel):
    hops: list[PathHop] = Field(description="從目標地址開始的資金路徑")
    note: str = Field(description="這條路徑代表的意義")


class ReportDraft(BaseModel):
    """地址風險調查報告。"""

    risk_level: RiskLevel
    summary: str
    findings: list[Finding]
    fund_paths: list[FundPath]
    recommendation: str
    limitations: list[str]


class RiskReport(ReportDraft):
    llm_risk_level: RiskLevel
    risk_floor: RiskLevel
    unverified_findings: list[int]
    version: int


def unknown_evidence(draft: ReportDraft, evidence_ids: set[str]) -> list[int]:
    return [
        index
        for index, finding in enumerate(draft.findings)
        if not {item.lower() for item in finding.evidence} <= evidence_ids
    ]


def finalize(
    draft: ReportDraft,
    *,
    floor: RiskLevel,
    unverified: list[int],
    version: int,
    labels: LabelStore,
) -> RiskReport:
    return RiskReport(
        **draft.model_dump(exclude={"risk_level", "fund_paths"}),
        risk_level=max_risk(draft.risk_level, floor),
        fund_paths=[_relabel(path, labels) for path in draft.fund_paths],
        llm_risk_level=draft.risk_level,
        risk_floor=floor,
        unverified_findings=unverified,
        version=version,
    )


def _relabel(path: FundPath, labels: LabelStore) -> FundPath:
    # LLM 填的標籤可能是編造的，一律以標籤庫覆寫
    hops = []
    for hop in path.hops:
        label = labels.get(hop.address)
        text = f"{label.category}: {label.name}" if label else None
        hops.append(PathHop(address=hop.address, label=text))
    return FundPath(hops=hops, note=path.note)


def to_markdown(report: RiskReport, target: str, reviews: list[Review]) -> str:
    lines = [
        f"# 地址風險報告：{target}",
        "",
        f"- 風險等級：**{report.risk_level}**"
        f"（LLM 判定 {report.llm_risk_level}，規則下限 {report.risk_floor}）",
        f"- 報告版本：{report.version}",
        "",
        "## 摘要",
        "",
        report.summary,
        "",
        "## 發現",
        "",
    ]
    for index, finding in enumerate(report.findings):
        mark = "（未驗證）" if index in report.unverified_findings else ""
        lines.append(f"{index + 1}. {finding.claim}{mark}  ")
        lines.append(f"   證據：{', '.join(finding.evidence)}")
    lines += ["", "## 資金路徑", ""]
    for path in report.fund_paths:
        hops = " → ".join(
            f"{hop.address}（{hop.label}）" if hop.label else hop.address for hop in path.hops
        )
        lines.append(f"- {hops}：{path.note}")
    lines += ["", "## 建議", "", report.recommendation, "", "## 調查限制", ""]
    lines += [f"- {item}" for item in report.limitations] or ["- 無"]
    lines += ["", "## 審核紀錄", ""]
    lines += [f"- 第 {r.round} 輪 {r.reviewer}：{r.decision} {r.comment}".rstrip() for r in reviews]
    return "\n".join(lines) + "\n"


def write_case_files(
    directory: Path, target: str, report: RiskReport, reviews: list[Review]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (directory / "report.md").write_text(to_markdown(report, target, reviews), encoding="utf-8")
