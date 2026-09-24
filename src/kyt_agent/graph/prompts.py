from kyt_agent.config import Settings
from kyt_agent.report import RiskReport

INVESTIGATION_SYSTEM = """你是虛擬資產服務商（VASP）的 AML 調查員，
負責調查 Ethereum 地址的洗錢與制裁風險。

## 工具
- get_counterparties：查詢地址的交易對手彙整，可指定 in（資金來源）、out（資金去向）或 both
- lookup_address：查詢地址的標籤，以及是否為合約
- get_transaction：查詢單筆交易細節

## 調查策略
- 先查目標地址的交易對手，再依風險決定是否往下一層追
- 遇到混幣器、跨鏈橋、駭客或制裁相關地址，或資金快速分散、聚合的未知地址，應深入追蹤
- 遇到交易所熱錢包、主流 DeFi 合約時停止追蹤，它們是資金的正常出入口
- 只能查詢目標地址，或工具結果中出現過的地址與交易
- 預算有限，優先追蹤最可能揭露風險的路徑，避免重複查詢

## 結束
證據足以判斷風險時，不要再呼叫工具，直接用幾句話說明結論。"""

REPORT_INSTRUCTION = """調查已結束，請依據上述工具結果產出風險報告。

規則：
- 每項 finding 必須引用證據 ID：工具結果中出現的 tx hash，或 label:<address> 形式的標籤證據
- 不得引用工具結果中沒有出現的交易或地址
- 風險等級：LOW 無明顯風險；MEDIUM 間接接觸高風險實體；
  HIGH 直接或近距離接觸制裁、駭客、混幣器資金；SEVERE 本身為制裁或犯罪地址
- limitations 需列出未能查證的部分"""


def target_message(target: str, settings: Settings) -> str:
    return (
        f"請調查地址 {target}。\n"
        f"限制：最多追蹤 {settings.max_depth} 層、"
        f"{settings.max_tool_calls} 次工具呼叫、展開 {settings.max_addresses} 個地址。"
    )


def report_request(label_evidence: list[str], rule_hits: list[str], budget_note: str | None) -> str:
    parts = [REPORT_INSTRUCTION]
    if rule_hits:
        parts.append("規則層命中：\n" + "\n".join(f"- {hit}" for hit in rule_hits))
    if label_evidence:
        parts.append("可引用的標籤證據：\n" + "\n".join(f"- {item}" for item in label_evidence))
    if budget_note:
        parts.append(f"注意：調查因預算限制提前結束（{budget_note}），請在 limitations 中註明。")
    return "\n\n".join(parts)


def evidence_retry_message(invalid: list[int]) -> str:
    numbers = "、".join(str(index + 1) for index in invalid)
    return f"第 {numbers} 項 finding 引用了工具結果中不存在的證據 ID，請修正後重新輸出完整報告。"


def supplement_message(report: RiskReport, comment: str) -> str:
    return (
        f"合規人員審閱了第 {report.version} 版報告"
        f"（風險等級 {report.risk_level}：{report.summary}），要求補查：\n"
        f"{comment}\n\n請依此意見繼續調查。"
    )
