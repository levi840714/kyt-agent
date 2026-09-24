from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from kyt_agent.chain.client import ChainClient
from kyt_agent.chain.etherscan import EtherscanError
from kyt_agent.chain.snapshot import SnapshotMissError
from kyt_agent.config import Settings
from kyt_agent.counterparties import (
    Counterparty,
    Direction,
    select_counterparties,
    summarize_counterparties,
)
from kyt_agent.labels import LabelStore
from kyt_agent.models import AddressNode, Evidence, Label

_UNKNOWN_ADDRESS = "拒絕：只能查詢目標地址，或工具結果中出現過的地址"


class GetCounterpartiesArgs(BaseModel):
    address: str = Field(description="要查詢的地址")
    direction: Direction = Field(
        default="both", description="in：資金來源，out：資金去向，both：兩者"
    )


class LookupAddressArgs(BaseModel):
    address: str = Field(description="要查詢的地址")


class GetTransactionArgs(BaseModel):
    tx_hash: str = Field(description="交易 hash")


_TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "get_counterparties": (
        GetCounterpartiesArgs,
        "查詢地址的交易對手彙整：筆數、金額、時間區間、範例交易與已知標籤",
    ),
    "lookup_address": (LookupAddressArgs, "查詢地址的標籤分類，以及是否為合約與合約名稱"),
    "get_transaction": (GetTransactionArgs, "查詢單筆交易的細節"),
}


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": args.model_json_schema(),
            },
        }
        for name, (args, description) in _TOOLS.items()
    ]


class ToolOutcome(BaseModel):
    content: str
    nodes: dict[str, AddressNode] = {}
    evidence: dict[str, Evidence] = {}
    miss: bool = False


class Investigator:
    def __init__(self, chain: ChainClient, labels: LabelStore, settings: Settings) -> None:
        self._chain = chain
        self._labels = labels
        self._settings = settings

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        nodes: dict[str, AddressNode],
        evidence: dict[str, Evidence],
    ) -> ToolOutcome:
        try:
            if name == "get_counterparties":
                return self._counterparties(GetCounterpartiesArgs.model_validate(args), nodes)
            if name == "lookup_address":
                return self._lookup(LookupAddressArgs.model_validate(args), nodes)
            if name == "get_transaction":
                return self._transaction(GetTransactionArgs.model_validate(args), evidence)
            return ToolOutcome(content=f"未知的工具：{name}")
        except ValidationError as error:
            return ToolOutcome(content=f"參數錯誤：{error}")
        except SnapshotMissError:
            return ToolOutcome(content="資料不可用：快照中沒有這筆資料", miss=True)
        except (EtherscanError, httpx.HTTPError) as error:
            return ToolOutcome(content=f"查詢失敗：{error}")

    def _counterparties(
        self, args: GetCounterpartiesArgs, nodes: dict[str, AddressNode]
    ) -> ToolOutcome:
        address = args.address.lower()
        node = nodes.get(address)
        if node is None:
            return ToolOutcome(content=_UNKNOWN_ADDRESS)
        if node.depth >= self._settings.max_depth:
            return ToolOutcome(content=f"拒絕：{address} 位於第 {node.depth} 層，已達最大追蹤深度")
        expanded = sum(item.expanded for item in nodes.values())
        if not node.expanded and expanded >= self._settings.max_addresses:
            return ToolOutcome(content=f"拒絕：已展開 {expanded} 個地址，達到上限")
        found = summarize_counterparties(
            address, self._chain.get_transfers(address), args.direction
        )
        shown = select_counterparties(found, self._labels, self._settings.top_counterparties)
        new_nodes = {address: node.model_copy(update={"expanded": True})}
        evidence: dict[str, Evidence] = {}
        lines = [
            f"{address}（第 {node.depth} 層）方向 {args.direction}：共 {len(found)} 個交易對手，"
            f"列出 {len(shown)} 個（已知標籤全列，其餘依互動次數排序）"
        ]
        for counterparty in shown:
            label = self._labels.get(counterparty.address)
            if counterparty.address not in nodes:
                new_nodes[counterparty.address] = AddressNode(
                    address=counterparty.address, depth=node.depth + 1, parent=address, label=label
                )
            evidence.update(_tx_evidence(address, counterparty))
            if label:
                evidence.update(label_evidence(label))
            lines.append(_format_counterparty(counterparty, label))
        return ToolOutcome(content="\n".join(lines), nodes=new_nodes, evidence=evidence)

    def _lookup(self, args: LookupAddressArgs, nodes: dict[str, AddressNode]) -> ToolOutcome:
        address = args.address.lower()
        if address not in nodes:
            return ToolOutcome(content=_UNKNOWN_ADDRESS)
        label = self._labels.get(address)
        info = self._chain.get_contract_info(address)
        kind = f"合約（{info.name or '未驗證'}）" if info.is_contract else "一般地址（EOA）"
        tag = f"{label.category}：{label.name}（來源 {label.source}）" if label else "無"
        return ToolOutcome(
            content=f"{address}\n類型：{kind}\n標籤：{tag}",
            evidence=label_evidence(label) if label else {},
        )

    def _transaction(self, args: GetTransactionArgs, evidence: dict[str, Evidence]) -> ToolOutcome:
        tx_hash = args.tx_hash.lower()
        if tx_hash not in evidence:
            return ToolOutcome(content="拒絕：只能查詢工具結果中出現過的交易")
        detail = self._chain.get_transaction(tx_hash)
        if detail is None:
            return ToolOutcome(content=f"查無交易 {tx_hash}")
        return ToolOutcome(
            content=(
                f"{detail.tx_hash}\n區塊：{detail.block_number}\n從：{detail.sender}\n"
                f"到：{detail.recipient or '（合約建立）'}\n"
                f"金額：{_amount(detail.value_eth)} ETH\n方法：{detail.method_id or '（無）'}"
            )
        )


def label_evidence(label: Label) -> dict[str, Evidence]:
    evidence_id = f"label:{label.address}"
    summary = f"{label.address} 為 {label.category}：{label.name}（來源 {label.source}）"
    return {evidence_id: Evidence(id=evidence_id, kind="label", summary=summary)}


def _tx_evidence(address: str, counterparty: Counterparty) -> dict[str, Evidence]:
    flow = {
        "in": f"{counterparty.address} → {address}",
        "out": f"{address} → {counterparty.address}",
        "both": f"{address} ↔ {counterparty.address}",
    }[counterparty.direction]
    return {
        tx_hash: Evidence(id=tx_hash, kind="tx", summary=flow)
        for tx_hash in counterparty.sample_hashes
    }


def _format_counterparty(counterparty: Counterparty, label: Label | None) -> str:
    tag = f"[{label.category}: {label.name}]" if label else "[未知]"
    totals = ", ".join(f"{_amount(value)} {asset}" for asset, value in counterparty.totals.items())
    period = f"{_date(counterparty.first_seen)}~{_date(counterparty.last_seen)}"
    samples = ", ".join(counterparty.sample_hashes)
    return (
        f"- {counterparty.address} {tag} {counterparty.direction} "
        f"{counterparty.transfer_count} 筆 | {totals} | {period} | 例：{samples}"
    )


def _amount(value: Decimal) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).date().isoformat()
