import re
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from kyt_agent.audit import digest
from kyt_agent.chain.client import ChainClient, Transfer
from kyt_agent.chain.etherscan import EtherscanError
from kyt_agent.chain.snapshot import SnapshotMissError
from kyt_agent.config import Settings
from kyt_agent.counterparties import (
    Counterparty,
    Direction,
    InflowComposition,
    inflow_composition,
    select_counterparties,
    summarize_counterparties,
)
from kyt_agent.labels import LabelStore
from kyt_agent.models import AddressNode, Evidence, Label, find_tx, tx_alias_number
from kyt_agent.tokens import (
    TokenRegistry,
    TransferFlag,
    classify_transfer,
    display_asset,
    is_plain_symbol,
)

_UNKNOWN_ADDRESS = "拒絕：只能查詢目標地址，或工具結果中出現過的地址"
_FLAG_NAMES: dict[TransferFlag, str] = {"spoofed_token": "偽冒代幣", "dust": "粉塵"}
_SOLIDITY_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_PREVIEW_CHARS = 40
SuspiciousField = Literal["token_symbol", "contract_name"]


class GetCounterpartiesArgs(BaseModel):
    address: str = Field(description="要查詢的地址")
    direction: Direction = Field(
        default="both", description="in：資金來源，out：資金去向，both：兩者"
    )


class LookupAddressArgs(BaseModel):
    address: str = Field(description="要查詢的地址")


class GetTransactionArgs(BaseModel):
    tx: str = Field(description="交易代號（如 T1），也接受工具結果中出現過的 hash")


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


class SuspiciousText(BaseModel):
    """格式不符的鏈上字串，可能夾帶給 LLM 的指令：只記入 audit log，不放進工具結果。"""

    field: SuspiciousField
    address: str
    sha256: str
    preview: str

    @classmethod
    def of(cls, field: SuspiciousField, address: str, text: str) -> "SuspiciousText":
        return cls(field=field, address=address, sha256=digest(text), preview=text[:_PREVIEW_CHARS])


class ToolOutcome(BaseModel):
    content: str
    nodes: dict[str, AddressNode] = {}
    evidence: dict[str, Evidence] = {}
    miss: bool = False
    resolved_tx: str | None = None
    suspicious: list[SuspiciousText] = []


class Investigator:
    def __init__(
        self, chain: ChainClient, labels: LabelStore, tokens: TokenRegistry, settings: Settings
    ) -> None:
        self._chain = chain
        self._labels = labels
        self._tokens = tokens
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
                return self._counterparties(
                    GetCounterpartiesArgs.model_validate(args), nodes, evidence
                )
            if name == "lookup_address":
                return self._lookup(LookupAddressArgs.model_validate(args), nodes)
            if name == "get_transaction":
                return self._transaction(GetTransactionArgs.model_validate(args), evidence)
            return ToolOutcome(content=f"未知的工具：{name}")
        except ValidationError as error:
            return ToolOutcome(content=f"參數錯誤：{error}")
        except SnapshotMissError:
            return ToolOutcome(content="資料不可用：快照中沒有這筆資料", miss=True)
        except EtherscanError as error:
            return ToolOutcome(content=f"查詢失敗：{error}")

    def _counterparties(
        self,
        args: GetCounterpartiesArgs,
        nodes: dict[str, AddressNode],
        known_evidence: dict[str, Evidence],
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
        transfers = list(self._chain.get_transfers(address))
        found = summarize_counterparties(
            address, transfers, args.direction, classify=self._flags, asset_name=self._asset_name
        )
        shown = select_counterparties(found, self._labels, self._settings.top_counterparties)
        new_nodes = {address: node.model_copy(update={"expanded": True})}
        graph = {**nodes, **new_nodes}
        aliases = assign_tx_aliases(
            (tx_hash for item in shown for tx_hash in item.sample_hashes), known_evidence
        )
        evidence: dict[str, Evidence] = {}
        inflow = ""
        if args.direction in ("in", "both"):
            composition = inflow_composition(address, transfers, self._labels, self._flags)
            inflow = f"{_format_inflow(composition)}；"
        lines = [
            f"{address}（第 {node.depth} 層）方向 {args.direction}：{inflow}"
            f"共 {len(found)} 個交易對手，列出 {len(shown)} 個"
            "（已知標籤全列，其餘依互動次數（不含偽冒代幣）排序）"
        ]
        for counterparty in shown:
            label = self._labels.get(counterparty.address)
            known = graph.get(counterparty.address)
            if known is None:
                created = AddressNode(
                    address=counterparty.address, depth=node.depth + 1, parent=address, label=label
                )
                new_nodes[created.address] = graph[created.address] = created
            elif known.depth > node.depth + 1:
                # 較淺層也出現時改掛到較近的路徑，否則直接接觸的高風險對手會被當成間接
                moved = known.model_copy(update={"depth": node.depth + 1, "parent": address})
                updated = _with_descendants(moved, graph)
                new_nodes.update(updated)
                graph.update(updated)
            evidence.update(_tx_evidence(address, counterparty, aliases))
            if label:
                evidence.update(label_evidence(label))
            lines.append(_format_counterparty(counterparty, label, aliases))
        return ToolOutcome(
            content="\n".join(lines),
            nodes=new_nodes,
            evidence=evidence,
            suspicious=_suspicious_symbols(transfers),
        )

    def _flags(self, item: Transfer) -> set[TransferFlag]:
        return classify_transfer(item, self._tokens, self._settings.native_dust_threshold)

    def _asset_name(self, item: Transfer) -> str:
        return display_asset(item, self._tokens)

    def _lookup(self, args: LookupAddressArgs, nodes: dict[str, AddressNode]) -> ToolOutcome:
        address = args.address.lower()
        if address not in nodes:
            return ToolOutcome(content=_UNKNOWN_ADDRESS)
        label = self._labels.get(address)
        info = self._chain.get_contract_info(address)
        suspicious: list[SuspiciousText] = []
        kind = "一般地址（EOA）"
        if info.is_contract:
            name = info.name or "未驗證"
            if info.name and not _SOLIDITY_IDENTIFIER.fullmatch(info.name):
                suspicious.append(SuspiciousText.of("contract_name", address, info.name))
                name = "名稱不符格式，已略過"
            kind = f"合約（{name}）"
        tag = f"{label.category}：{label.name}（來源 {label.source}）" if label else "無"
        return ToolOutcome(
            content=f"{address}\n類型：{kind}\n標籤：{tag}",
            evidence=label_evidence(label) if label else {},
            suspicious=suspicious,
        )

    def _transaction(self, args: GetTransactionArgs, evidence: dict[str, Evidence]) -> ToolOutcome:
        item = find_tx(args.tx, evidence)
        if item is None:
            return ToolOutcome(content="拒絕：只能查詢工具結果中出現過的交易")
        detail = self._chain.get_transaction(item.ref)
        if detail is None:
            return ToolOutcome(content=f"查無交易 {item.id}", resolved_tx=item.ref)
        return ToolOutcome(
            content=(
                f"{item.id}（{item.ref}）\n區塊：{detail.block_number}\n從：{detail.sender}\n"
                f"到：{detail.recipient or '（合約建立）'}\n"
                f"金額：{_amount(detail.value_eth)} ETH\n方法：{detail.method_id or '（無）'}"
            ),
            resolved_tx=item.ref,
        )


def _suspicious_symbols(transfers: Iterable[Transfer]) -> list[SuspiciousText]:
    found: dict[tuple[str, str], SuspiciousText] = {}
    for item in transfers:
        if item.token_contract and not is_plain_symbol(item.asset):
            finding = SuspiciousText.of("token_symbol", item.token_contract, item.asset)
            found.setdefault((finding.address, finding.sha256), finding)
    return list(found.values())


def _with_descendants(moved: AddressNode, nodes: dict[str, AddressNode]) -> dict[str, AddressNode]:
    # 子孫深度不跟著更新的話，深度上限與回報的層數會用到過時的值
    updated = {moved.address: moved}
    frontier = [moved]
    while frontier:
        parent = frontier.pop()
        for child in nodes.values():
            if child.parent == parent.address and child.depth > parent.depth + 1:
                updated[child.address] = child.model_copy(update={"depth": parent.depth + 1})
                frontier.append(updated[child.address])
    return updated


def label_evidence(label: Label) -> dict[str, Evidence]:
    evidence_id = f"label:{label.address}"
    summary = f"{label.address} 為 {label.category}：{label.name}（來源 {label.source}）"
    return {evidence_id: Evidence(id=evidence_id, kind="label", summary=summary, ref=label.address)}


def assign_tx_aliases(hashes: Iterable[str], evidence: dict[str, Evidence]) -> dict[str, str]:
    """hash → 代號：已登記的沿用，新的依出現順序從目前最大編號往後配。"""
    aliases = {item.ref: item.id for item in evidence.values() if item.kind == "tx"}
    numbers = (tx_alias_number(alias) for alias in aliases.values())
    next_number = max((number for number in numbers if number is not None), default=0) + 1
    assigned: dict[str, str] = {}
    for tx_hash in map(str.lower, hashes):
        if tx_hash in assigned:
            continue
        if tx_hash not in aliases:
            aliases[tx_hash] = f"T{next_number}"
            next_number += 1
        assigned[tx_hash] = aliases[tx_hash]
    return assigned


def _tx_evidence(
    address: str, counterparty: Counterparty, aliases: dict[str, str]
) -> dict[str, Evidence]:
    flow = {
        "in": f"{counterparty.address} → {address}",
        "out": f"{address} → {counterparty.address}",
        "both": f"{address} ↔ {counterparty.address}",
    }[counterparty.direction]
    return {
        aliases[tx_hash]: Evidence(id=aliases[tx_hash], kind="tx", summary=flow, ref=tx_hash)
        for tx_hash in counterparty.sample_hashes
    }


def _format_counterparty(
    counterparty: Counterparty, label: Label | None, aliases: dict[str, str]
) -> str:
    tag = f"[{label.category}: {label.name}]" if label else "[未知]"
    totals = ", ".join(f"{_amount(value)} {asset}" for asset, value in counterparty.totals.items())
    period = f"{_date(counterparty.first_seen)}~{_date(counterparty.last_seen)}"
    samples = ", ".join(aliases[tx_hash] for tx_hash in counterparty.sample_hashes)
    line = (
        f"- {counterparty.address} {tag} {counterparty.direction} "
        f"{counterparty.transfer_count} 筆 | {totals or '-'} | {period} | 例：{samples}"
    )
    flags = [
        f"{name} {counterparty.flags[flag]} 筆"
        for flag, name in _FLAG_NAMES.items()
        if flag in counterparty.flags
    ]
    return f"{line} | ⚠ {'、'.join(flags)}" if flags else line


def _format_inflow(composition: InflowComposition) -> str:
    window = f"最近 {composition.total} 筆轉入（不含偽冒代幣與粉塵）"
    if composition.total == 0:
        return window
    ranked = sorted(composition.by_category.items(), key=lambda kv: (-kv[1], kv[0]))
    breakdown = "、".join(f"{name} {count}" for name, count in ranked)
    detail = f"：{breakdown}" if breakdown else ""
    majority = "（過半）" if composition.majority_risky else ""
    return (
        f"{window}，來自風險標籤地址 {composition.risky} 筆"
        f"（{composition.share_pct}%）{majority}{detail}"
    )


def _amount(value: Decimal) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).date().isoformat()
