from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from kyt_agent.chain.client import Transfer
from kyt_agent.labels import LabelStore
from kyt_agent.models import HIGH_RISK_CATEGORIES
from kyt_agent.tokens import TransferFlag

Direction = Literal["in", "out", "both"]
Classifier = Callable[[Transfer], set[TransferFlag]]
AssetName = Callable[[Transfer], str]


class Counterparty(BaseModel):
    address: str
    direction: Direction
    transfer_count: int
    totals: dict[str, Decimal]
    first_seen: int
    last_seen: int
    sample_hashes: list[str]
    flags: dict[TransferFlag, int] = {}


def _no_flags(item: Transfer) -> set[TransferFlag]:
    return set()


def _raw_asset(item: Transfer) -> str:
    return item.asset


def summarize_counterparties(
    address: str,
    transfers: Iterable[Transfer],
    direction: Direction = "both",
    samples: int = 2,
    classify: Classifier = _no_flags,
    asset_name: AssetName = _raw_asset,
) -> list[Counterparty]:
    address = address.lower()
    groups: dict[str, list[Transfer]] = defaultdict(list)
    for item in transfers:
        incoming = item.recipient == address
        other = item.sender if incoming else item.recipient
        if other == address or (direction == "in" and not incoming):
            continue
        if direction == "out" and incoming:
            continue
        groups[other].append(item)
    # 依非偽冒代幣筆數排序，避免偽冒代幣洗版把真實交易對手擠出列表；transfer_count 仍為原始筆數
    ranked = sorted(
        groups.items(), key=lambda group: (-_non_spoofed_count(group[1], classify), group[0])
    )
    return [
        _summarize(address, other, items, samples, classify, asset_name) for other, items in ranked
    ]


def _non_spoofed_count(items: list[Transfer], classify: Classifier) -> int:
    return sum(1 for item in items if "spoofed_token" not in classify(item))


class InflowComposition(BaseModel):
    """已取得的最近轉入（不只列出的交易對手）的風險組成，排除偽冒代幣、粉塵與自轉。"""

    total: int
    risky: int
    by_category: dict[str, int]

    @property
    def share_pct(self) -> int:
        # 只供顯示且無條件捨去；是否過半以 majority_risky 的筆數比較為準
        return self.risky * 100 // self.total if self.total else 0

    @property
    def majority_risky(self) -> bool:
        return self.risky * 2 > self.total


def inflow_composition(
    address: str,
    transfers: Iterable[Transfer],
    labels: LabelStore,
    classify: Classifier = _no_flags,
) -> InflowComposition:
    address = address.lower()
    total = 0
    by_category: Counter[str] = Counter()
    for item in transfers:
        if item.recipient != address or item.sender == address:
            continue
        if classify(item) & {"spoofed_token", "dust"}:
            continue
        total += 1
        label = labels.get(item.sender)
        if label and label.category in HIGH_RISK_CATEGORIES:
            by_category[label.category] += 1
    return InflowComposition(
        total=total, risky=sum(by_category.values()), by_category=dict(by_category)
    )


def select_counterparties(
    counterparties: list[Counterparty], labels: LabelStore, top_n: int
) -> list[Counterparty]:
    labeled = [c for c in counterparties if labels.get(c.address)]
    unlabeled = [c for c in counterparties if not labels.get(c.address)]
    return labeled + unlabeled[:top_n]


def _summarize(
    address: str,
    other: str,
    items: list[Transfer],
    samples: int,
    classify: Classifier,
    asset_name: AssetName,
) -> Counterparty:
    totals: dict[str, Decimal] = defaultdict(Decimal)
    flags: Counter[TransferFlag] = Counter()
    for item in items:
        item_flags = classify(item)
        flags.update(item_flags)
        # 偽冒代幣沒有實際價值，計入總額會讓往來金額看起來比實際大
        if "spoofed_token" not in item_flags:
            totals[asset_name(item)] += item.amount
    incoming = {item.recipient == address for item in items}
    direction: Direction = "both" if len(incoming) == 2 else ("in" if True in incoming else "out")
    recent = sorted(items, key=lambda item: item.timestamp, reverse=True)
    return Counterparty(
        address=other,
        direction=direction,
        transfer_count=len(items),
        totals=dict(totals),
        first_seen=recent[-1].timestamp,
        last_seen=recent[0].timestamp,
        sample_hashes=list(dict.fromkeys(item.tx_hash for item in recent))[:samples],
        flags=dict(flags),
    )
