from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from kyt_agent.chain.client import Transfer
from kyt_agent.labels import LabelStore
from kyt_agent.models import Category
from kyt_agent.tokens import TransferFlag

Direction = Literal["in", "out", "both"]
Classifier = Callable[[Transfer], set[TransferFlag]]
# 中間地址判斷「高度依賴風險來源」時計入的標籤分類（不含 bridge/exchange/defi）
INFLOW_RISK_CATEGORIES: tuple[Category, ...] = ("sanctioned", "hack", "mixer")


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


def summarize_counterparties(
    address: str,
    transfers: Iterable[Transfer],
    direction: Direction = "both",
    samples: int = 2,
    classify: Classifier = _no_flags,
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
    ranked = sorted(groups.items(), key=lambda group: (-len(group[1]), group[0]))
    return [_summarize(address, other, items, samples, classify) for other, items in ranked]


class InflowComposition(BaseModel):
    """查詢地址「全部」轉入（不只列出的交易對手）的風險組成，排除偽冒代幣。"""

    total: int
    risky: int
    by_category: dict[str, int]

    @property
    def share_pct(self) -> int:
        return round(self.risky / self.total * 100) if self.total else 0


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
        if item.recipient != address or "spoofed_token" in classify(item):
            continue
        total += 1
        label = labels.get(item.sender)
        if label and label.category in INFLOW_RISK_CATEGORIES:
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
    address: str, other: str, items: list[Transfer], samples: int, classify: Classifier
) -> Counterparty:
    totals: dict[str, Decimal] = defaultdict(Decimal)
    flags: Counter[TransferFlag] = Counter()
    for item in items:
        item_flags = classify(item)
        flags.update(item_flags)
        # 偽冒代幣沒有實際價值，計入總額會讓往來金額看起來比實際大
        if "spoofed_token" not in item_flags:
            totals[item.asset] += item.amount
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
