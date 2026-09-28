import csv
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kyt_agent.chain.client import Transfer
from kyt_agent.models import LowerStr

TransferFlag = Literal["spoofed_token", "dust"]
NATIVE_SYMBOL = "ETH"


class KnownToken(BaseModel):
    contract: LowerStr
    symbol: str
    decimals: int
    dust_threshold: Decimal
    source: str


class TokenRegistry:
    """經查證的主流代幣：依合約查代幣、依 symbol 查正牌合約。"""

    def __init__(self, tokens: Iterable[KnownToken]) -> None:
        self._by_contract: dict[str, KnownToken] = {}
        self._by_symbol: dict[str, KnownToken] = {}
        for token in tokens:
            symbol = token.symbol.upper()
            if symbol in self._by_symbol:
                raise ValueError(f"代幣 symbol 重複：{symbol}")
            self._by_contract[token.contract] = token
            self._by_symbol[symbol] = token

    @classmethod
    def from_csv(cls, path: Path) -> "TokenRegistry":
        if not path.exists():
            return cls([])
        with path.open(newline="", encoding="utf-8") as file:
            return cls([KnownToken.model_validate(row) for row in csv.DictReader(file)])

    def by_contract(self, contract: str) -> KnownToken | None:
        return self._by_contract.get(contract.lower())

    def by_symbol(self, symbol: str) -> KnownToken | None:
        return self._by_symbol.get(symbol.upper())

    def __len__(self) -> int:
        return len(self._by_contract)


def classify_transfer(
    item: Transfer, registry: TokenRegistry, native_dust_threshold: Decimal
) -> set[TransferFlag]:
    if item.token_contract is None:
        return {"dust"} if item.amount < native_dust_threshold else set()
    known = registry.by_contract(item.token_contract)
    if known is not None:
        return {"dust"} if item.amount < known.dust_threshold else set()
    # 冒用原生幣或主流代幣 symbol 的合約，常見於地址投毒
    if item.asset.upper() == NATIVE_SYMBOL or registry.by_symbol(item.asset) is not None:
        return {"spoofed_token"}
    return set()
