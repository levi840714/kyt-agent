from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter

from kyt_agent.chain.client import ChainClient, ContractInfo, TransactionDetail, Transfer

T = TypeVar("T")

_TRANSFERS = TypeAdapter(list[Transfer])
_TRANSACTION: TypeAdapter[TransactionDetail | None] = TypeAdapter(TransactionDetail | None)
_CONTRACT = TypeAdapter(ContractInfo)


class SnapshotMissError(LookupError):
    """重播模式下快照中沒有對應資料。"""


class SnapshotClient:
    """有 inner 時為 read-through 錄製，沒有時只讀快照。"""

    def __init__(self, root: Path, inner: ChainClient | None = None) -> None:
        self._root = root
        self._inner = inner

    def get_transfers(self, address: str) -> list[Transfer]:
        return self._fetch(
            "transfers", address, _TRANSFERS, lambda chain: chain.get_transfers(address)
        )

    def get_transaction(self, tx_hash: str) -> TransactionDetail | None:
        return self._fetch(
            "transactions", tx_hash, _TRANSACTION, lambda chain: chain.get_transaction(tx_hash)
        )

    def get_contract_info(self, address: str) -> ContractInfo:
        return self._fetch(
            "contracts", address, _CONTRACT, lambda chain: chain.get_contract_info(address)
        )

    def _fetch(
        self, kind: str, key: str, adapter: TypeAdapter[T], load: Callable[[ChainClient], T]
    ) -> T:
        path = self._root / kind / f"{key.lower()}.json"
        if path.exists():
            return adapter.validate_json(path.read_bytes())
        if self._inner is None:
            raise SnapshotMissError(f"{kind}/{key.lower()}")
        value = load(self._inner)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(adapter.dump_json(value, indent=2))
        return value
