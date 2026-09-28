from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel

from kyt_agent.models import LowerStr


class Transfer(BaseModel):
    tx_hash: LowerStr
    timestamp: int
    sender: LowerStr
    recipient: LowerStr
    amount: Decimal
    asset: str


class TransactionDetail(BaseModel):
    tx_hash: LowerStr
    sender: LowerStr
    recipient: LowerStr | None
    value_eth: Decimal
    block_number: int
    method_id: str


class ContractInfo(BaseModel):
    address: LowerStr
    is_contract: bool
    name: str | None = None


class ChainClient(Protocol):
    def get_transfers(self, address: str) -> list[Transfer]: ...

    def get_transaction(self, tx_hash: str) -> TransactionDetail | None: ...

    def get_contract_info(self, address: str) -> ContractInfo: ...
