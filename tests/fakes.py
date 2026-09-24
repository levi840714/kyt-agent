from decimal import Decimal

from kyt_agent.chain.client import ContractInfo, TransactionDetail, Transfer
from kyt_agent.models import Category, Label

TARGET = "0x" + "1" * 40
MIXER = "0x" + "2" * 40
EXCHANGE = "0x" + "3" * 40
UNKNOWN = "0x" + "4" * 40
SANCTIONED = "0x" + "5" * 40
MIXED_CASE = "0x" + "aB" * 20


def tx_hash(n: int) -> str:
    return "0x" + f"{n:064x}"


def label(address: str, category: Category, name: str = "測試標籤") -> Label:
    return Label(address=address, name=name, category=category, source="test")


def transfer(
    n: int, sender: str, recipient: str, amount: str = "1", asset: str = "ETH"
) -> Transfer:
    return Transfer(
        tx_hash=tx_hash(n),
        timestamp=1_700_000_000 + n,
        sender=sender,
        recipient=recipient,
        amount=Decimal(amount),
        asset=asset,
    )


class StubChain:
    def __init__(
        self,
        transfers: dict[str, list[Transfer]] | None = None,
        contracts: dict[str, ContractInfo] | None = None,
        transactions: dict[str, TransactionDetail] | None = None,
    ) -> None:
        self.transfers = transfers or {}
        self.contracts = contracts or {}
        self.transactions = transactions or {}
        self.calls: list[tuple[str, str]] = []

    def get_transfers(self, address: str) -> list[Transfer]:
        self.calls.append(("transfers", address))
        return self.transfers.get(address, [])

    def get_transaction(self, tx_hash: str) -> TransactionDetail | None:
        self.calls.append(("transaction", tx_hash))
        return self.transactions.get(tx_hash)

    def get_contract_info(self, address: str) -> ContractInfo:
        self.calls.append(("contract", address))
        return self.contracts.get(address, ContractInfo(address=address, is_contract=False))
