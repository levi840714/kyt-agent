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
