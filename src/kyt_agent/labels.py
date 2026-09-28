import csv
from collections.abc import Iterable
from pathlib import Path

import httpx

from kyt_agent.models import Category, Label

OFAC_ETH_URL = (
    "https://raw.githubusercontent.com/0xB10C/ofac-sanctioned-digital-currency-addresses"
    "/lists/sanctioned_addresses_ETH.txt"
)
FIELDS = ("address", "name", "category", "source")
_SEVERITY: tuple[Category, ...] = ("defi", "exchange", "bridge", "mixer", "hack", "sanctioned")


class LabelStore:
    def __init__(self, labels: Iterable[Label]) -> None:
        self._labels: dict[str, Label] = {}
        for label in labels:
            current = self._labels.get(label.address)
            # 同一地址有多個標籤時保留最嚴重的分類
            if current is None or _SEVERITY.index(label.category) > _SEVERITY.index(
                current.category
            ):
                self._labels[label.address] = label

    @classmethod
    def from_dir(cls, directory: Path) -> "LabelStore":
        return cls(
            label for path in sorted(directory.glob("*.csv")) for label in read_labels_csv(path)
        )

    def get(self, address: str) -> Label | None:
        return self._labels.get(address.lower())

    def __len__(self) -> int:
        return len(self._labels)


def read_labels_csv(path: Path) -> list[Label]:
    with path.open(newline="", encoding="utf-8") as file:
        return [Label.model_validate(row) for row in csv.DictReader(file)]


def write_labels_csv(path: Path, labels: Iterable[Label]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(label.model_dump() for label in labels)


def parse_ofac_list(text: str) -> list[Label]:
    return [
        Label(address=line.strip(), name="OFAC SDN", category="sanctioned", source=OFAC_ETH_URL)
        for line in text.splitlines()
        if line.strip().startswith("0x")
    ]


def fetch_ofac_labels(http: httpx.Client | None = None) -> list[Label]:
    response = (http or httpx.Client(timeout=30)).get(OFAC_ETH_URL)
    response.raise_for_status()
    return parse_ofac_list(response.text)
