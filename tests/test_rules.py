import pytest

from kyt_agent.labels import LabelStore
from kyt_agent.models import AddressNode, Category
from kyt_agent.rules import risk_floor, screen
from tests.fakes import EXCHANGE, MIXER, SANCTIONED, TARGET, UNKNOWN, label


def node(address: str, depth: int, category: Category | None = None) -> AddressNode:
    return AddressNode(
        address=address, depth=depth, label=label(address, category) if category else None
    )


def test_screen_hits_only_sanctioned_target():
    labels = LabelStore([label(SANCTIONED, "sanctioned", "OFAC SDN"), label(MIXER, "mixer")])
    hits = screen(SANCTIONED, labels)
    assert [(hit.rule, hit.level) for hit in hits] == [("target_sanctioned", "SEVERE")]
    assert screen(MIXER, labels) == []
    assert screen(UNKNOWN, labels) == []


@pytest.mark.parametrize(
    ("nodes", "expected"),
    [
        ([node(TARGET, 0)], "LOW"),
        ([node(TARGET, 0, "sanctioned")], "SEVERE"),
        ([node(TARGET, 0, "mixer")], "HIGH"),
        ([node(TARGET, 0, "hack")], "HIGH"),
        ([node(TARGET, 0, "exchange")], "LOW"),
        ([node(TARGET, 0), node(SANCTIONED, 1, "sanctioned")], "HIGH"),
        ([node(TARGET, 0), node(MIXER, 1, "mixer")], "HIGH"),
        ([node(TARGET, 0), node(MIXER, 1, "hack")], "HIGH"),
        ([node(TARGET, 0), node(MIXER, 2, "mixer")], "LOW"),
        ([node(TARGET, 0), node(EXCHANGE, 1, "exchange")], "LOW"),
    ],
)
def test_risk_floor(nodes, expected):
    assert risk_floor(TARGET, {item.address: item for item in nodes}) == expected
