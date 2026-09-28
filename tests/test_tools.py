from decimal import Decimal

import httpx
import pytest

from kyt_agent.chain.client import ContractInfo, TransactionDetail
from kyt_agent.chain.etherscan import EtherscanClient
from kyt_agent.chain.snapshot import SnapshotClient
from kyt_agent.graph.tools import Investigator, tool_schemas
from kyt_agent.labels import LabelStore
from kyt_agent.models import AddressNode
from kyt_agent.rules import risk_floor
from kyt_agent.tokens import TokenRegistry
from tests.fakes import (
    EXCHANGE,
    MIXER,
    SANCTIONED,
    TARGET,
    UNKNOWN,
    StubChain,
    label,
    transfer,
    tx_hash,
)

ROOT = {TARGET: AddressNode(address=TARGET, depth=0)}
NO_TOKENS = TokenRegistry([])


@pytest.fixture
def chain() -> StubChain:
    return StubChain(
        transfers={
            TARGET: [
                transfer(1, TARGET, MIXER),
                transfer(2, EXCHANGE, TARGET),
                transfer(3, EXCHANGE, TARGET),
            ]
        },
        contracts={MIXER: ContractInfo(address=MIXER, is_contract=True, name="TornadoCash_Eth")},
        transactions={
            tx_hash(1): TransactionDetail(
                tx_hash=tx_hash(1),
                sender=TARGET,
                recipient=MIXER,
                value_eth=Decimal("1"),
                block_number=100,
                method_id="0xb214faa5",
            )
        },
    )


@pytest.fixture
def investigator(chain, settings) -> Investigator:
    labels = LabelStore([label(MIXER, "mixer", "Tornado Cash")])
    return Investigator(chain, labels, NO_TOKENS, settings)


def test_tool_schemas_expose_three_tools():
    names = [schema["function"]["name"] for schema in tool_schemas()]
    assert names == ["get_counterparties", "lookup_address", "get_transaction"]


def test_rejects_address_outside_case_graph(investigator, chain):
    outcome = investigator.execute("get_counterparties", {"address": UNKNOWN}, ROOT, {})
    assert "拒絕" in outcome.content
    assert chain.calls == []


def test_expands_target_and_registers_counterparties(investigator):
    outcome = investigator.execute("get_counterparties", {"address": TARGET}, ROOT, {})
    assert outcome.nodes[TARGET].expanded
    assert (outcome.nodes[MIXER].depth, outcome.nodes[MIXER].label.category) == (1, "mixer")
    assert outcome.nodes[EXCHANGE].parent == TARGET
    assert tx_hash(1) in outcome.evidence
    assert f"label:{MIXER}" in outcome.evidence
    assert "Tornado Cash" in outcome.content


def test_labeled_counterparty_is_listed_beyond_top_n(chain, settings):
    narrow = settings.model_copy(update={"top_counterparties": 1})
    labels = LabelStore([label(MIXER, "mixer")])
    outcome = Investigator(chain, labels, NO_TOKENS, narrow).execute(
        "get_counterparties", {"address": TARGET}, ROOT, {}
    )
    assert {MIXER, EXCHANGE} <= outcome.nodes.keys()


def test_rejects_expansion_beyond_max_depth(investigator, settings):
    nodes = {**ROOT, UNKNOWN: AddressNode(address=UNKNOWN, depth=settings.max_depth)}
    outcome = investigator.execute("get_counterparties", {"address": UNKNOWN}, nodes, {})
    assert "最大追蹤深度" in outcome.content


def test_rejects_expansion_beyond_address_limit(chain, settings):
    limited = settings.model_copy(update={"max_addresses": 1})
    nodes = {
        TARGET: AddressNode(address=TARGET, depth=0, expanded=True),
        UNKNOWN: AddressNode(address=UNKNOWN, depth=1),
    }
    outcome = Investigator(chain, LabelStore([]), NO_TOKENS, limited).execute(
        "get_counterparties", {"address": UNKNOWN}, nodes, {}
    )
    assert "上限" in outcome.content


def test_lookup_address_reports_contract_and_label(investigator):
    nodes = {**ROOT, MIXER: AddressNode(address=MIXER, depth=1)}
    outcome = investigator.execute("lookup_address", {"address": MIXER}, nodes, {})
    assert "TornadoCash_Eth" in outcome.content
    assert f"label:{MIXER}" in outcome.evidence


def test_get_transaction_only_for_known_hash(investigator):
    rejected = investigator.execute("get_transaction", {"tx_hash": tx_hash(1)}, ROOT, {})
    assert "拒絕" in rejected.content
    found = investigator.execute("get_counterparties", {"address": TARGET}, ROOT, {}).evidence
    outcome = investigator.execute("get_transaction", {"tx_hash": tx_hash(1)}, ROOT, found)
    assert "0xb214faa5" in outcome.content


def test_invalid_arguments_and_unknown_tool(investigator):
    assert "參數錯誤" in investigator.execute("get_transaction", {}, ROOT, {}).content
    assert "未知的工具" in investigator.execute("drop_table", {}, ROOT, {}).content


def test_snapshot_miss_is_reported(settings, tmp_path):
    snapshots = SnapshotClient(tmp_path / "snapshots")
    replay = Investigator(snapshots, LabelStore([]), NO_TOKENS, settings)
    outcome = replay.execute("get_counterparties", {"address": TARGET}, ROOT, {})
    assert outcome.miss
    assert "資料不可用" in outcome.content


@pytest.mark.parametrize(
    "failure",
    [httpx.Response(502), httpx.ConnectTimeout("timeout")],
)
def test_etherscan_failure_does_not_leak_api_key(settings, failure):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(failure, Exception):
            raise failure
        return failure

    client = EtherscanClient(
        "super-secret-key",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        min_interval=0,
        sleep=lambda _: None,
    )
    outcome = Investigator(client, LabelStore([]), NO_TOKENS, settings).execute(
        "get_counterparties", {"address": TARGET}, ROOT, {}
    )
    assert outcome.content.startswith("查詢失敗")
    assert "super-secret-key" not in outcome.content


def test_counterparty_found_at_shallower_depth_is_reparented(settings):
    chain = StubChain(
        transfers={
            TARGET: [transfer(1, EXCHANGE, TARGET), transfer(2, TARGET, MIXER)],
            EXCHANGE: [transfer(3, MIXER, EXCHANGE)],
        }
    )
    investigator = Investigator(chain, LabelStore([label(MIXER, "mixer")]), NO_TOKENS, settings)
    nodes = dict(ROOT)
    for address, direction in [(TARGET, "in"), (EXCHANGE, "both"), (TARGET, "out")]:
        args = {"address": address, "direction": direction}
        nodes.update(investigator.execute("get_counterparties", args, nodes, {}).nodes)
        if address == EXCHANGE:
            assert (nodes[MIXER].depth, nodes[MIXER].parent) == (2, EXCHANGE)
    assert (nodes[MIXER].depth, nodes[MIXER].parent) == (1, TARGET)
    assert nodes[MIXER].label.category == "mixer"
    assert risk_floor(TARGET, nodes) == "HIGH"


def test_counterparty_flags_are_shown_and_spoofed_amounts_excluded(settings):
    fake_eth = "0x" + "e" * 40
    chain = StubChain(
        transfers={
            TARGET: [
                transfer(1, UNKNOWN, TARGET, "0.3", "ETH", token_contract=fake_eth),
                transfer(2, UNKNOWN, TARGET, "0.3", "ETH", token_contract=fake_eth),
                transfer(3, UNKNOWN, TARGET, "0.00001"),
                transfer(4, UNKNOWN, TARGET, "1"),
            ]
        }
    )
    investigator = Investigator(chain, LabelStore([]), NO_TOKENS, settings)
    outcome = investigator.execute("get_counterparties", {"address": TARGET}, ROOT, {})
    line = next(row for row in outcome.content.splitlines() if row.startswith(f"- {UNKNOWN}"))
    assert line.endswith("| ⚠ 偽冒代幣 2 筆、粉塵 1 筆")
    assert "| 1 ETH |" in line
    assert tx_hash(4) in outcome.evidence


def test_reparent_updates_registered_descendants(settings):
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    nodes = {
        **ROOT,
        EXCHANGE: AddressNode(address=EXCHANGE, depth=1, parent=TARGET),
        UNKNOWN: AddressNode(address=UNKNOWN, depth=2, parent=EXCHANGE),
        MIXER: AddressNode(address=MIXER, depth=3, parent=UNKNOWN, label=label(MIXER, "mixer")),
        SANCTIONED: AddressNode(address=SANCTIONED, depth=4, parent=MIXER),
    }
    investigator = Investigator(chain, LabelStore([]), NO_TOKENS, settings)
    outcome = investigator.execute("get_counterparties", {"address": TARGET}, nodes, {})
    assert (outcome.nodes[UNKNOWN].depth, outcome.nodes[UNKNOWN].parent) == (1, TARGET)
    assert (outcome.nodes[MIXER].depth, outcome.nodes[MIXER].parent) == (2, UNKNOWN)
    assert (outcome.nodes[SANCTIONED].depth, outcome.nodes[SANCTIONED].parent) == (3, MIXER)
    assert EXCHANGE not in outcome.nodes


def test_descendant_reparented_directly_keeps_shallower_depth(settings):
    chain = StubChain(
        transfers={
            TARGET: [
                transfer(1, TARGET, UNKNOWN),
                transfer(2, TARGET, UNKNOWN),
                transfer(3, TARGET, MIXER),
            ]
        }
    )
    nodes = {
        **ROOT,
        EXCHANGE: AddressNode(address=EXCHANGE, depth=1, parent=TARGET),
        UNKNOWN: AddressNode(address=UNKNOWN, depth=2, parent=EXCHANGE),
        MIXER: AddressNode(address=MIXER, depth=3, parent=UNKNOWN),
    }
    labels = LabelStore([label(MIXER, "mixer")])
    outcome = Investigator(chain, labels, NO_TOKENS, settings).execute(
        "get_counterparties", {"address": TARGET}, nodes, {}
    )
    assert outcome.content.index(MIXER) < outcome.content.index(UNKNOWN)
    assert (outcome.nodes[MIXER].depth, outcome.nodes[MIXER].parent) == (1, TARGET)
    assert (outcome.nodes[UNKNOWN].depth, outcome.nodes[UNKNOWN].parent) == (1, TARGET)
