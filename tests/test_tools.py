import re
from decimal import Decimal

import httpx
import pytest

from kyt_agent.chain.client import ContractInfo, TransactionDetail
from kyt_agent.chain.etherscan import EtherscanClient
from kyt_agent.chain.snapshot import SnapshotClient
from kyt_agent.graph.tools import Investigator, assign_tx_aliases, tool_schemas
from kyt_agent.labels import LabelStore
from kyt_agent.models import AddressNode, Evidence
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
    tx_refs,
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
    assert tx_hash(1) in tx_refs(outcome.evidence)
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


FULL_HASH = re.compile(r"0x[0-9a-f]{64}")


def test_counterparties_cite_transactions_by_alias(investigator):
    outcome = investigator.execute("get_counterparties", {"address": TARGET}, ROOT, {})
    assert FULL_HASH.search(outcome.content) is None
    assert set(outcome.evidence) == {"T1", "T2", "T3", f"label:{MIXER}"}
    assert set(tx_refs(outcome.evidence)) == {tx_hash(1), tx_hash(2), tx_hash(3)}
    mixer_line = next(row for row in outcome.content.splitlines() if row.startswith(f"- {MIXER}"))
    alias = tx_refs(outcome.evidence)[tx_hash(1)]
    assert mixer_line.endswith(f"例：{alias}")
    assert outcome.evidence[f"label:{MIXER}"].ref == MIXER


def test_aliases_are_stable_across_calls_and_continue_numbering(investigator):
    inbound = investigator.execute(
        "get_counterparties", {"address": TARGET, "direction": "in"}, ROOT, {}
    )
    assert set(tx_refs(inbound.evidence)) == {tx_hash(2), tx_hash(3)}
    both = investigator.execute(
        "get_counterparties", {"address": TARGET, "direction": "both"}, ROOT, inbound.evidence
    )
    first, second = tx_refs(inbound.evidence), tx_refs(both.evidence)
    assert {second[tx_hash(2)], second[tx_hash(3)]} == {first[tx_hash(2)], first[tx_hash(3)]}
    assert second[tx_hash(1)] == "T3"


def test_assign_tx_aliases_reuses_known_and_skips_duplicates():
    known = {"T5": Evidence(id="T5", kind="tx", summary="s", ref=tx_hash(9))}
    aliases = assign_tx_aliases([tx_hash(1), tx_hash(9), tx_hash(1).upper(), tx_hash(2)], known)
    assert aliases == {tx_hash(1): "T6", tx_hash(9): "T5", tx_hash(2): "T7"}


def test_get_transaction_accepts_alias_or_registered_hash(investigator):
    found = investigator.execute("get_counterparties", {"address": TARGET}, ROOT, {}).evidence
    alias = tx_refs(found)[tx_hash(1)]
    by_alias = investigator.execute("get_transaction", {"tx": alias.lower()}, ROOT, found)
    assert by_alias.content.splitlines()[0] == f"{alias}（{tx_hash(1)}）"
    assert "0xb214faa5" in by_alias.content
    by_hash = investigator.execute("get_transaction", {"tx": tx_hash(1)}, ROOT, found)
    assert by_hash.content == by_alias.content


@pytest.mark.parametrize("tx", ["T9", tx_hash(7), "label:" + MIXER])
def test_get_transaction_rejects_unregistered_reference(investigator, chain, tx):
    found = investigator.execute("get_counterparties", {"address": TARGET}, ROOT, {}).evidence
    outcome = investigator.execute("get_transaction", {"tx": tx}, ROOT, found)
    assert "拒絕" in outcome.content
    assert ("transaction", tx_hash(7)) not in chain.calls


def test_get_transaction_rejected_without_evidence(investigator, chain):
    outcome = investigator.execute("get_transaction", {"tx": "T1"}, ROOT, {})
    assert "拒絕" in outcome.content
    assert chain.calls == []


def test_evidence_from_old_checkpoint_falls_back_to_id_as_ref(investigator):
    old = {
        tx_hash(1): Evidence.model_validate({"id": tx_hash(1), "kind": "tx", "summary": "s"}),
        f"label:{MIXER}": Evidence.model_validate(
            {"id": f"label:{MIXER}", "kind": "label", "summary": "s"}
        ),
    }
    assert (old[tx_hash(1)].ref, old[f"label:{MIXER}"].ref) == (tx_hash(1), MIXER)
    assert assign_tx_aliases([tx_hash(2), tx_hash(1)], old) == {
        tx_hash(2): "T1",
        tx_hash(1): tx_hash(1),
    }
    outcome = investigator.execute("get_transaction", {"tx": tx_hash(1)}, ROOT, old)
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
    assert tx_hash(4) in tx_refs(outcome.evidence)


def test_header_shows_inflow_composition_for_in_and_both_directions(settings):
    chain = StubChain(
        transfers={
            TARGET: [
                transfer(1, MIXER, TARGET),
                transfer(2, SANCTIONED, TARGET),
                transfer(3, EXCHANGE, TARGET),
                transfer(4, EXCHANGE, TARGET),
                transfer(5, TARGET, EXCHANGE),
            ]
        }
    )
    labels = LabelStore([label(MIXER, "mixer"), label(SANCTIONED, "sanctioned")])
    investigator = Investigator(chain, labels, NO_TOKENS, settings)
    outcome = investigator.execute(
        "get_counterparties", {"address": TARGET, "direction": "both"}, ROOT, {}
    )
    header = outcome.content.splitlines()[0]
    assert "轉入 4 筆（不含偽冒代幣），來自風險標籤地址 2 筆（50%）" in header
    assert "mixer 1" in header
    assert "sanctioned 1" in header


def test_header_omits_inflow_for_out_direction(settings):
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, MIXER)]})
    investigator = Investigator(chain, LabelStore([]), NO_TOKENS, settings)
    outcome = investigator.execute(
        "get_counterparties", {"address": TARGET, "direction": "out"}, ROOT, {}
    )
    header = outcome.content.splitlines()[0]
    assert "轉入" not in header


def test_header_shows_zero_inflow(settings):
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, MIXER)]})
    investigator = Investigator(chain, LabelStore([]), NO_TOKENS, settings)
    outcome = investigator.execute(
        "get_counterparties", {"address": TARGET, "direction": "in"}, ROOT, {}
    )
    header = outcome.content.splitlines()[0]
    assert "轉入 0 筆" in header


def test_counterparty_with_only_spoofed_transfers_shows_dash_totals(settings):
    fake_eth = "0x" + "e" * 40
    chain = StubChain(
        transfers={
            TARGET: [
                transfer(1, UNKNOWN, TARGET, "0.3", "ETH", token_contract=fake_eth),
                transfer(2, UNKNOWN, TARGET, "0.5", "ETH", token_contract=fake_eth),
            ]
        }
    )
    investigator = Investigator(chain, LabelStore([]), NO_TOKENS, settings)
    outcome = investigator.execute("get_counterparties", {"address": TARGET}, ROOT, {})
    line = next(row for row in outcome.content.splitlines() if row.startswith(f"- {UNKNOWN}"))
    assert "| - |" in line
    assert line.endswith("⚠ 偽冒代幣 2 筆")


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
