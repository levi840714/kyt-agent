from decimal import Decimal

from kyt_agent.counterparties import select_counterparties, summarize_counterparties
from kyt_agent.labels import LabelStore
from tests.fakes import EXCHANGE, MIXER, TARGET, UNKNOWN, label, transfer, tx_hash


def test_groups_transfers_by_counterparty():
    transfers = [
        transfer(1, TARGET, MIXER, "1"),
        transfer(2, TARGET, MIXER, "2"),
        transfer(3, EXCHANGE, TARGET, "5", "USDT"),
        transfer(4, TARGET, EXCHANGE, "1"),
    ]
    mixer, exchange = summarize_counterparties(TARGET, transfers)
    assert (mixer.address, mixer.direction, mixer.transfer_count) == (MIXER, "out", 2)
    assert mixer.totals == {"ETH": Decimal("3")}
    assert mixer.sample_hashes == [tx_hash(2), tx_hash(1)]
    assert (exchange.direction, exchange.totals) == (
        "both", {"USDT": Decimal("5"), "ETH": Decimal("1")}
    )
    assert (exchange.first_seen, exchange.last_seen) == (1_700_000_003, 1_700_000_004)


def test_direction_filter_and_self_transfers():
    transfers = [transfer(1, TARGET, MIXER), transfer(2, EXCHANGE, TARGET), transfer(3, TARGET, TARGET)]
    assert [c.address for c in summarize_counterparties(TARGET, transfers, "in")] == [EXCHANGE]
    assert [c.address for c in summarize_counterparties(TARGET, transfers, "out")] == [MIXER]
    assert len(summarize_counterparties(TARGET, transfers)) == 2


def test_ranks_by_count_and_select_keeps_labeled_counterparties():
    transfers = [
        transfer(1, TARGET, EXCHANGE),
        transfer(2, TARGET, EXCHANGE),
        transfer(3, TARGET, UNKNOWN),
        transfer(4, TARGET, MIXER),
    ]
    ranked = summarize_counterparties(TARGET, transfers)
    assert [c.address for c in ranked] == [EXCHANGE, MIXER, UNKNOWN]
    labels = LabelStore([label(MIXER, "mixer")])
    selected = select_counterparties(ranked, labels, top_n=1)
    assert [c.address for c in selected] == [MIXER, EXCHANGE]
