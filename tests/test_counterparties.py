from decimal import Decimal

from kyt_agent.counterparties import (
    inflow_composition,
    select_counterparties,
    summarize_counterparties,
)
from kyt_agent.labels import LabelStore
from tests.fakes import EXCHANGE, MIXER, SANCTIONED, TARGET, UNKNOWN, label, transfer, tx_hash


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
        "both",
        {"USDT": Decimal("5"), "ETH": Decimal("1")},
    )
    assert (exchange.first_seen, exchange.last_seen) == (1_700_000_003, 1_700_000_004)


def test_direction_filter_and_self_transfers():
    transfers = [
        transfer(1, TARGET, MIXER),
        transfer(2, EXCHANGE, TARGET),
        transfer(3, TARGET, TARGET),
    ]
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


def test_counts_flags_and_excludes_spoofed_amounts_from_totals():
    def classify(item):
        if item.asset == "FAKE":
            return {"spoofed_token"}
        return {"dust"} if item.amount < 1 else set()

    transfers = [
        transfer(1, MIXER, TARGET, "0.3", "FAKE"),
        transfer(2, MIXER, TARGET, "0.3", "FAKE"),
        transfer(3, MIXER, TARGET, "0.5"),
        transfer(4, MIXER, TARGET, "2"),
    ]
    (mixer,) = summarize_counterparties(TARGET, transfers, classify=classify)
    assert mixer.transfer_count == 4
    assert mixer.flags == {"spoofed_token": 2, "dust": 1}
    assert mixer.totals == {"ETH": Decimal("2.5")}
    assert summarize_counterparties(TARGET, transfers)[0].flags == {}


def test_ranks_by_non_spoofed_count_so_spoofed_spam_cannot_crowd_out_real_counterparties():
    def classify(item):
        return {"spoofed_token"} if item.asset == "FAKE" else set()

    transfers = [
        transfer(1, TARGET, EXCHANGE, "1", "FAKE"),
        transfer(2, TARGET, EXCHANGE, "1", "FAKE"),
        transfer(3, TARGET, EXCHANGE, "1", "FAKE"),
        transfer(4, TARGET, MIXER),
        transfer(5, TARGET, MIXER),
    ]
    ranked = summarize_counterparties(TARGET, transfers, classify=classify)
    assert [c.address for c in ranked] == [MIXER, EXCHANGE]
    exchange = next(c for c in ranked if c.address == EXCHANGE)
    assert exchange.transfer_count == 3  # 原始筆數不受排序影響


def test_inflow_composition_counts_risky_share_by_category():
    labels = LabelStore([label(MIXER, "mixer"), label(SANCTIONED, "sanctioned")])
    transfers = [
        transfer(1, MIXER, TARGET),
        transfer(2, MIXER, TARGET),
        transfer(3, SANCTIONED, TARGET),
        transfer(4, EXCHANGE, TARGET),
        transfer(5, EXCHANGE, TARGET),
        transfer(6, EXCHANGE, TARGET),
    ]
    composition = inflow_composition(TARGET, transfers, labels)
    assert (composition.total, composition.risky) == (6, 3)
    assert composition.by_category == {"mixer": 2, "sanctioned": 1}
    assert composition.share_pct == 50


def test_inflow_composition_rounds_share_to_nearest_percent():
    labels = LabelStore([label(MIXER, "mixer")])
    transfers = [
        transfer(1, MIXER, TARGET),
        transfer(2, MIXER, TARGET),
        transfer(3, EXCHANGE, TARGET),
    ]
    composition = inflow_composition(TARGET, transfers, labels)
    assert (composition.total, composition.risky) == (3, 2)
    assert composition.share_pct == 67  # 2/3 = 66.67% -> 67


def test_inflow_composition_zero_inflow():
    composition = inflow_composition(TARGET, [transfer(1, TARGET, MIXER)], LabelStore([]))
    assert (composition.total, composition.risky, composition.by_category) == (0, 0, {})
    assert composition.share_pct == 0


def test_inflow_composition_excludes_spoofed_transfers_and_outgoing():
    def classify(item):
        return {"spoofed_token"} if item.asset == "FAKE" else set()

    labels = LabelStore([label(MIXER, "mixer")])
    transfers = [
        transfer(1, MIXER, TARGET, "1", "FAKE"),
        transfer(2, MIXER, TARGET),
        transfer(3, TARGET, MIXER),
    ]
    composition = inflow_composition(TARGET, transfers, labels, classify=classify)
    assert (composition.total, composition.risky) == (1, 1)
