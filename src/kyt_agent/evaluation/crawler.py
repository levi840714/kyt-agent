from kyt_agent import rules
from kyt_agent.chain.client import ChainClient
from kyt_agent.counterparties import (
    Classifier,
    Direction,
    select_counterparties,
    summarize_counterparties,
)
from kyt_agent.labels import LabelStore
from kyt_agent.models import AddressNode, RiskLevel

_DIRECTIONS: tuple[Direction, ...] = ("in", "out", "both")


def crawl(
    chain: ChainClient,
    labels: LabelStore,
    address: str,
    depth: int,
    top_n: int,
    classify: Classifier,
) -> int:
    """依 agent 工具的挑選規則展開 depth 層，回傳涵蓋的地址數。
    目標地址的交易對手樣本交易也一併錄製，供 agent 重播 get_transaction。"""
    frontier = [address.lower()]
    seen = set(frontier)
    for layer in range(depth):
        discovered: list[str] = []
        sample_hashes: set[str] = set()
        for current in frontier:
            chain.get_contract_info(current)
            transfers = chain.get_transfers(current)
            for direction in _DIRECTIONS:
                found = summarize_counterparties(current, transfers, direction, classify=classify)
                for counterparty in select_counterparties(found, labels, top_n):
                    if layer == 0:
                        sample_hashes.update(counterparty.sample_hashes)
                    if counterparty.address not in seen:
                        seen.add(counterparty.address)
                        discovered.append(counterparty.address)
        for tx_hash in sample_hashes:
            chain.get_transaction(tx_hash)
        frontier = discovered
    return len(seen)


def baseline_level(chain: ChainClient, labels: LabelStore, address: str) -> RiskLevel:
    """不經 LLM 的純規則判斷：目標篩選加上所有直接交易對手的風險下限。"""
    address = address.lower()
    if rules.screen(address, labels):
        return "SEVERE"
    nodes = {address: AddressNode(address=address, depth=0, label=labels.get(address))}
    for counterparty in summarize_counterparties(address, chain.get_transfers(address)):
        nodes[counterparty.address] = AddressNode(
            address=counterparty.address,
            depth=1,
            parent=address,
            label=labels.get(counterparty.address),
        )
    return rules.risk_floor(address, nodes)
