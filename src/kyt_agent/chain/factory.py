from kyt_agent.chain.client import ChainClient
from kyt_agent.chain.etherscan import EtherscanClient
from kyt_agent.chain.snapshot import SnapshotClient
from kyt_agent.config import Settings


def make_chain_client(settings: Settings) -> ChainClient:
    snapshots = settings.data_dir / "snapshots"
    if settings.chain_mode == "replay":
        return SnapshotClient(snapshots)
    if not settings.etherscan_api_key:
        raise ValueError("ETHERSCAN_API_KEY 未設定")
    live = EtherscanClient(
        settings.etherscan_api_key,
        page_size=settings.tx_page_size,
        min_interval=settings.etherscan_min_interval,
    )
    return SnapshotClient(snapshots, live) if settings.chain_mode == "record" else live
