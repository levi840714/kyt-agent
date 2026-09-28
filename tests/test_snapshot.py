import pytest

from kyt_agent.chain.client import ContractInfo
from kyt_agent.chain.etherscan import EtherscanClient
from kyt_agent.chain.factory import make_chain_client
from kyt_agent.chain.snapshot import SnapshotClient, SnapshotMissError
from tests.fakes import MIXER, TARGET, StubChain, transfer, tx_hash


def test_record_mode_persists_and_reuses_snapshots(tmp_path):
    inner = StubChain(transfers={TARGET: [transfer(1, TARGET, MIXER)]})
    recorder = SnapshotClient(tmp_path, inner)
    assert recorder.get_transfers(TARGET) == inner.transfers[TARGET]
    assert recorder.get_transfers(TARGET) == inner.transfers[TARGET]
    assert inner.calls == [("transfers", TARGET)]
    assert (tmp_path / "transfers" / f"{TARGET}.json").exists()


def test_replay_reads_recorded_data(tmp_path):
    contract = ContractInfo(address=MIXER, is_contract=True, name="Tornado")
    SnapshotClient(tmp_path, StubChain(contracts={MIXER: contract})).get_contract_info(MIXER)
    assert SnapshotClient(tmp_path).get_contract_info(MIXER) == contract


def test_missing_transaction_is_recorded_as_none(tmp_path):
    SnapshotClient(tmp_path, StubChain()).get_transaction(tx_hash(9))
    assert SnapshotClient(tmp_path).get_transaction(tx_hash(9)) is None


def test_replay_miss_raises(tmp_path):
    with pytest.raises(SnapshotMissError):
        SnapshotClient(tmp_path).get_transfers(TARGET)


def test_factory_builds_client_for_each_mode(settings):
    assert isinstance(make_chain_client(settings), EtherscanClient)
    for mode in ("record", "replay"):
        client = make_chain_client(settings.model_copy(update={"chain_mode": mode}))
        assert isinstance(client, SnapshotClient)


def test_factory_requires_api_key_outside_replay(settings):
    with pytest.raises(ValueError, match="ETHERSCAN_API_KEY"):
        make_chain_client(settings.model_copy(update={"etherscan_api_key": ""}))
    no_key_replay = settings.model_copy(update={"etherscan_api_key": "", "chain_mode": "replay"})
    assert isinstance(make_chain_client(no_key_replay), SnapshotClient)
