from decimal import Decimal
from typing import Any

import httpx
import pytest

from kyt_agent.chain.client import ContractInfo, TransactionDetail
from kyt_agent.chain.etherscan import EtherscanClient, EtherscanError

A = "0x" + "A" * 40
B = "0x" + "B" * 40
TOKEN = "0x" + "C" * 40
EMPTY = {"status": "0", "message": "No transactions found", "result": []}


def ok(result: Any) -> dict[str, Any]:
    return {"status": "1", "message": "OK", "result": result}


def rpc(result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def native(hash_: str, value: str, is_error: str = "0") -> dict[str, str]:
    return {
        "hash": hash_,
        "timeStamp": "1700000000",
        "from": A,
        "to": B,
        "value": value,
        "isError": is_error,
        "contractAddress": "",
    }


def token(hash_: str, value: str, decimals: str, symbol: str) -> dict[str, str]:
    return {
        "hash": hash_,
        "timeStamp": "1700000001",
        "from": B,
        "to": A,
        "value": value,
        "tokenDecimal": decimals,
        "tokenSymbol": symbol,
        "contractAddress": TOKEN,
    }


def make_client(responses: dict[str, list[dict[str, Any]]]) -> tuple[EtherscanClient, list[float]]:
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["chainid"] == "1"
        assert request.url.params["apikey"] == "key"
        return httpx.Response(200, json=responses[request.url.params["action"]].pop(0))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return EtherscanClient("key", http=http, min_interval=0, sleep=sleeps.append), sleeps


def test_get_transfers_merges_native_internal_and_token_transfers():
    client, _ = make_client(
        {
            "txlist": [
                ok(
                    [
                        native("0xAA", "1500000000000000000"),
                        native("0xBB", "1000", is_error="1"),
                        native("0xCC", "0"),
                    ]
                )
            ],
            "txlistinternal": [ok([native("0xEE", "2000000000000000000")])],
            "tokentx": [ok([token("0xDD", "2500000", "6", "USDT")])],
        }
    )
    transfers = client.get_transfers(A)
    assert [(t.tx_hash, t.amount, t.asset, t.token_contract) for t in transfers] == [
        ("0xaa", Decimal("1.5"), "ETH", None),
        ("0xee", Decimal("2"), "ETH", None),
        ("0xdd", Decimal("2.5"), "USDT", TOKEN.lower()),
    ]
    assert (transfers[0].sender, transfers[0].recipient) == (A.lower(), B.lower())


def test_no_transactions_found_returns_empty_list():
    client, _ = make_client({"txlist": [EMPTY], "txlistinternal": [EMPTY], "tokentx": [EMPTY]})
    assert client.get_transfers(A) == []


def test_rate_limit_is_retried_with_backoff():
    limited = {"status": "0", "message": "NOTOK", "result": "Max rate limit reached"}
    client, sleeps = make_client(
        {
            "txlist": [limited, ok([])],
            "txlistinternal": [ok([])],
            "tokentx": [ok([])],
        }
    )
    assert client.get_transfers(A) == []
    assert sleeps == [1.0]


def test_other_errors_raise():
    invalid = {"status": "0", "message": "NOTOK", "result": "Invalid API Key"}
    client, _ = make_client({"txlist": [invalid]})
    with pytest.raises(EtherscanError, match="Invalid API Key"):
        client.get_transfers(A)


def test_contract_info_distinguishes_eoa_and_contract():
    client, _ = make_client(
        {
            "eth_getCode": [rpc("0x"), rpc("0x6080")],
            "getsourcecode": [ok([{"ContractName": "TornadoCash_Eth"}])],
        }
    )
    assert client.get_contract_info(A) == ContractInfo(address=A, is_contract=False)
    assert client.get_contract_info(B) == ContractInfo(
        address=B, is_contract=True, name="TornadoCash_Eth"
    )


def test_get_transaction_parses_hex_fields():
    client, _ = make_client(
        {
            "eth_getTransactionByHash": [
                rpc(
                    {
                        "hash": "0xAA",
                        "from": A,
                        "to": B,
                        "value": hex(10**18),
                        "blockNumber": "0x10",
                        "input": "0xa9059cbb0000",
                    }
                )
            ]
        }
    )
    assert client.get_transaction("0xaa") == TransactionDetail(
        tx_hash="0xaa",
        sender=A,
        recipient=B,
        value_eth=Decimal(1),
        block_number=16,
        method_id="0xa9059cbb",
    )


def test_get_transaction_returns_none_when_missing():
    client, _ = make_client({"eth_getTransactionByHash": [rpc(None)]})
    assert client.get_transaction("0xaa") is None


SECRET = "super-secret-key"


def flaky_client(*failures: httpx.Response | Exception) -> tuple[EtherscanClient, list[float]]:
    queue = list(failures)
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if queue:
            failure = queue.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return failure
        return httpx.Response(200, json=rpc(None))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = EtherscanClient(SECRET, http=http, min_interval=0, sleep=sleeps.append)
    return client, sleeps


def assert_sanitized(error: BaseException) -> None:
    assert SECRET not in str(error)
    assert error.__cause__ is None
    assert error.__suppress_context__


def test_server_error_is_retried_then_succeeds():
    client, sleeps = flaky_client(httpx.Response(502), httpx.Response(503))
    assert client.get_transaction("0xaa") is None
    assert sleeps == [1.0, 2.0]


def test_persistent_server_error_hides_api_key():
    client, sleeps = flaky_client(*[httpx.Response(502)] * 4)
    with pytest.raises(EtherscanError, match="HTTP 502") as caught:
        client.get_transaction("0xaa")
    assert_sanitized(caught.value)
    assert sleeps == [1.0, 2.0, 4.0]


def test_client_error_is_not_retried_and_hides_api_key():
    client, sleeps = flaky_client(httpx.Response(403))
    with pytest.raises(EtherscanError, match="HTTP 403") as caught:
        client.get_transaction("0xaa")
    assert_sanitized(caught.value)
    assert sleeps == []


def test_timeout_is_retried_and_hides_api_key():
    client, sleeps = flaky_client(*[httpx.ConnectTimeout(f"timeout apikey={SECRET}")] * 4)
    with pytest.raises(EtherscanError, match="ConnectTimeout") as caught:
        client.get_transaction("0xaa")
    assert_sanitized(caught.value)
    assert len(sleeps) == 3
